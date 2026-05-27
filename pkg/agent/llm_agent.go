package agent

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// LLMAgent is a reasoning agent powered by an LLM. It iteratively calls the
// model, executes tool calls, and can delegate to sub-agents.
type LLMAgent struct {
	name         string
	description  string
	prompt       string
	outputSchema map[string]interface{} // JSON schema for structured output
	model        ModelProvider
	tools        []Tool
	subAgents    []Agent
	maxTurns         int
	twoPhase         bool
	gracefulMaxTurns bool
}

// MaxTurnsConfig controls the turn limit and what happens when it is reached.
type MaxTurnsConfig struct {
	Limit   int  // maximum number of turns (0 = default 10)
	Graceful bool // if true, strip tools on the final turn and return Success=true with Truncated=true instead of an error
}

// LLMAgentConfig holds configuration for creating an LLMAgent.
type LLMAgentConfig struct {
	Name         string
	Description  string
	Prompt       string                 // System prompt/instruction
	OutputSchema map[string]interface{} // JSON schema for structured output (optional)
	Model        ModelProvider
	Tools        []Tool
	SubAgents    []Agent
	MaxTurns     MaxTurnsConfig
	// TwoPhase enables two-phase execution when both Tools and OutputSchema are set.
	// Phase 1: real tools run freely (OutputSchema suppressed).
	// Phase 2: one extra LLM call with OutputSchema forced and no real tools.
	// Use when your provider enforces structured output via a forced tool call,
	// which would otherwise block real tools from being called.
	// Default false — both fields sent every turn (standard behavior).
	TwoPhase bool
}

// NewLLMAgent creates a new LLMAgent from the given configuration.
func NewLLMAgent(cfg LLMAgentConfig) *LLMAgent {
	if cfg.MaxTurns.Limit == 0 {
		cfg.MaxTurns.Limit = 10
	}
	return &LLMAgent{
		name:             cfg.Name,
		description:      cfg.Description,
		prompt:           cfg.Prompt,
		outputSchema:     cfg.OutputSchema,
		model:            cfg.Model,
		tools:            cfg.Tools,
		subAgents:        cfg.SubAgents,
		maxTurns:         cfg.MaxTurns.Limit,
		twoPhase:         cfg.TwoPhase,
		gracefulMaxTurns: cfg.MaxTurns.Graceful,
	}
}

func (a *LLMAgent) Name() string {
	return a.name
}

func (a *LLMAgent) SubAgents() []Agent {
	return a.subAgents
}

func (a *LLMAgent) OutputSchema() map[string]interface{} {
	return a.outputSchema
}

func (a *LLMAgent) Execute(ctx context.Context, task *Task) (*Result, error) {
	result := &Result{
		TaskID:   task.ID,
		Success:  false,
		Metadata: make(map[string]interface{}),
		Steps:    []ExecutionStep{},
	}

	// Build initial prompt with state injection
	systemPrompt := a.injectState(task.State)

	// Build user message with files
	userMsg := Message{
		Role:    "user",
		Content: task.Input,
		Parts:   []Part{},
	}

	// Add file parts to user message
	for _, file := range task.Files {
		userMsg.Parts = append(userMsg.Parts, Part{
			Type: file.Type,
			Data: file.Content,
		})
	}

	history := []Message{
		{Role: "system", Content: systemPrompt},
		userMsg,
	}

	// dualMode activates only when TwoPhase is explicitly enabled (opt-in) and the agent
	// has both real tools and an output schema. Set TwoPhase:true in LLMAgentConfig when
	// your provider cannot handle tools and structured output in the same turn — e.g.
	// when structured output is enforced via a forced tool call that blocks real tools.
	dualMode := a.twoPhase && len(a.tools) > 0 && a.outputSchema != nil

	for turn := 0; turn < a.maxTurns; turn++ {
		stepStart := time.Now()
		step := ExecutionStep{
			AgentName: a.name,
			Timestamp: stepStart,
			ToolCalls: []ToolCall{},
		}

		// Call LLM and track latency
		llmStart := time.Now()

		// Build completion request
		req := &CompletionRequest{
			Prompt:       task.Input,
			Files:        task.Files,
			Tools:        a.tools,
			History:      history,
			OutputSchema: a.outputSchema,
		}

		// In dual-mode, suppress OutputSchema so the provider uses AUTO tool
		// choice and real tools are reachable. OutputSchema is restored in the
		// dedicated structured-output turn fired after tool work is complete.
		if dualMode {
			req.OutputSchema = nil
		}

		// Add temperature from config if available
		if task.Config != nil && task.Config.Temperature > 0 {
			req.Temperature = &task.Config.Temperature
		}

		// On the final turn, strip tools so the model cannot make further tool calls.
		if a.gracefulMaxTurns && turn == a.maxTurns-1 {
			req.Tools = nil
		}

		resp, err := a.model.Complete(ctx, req)
		step.LLMLatency = time.Since(llmStart)
		if err != nil {
			step.Error = err.Error()
			step.Duration = time.Since(stepStart)
			result.Steps = append(result.Steps, step)
			result.Error = fmt.Sprintf("LLM error: %v", err)
			return result, err
		}

		// Record token usage from response
		step.TokenUsage = resp.Usage

		step.Action = "reasoning"
		step.Output = resp.Content

		// Handle tool calls
		if len(resp.ToolCalls) > 0 {
			step.Action = "tool_execution"
			var totalToolsLatency time.Duration

			for i := range resp.ToolCalls {
				tc := &resp.ToolCalls[i]
				tool := a.findTool(tc.Name)

				if tool == nil {
					tc.Error = fmt.Errorf("tool not found: %s", tc.Name)
					continue
				}

				tcStart := time.Now()
				tcResult, tcErr := tool.Execute(ctx, tc.Arguments)
				tc.Duration = time.Since(tcStart)
				totalToolsLatency += tc.Duration
				tc.Result = tcResult
				tc.Error = tcErr

				// Update task state with result
				if tcErr == nil && tcResult != nil {
					if resultMap, ok := tcResult.(map[string]interface{}); ok {
						for k, v := range resultMap {
							task.State[k] = v
						}
					} else {
						task.State[tc.Name+"_result"] = tcResult
					}
				}

				step.ToolCalls = append(step.ToolCalls, *tc)
			}
			step.ToolsLatency = totalToolsLatency

			// Add results to conversation
			history = append(history, Message{
				Role:    "assistant",
				Content: formatToolCalls(resp.ToolCalls),
			})
			history = append(history, Message{
				Role:    "user",
				Content: formatToolResults(resp.ToolCalls),
			})

			step.Duration = time.Since(stepStart)
			result.Steps = append(result.Steps, step)
			continue
		}

		// No tool calls returned — LLM has finished tool-based research.
		// In dual-mode, fire one final structured-output turn: real tools removed,
		// OutputSchema restored so the provider forces produce_output.
		if dualMode {
			step.Duration = time.Since(stepStart)
			result.Steps = append(result.Steps, step)

			finalStepStart := time.Now()
			finalStep := ExecutionStep{
				AgentName: a.name,
				Action:    "structured_output",
				Timestamp: finalStepStart,
				ToolCalls: []ToolCall{},
			}

			finalReq := &CompletionRequest{
				Prompt:       task.Input,
				Files:        task.Files,
				Tools:        nil,
				History:      history,
				OutputSchema: a.outputSchema,
			}
			if task.Config != nil && task.Config.Temperature > 0 {
				finalReq.Temperature = &task.Config.Temperature
			}

			finalLLMStart := time.Now()
			finalResp, finalErr := a.model.Complete(ctx, finalReq)
			finalStep.LLMLatency = time.Since(finalLLMStart)
			if finalErr != nil {
				finalStep.Error = finalErr.Error()
				finalStep.Duration = time.Since(finalStepStart)
				result.Steps = append(result.Steps, finalStep)
				result.Error = fmt.Sprintf("LLM error on structured output turn: %v", finalErr)
				return result, finalErr
			}

			finalStep.TokenUsage = finalResp.Usage
			finalStep.Output = finalResp.Content
			finalStep.Duration = time.Since(finalStepStart)
			result.Steps = append(result.Steps, finalStep)

			result.Output = finalResp.Content
			result.Success = true
			if a.gracefulMaxTurns && turn == a.maxTurns-1 {
				result.Truncated = true
			}
			result.Artifacts = a.extractArtifacts(task.State)
			result.aggregateMetrics()
			return result, nil
		}

		// Check for sub-agent delegation
		if strings.Contains(strings.ToLower(resp.Content), "delegate to") {
			for _, sub := range a.subAgents {
				if strings.Contains(strings.ToLower(resp.Content), strings.ToLower(sub.Name())) {
					step.Action = "delegate"
					step.Output = fmt.Sprintf("Delegating to %s", sub.Name())
					step.Duration = time.Since(stepStart)
					result.Steps = append(result.Steps, step)

					// Execute sub-agent
					subResult, subErr := sub.Execute(ctx, task)
					if subErr != nil {
						result.Error = fmt.Sprintf("sub-agent failed: %v", subErr)
						return result, subErr
					}

					// Merge results
					result.Steps = append(result.Steps, subResult.Steps...)
					result.Output = subResult.Output
					result.Artifacts = subResult.Artifacts
					result.Success = subResult.Success
					return result, nil
				}
			}
		}

		// Task complete
		step.Duration = time.Since(stepStart)
		result.Steps = append(result.Steps, step)
		result.Output = resp.Content
		result.Success = true
		if turn == a.maxTurns-1 {
			result.Truncated = true
		}

		// Extract artifacts from state
		result.Artifacts = a.extractArtifacts(task.State)

		// Aggregate metrics
		result.aggregateMetrics()

		return result, nil
	}

	if !a.gracefulMaxTurns {
		result.Error = "max iterations reached"
		return result, fmt.Errorf("max iterations reached")
	}
	if len(result.Steps) > 0 {
		result.Output = result.Steps[len(result.Steps)-1].Output
	}
	result.Truncated = true
	result.Success = true
	result.aggregateMetrics()
	return result, nil
}

func (a *LLMAgent) injectState(state map[string]interface{}) string {
	prompt := a.prompt
	for key, val := range state {
		placeholder := fmt.Sprintf("{%s}", key)
		prompt = strings.ReplaceAll(prompt, placeholder, fmt.Sprint(val))
	}
	return prompt
}

func (a *LLMAgent) findTool(name string) Tool {
	for _, t := range a.tools {
		if t.Name() == name {
			return t
		}
	}
	return nil
}

func (a *LLMAgent) extractArtifacts(state map[string]interface{}) []Artifact {
	var artifacts []Artifact

	// Look for known artifact patterns in state
	for key, val := range state {
		if strings.HasPrefix(key, "artifact_") ||
			strings.HasSuffix(key, "_content") ||
			strings.HasSuffix(key, "_output") {

			artifact := Artifact{
				Type:     inferType(key, val),
				Content:  val,
				Metadata: map[string]interface{}{"key": key},
			}
			artifacts = append(artifacts, artifact)
		}
	}

	return artifacts
}

func inferType(key string, val interface{}) string {
	switch val.(type) {
	case string:
		if strings.Contains(strings.ToLower(key), "image") {
			return "image"
		}
		if strings.Contains(strings.ToLower(key), "video") {
			return "video"
		}
		return "text"
	default:
		return "unknown"
	}
}

func formatToolCalls(calls []ToolCall) string {
	var parts []string
	for _, tc := range calls {
		parts = append(parts, fmt.Sprintf("Calling: %s(%v)", tc.Name, tc.Arguments))
	}
	return strings.Join(parts, "\n")
}

func formatToolResults(calls []ToolCall) string {
	var parts []string
	for _, tc := range calls {
		if tc.Error != nil {
			parts = append(parts, fmt.Sprintf("%s failed: %v", tc.Name, tc.Error))
		} else {
			parts = append(parts, fmt.Sprintf("%s result: %v", tc.Name, tc.Result))
		}
	}
	return strings.Join(parts, "\n")
}

// aggregateMetrics aggregates token usage and latencies from all steps.
func (r *Result) aggregateMetrics() {
	for _, step := range r.Steps {
		r.TotalLLMLatency += step.LLMLatency
		r.TotalToolsLatency += step.ToolsLatency

		if step.TokenUsage != nil {
			r.TotalTokenUsage.PromptTokens += step.TokenUsage.PromptTokens
			r.TotalTokenUsage.CompletionTokens += step.TokenUsage.CompletionTokens
			r.TotalTokenUsage.TotalTokens += step.TokenUsage.TotalTokens
		}
	}
}
