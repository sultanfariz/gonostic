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
	name        string
	description string
	prompt      string
	model       ModelProvider
	tools       []Tool
	subAgents   []Agent
	maxTurns    int
}

// LLMAgentConfig holds configuration for creating an LLMAgent.
type LLMAgentConfig struct {
	Name        string
	Description string
	Prompt      string // System prompt/instruction
	Model       ModelProvider
	Tools       []Tool
	SubAgents   []Agent
	MaxTurns    int
}

// NewLLMAgent creates a new LLMAgent from the given configuration.
func NewLLMAgent(cfg LLMAgentConfig) *LLMAgent {
	if cfg.MaxTurns == 0 {
		cfg.MaxTurns = 10
	}
	return &LLMAgent{
		name:        cfg.Name,
		description: cfg.Description,
		prompt:      cfg.Prompt,
		model:       cfg.Model,
		tools:       cfg.Tools,
		subAgents:   cfg.SubAgents,
		maxTurns:    cfg.MaxTurns,
	}
}

func (a *LLMAgent) Name() string {
	return a.name
}

func (a *LLMAgent) SubAgents() []Agent {
	return a.subAgents
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

	for turn := 0; turn < a.maxTurns; turn++ {
		stepStart := time.Now()
		step := ExecutionStep{
			AgentName: a.name,
			Timestamp: stepStart,
			ToolCalls: []ToolCall{},
		}

		// Call LLM and track latency
		llmStart := time.Now()
		resp, err := a.model.Complete(ctx, task.Input, task.Files, a.tools, history)
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

		// Extract artifacts from state
		result.Artifacts = a.extractArtifacts(task.State)

		// Aggregate metrics
		result.aggregateMetrics()

		return result, nil
	}

	result.Error = "max iterations reached"
	return result, fmt.Errorf("max iterations reached")
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
