package agent

import (
	"context"
	"sync/atomic"
	"testing"
)

// truncTestProvider returns tool calls when tools are present, text when stripped.
type truncTestProvider struct {
	calls atomic.Int32
}

func (p *truncTestProvider) Complete(_ context.Context, req *CompletionRequest) (*ModelResponse, error) {
	p.calls.Add(1)
	if len(req.Tools) > 0 {
		return &ModelResponse{
			Content: "I'll call the tool",
			ToolCalls: []ToolCall{
				{ID: "1", Name: "stub_tool", Arguments: map[string]interface{}{}},
			},
		}, nil
	}
	return &ModelResponse{Content: "final summary", Finished: true}, nil
}

// textOnlyProvider always returns text with no tool calls.
type textOnlyProvider struct{}

func (p *textOnlyProvider) Complete(_ context.Context, _ *CompletionRequest) (*ModelResponse, error) {
	return &ModelResponse{Content: "done", Finished: true}, nil
}

// stubTool satisfies the Tool interface.
type stubTool struct{}

func (t *stubTool) Name() string                                                    { return "stub_tool" }
func (t *stubTool) Description() string                                             { return "stub" }
func (t *stubTool) Schema() interface{}                                             { return nil }
func (t *stubTool) Execute(_ context.Context, _ map[string]interface{}) (interface{}, error) {
	return "stub result", nil
}

func TestLLMAgentMaxTurnsTruncation(t *testing.T) {
	provider := &truncTestProvider{}

	ag := NewLLMAgent(LLMAgentConfig{
		Name:     "trunc-agent",
		Prompt:   "test",
		Model:    provider,
		Tools:    []Tool{&stubTool{}},
		MaxTurns: MaxTurnsConfig{Limit: 3, Graceful: true},
	})

	task := &Task{ID: "t1", Input: "go", State: map[string]interface{}{}}

	result, err := ag.Execute(context.Background(), task)

	if err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if !result.Success {
		t.Error("expected Success=true")
	}
	if !result.Truncated {
		t.Error("expected Truncated=true")
	}
	if result.Output != "final summary" {
		t.Errorf("expected output 'final summary', got: %v", result.Output)
	}
	if len(result.Steps) != 3 {
		t.Errorf("expected 3 steps, got %d", len(result.Steps))
	}
	if int(provider.calls.Load()) != 3 {
		t.Errorf("expected 3 provider calls, got %d", provider.calls.Load())
	}
}

func TestLLMAgentMaxTurnsError(t *testing.T) {
	provider := &truncTestProvider{}

	ag := NewLLMAgent(LLMAgentConfig{
		Name:     "error-agent",
		Prompt:   "test",
		Model:    provider,
		Tools:    []Tool{&stubTool{}},
		MaxTurns: MaxTurnsConfig{Limit: 3}, // Graceful omitted — original error behavior expected
	})

	task := &Task{ID: "t3", Input: "go", State: map[string]interface{}{}}

	result, err := ag.Execute(context.Background(), task)

	if err == nil {
		t.Fatal("expected an error when GracefulMaxTurns is false")
	}
	if result.Success {
		t.Error("expected Success=false")
	}
	if result.Truncated {
		t.Error("expected Truncated=false")
	}
}

func TestLLMAgentNormalCompletion(t *testing.T) {
	ag := NewLLMAgent(LLMAgentConfig{
		Name:     "normal-agent",
		Prompt:   "test",
		Model:    &textOnlyProvider{},
		MaxTurns: MaxTurnsConfig{Limit: 5},
	})

	task := &Task{ID: "t2", Input: "go", State: map[string]interface{}{}}

	result, err := ag.Execute(context.Background(), task)

	if err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if !result.Success {
		t.Error("expected Success=true")
	}
	if result.Truncated {
		t.Error("expected Truncated=false for normal completion")
	}
	if len(result.Steps) != 1 {
		t.Errorf("expected 1 step, got %d", len(result.Steps))
	}
}

// callsProvider returns a response carrying a ProviderCalls breakdown, as a
// provider that fanned out internally would.
type callsProvider struct {
	gotSchema bool
	gotTools  int
}

func (p *callsProvider) Complete(_ context.Context, req *CompletionRequest) (*ModelResponse, error) {
	p.gotSchema = req.OutputSchema != nil
	p.gotTools = len(req.Tools)
	return &ModelResponse{
		Content:    `{"ok":true}`,
		Finished:   true,
		StopReason: "end_turn",
		Usage:      &TokenUsage{TotalTokens: 128},
		Calls: []ProviderCall{
			{Label: "two_phase:tools", Usage: &TokenUsage{TotalTokens: 120}},
			{Label: "two_phase:schema", Usage: &TokenUsage{TotalTokens: 8}},
		},
	}, nil
}

func TestLLMAgentPropagatesProviderCalls(t *testing.T) {
	provider := &callsProvider{}
	agent := NewLLMAgent(LLMAgentConfig{
		Name:         "test",
		Model:        provider,
		Tools:        []Tool{&stubTool{}},
		OutputSchema: map[string]interface{}{"type": "object"},
	})

	result, err := agent.Execute(context.Background(), &Task{ID: "t1", State: map[string]interface{}{}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Steps) != 1 {
		t.Fatalf("expected 1 step, got %d", len(result.Steps))
	}
	calls := result.Steps[0].ProviderCalls
	if len(calls) != 2 {
		t.Fatalf("expected step to carry 2 provider calls, got %d", len(calls))
	}
	if calls[0].Label != "two_phase:tools" || calls[1].Label != "two_phase:schema" {
		t.Errorf("labels not propagated: %+v", calls)
	}
	if result.TotalTokenUsage.TotalTokens != 128 {
		t.Errorf("step usage should stay the merged total, got %+v", result.TotalTokenUsage)
	}
}

// TestLLMAgentSendsToolsAndSchemaTogether pins the behavior change from
// removing dualMode: the agent no longer suppresses OutputSchema when tools
// are present. Providers that cannot serve both must be wrapped in
// WithTwoPhase instead.
func TestLLMAgentSendsToolsAndSchemaTogether(t *testing.T) {
	provider := &callsProvider{}
	agent := NewLLMAgent(LLMAgentConfig{
		Name:         "test",
		Model:        provider,
		Tools:        []Tool{&stubTool{}},
		OutputSchema: map[string]interface{}{"type": "object"},
	})

	if _, err := agent.Execute(context.Background(), &Task{ID: "t1", State: map[string]interface{}{}}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !provider.gotSchema {
		t.Error("OutputSchema must reach the provider unsuppressed")
	}
	if provider.gotTools != 1 {
		t.Errorf("Tools must reach the provider, got %d", provider.gotTools)
	}
}
