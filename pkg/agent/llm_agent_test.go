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
