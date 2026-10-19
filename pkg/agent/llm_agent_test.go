package agent

import (
	"context"
	"errors"
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

func (t *stubTool) Name() string        { return "stub_tool" }
func (t *stubTool) Description() string { return "stub" }
func (t *stubTool) Schema() interface{} { return nil }
func (t *stubTool) Execute(_ context.Context, _ map[string]interface{}) (ToolResult, error) {
	return ToolResult{Data: "stub result"}, nil
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

func TestFormatToolResults(t *testing.T) {
	tests := []struct {
		name  string
		calls []ToolCall
		want  string
	}{
		{
			name:  "Text_set_with_data",
			calls: []ToolCall{{Name: "t", Result: ToolResult{Text: "hello", Data: "ignored"}}},
			want:  "hello",
		},
		{
			name:  "Text_empty_data_string",
			calls: []ToolCall{{Name: "t", Result: ToolResult{Data: "payload"}}},
			want:  "t result: payload",
		},
		{
			name:  "Text_empty_data_nil",
			calls: []ToolCall{{Name: "t", Result: ToolResult{}}},
			want:  "t result: ",
		},
		{
			name:  "Error_set",
			calls: []ToolCall{{Name: "t", Error: errors.New("boom")}},
			want:  "t failed: boom",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := formatToolResults(tt.calls); got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

// namedTool returns a fixed ToolResult under a given name.
type namedTool struct {
	name string
	res  ToolResult
}

func (t *namedTool) Name() string        { return t.name }
func (t *namedTool) Description() string { return "named" }
func (t *namedTool) Schema() interface{} { return nil }
func (t *namedTool) Execute(_ context.Context, _ map[string]interface{}) (ToolResult, error) {
	return t.res, nil
}

// toolThenTextProvider returns the given tool calls once, then text.
type toolThenTextProvider struct {
	calls atomic.Int32
	tools []ToolCall
}

func (p *toolThenTextProvider) Complete(_ context.Context, _ *CompletionRequest) (*ModelResponse, error) {
	if p.calls.Add(1) == 1 {
		return &ModelResponse{ToolCalls: p.tools}, nil
	}
	return &ModelResponse{Content: "done", Finished: true}, nil
}

func TestLLMAgentStateMerge(t *testing.T) {
	provider := &toolThenTextProvider{tools: []ToolCall{
		{ID: "1", Name: "map_tool", Arguments: map[string]interface{}{}},
		{ID: "2", Name: "str_tool", Arguments: map[string]interface{}{}},
	}}
	ag := NewLLMAgent(LLMAgentConfig{
		Name:   "merge-agent",
		Prompt: "test",
		Model:  provider,
		Tools: []Tool{
			&namedTool{name: "map_tool", res: ToolResult{Data: map[string]interface{}{"k": "v"}}},
			&namedTool{name: "str_tool", res: ToolResult{Data: "x"}},
		},
	})
	task := &Task{ID: "t1", Input: "go", State: map[string]interface{}{}}

	if _, err := ag.Execute(context.Background(), task); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if task.State["k"] != "v" {
		t.Errorf("State[k] = %v, want v", task.State["k"])
	}
	if task.State["str_tool_result"] != "x" {
		t.Errorf("State[str_tool_result] = %v, want x", task.State["str_tool_result"])
	}
}
