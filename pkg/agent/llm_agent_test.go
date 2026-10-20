package agent

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
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

// stubTool is a tool that returns fixed data and counts its calls.
type stubTool struct{ calls atomic.Int32 }

func (t *stubTool) Name() string        { return "stub_tool" }
func (t *stubTool) Description() string { return "stub" }
func (t *stubTool) Schema() interface{} { return nil }
func (t *stubTool) Execute(_ context.Context, _ map[string]interface{}) (ToolResult, error) {
	t.calls.Add(1)
	return ToolResult{Data: "stub result"}, nil
}

// namedTool returns a fixed ToolResult under a given name and counts its calls.
type namedTool struct {
	name  string
	res   ToolResult
	calls atomic.Int32
}

func (t *namedTool) Name() string        { return t.name }
func (t *namedTool) Description() string { return "named" }
func (t *namedTool) Schema() interface{} { return nil }
func (t *namedTool) Execute(_ context.Context, _ map[string]interface{}) (ToolResult, error) {
	t.calls.Add(1)
	return t.res, nil
}

// failingTool returns a result together with err and counts its calls.
type failingTool struct {
	name  string
	res   ToolResult
	err   error
	calls atomic.Int32
}

func (t *failingTool) Name() string        { return t.name }
func (t *failingTool) Description() string { return "failing" }
func (t *failingTool) Schema() interface{} { return nil }
func (t *failingTool) Execute(_ context.Context, _ map[string]interface{}) (ToolResult, error) {
	t.calls.Add(1)
	return t.res, t.err
}

// toolThenTextProvider returns the given tool calls once, then text, and counts its calls.
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

// recordingProvider returns the given tool calls once, then text. It stores a
// copy of the History slice of every request and counts its calls.
type recordingProvider struct {
	calls   atomic.Int32
	tools   []ToolCall
	history [][]Message
}

func (p *recordingProvider) Complete(_ context.Context, req *CompletionRequest) (*ModelResponse, error) {
	p.history = append(p.history, append([]Message(nil), req.History...))
	if p.calls.Add(1) == 1 {
		return &ModelResponse{ToolCalls: p.tools}, nil
	}
	return &ModelResponse{Content: "done", Finished: true}, nil
}

// errBoom is the error returned by failingTool in tests.
var errBoom = errors.New("boom")

func TestLLMAgentMaxTurnsTruncation(t *testing.T) {
	provider := &truncTestProvider{}

	ag := NewLLMAgent(LLMAgentConfig{
		Name:     "trunc-agent",
		Prompt:   "test",
		Model:    provider,
		Tools:    []Tool{&stubTool{}},
		MaxTurns: MaxTurnsConfig{Limit: 3, Graceful: true},
	})

	task := &Task{ID: "t1", Input: "go"}

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

	task := &Task{ID: "t3", Input: "go"}

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

	task := &Task{ID: "t2", Input: "go"}

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

// spec describes one fake tool; a non-nil err makes it a failingTool.
type spec struct {
	name string
	res  ToolResult
	err  error
}

func TestAgent_formatToolResults(t *testing.T) {
	tests := []struct {
		name  string
		calls []ToolCall
		want  string
	}{
		{"Success_text_wins_over_data", []ToolCall{{Name: "t", Result: ToolResult{Text: "hello", Data: "ignored"}}}, "hello"},
		{"Success_data_fallback", []ToolCall{{Name: "t", Result: ToolResult{Data: "payload"}}}, "t result: payload"},
		{"Success_nil_data", []ToolCall{{Name: "t", Result: ToolResult{}}}, "t result: "},
		{"Failed_tool_error", []ToolCall{{Name: "t", Error: errBoom}}, "t failed: boom"},
		{"Failed_error_with_result_present", []ToolCall{{Name: "t", Result: ToolResult{Text: "hello", Data: "payload"}, Error: errBoom}}, "t failed: boom"},
		{"Success_two_calls_joined_by_newline", []ToolCall{
			{Name: "a", Result: ToolResult{Text: "first"}},
			{Name: "b", Result: ToolResult{Data: 7}},
		}, "first\nb result: 7"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := formatToolResults(tt.calls); got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestLLMAgent_Execute_state(t *testing.T) {
	tests := []struct {
		name    string
		state   TaskState
		tools   []spec
		calls   []ToolCall
		want    map[string]interface{}
		wantErr error
	}{
		{
			name:  "Success_map_data_merges_keys",
			state: TaskState{data: map[string]interface{}{"seed": 1}},
			tools: []spec{{name: "map_tool", res: ToolResult{Data: map[string]interface{}{"k": "v"}}}},
			calls: []ToolCall{{ID: "1", Name: "map_tool", Arguments: map[string]interface{}{}}},
			want:  map[string]interface{}{"seed": 1, "k": "v"},
		},
		{
			name:  "Success_non_map_data_stored_under_tool_result_key",
			tools: []spec{{name: "str_tool", res: ToolResult{Data: "x"}}},
			calls: []ToolCall{{ID: "1", Name: "str_tool", Arguments: map[string]interface{}{}}},
			want:  map[string]interface{}{"str_tool_result": "x"},
		},
		{
			name:  "Success_nil_data_with_text_leaves_state_empty",
			tools: []spec{{name: "text_tool", res: ToolResult{Text: "hi"}}},
			calls: []ToolCall{{ID: "1", Name: "text_tool", Arguments: map[string]interface{}{}}},
			want:  map[string]interface{}{},
		},
		{
			name:  "Success_zero_state_task",
			state: TaskState{},
			tools: []spec{
				{name: "map_tool", res: ToolResult{Data: map[string]interface{}{"k": "v"}}},
				{name: "str_tool", res: ToolResult{Data: "x"}},
			},
			calls: []ToolCall{
				{ID: "1", Name: "map_tool", Arguments: map[string]interface{}{}},
				{ID: "2", Name: "str_tool", Arguments: map[string]interface{}{}},
			},
			want: map[string]interface{}{"k": "v", "str_tool_result": "x"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := &toolThenTextProvider{tools: tt.calls}
			var tools []Tool
			named := map[string]*namedTool{}
			for _, s := range tt.tools {
				nt := &namedTool{name: s.name, res: s.res}
				named[s.name] = nt
				tools = append(tools, nt)
			}
			ag := NewLLMAgent(LLMAgentConfig{Name: "state", Prompt: "test", Model: provider, Tools: tools})
			task := &Task{ID: "t1", Input: "go", State: tt.state}

			_, err := ag.Execute(context.Background(), task)

			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("err = %v, want %v", err, tt.wantErr)
			}
			if got := provider.calls.Load(); got != 2 {
				t.Errorf("provider calls = %d, want 2", got)
			}
			for name, nt := range named {
				if got := nt.calls.Load(); got != 1 {
					t.Errorf("tool %s calls = %d, want 1", name, got)
				}
			}
			got := map[string]interface{}{}
			task.State.Range(func(k string, v interface{}) { got[k] = v })
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("state = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLLMAgent_Execute_toolResultParts(t *testing.T) {
	png1 := Part{Type: "image/png", Data: []byte{1}}
	jpg2 := Part{Type: "image/jpeg", Data: []byte{2}}
	notFound := errors.New("tool not found: ghost")

	tests := []struct {
		name string
		// tools are the fakes registered with the agent; calls may name tools not listed here.
		tools []spec
		calls []ToolCall
		// wantToolCalls is the expected Execute count per registered tool.
		wantToolCalls map[string]int32
		// wantCallErrs is the expected Error per model tool call, by call ID.
		wantCallErrs map[string]error
		want         Message
	}{
		{
			name:          "Success_one_png_part",
			tools:         []spec{{name: "snap", res: ToolResult{Text: "frame ok", Parts: []Part{{Type: "image/png", Data: []byte{1, 2, 3}}}}}},
			calls:         []ToolCall{{ID: "1", Name: "snap", Arguments: map[string]interface{}{}}},
			wantToolCalls: map[string]int32{"snap": 1},
			wantCallErrs:  map[string]error{"1": nil},
			want:          Message{Role: "user", Content: "frame ok", Parts: []Part{{Type: "image/png", Data: []byte{1, 2, 3}}}},
		},
		{
			name: "Success_parts_keep_call_order",
			tools: []spec{
				{name: "a", res: ToolResult{Text: "a ok", Parts: []Part{png1}}},
				{name: "b", res: ToolResult{Text: "b ok", Parts: []Part{jpg2}}},
			},
			calls: []ToolCall{
				{ID: "1", Name: "a", Arguments: map[string]interface{}{}},
				{ID: "2", Name: "b", Arguments: map[string]interface{}{}},
			},
			wantToolCalls: map[string]int32{"a": 1, "b": 1},
			wantCallErrs:  map[string]error{"1": nil, "2": nil},
			want:          Message{Role: "user", Content: "a ok\nb ok", Parts: []Part{png1, jpg2}},
		},
		{
			name:          "Failed_tool_adds_no_parts",
			tools:         []spec{{name: "bad", res: ToolResult{Parts: []Part{png1}}, err: errBoom}},
			calls:         []ToolCall{{ID: "1", Name: "bad", Arguments: map[string]interface{}{}}},
			wantToolCalls: map[string]int32{"bad": 1},
			wantCallErrs:  map[string]error{"1": errBoom},
			want:          Message{Role: "user", Content: "bad failed: boom", Parts: nil},
		},
		{
			name:          "Failed_tool_not_found_adds_no_parts",
			calls:         []ToolCall{{ID: "1", Name: "ghost", Arguments: map[string]interface{}{}}},
			wantToolCalls: map[string]int32{},
			wantCallErrs:  map[string]error{"1": notFound},
			want:          Message{Role: "user", Content: "ghost failed: tool not found: ghost", Parts: nil},
		},
		{
			name: "Success_mixed_turn_failed_call_skipped",
			tools: []spec{
				{name: "a", res: ToolResult{Text: "a ok", Parts: []Part{png1}}},
				{name: "bad", res: ToolResult{Parts: []Part{{Type: "image/gif", Data: []byte{9}}}}, err: errBoom},
				{name: "b", res: ToolResult{Text: "b ok", Parts: []Part{jpg2}}},
			},
			calls: []ToolCall{
				{ID: "1", Name: "a", Arguments: map[string]interface{}{}},
				{ID: "2", Name: "bad", Arguments: map[string]interface{}{}},
				{ID: "3", Name: "ghost", Arguments: map[string]interface{}{}},
				{ID: "4", Name: "b", Arguments: map[string]interface{}{}},
			},
			wantToolCalls: map[string]int32{"a": 1, "bad": 1, "b": 1},
			wantCallErrs:  map[string]error{"1": nil, "2": errBoom, "3": notFound, "4": nil},
			want: Message{
				Role:    "user",
				Content: "a ok\nbad failed: boom\nghost failed: tool not found: ghost\nb ok",
				Parts:   []Part{png1, jpg2},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := &recordingProvider{tools: tt.calls}
			var tools []Tool
			counts := map[string]func() int32{}
			for _, s := range tt.tools {
				if s.err != nil {
					ft := &failingTool{name: s.name, res: s.res, err: s.err}
					tools = append(tools, ft)
					counts[s.name] = ft.calls.Load
				} else {
					nt := &namedTool{name: s.name, res: s.res}
					tools = append(tools, nt)
					counts[s.name] = nt.calls.Load
				}
			}
			ag := NewLLMAgent(LLMAgentConfig{Name: "rec", Prompt: "test", Model: provider, Tools: tools})
			task := &Task{ID: "t1", Input: "go"}

			if _, err := ag.Execute(context.Background(), task); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if got := provider.calls.Load(); got != 2 {
				t.Fatalf("provider calls = %d, want 2", got)
			}
			for name, want := range tt.wantToolCalls {
				if got := counts[name](); got != want {
					t.Errorf("tool %s calls = %d, want %d", name, got, want)
				}
			}
			// The provider shares its ToolCall slice with the agent, so Error is visible here.
			for _, tc := range provider.tools {
				wantErr := tt.wantCallErrs[tc.ID]
				if errors.Is(wantErr, errBoom) || wantErr == nil {
					if !errors.Is(tc.Error, wantErr) {
						t.Errorf("call %s Error = %v, want %v", tc.ID, tc.Error, wantErr)
					}
				} else if tc.Error == nil || tc.Error.Error() != wantErr.Error() {
					t.Errorf("call %s Error = %v, want %v", tc.ID, tc.Error, wantErr)
				}
			}

			first, second := provider.history[0], provider.history[1]
			if len(second) != len(first)+2 {
				t.Fatalf("second request has %d messages, want %d", len(second), len(first)+2)
			}
			if !reflect.DeepEqual(second[:len(first)], first) {
				t.Errorf("earlier messages changed: got %+v, want %+v", second[:len(first)], first)
			}
			if got := second[len(second)-1]; !reflect.DeepEqual(got, tt.want) {
				t.Errorf("last message = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestToolResult_Parts_json(t *testing.T) {
	tests := []struct {
		name         string
		call         ToolCall
		wantContains []string
		wantAbsent   []string
		wantErr      error
	}{
		{
			name:         "Success_parts_not_serialised",
			call:         ToolCall{Name: "snap", Result: ToolResult{Text: "ok", Parts: []Part{{Type: "image/png", Data: []byte{1, 2, 3}}}}},
			wantContains: []string{`"Text":"ok"`},
			wantAbsent:   []string{"Parts", "image/png", "AQID"},
		},
		{
			name:         "Success_text_and_data_serialised",
			call:         ToolCall{Name: "snap", Result: ToolResult{Text: "frame ok", Data: "payload-42", Parts: []Part{{Type: "image/png", Data: []byte{1, 2, 3}}}}},
			wantContains: []string{`"Text":"frame ok"`, `"Data":"payload-42"`},
			wantAbsent:   []string{"Parts", "AQID"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, err := json.Marshal(tt.call)

			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("err = %v, want %v", err, tt.wantErr)
			}
			for _, s := range tt.wantContains {
				if !strings.Contains(string(b), s) {
					t.Errorf("json %s does not contain %s", b, s)
				}
			}
			for _, s := range tt.wantAbsent {
				if strings.Contains(string(b), s) {
					t.Errorf("json %s contains %s", b, s)
				}
			}
		})
	}
}
