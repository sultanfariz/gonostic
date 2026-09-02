package agent

import (
	"context"
	"testing"
)

// recordingProvider captures every request it receives and replays a scripted
// response per call. Each response is a fresh value so callers can mutate the
// returned pointer without affecting later calls.
type recordingProvider struct {
	responses []ModelResponse
	requests  []CompletionRequest
}

func (p *recordingProvider) Complete(_ context.Context, req *CompletionRequest) (*ModelResponse, error) {
	p.requests = append(p.requests, *req)
	resp := p.responses[len(p.requests)-1]
	return &resp, nil
}

func (p *recordingProvider) calls() int { return len(p.requests) }

var twoPhaseSchema = map[string]interface{}{"type": "object"}

func twoPhaseTools() []Tool { return []Tool{&stubTool{}} }

func TestTwoPhaseToolCallReturnedAsIs(t *testing.T) {
	inner := &recordingProvider{responses: []ModelResponse{
		{Content: "calling", ToolCalls: []ToolCall{{Name: "stub_tool"}}, Usage: &TokenUsage{TotalTokens: 10}},
	}}
	p := WithTwoPhase(inner)

	resp, err := p.Complete(context.Background(), &CompletionRequest{
		Prompt: "hi", Tools: twoPhaseTools(), OutputSchema: twoPhaseSchema,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if inner.calls() != 1 {
		t.Fatalf("expected 1 inner call, got %d", inner.calls())
	}
	if inner.requests[0].OutputSchema != nil {
		t.Error("phase 1 must drop OutputSchema")
	}
	if len(inner.requests[0].Tools) != 1 {
		t.Error("phase 1 must keep Tools")
	}
	if len(resp.ToolCalls) != 1 || resp.Content != "calling" {
		t.Errorf("response altered: %+v", resp)
	}
	if resp.Usage.TotalTokens != 10 {
		t.Errorf("usage altered: %+v", resp.Usage)
	}
}

func TestTwoPhaseZeroToolCallsFiresSchemaTurn(t *testing.T) {
	inner := &recordingProvider{responses: []ModelResponse{
		{Content: "done researching", Usage: &TokenUsage{PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120}},
		{Content: `{"ok":true}`, Finished: true, StopReason: "stop", Reasoning: "because",
			Usage: &TokenUsage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8}},
	}}
	p := WithTwoPhase(inner)

	temp := float32(0.3)
	maxTok := 4096
	resp, err := p.Complete(context.Background(), &CompletionRequest{
		Prompt:       "hi",
		Tools:        twoPhaseTools(),
		OutputSchema: twoPhaseSchema,
		History:      []Message{{Role: "user", Content: "earlier"}},
		Temperature:  &temp,
		MaxTokens:    &maxTok,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if inner.calls() != 2 {
		t.Fatalf("expected 2 inner calls, got %d", inner.calls())
	}

	final := inner.requests[1]
	if final.Tools != nil {
		t.Error("phase 2 must drop Tools")
	}
	if final.OutputSchema == nil {
		t.Error("phase 2 must restore OutputSchema")
	}
	if final.Temperature != &temp || final.MaxTokens != &maxTok {
		t.Error("phase 2 must carry Temperature and MaxTokens")
	}
	if len(final.History) != 2 {
		t.Fatalf("expected history of 2, got %d", len(final.History))
	}
	if final.History[1].Role != "assistant" || final.History[1].Content != "done researching" {
		t.Errorf("phase 1 content not folded into history: %+v", final.History[1])
	}

	if resp.Content != `{"ok":true}` || resp.StopReason != "stop" || resp.Reasoning != "because" {
		t.Errorf("must return phase 2 response verbatim: %+v", resp)
	}
	want := TokenUsage{PromptTokens: 105, CompletionTokens: 23, TotalTokens: 128}
	if *resp.Usage != want {
		t.Errorf("usage: got %+v want %+v", *resp.Usage, want)
	}
}

func TestTwoPhaseDoesNotMutateCallerHistory(t *testing.T) {
	inner := &recordingProvider{responses: []ModelResponse{{Content: "a"}, {Content: "b"}}}
	p := WithTwoPhase(inner)

	// Spare capacity: a naive append would write into this backing array.
	history := make([]Message, 1, 4)
	history[0] = Message{Role: "user", Content: "earlier"}
	req := &CompletionRequest{Tools: twoPhaseTools(), OutputSchema: twoPhaseSchema, History: history}

	if _, err := p.Complete(context.Background(), req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(req.History) != 1 || req.OutputSchema == nil || len(req.Tools) != 1 {
		t.Errorf("caller request mutated: %+v", req)
	}
	if history[:2][1].Content != "" {
		t.Error("assistant message written into caller's backing array")
	}
}

func TestTwoPhasePassthrough(t *testing.T) {
	cases := []struct {
		name string
		req  CompletionRequest
	}{
		{"no tools", CompletionRequest{Prompt: "hi", OutputSchema: twoPhaseSchema}},
		{"no schema", CompletionRequest{Prompt: "hi", Tools: twoPhaseTools()}},
		{"neither", CompletionRequest{Prompt: "hi"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			inner := &recordingProvider{responses: []ModelResponse{{Content: "plain"}}}
			resp, err := WithTwoPhase(inner).Complete(context.Background(), &tc.req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if inner.calls() != 1 {
				t.Fatalf("expected 1 inner call, got %d", inner.calls())
			}
			got := inner.requests[0]
			if (got.OutputSchema == nil) != (tc.req.OutputSchema == nil) || len(got.Tools) != len(tc.req.Tools) {
				t.Error("decorator altered a passthrough request")
			}
			if resp.Content != "plain" {
				t.Errorf("decorator altered a passthrough response: %+v", resp)
			}
		})
	}
}

func TestSumTokenUsageNilHandling(t *testing.T) {
	a := &TokenUsage{TotalTokens: 1}
	if got := sumTokenUsage(nil, a); got != a {
		t.Error("nil first operand should return second")
	}
	if got := sumTokenUsage(a, nil); got != a {
		t.Error("nil second operand should return first")
	}
	if got := sumTokenUsage(nil, nil); got != nil {
		t.Errorf("both nil should return nil, got %+v", got)
	}
}

func TestTwoPhaseRecordsProviderCalls(t *testing.T) {
	inner := &recordingProvider{responses: []ModelResponse{
		{Content: "prose", StopReason: "stop", Usage: &TokenUsage{TotalTokens: 120}},
		{Content: `{"ok":true}`, StopReason: "end_turn", Usage: &TokenUsage{TotalTokens: 8}},
	}}

	resp, err := WithTwoPhase(inner).Complete(context.Background(), &CompletionRequest{
		Tools: twoPhaseTools(), OutputSchema: twoPhaseSchema,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Calls) != 2 {
		t.Fatalf("expected 2 provider calls, got %d", len(resp.Calls))
	}

	// Phase 1's content and stop reason are invisible on the merged response.
	// Calls is the only place they survive.
	if resp.Calls[0].Label != "two_phase:tools" || resp.Calls[0].Content != "prose" ||
		resp.Calls[0].StopReason != "stop" || resp.Calls[0].Usage.TotalTokens != 120 {
		t.Errorf("phase 1 call: %+v", resp.Calls[0])
	}
	if resp.Calls[1].Label != "two_phase:schema" || resp.Calls[1].Content != `{"ok":true}` ||
		resp.Calls[1].StopReason != "end_turn" || resp.Calls[1].Usage.TotalTokens != 8 {
		t.Errorf("phase 2 call: %+v", resp.Calls[1])
	}
	if resp.Usage.TotalTokens != 128 {
		t.Errorf("merged usage: %+v", resp.Usage)
	}
}

func TestTwoPhaseSingleCallLeavesCallsNil(t *testing.T) {
	cases := map[string]CompletionRequest{
		"tool call returned": {Tools: twoPhaseTools(), OutputSchema: twoPhaseSchema},
		"passthrough":        {Prompt: "hi"},
	}
	for name, req := range cases {
		t.Run(name, func(t *testing.T) {
			inner := &recordingProvider{responses: []ModelResponse{
				{Content: "x", ToolCalls: []ToolCall{{Name: "stub_tool"}}},
			}}
			if name == "passthrough" {
				inner.responses[0].ToolCalls = nil
			}
			resp, err := WithTwoPhase(inner).Complete(context.Background(), &req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resp.Calls != nil {
				t.Errorf("single-call response should leave Calls nil, got %+v", resp.Calls)
			}
		})
	}
}

func TestTwoPhaseDoesNotMutateInnerResponse(t *testing.T) {
	// A provider that hands back a response it also retains must not see its
	// Usage rewritten by the decorator's summing.
	shared := &ModelResponse{Content: "final", Usage: &TokenUsage{TotalTokens: 8}}
	inner := &sharedRespProvider{first: &ModelResponse{Content: "prose", Usage: &TokenUsage{TotalTokens: 120}}, second: shared}

	resp, err := WithTwoPhase(inner).Complete(context.Background(), &CompletionRequest{
		Tools: twoPhaseTools(), OutputSchema: twoPhaseSchema,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Usage.TotalTokens != 128 {
		t.Errorf("returned usage: %+v", resp.Usage)
	}
	if shared.Usage.TotalTokens != 8 {
		t.Errorf("inner response mutated: %+v", shared.Usage)
	}
	if shared.Calls != nil {
		t.Error("inner response gained Calls")
	}
}

type sharedRespProvider struct {
	first, second *ModelResponse
	n             int
}

func (p *sharedRespProvider) Complete(_ context.Context, _ *CompletionRequest) (*ModelResponse, error) {
	p.n++
	if p.n == 1 {
		return p.first, nil
	}
	return p.second, nil
}
