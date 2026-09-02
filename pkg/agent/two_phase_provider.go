package agent

import (
	"context"
	"time"
)

// twoPhaseProvider decorates a ModelProvider that cannot honor Tools and
// OutputSchema in the same request.
//
// It holds no state — the decision is made per Complete call — so it is safe
// for concurrent use if inner is. Unlike LLMAgentConfig.TwoPhase, which
// applies to whichever provider serves a turn, this is scoped to one specific
// provider instance: wrap only the leaf provider that needs it before it goes
// into a TieredProvider.
type twoPhaseProvider struct {
	inner ModelProvider
}

// WithTwoPhase wraps a ModelProvider that cannot honor both Tools and
// OutputSchema in one request. When a request carries both, it drops the
// schema, lets the model call tools, and — only once a turn comes back with
// zero tool calls — fires one more call to inner with tools removed and the
// schema restored, folding the dropped-schema turn's own content into history
// as an assistant message.
//
// Requests without tools, or without a schema, pass through unchanged.
func WithTwoPhase(inner ModelProvider) ModelProvider {
	return &twoPhaseProvider{inner: inner}
}

func (p *twoPhaseProvider) Complete(ctx context.Context, req *CompletionRequest) (*ModelResponse, error) {
	if len(req.Tools) == 0 || req.OutputSchema == nil {
		return p.inner.Complete(ctx, req)
	}

	// Phase 1: drop the schema so real tools stay reachable.
	toolReq := *req
	toolReq.OutputSchema = nil

	toolStart := time.Now()
	toolResp, err := p.inner.Complete(ctx, &toolReq)
	toolLatency := time.Since(toolStart)
	if err != nil {
		return nil, err
	}
	if len(toolResp.ToolCalls) > 0 {
		return toolResp, nil
	}

	// Phase 2: model stopped calling tools and would hand back unenforced
	// prose. Re-ask with tools removed and the schema restored.
	finalReq := *req
	finalReq.Tools = nil
	// Fresh slice: req.History may have spare capacity, and a TieredProvider
	// hands the same req to later providers.
	finalReq.History = append(append([]Message{}, req.History...), Message{
		Role:    "assistant",
		Content: toolResp.Content,
	})

	finalStart := time.Now()
	finalResp, err := p.inner.Complete(ctx, &finalReq)
	finalLatency := time.Since(finalStart)
	if err != nil {
		return nil, err
	}

	// Two calls happened for one turn, but the caller (LLMAgent) records only
	// the single ModelResponse returned here. Without summing, phase 1's
	// tokens vanish from execution steps and cost tracking. Calls keeps the
	// per-call split that the summed totals would otherwise hide.
	out := *finalResp
	out.Usage = sumTokenUsage(toolResp.Usage, finalResp.Usage)
	out.Calls = []ProviderCall{
		{
			Label:      "two_phase:tools",
			Latency:    toolLatency,
			Usage:      toolResp.Usage,
			StopReason: toolResp.StopReason,
			Content:    toolResp.Content,
		},
		{
			Label:      "two_phase:schema",
			Latency:    finalLatency,
			Usage:      finalResp.Usage,
			StopReason: finalResp.StopReason,
			Content:    finalResp.Content,
		},
	}
	return &out, nil
}

func sumTokenUsage(a, b *TokenUsage) *TokenUsage {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	return &TokenUsage{
		PromptTokens:     a.PromptTokens + b.PromptTokens,
		CompletionTokens: a.CompletionTokens + b.CompletionTokens,
		TotalTokens:      a.TotalTokens + b.TotalTokens,
	}
}
