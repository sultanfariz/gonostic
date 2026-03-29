package agent

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// RetryConfig controls retry and fallback behavior for TieredProvider.
type RetryConfig struct {
	MaxRetries              int           // Per-provider retry count. 0 = one attempt, no retries. Default: 2.
	InitialDelay            time.Duration // Delay before first retry. Default: 500ms.
	BackoffMultiplier       float64       // Multiplier applied to delay after each retry. Default: 2.0.
	MaxDelay                time.Duration // Cap on backoff delay. Default: 10s.
	MaxTiers                int           // Max tiers to attempt. 0 = unlimited (try all tiers). Default: 0.
	SkipTiersOnNonRetryable bool          // If true, fall through to the next tier on non-retryable error. Default: false.
}

// DefaultRetryConfig returns sensible defaults for retry behavior.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:        2,
		InitialDelay:      500 * time.Millisecond,
		BackoffMultiplier: 2.0,
		MaxDelay:          10 * time.Second,
		MaxTiers:          0,
	}
}

// Attempt records a single call to a provider during TieredProvider.Complete().
type Attempt struct {
	Tier          int           // 0-indexed tier
	ProviderIndex int           // 0-indexed provider within tier
	Retry         int           // Which retry this was (0 = first attempt)
	Duration      time.Duration // Wall-clock time for this attempt
	Error         error         // nil on success
}

// nonRetryableErr wraps a non-retryable provider error so Complete() can
// distinguish it from ordinary errors and stop immediately.
type nonRetryableErr struct{ err error }

func (e *nonRetryableErr) Error() string { return e.err.Error() }
func (e *nonRetryableErr) Unwrap() error { return e.err }

// TieredProvider implements ModelProvider with tiered fallback and per-provider retry.
// Providers are organized in tiers (priority levels). Tier 0 is tried first.
// Within a tier, providers are tried left-to-right. Each provider is retried
// with exponential backoff before moving to the next.
type TieredProvider struct {
	tiers    [][]ModelProvider
	config   RetryConfig
	mu       sync.Mutex
	attempts []Attempt
}

// NewTieredProvider creates a TieredProvider with the given tiers and config.
// Each inner slice is a tier of equivalent providers tried left-to-right.
// Tiers are tried in order (index 0 first, then 1, etc.).
func NewTieredProvider(tiers [][]ModelProvider, config RetryConfig) *TieredProvider {
	return &TieredProvider{
		tiers:  tiers,
		config: config,
	}
}

// NewTieredProviderWithDefaults creates a TieredProvider using DefaultRetryConfig.
func NewTieredProviderWithDefaults(tiers [][]ModelProvider) *TieredProvider {
	return NewTieredProvider(tiers, DefaultRetryConfig())
}

// NewFallbackProvider is a convenience for a single tier with multiple providers
// (no tiering, just left-to-right fallback within one priority level).
func NewFallbackProvider(providers []ModelProvider, config RetryConfig) *TieredProvider {
	return NewTieredProvider([][]ModelProvider{providers}, config)
}

// NewRetryProvider is a convenience for a single provider with retries (no fallback).
func NewRetryProvider(provider ModelProvider, config RetryConfig) *TieredProvider {
	return NewTieredProvider([][]ModelProvider{{provider}}, config)
}

// Attempts returns a copy of the attempt log from the most recent Complete() call.
// Returns nil if Complete() has not been called yet.
func (tp *TieredProvider) Attempts() []Attempt {
	tp.mu.Lock()
	defer tp.mu.Unlock()
	if tp.attempts == nil {
		return nil
	}
	out := make([]Attempt, len(tp.attempts))
	copy(out, tp.attempts)
	return out
}

// Complete implements ModelProvider. It tries providers across tiers with
// per-provider retry and exponential backoff.
func (tp *TieredProvider) Complete(ctx context.Context, req *CompletionRequest) (*ModelResponse, error) {
	tp.mu.Lock()
	tp.attempts = nil
	tp.mu.Unlock()

	tierLimit := len(tp.tiers)
	if tp.config.MaxTiers > 0 && tp.config.MaxTiers < tierLimit {
		tierLimit = tp.config.MaxTiers
	}

	var lastErr error

	for tierIdx := 0; tierIdx < tierLimit; tierIdx++ {
		tier := tp.tiers[tierIdx]
		if len(tier) == 0 {
			continue
		}

		for provIdx, provider := range tier {
			resp, err := tp.attemptWithRetry(ctx, provider, req, tierIdx, provIdx)
			if err == nil {
				return resp, nil
			}

			// Check for non-retryable error signal
			var nre *nonRetryableErr
			if errors.As(err, &nre) {
				if !tp.config.SkipTiersOnNonRetryable {
					// Stop immediately — return the original error intact
					return nil, nre.err
				}
				// Skip remaining providers in this tier, fall through to next tier
				lastErr = nre.err
				break
			}

			lastErr = err

			if ctx.Err() != nil {
				return nil, tp.wrapError(lastErr)
			}
		}
	}

	if lastErr == nil {
		return nil, fmt.Errorf("tiered provider: no providers configured")
	}
	return nil, tp.wrapError(lastErr)
}

func (tp *TieredProvider) attemptWithRetry(ctx context.Context, provider ModelProvider, req *CompletionRequest, tierIdx, provIdx int) (*ModelResponse, error) {
	totalAttempts := tp.config.MaxRetries + 1
	delay := tp.config.InitialDelay

	var lastErr error

	for attempt := range totalAttempts {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		start := time.Now()
		resp, err := provider.Complete(ctx, req)
		duration := time.Since(start)

		tp.mu.Lock()
		tp.attempts = append(tp.attempts, Attempt{
			Tier:          tierIdx,
			ProviderIndex: provIdx,
			Retry:         attempt,
			Duration:      duration,
			Error:         err,
		})
		tp.mu.Unlock()

		if err == nil {
			return resp, nil
		}
		lastErr = err

		// Stop immediately if the error signals it is not retryable
		var re RetryableError
		if errors.As(err, &re) && !re.Retryable() {
			return nil, &nonRetryableErr{err: err}
		}

		// Don't sleep after the last attempt
		if attempt < totalAttempts-1 {
			select {
			case <-time.After(delay):
				delay = time.Duration(float64(delay) * tp.config.BackoffMultiplier)
				if tp.config.MaxDelay > 0 && delay > tp.config.MaxDelay {
					delay = tp.config.MaxDelay
				}
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}

	return nil, lastErr
}

func (tp *TieredProvider) wrapError(lastErr error) error {
	tp.mu.Lock()
	totalAttempts := len(tp.attempts)
	tp.mu.Unlock()
	return fmt.Errorf("all providers exhausted (%d attempts): %w", totalAttempts, lastErr)
}
