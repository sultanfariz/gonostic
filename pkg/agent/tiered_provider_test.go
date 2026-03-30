package agent

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"
)

// permanentErr is a non-retryable error.
type permanentErr struct{ msg string }

func (e *permanentErr) Error() string  { return e.msg }
func (e *permanentErr) Retryable() bool { return false }

// nonRetryableProvider always returns a non-retryable error.
type nonRetryableProvider struct {
	calls atomic.Int32
	msg   string
}

func (p *nonRetryableProvider) Complete(ctx context.Context, req *CompletionRequest) (*ModelResponse, error) {
	p.calls.Add(1)
	return nil, &permanentErr{msg: p.msg}
}

// mockProvider fails failCount times then succeeds.
type mockProvider struct {
	failCount int
	calls     atomic.Int32
	response  *ModelResponse
}

func newMockProvider(failCount int) *mockProvider {
	return &mockProvider{
		failCount: failCount,
		response: &ModelResponse{
			Content:  "ok",
			Finished: true,
		},
	}
}

func (m *mockProvider) Complete(ctx context.Context, req *CompletionRequest) (*ModelResponse, error) {
	n := int(m.calls.Add(1))
	if n <= m.failCount {
		return nil, fmt.Errorf("mock error (call %d)", n)
	}
	return m.response, nil
}

// alwaysFailProvider always returns an error.
type alwaysFailProvider struct {
	calls atomic.Int32
}

func (p *alwaysFailProvider) Complete(ctx context.Context, req *CompletionRequest) (*ModelResponse, error) {
	p.calls.Add(1)
	return nil, fmt.Errorf("always fail")
}

func TestSingleProviderSuccess(t *testing.T) {
	p := newMockProvider(0)
	tp := NewTieredProviderWithDefaults([][]ModelProvider{{p}})

	resp, err := tp.Complete(context.Background(), &CompletionRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "ok" {
		t.Fatalf("unexpected content: %s", resp.Content)
	}

	attempts := tp.Attempts()
	if len(attempts) != 1 {
		t.Fatalf("expected 1 attempt, got %d", len(attempts))
	}
	if attempts[0].Error != nil {
		t.Fatalf("expected nil error, got %v", attempts[0].Error)
	}
	if attempts[0].Tier != 0 || attempts[0].ProviderIndex != 0 || attempts[0].Retry != 0 {
		t.Fatalf("unexpected attempt fields: %+v", attempts[0])
	}
}

func TestRetryThenSuccess(t *testing.T) {
	p := newMockProvider(2) // fails twice, succeeds on 3rd
	cfg := RetryConfig{
		MaxRetries:        2,
		InitialDelay:      1 * time.Millisecond,
		BackoffMultiplier: 2.0,
		MaxDelay:          100 * time.Millisecond,
	}
	tp := NewTieredProvider([][]ModelProvider{{p}}, cfg)

	resp, err := tp.Complete(context.Background(), &CompletionRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "ok" {
		t.Fatalf("unexpected content: %s", resp.Content)
	}

	attempts := tp.Attempts()
	if len(attempts) != 3 {
		t.Fatalf("expected 3 attempts, got %d", len(attempts))
	}
	for i := range 2 {
		if attempts[i].Error == nil {
			t.Fatalf("attempt %d should have error", i)
		}
		if attempts[i].Retry != i {
			t.Fatalf("attempt %d: expected retry=%d, got %d", i, i, attempts[i].Retry)
		}
	}
	if attempts[2].Error != nil {
		t.Fatalf("attempt 2 should succeed")
	}
}

func TestRetryExhaustedFallsToNextProvider(t *testing.T) {
	fail := &alwaysFailProvider{}
	succeed := newMockProvider(0)
	cfg := RetryConfig{
		MaxRetries:        1,
		InitialDelay:      1 * time.Millisecond,
		BackoffMultiplier: 1.0,
		MaxDelay:          10 * time.Millisecond,
	}
	tp := NewTieredProvider([][]ModelProvider{{fail, succeed}}, cfg)

	resp, err := tp.Complete(context.Background(), &CompletionRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "ok" {
		t.Fatalf("unexpected content: %s", resp.Content)
	}

	attempts := tp.Attempts()
	// fail: 2 attempts (1 + 1 retry), succeed: 1 attempt
	if len(attempts) != 3 {
		t.Fatalf("expected 3 attempts, got %d", len(attempts))
	}
	if attempts[0].ProviderIndex != 0 || attempts[1].ProviderIndex != 0 {
		t.Fatal("first two attempts should be provider 0")
	}
	if attempts[2].ProviderIndex != 1 {
		t.Fatal("third attempt should be provider 1")
	}
}

func TestTierFallback(t *testing.T) {
	fail := &alwaysFailProvider{}
	succeed := newMockProvider(0)
	cfg := RetryConfig{
		MaxRetries:        0, // no retries
		InitialDelay:      1 * time.Millisecond,
		BackoffMultiplier: 1.0,
	}
	tp := NewTieredProvider([][]ModelProvider{
		{fail},     // tier 0
		{succeed},  // tier 1
	}, cfg)

	resp, err := tp.Complete(context.Background(), &CompletionRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "ok" {
		t.Fatalf("unexpected content: %s", resp.Content)
	}

	attempts := tp.Attempts()
	if len(attempts) != 2 {
		t.Fatalf("expected 2 attempts, got %d", len(attempts))
	}
	if attempts[0].Tier != 0 {
		t.Fatal("first attempt should be tier 0")
	}
	if attempts[1].Tier != 1 {
		t.Fatal("second attempt should be tier 1")
	}
}

func TestAllProvidersExhausted(t *testing.T) {
	fail1 := &alwaysFailProvider{}
	fail2 := &alwaysFailProvider{}
	cfg := RetryConfig{
		MaxRetries:        1,
		InitialDelay:      1 * time.Millisecond,
		BackoffMultiplier: 1.0,
	}
	tp := NewTieredProvider([][]ModelProvider{
		{fail1},
		{fail2},
	}, cfg)

	_, err := tp.Complete(context.Background(), &CompletionRequest{})
	if err == nil {
		t.Fatal("expected error")
	}

	attempts := tp.Attempts()
	// 2 providers x (1 + 1 retry) = 4
	if len(attempts) != 4 {
		t.Fatalf("expected 4 attempts, got %d", len(attempts))
	}
}

func TestMaxRetriesZero(t *testing.T) {
	fail := &alwaysFailProvider{}
	succeed := newMockProvider(0)
	cfg := RetryConfig{
		MaxRetries:        0,
		InitialDelay:      1 * time.Millisecond,
		BackoffMultiplier: 1.0,
	}
	tp := NewTieredProvider([][]ModelProvider{{fail, succeed}}, cfg)

	resp, err := tp.Complete(context.Background(), &CompletionRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "ok" {
		t.Fatalf("unexpected content: %s", resp.Content)
	}

	attempts := tp.Attempts()
	if len(attempts) != 2 {
		t.Fatalf("expected 2 attempts (1 per provider), got %d", len(attempts))
	}
	if int(fail.calls.Load()) != 1 {
		t.Fatalf("fail provider should be called exactly once, got %d", fail.calls.Load())
	}
}

func TestMaxTiersLimit(t *testing.T) {
	fail := &alwaysFailProvider{}
	succeed := newMockProvider(0)
	cfg := RetryConfig{
		MaxRetries:        0,
		InitialDelay:      1 * time.Millisecond,
		BackoffMultiplier: 1.0,
		MaxTiers:          1, // only try tier 0
	}
	tp := NewTieredProvider([][]ModelProvider{
		{fail},
		{succeed}, // should not be reached
	}, cfg)

	_, err := tp.Complete(context.Background(), &CompletionRequest{})
	if err == nil {
		t.Fatal("expected error since tier 1 is unreachable")
	}

	if int(succeed.calls.Load()) != 0 {
		t.Fatalf("tier 1 provider should not be called, got %d calls", succeed.calls.Load())
	}
}

func TestEmptyTierSkipped(t *testing.T) {
	succeed := newMockProvider(0)
	cfg := RetryConfig{
		MaxRetries:        0,
		InitialDelay:      1 * time.Millisecond,
		BackoffMultiplier: 1.0,
	}
	tp := NewTieredProvider([][]ModelProvider{
		{},        // empty tier 0
		{succeed}, // tier 1
	}, cfg)

	resp, err := tp.Complete(context.Background(), &CompletionRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "ok" {
		t.Fatalf("unexpected content: %s", resp.Content)
	}

	attempts := tp.Attempts()
	if len(attempts) != 1 {
		t.Fatalf("expected 1 attempt, got %d", len(attempts))
	}
	if attempts[0].Tier != 1 {
		t.Fatal("attempt should be from tier 1")
	}
}

func TestAllTiersEmpty(t *testing.T) {
	cfg := RetryConfig{MaxRetries: 0, InitialDelay: 1 * time.Millisecond, BackoffMultiplier: 1.0}
	tp := NewTieredProvider([][]ModelProvider{{}, {}}, cfg)

	_, err := tp.Complete(context.Background(), &CompletionRequest{})
	if err == nil {
		t.Fatal("expected error for all empty tiers")
	}
}

func TestNoProviders(t *testing.T) {
	cfg := RetryConfig{MaxRetries: 0, InitialDelay: 1 * time.Millisecond, BackoffMultiplier: 1.0}
	tp := NewTieredProvider([][]ModelProvider{}, cfg)

	_, err := tp.Complete(context.Background(), &CompletionRequest{})
	if err == nil {
		t.Fatal("expected error for no providers")
	}
}

func TestContextCancelledBeforeFirstAttempt(t *testing.T) {
	succeed := newMockProvider(0)
	cfg := RetryConfig{MaxRetries: 2, InitialDelay: 1 * time.Millisecond, BackoffMultiplier: 1.0}
	tp := NewTieredProvider([][]ModelProvider{{succeed}}, cfg)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	_, err := tp.Complete(ctx, &CompletionRequest{})
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
	if int(succeed.calls.Load()) != 0 {
		t.Fatal("provider should not be called with cancelled context")
	}
}

func TestContextCancelledDuringBackoff(t *testing.T) {
	fail := &alwaysFailProvider{}
	cfg := RetryConfig{
		MaxRetries:        5,
		InitialDelay:      1 * time.Second, // long delay
		BackoffMultiplier: 1.0,
	}
	tp := NewTieredProvider([][]ModelProvider{{fail}}, cfg)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := tp.Complete(ctx, &CompletionRequest{})
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected error")
	}
	// Should return well before the 1s backoff
	if elapsed > 500*time.Millisecond {
		t.Fatalf("should have returned quickly, took %v", elapsed)
	}
}

func TestAttemptLogResetBetweenCalls(t *testing.T) {
	p := newMockProvider(0)
	cfg := RetryConfig{MaxRetries: 0, InitialDelay: 1 * time.Millisecond, BackoffMultiplier: 1.0}
	tp := NewTieredProvider([][]ModelProvider{{p}}, cfg)

	tp.Complete(context.Background(), &CompletionRequest{})
	if len(tp.Attempts()) != 1 {
		t.Fatal("expected 1 attempt after first call")
	}

	tp.Complete(context.Background(), &CompletionRequest{})
	if len(tp.Attempts()) != 1 {
		t.Fatal("expected 1 attempt after second call (log should reset)")
	}
}

func TestAttemptsReturnsNilBeforeCall(t *testing.T) {
	cfg := RetryConfig{MaxRetries: 0, InitialDelay: 1 * time.Millisecond, BackoffMultiplier: 1.0}
	tp := NewTieredProvider([][]ModelProvider{{newMockProvider(0)}}, cfg)

	if tp.Attempts() != nil {
		t.Fatal("expected nil before any Complete() call")
	}
}

func TestAttemptsReturnsCopy(t *testing.T) {
	p := newMockProvider(0)
	cfg := RetryConfig{MaxRetries: 0, InitialDelay: 1 * time.Millisecond, BackoffMultiplier: 1.0}
	tp := NewTieredProvider([][]ModelProvider{{p}}, cfg)

	tp.Complete(context.Background(), &CompletionRequest{})

	a1 := tp.Attempts()
	a2 := tp.Attempts()

	// Mutate a1 and verify a2 is unaffected
	a1[0].Tier = 999
	if a2[0].Tier == 999 {
		t.Fatal("Attempts() should return independent copies")
	}
}

func TestAttemptDurationPositive(t *testing.T) {
	p := newMockProvider(0)
	cfg := RetryConfig{MaxRetries: 0, InitialDelay: 1 * time.Millisecond, BackoffMultiplier: 1.0}
	tp := NewTieredProvider([][]ModelProvider{{p}}, cfg)

	tp.Complete(context.Background(), &CompletionRequest{})

	for _, a := range tp.Attempts() {
		if a.Duration <= 0 {
			t.Fatal("attempt duration should be positive")
		}
	}
}

func TestConvenienceConstructors(t *testing.T) {
	p := newMockProvider(0)
	cfg := RetryConfig{MaxRetries: 0, InitialDelay: 1 * time.Millisecond, BackoffMultiplier: 1.0}

	// NewRetryProvider
	tp1 := NewRetryProvider(p, cfg)
	resp, err := tp1.Complete(context.Background(), &CompletionRequest{})
	if err != nil || resp.Content != "ok" {
		t.Fatal("NewRetryProvider failed")
	}

	// NewFallbackProvider
	p2 := newMockProvider(0)
	tp2 := NewFallbackProvider([]ModelProvider{p2}, cfg)
	resp, err = tp2.Complete(context.Background(), &CompletionRequest{})
	if err != nil || resp.Content != "ok" {
		t.Fatal("NewFallbackProvider failed")
	}
}

func TestBackoffGrows(t *testing.T) {
	fail := &alwaysFailProvider{}
	cfg := RetryConfig{
		MaxRetries:        3,
		InitialDelay:      10 * time.Millisecond,
		BackoffMultiplier: 2.0,
		MaxDelay:          1 * time.Second,
	}
	tp := NewTieredProvider([][]ModelProvider{{fail}}, cfg)

	start := time.Now()
	tp.Complete(context.Background(), &CompletionRequest{})
	elapsed := time.Since(start)

	// Expected waits: 10ms + 20ms + 40ms = 70ms minimum
	if elapsed < 60*time.Millisecond {
		t.Fatalf("backoff should accumulate, elapsed only %v", elapsed)
	}
}

func TestNonRetryableStopsImmediately(t *testing.T) {
	p := &nonRetryableProvider{msg: "content policy violation"}
	backup := newMockProvider(0) // should never be reached
	cfg := RetryConfig{
		MaxRetries:        2,
		InitialDelay:      1 * time.Millisecond,
		BackoffMultiplier: 1.0,
	}
	tp := NewTieredProvider([][]ModelProvider{{p, backup}}, cfg)

	_, err := tp.Complete(context.Background(), &CompletionRequest{})
	if err == nil {
		t.Fatal("expected error")
	}
	if err.Error() != "content policy violation" {
		t.Fatalf("expected original error, got: %v", err)
	}

	// Only one attempt — no retries, no fallback to backup
	attempts := tp.Attempts()
	if len(attempts) != 1 {
		t.Fatalf("expected 1 attempt, got %d", len(attempts))
	}
	if int(p.calls.Load()) != 1 {
		t.Fatalf("provider should be called exactly once, got %d", p.calls.Load())
	}
	if int(backup.calls.Load()) != 0 {
		t.Fatalf("backup should not be called, got %d calls", backup.calls.Load())
	}
}

func TestNonRetryableNoTierFallback(t *testing.T) {
	p := &nonRetryableProvider{msg: "auth failure"}
	fallback := newMockProvider(0) // should never be reached
	cfg := RetryConfig{
		MaxRetries:        1,
		InitialDelay:      1 * time.Millisecond,
		BackoffMultiplier: 1.0,
	}
	tp := NewTieredProvider([][]ModelProvider{{p}, {fallback}}, cfg)

	_, err := tp.Complete(context.Background(), &CompletionRequest{})
	if err == nil {
		t.Fatal("expected error")
	}

	if int(fallback.calls.Load()) != 0 {
		t.Fatalf("tier 1 should not be called, got %d calls", fallback.calls.Load())
	}
}

func TestNonRetryableSkipTiersEnabled(t *testing.T) {
	p := &nonRetryableProvider{msg: "content policy violation"}
	fallback := newMockProvider(0)
	cfg := RetryConfig{
		MaxRetries:              0,
		InitialDelay:            1 * time.Millisecond,
		BackoffMultiplier:       1.0,
		SkipTiersOnNonRetryable: true,
	}
	tp := NewTieredProvider([][]ModelProvider{
		{p},        // tier 0: non-retryable
		{fallback}, // tier 1: succeeds
	}, cfg)

	resp, err := tp.Complete(context.Background(), &CompletionRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "ok" {
		t.Fatalf("unexpected content: %s", resp.Content)
	}

	if int(fallback.calls.Load()) != 1 {
		t.Fatalf("tier 1 should be called once, got %d calls", fallback.calls.Load())
	}
}

func TestNonRetryableSkipTiersSkipsRemainingTierProviders(t *testing.T) {
	p := &nonRetryableProvider{msg: "content policy violation"}
	sameTierProvider := newMockProvider(0) // in same tier as p, should be skipped
	fallback := newMockProvider(0)
	cfg := RetryConfig{
		MaxRetries:              0,
		InitialDelay:            1 * time.Millisecond,
		BackoffMultiplier:       1.0,
		SkipTiersOnNonRetryable: true,
	}
	tp := NewTieredProvider([][]ModelProvider{
		{p, sameTierProvider}, // tier 0: non-retryable + provider in same tier
		{fallback},            // tier 1: succeeds
	}, cfg)

	resp, err := tp.Complete(context.Background(), &CompletionRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "ok" {
		t.Fatalf("unexpected content: %s", resp.Content)
	}

	// sameTierProvider should be skipped — only fallback (tier 1) called
	if int(sameTierProvider.calls.Load()) != 0 {
		t.Fatalf("same-tier provider should be skipped, got %d calls", sameTierProvider.calls.Load())
	}
	if int(fallback.calls.Load()) != 1 {
		t.Fatalf("tier 1 should be called once, got %d calls", fallback.calls.Load())
	}
}

func TestMaxDelayCap(t *testing.T) {
	fail := &alwaysFailProvider{}
	cfg := RetryConfig{
		MaxRetries:        4,
		InitialDelay:      50 * time.Millisecond,
		BackoffMultiplier: 10.0,
		MaxDelay:          60 * time.Millisecond, // cap near initial
	}
	tp := NewTieredProvider([][]ModelProvider{{fail}}, cfg)

	start := time.Now()
	tp.Complete(context.Background(), &CompletionRequest{})
	elapsed := time.Since(start)

	// Without cap: 50 + 500 + 5000 + 50000 = way too long
	// With cap: 50 + 60 + 60 + 60 = 230ms
	if elapsed > 500*time.Millisecond {
		t.Fatalf("MaxDelay cap not working, elapsed %v", elapsed)
	}
}
