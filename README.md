# gonostic

A Go framework for building AI agent systems with tool use, orchestration, and async execution.

## Installation

```bash
go get github.com/sultanfariz/gonostic
```

## Core Concepts

### Agent Interface

All agents implement this interface:

```go
type Agent interface {
    Name() string
    Execute(ctx context.Context, task *Task) (*Result, error)
    SubAgents() []Agent
}
```

### Task & Result

```go
// Task is the input to an agent
task := &agent.Task{
    ID:     "task-123",
    Input:  "Summarize this document",
    Params: map[string]interface{}{"format": "bullet"},
    State:  make(map[string]interface{}),
    Config: &agent.ExecutionConfig{
        MaxIterations:  5,
        TimeoutSeconds: 60,
    },
}

// Result contains output, artifacts, and execution audit trail
result, err := myAgent.Execute(ctx, task)
fmt.Println(result.Output)
fmt.Println(result.Success)
for _, step := range result.Steps {
    fmt.Printf("%s: %s (%v)\n", step.AgentName, step.Action, step.Duration)
}
```

### Tools

Agents can invoke tools during execution:

```go
type Tool interface {
    Name() string
    Description() string
    Schema() interface{}
    Execute(ctx context.Context, args map[string]interface{}) (interface{}, error)
}
```

## Agent Types

### LLMAgent

A reasoning agent powered by an LLM with tool use and sub-agent delegation:

```go
agent := agent.NewLLMAgent(agent.LLMAgentConfig{
    Name:   "assistant",
    Prompt: "You are a helpful assistant. User context: {user_name}",
    Model:  myModelProvider, // implements ModelProvider interface
    Tools:  []agent.Tool{searchTool, calcTool},
    MaxTurns: 10,
})

result, err := agent.Execute(ctx, task)
```

**Features:**
- State injection into prompts via `{placeholder}` syntax
- Automatic tool execution and state updates
- Sub-agent delegation (responds to "delegate to <agent-name>" in LLM output)
- Artifact extraction from state

### SequentialAgent

Runs agents in order, passing accumulated state:

```go
pipeline := agent.NewSequentialAgent("data-pipeline", []agent.Agent{
    fetchAgent,
    transformAgent,
    storeAgent,
})
```

### ParallelAgent

Runs agents concurrently with isolated state copies:

```go
parallel := agent.NewParallelAgent("multi-search", []agent.Agent{
    webSearchAgent,
    dbSearchAgent,
    cacheSearchAgent,
})

// Result.Output is map[string]interface{} with each agent's output
```

### PipelineAgent

Chains agents where each output becomes the next input:

```go
pipeline := agent.NewPipelineAgent("etl", []agent.Agent{
    extractAgent,  // output: raw data
    transformAgent, // input: raw data, output: cleaned data
    loadAgent,      // input: cleaned data
})
```

## Async Execution

The `Executor` manages async task execution with a worker pool:

```go
exec := agent.NewExecutor(myAgent, 5) // 5 workers

// Async submission
taskID, _ := exec.Submit("Process this", params, config)

// Check status
status, _ := exec.GetStatus(taskID)
// JobPending | JobRunning | JobCompleted | JobFailed

// Get result (blocks until complete)
result, err := exec.GetResult(taskID)

// Or execute synchronously
result, err := exec.ExecuteSync(ctx, "Process this", params)
```

## Session-Based Agents

For interactive, stateful conversations, use `SessionAgent`:

```go
type SessionAgent interface {
    Name() string
    Run(ctx context.Context, inv *Invocation) (*Response, error)
    Agents() []SessionAgent
}
```

With thread-safe state management:

```go
state := agent.NewMapState()
state.Set("user_id", "123")

inv := &agent.Invocation{
    SessionID: "session-abc",
    UserID:    "user-123",
    Input:     &agent.Message{Role: "user", Content: "Hello"},
    State:     state,
    Config: &agent.RunConfig{
        MaxIterations: 10,
        StreamingMode: agent.StreamingModeFull,
        EnableMemory:  true,
    },
}
```

## TieredProvider

`TieredProvider` wraps multiple `ModelProvider` implementations into a single provider with automatic retry and tiered fallback. Pass it anywhere a `ModelProvider` is accepted.

### How it works

Providers are organised in **tiers** (priority groups). Tier 0 is always tried first. Within a tier, providers are tried left-to-right. Each provider is retried with exponential backoff before the next is attempted. The framework only advances to the next tier after every provider in the current one has been exhausted.

```
Tier 0: [GPT-4o, GPT-4o-backup]  ← tried first
Tier 1: [Claude Sonnet]           ← fallback if all of tier 0 fails
```

### Quick start

```go
primary  := NewOpenAIProvider(apiKey, "gpt-4o")
backup   := NewOpenAIProvider(apiKey, "gpt-4o-mini")
fallback := NewAnthropicProvider(apiKey, "claude-sonnet-4-6")

tiered := agent.NewTieredProvider(
    [][]agent.ModelProvider{
        {primary, backup}, // tier 0
        {fallback},        // tier 1
    },
    agent.DefaultRetryConfig(),
)

myAgent := agent.NewLLMAgent(agent.LLMAgentConfig{
    Name:  "assistant",
    Model: tiered, // drop-in replacement
})
```

### Retry config

```go
cfg := agent.RetryConfig{
    MaxRetries:              2,               // retries per provider (0 = one attempt)
    InitialDelay:            500 * time.Millisecond,
    BackoffMultiplier:       2.0,             // delay doubles each retry
    MaxDelay:                10 * time.Second,
    MaxTiers:                0,               // 0 = try all tiers
    SkipTiersOnNonRetryable: false,           // see non-retryable errors below
}
```

Backoff resets when moving to a new provider — each provider always starts from `InitialDelay`.

### Non-retryable errors

Providers can signal that an error is permanent (e.g. content policy violation, invalid API key) by implementing `RetryableError`:

```go
type RetryableError interface {
    error
    Retryable() bool
}
```

When `Retryable()` returns `false`, the framework stops immediately — no retries, no fallback to other providers or tiers — and returns the error to the caller.

```go
type MyProviderError struct {
    StatusCode int
    Message    string
}

func (e *MyProviderError) Error() string   { return e.Message }
func (e *MyProviderError) Retryable() bool {
    return e.StatusCode == 429 || e.StatusCode >= 500
}
```

If the error does **not** implement `RetryableError`, the framework treats it as retryable (safe default).

**`SkipTiersOnNonRetryable`** — set to `true` to fall through to the next tier even on a non-retryable error. Useful when providers have different content policies and a second opinion is desirable.

### Observability

```go
result, err := tiered.Complete(ctx, req)

for _, a := range tiered.Attempts() {
    fmt.Printf("tier=%d provider=%d retry=%d duration=%v err=%v\n",
        a.Tier, a.ProviderIndex, a.Retry, a.Duration, a.Error)
}
```

### Convenience constructors

```go
// Single provider with retries, no fallback
agent.NewRetryProvider(provider, cfg)

// Multiple providers in one tier, no tiering
agent.NewFallbackProvider([]agent.ModelProvider{p1, p2}, cfg)

// Defaults: 2 retries, 500ms initial delay, 2× backoff, 10s cap
agent.NewTieredProviderWithDefaults(tiers)
```

## Implementing ModelProvider

To use `LLMAgent`, implement the `ModelProvider` interface:

```go
type ModelProvider interface {
    Complete(ctx context.Context, req *CompletionRequest) (*ModelResponse, error)
}
```

## License

MIT
