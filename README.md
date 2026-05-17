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
    Files:  []agent.FileInput{imageFile}, // Optional multimodal inputs
    Params: map[string]interface{}{"format": "bullet"},
    State:  make(map[string]interface{}),
    Config: &agent.ExecutionConfig{
        MaxIterations:  5,
        TimeoutSeconds: 60,
        Temperature:    0.7, // Passed to model provider
    },
}

// Result contains output, artifacts, execution audit trail, and metrics
result, err := myAgent.Execute(ctx, task)
fmt.Println(result.Output)
fmt.Println(result.Success)

// Metrics tracking
fmt.Printf("Tokens: %d, LLM Latency: %v\n",
    result.TotalTokenUsage.TotalTokens, result.TotalLLMLatency)

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

**Structured Output:**
Enforce JSON schema on LLM responses:

```go
agent := agent.NewLLMAgent(agent.LLMAgentConfig{
    Name:   "extractor",
    Prompt: "Extract user information from text",
    OutputSchema: map[string]interface{}{
        "type": "object",
        "properties": map[string]interface{}{
            "name": map[string]interface{}{"type": "string"},
            "age":  map[string]interface{}{"type": "integer"},
        },
        "required": []string{"name", "age"},
    },
    Model: myModelProvider,
})
```

**Features:**
- State injection into prompts via `{placeholder}` syntax
- Automatic tool execution and state updates
- Sub-agent delegation (responds to "delegate to <agent-name>" in LLM output)
- Artifact extraction from state
- Multimodal input support (images, PDFs, etc.)
- Structured output via JSON schema

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

## Implementing ModelProvider

To use `LLMAgent`, implement the `ModelProvider` interface:

```go
type ModelProvider interface {
    Complete(ctx context.Context, req *CompletionRequest) (*ModelResponse, error)
}

type CompletionRequest struct {
    Prompt       string                 // User input prompt
    Files        []FileInput            // Multimodal inputs (images, PDFs)
    Tools        []Tool                 // Available function calling tools
    History      []Message              // Conversation history
    OutputSchema map[string]interface{} // JSON schema for structured output
    Temperature  *float32               // Sampling temperature (nil = use default)
    MaxTokens    *int                   // Max completion tokens (nil = use default)
}

type ModelResponse struct {
    Content   string
    ToolCalls []ToolCall
    Reasoning string
    Finished  bool
    Usage     *TokenUsage // Token usage metadata
}
```

**Example implementation:**
```go
func (p *MyProvider) Complete(ctx context.Context, req *CompletionRequest) (*ModelResponse, error) {
    // Use req.OutputSchema for structured output if provided
    if len(req.OutputSchema) > 0 {
        // Configure your LLM for structured output
    }
    // Use req.Temperature if provided
    if req.Temperature != nil {
        // Apply temperature setting
    }
    // Process req.Files for multimodal input
    // ... implementation
}
```

## License

MIT
