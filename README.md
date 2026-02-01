# llemme-go

A Go framework for building AI agent systems with tool use, orchestration, and async execution.

## Installation

```bash
go get github.com/sultanfariz/llemme-go
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
