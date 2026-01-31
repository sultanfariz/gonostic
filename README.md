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
