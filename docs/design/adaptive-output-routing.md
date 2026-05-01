# Adaptive Output Routing (AOR)

## Overview

Adaptive Output Routing (AOR) is the core execution strategy used by `LLMAgent`
in gonostic. Rather than always passing an `OutputSchema` on every LLM call,
AOR routes each model response through one of three paths determined at runtime
by the model's `StopReason` and the agent's schema configuration. This keeps
agentic (multi-turn tool-use) loops lean and reserves structured-output
enforcement for a single, targeted extraction call at the very end.

---

## Motivation

Early versions of `LLMAgent.Execute()` forwarded `OutputSchema` on every call
inside the tool-use loop. This caused two problems:

1. **Schema interference with tool calls.** Many providers (e.g. Anthropic)
   treat `OutputSchema` as mutually exclusive with `tools`. Sending both
   simultaneously either errors or silently suppresses tool calls.
2. **Wasted tokens.** Asking the model to produce structured JSON on every
   intermediate reasoning step inflates prompt and completion token counts with
   no benefit.

AOR solves both issues by applying the schema exactly once — only when needed.

---

## Three-Path Routing

```
tools + messages
       │
       ▼
   LLM call (no schema)
       │
       ├─ StopReason == "tool_use"  ─────────────────────► execute tools → loop
       │
       ├─ StopReason == "end_turn", no outputSchema ──────► return resp.Content
       │
       └─ StopReason == "end_turn", outputSchema set ─────► schema-enforcement call
                                                                    │
                                                                    ▼
                                                            LLM call (schema only)
                                                                    │
                                                                    ▼
                                                            return structured JSON
```

### Path 1 — `tool_use`

**Condition:** `resp.StopReason == "tool_use"` or `len(resp.ToolCalls) > 0`

The model has decided to call one or more tools. Each tool is looked up by name,
executed, and its result is stored in `task.State`. The exchange is appended to
the conversation history as an assistant message (the tool calls) followed by a
user message (the tool results). The loop continues to the next turn.

No `OutputSchema` is sent during this path, so tool use is never suppressed.

### Path 2 — `end_turn`, no schema

**Condition:** `resp.StopReason == "end_turn"` and `a.outputSchema == nil`

The model has finished reasoning and no structured output is required. The plain
`resp.Content` string is set as `result.Output` and the agent returns
immediately. This is the zero-overhead fast path for open-ended tasks.

### Path 3 — `end_turn`, schema present

**Condition:** `resp.StopReason == "end_turn"` and `a.outputSchema != nil`

The model has finished its reasoning but the caller expects a structured
response. AOR appends the assistant's reasoning to history, adds a brief
extraction prompt, then makes **one additional LLM call** with `OutputSchema`
enforced (and no tools). The model uses its own conclusion as context to produce
a valid JSON object that conforms to the schema.

This two-call pattern keeps the agentic loop clean while guaranteeing a
well-formed structured response at the end.

---

## Implementation Details

### `ModelResponse.StopReason`

`ModelResponse` exposes a `StopReason string` field instead of the previous
`Finished bool`. Model providers set this to the canonical stop-reason string
returned by the underlying API (e.g. `"tool_use"`, `"end_turn"`,
`"max_tokens"`). The AOR router branches on this value.

```go
// pkg/agent/types.go
type ModelResponse struct {
    Content    string
    ToolCalls  []ToolCall
    Reasoning  string
    StopReason string      // "tool_use", "end_turn", "max_tokens", etc.
    Usage      *TokenUsage
}
```

### Schema isolation in `CompletionRequest`

During the main agentic loop (Paths 1 & 2) `OutputSchema` is **not** set on
`CompletionRequest`. It is set only in the dedicated extraction call (Path 3).
This ensures tool-use capability is never inadvertently disabled by the
presence of a schema.

### Execution steps

AOR produces distinct `ExecutionStep` records for observability:

| Step action            | Path  | Description                                   |
|------------------------|-------|-----------------------------------------------|
| `reasoning`            | 1/2/3 | Normal LLM call result                        |
| `tool_execution`       | 1     | Tool calls executed in this turn              |
| `structured_extraction`| 3     | Reasoning turn that triggered schema path     |
| `schema_enforcement`   | 3     | Final LLM call with `OutputSchema` applied    |

All steps contribute to `Result.TotalLLMLatency`, `TotalToolsLatency`, and
`TotalTokenUsage` via `aggregateMetrics()`.

---

## Configuration

`LLMAgentConfig.OutputSchema` controls whether AOR takes Path 2 or Path 3 upon
`end_turn`. Agents that do not need structured output should leave this field
`nil` to avoid the extra extraction call.

```go
// Unstructured agent — always Path 2
agent := agent.NewLLMAgent(agent.LLMAgentConfig{
    Name:  "researcher",
    Model: myModel,
    Tools: []agent.Tool{searchTool},
})

// Structured agent — Path 3 on end_turn
agent := agent.NewLLMAgent(agent.LLMAgentConfig{
    Name:  "extractor",
    Model: myModel,
    OutputSchema: map[string]interface{}{
        "type": "object",
        "properties": map[string]interface{}{
            "summary": map[string]interface{}{"type": "string"},
            "score":   map[string]interface{}{"type": "number"},
        },
        "required": []string{"summary", "score"},
    },
})
```

---

## Trade-offs

| Concern               | Decision                                              |
|-----------------------|-------------------------------------------------------|
| Extra LLM call (Path 3) | Accepted: schema correctness outweighs one extra call |
| Schema on tool turns  | Explicitly avoided to preserve tool-use capability    |
| `StopReason` as string | Extensible; providers add new reasons without code change |
| Fallback for unknown `StopReason` | Treated as `end_turn`; safe default        |
