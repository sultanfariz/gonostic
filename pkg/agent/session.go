package agent

import "context"

// SessionAgent is an agent designed for interactive, session-based execution.
// It operates on invocations rather than tasks, supporting streaming and
// stateful conversation flows.
type SessionAgent interface {
	Name() string
	Run(ctx context.Context, inv *Invocation) (*Response, error)
	Agents() []SessionAgent // Sub-agents for orchestration
}

// Invocation represents a single agent execution request within a session.
type Invocation struct {
	SessionID string
	UserID    string
	Input     *Message
	State     State
	Config    *RunConfig
}

// Response represents the agent's output from an invocation.
type Response struct {
	Content   string
	ToolCalls []ToolCall
	Artifacts []Artifact
	Actions   *EventActions
	Finished  bool
}

// EventActions captures state changes and control flow actions.
type EventActions struct {
	StateDelta    map[string]interface{}
	Escalate      bool
	TransferTo    string
	ExitLoop      bool
	SkipRemaining bool
}

// State represents session state with get/set/delete semantics.
type State interface {
	Get(key string) (interface{}, bool)
	Set(key string, value interface{})
	Delete(key string)
	Keys() []string
	Merge(delta map[string]interface{})
}

// RunConfig controls execution behavior for session-based agents.
type RunConfig struct {
	MaxIterations  int
	StreamingMode  StreamingMode
	Temperature    float32
	EnablePlan     bool
	EnableMemory   bool
	TimeoutSeconds int
}

// StreamingMode defines how output is streamed back to the caller.
type StreamingMode int

const (
	// StreamingModeNone disables streaming; full response is returned at once.
	StreamingModeNone StreamingMode = iota
	// StreamingModePartial streams partial token updates.
	StreamingModePartial
	// StreamingModeFull streams all events including tool calls and state changes.
	StreamingModeFull
)
