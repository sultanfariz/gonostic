// Package agent provides types and implementations for building AI agent
// systems with tool use, orchestration, and async execution.
package agent

import (
	"context"
	"time"
)

// Agent executes tasks with tools and reasoning.
type Agent interface {
	Name() string
	Execute(ctx context.Context, task *Task) (*Result, error)
	SubAgents() []Agent
}

// FileInput represents a file to be passed as input to the LLM.
type FileInput struct {
	Name     string      // Filename or identifier
	Type     string      // MIME type (e.g., "image/png", "application/pdf", "text/plain")
	Content  []byte      // Raw file content
	URI      string      // Optional URI if file is referenced by path
	Metadata interface{} // Additional metadata about the file
}

// Task represents a unit of work (API-triggered).
type Task struct {
	ID          string
	Input       string                 // User's minimal prompt
	Files       []FileInput            // Files to pass as input to LLM (images, PDFs, etc.)
	Params      map[string]interface{} // Additional parameters
	State       map[string]interface{} // Working state
	Config      *ExecutionConfig
	StartedAt   time.Time
	CompletedAt time.Time
}

// Result is the final output of an agent execution.
type Result struct {
	TaskID    string
	Success   bool
	Output    interface{}            // Final result (can be struct, string, map)
	Artifacts []Artifact             // Generated files, images, etc.
	Metadata  map[string]interface{} // Processing metadata
	Error     string
	Steps     []ExecutionStep // Audit trail
}

// ExecutionStep tracks what happened during a single turn.
type ExecutionStep struct {
	AgentName  string
	Action     string
	Input      interface{}
	Output     interface{}
	Error      string
	Duration   time.Duration
	Timestamp  time.Time
	ToolCalls  []ToolCall
	StateDelta map[string]interface{}
}

// ExecutionConfig controls how a task is executed.
type ExecutionConfig struct {
	MaxIterations  int
	TimeoutSeconds int
	Temperature    float32
	EnablePlan     bool
	CallbackURL    string // For async notifications
}

// Artifact represents generated content (files, images, etc.).
type Artifact struct {
	Type     string      // "text", "image", "video", "code", etc.
	MimeType string
	Content  interface{} // Content or reference
	Metadata map[string]interface{}
}

// ToolCall represents a function call made by an agent.
type ToolCall struct {
	ID        string
	Name      string
	Arguments map[string]interface{}
	Result    interface{}
	Error     error
	Duration  time.Duration
}

// Tool is a function that agents can invoke.
type Tool interface {
	Name() string
	Description() string
	Schema() interface{} // JSON schema for parameters
	Execute(ctx context.Context, args map[string]interface{}) (interface{}, error)
}

// ModelProvider interfaces with LLM backends.
type ModelProvider interface {
	Complete(ctx context.Context, prompt string, files []FileInput, tools []Tool, history []Message) (*ModelResponse, error)
}

// ModelResponse is the response from an LLM.
type ModelResponse struct {
	Content   string
	ToolCalls []ToolCall
	Reasoning string
	Finished  bool
}

// Message represents a conversation message.
type Message struct {
	Role    string
	Content string
	Parts   []Part
}

// Part represents a segment of a multimodal message.
type Part struct {
	Type string
	Text string
	Data interface{}
}
