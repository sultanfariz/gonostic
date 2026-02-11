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
	TaskID        string
	Success       bool
	Output        interface{}            // Final result (can be struct, string, map)
	Artifacts     []Artifact             // Generated files, images, etc.
	Metadata      map[string]interface{} // Processing metadata
	Error         string
	Steps         []ExecutionStep        // Audit trail

	// Aggregated metrics
	TotalLLMLatency   time.Duration // Total time spent on LLM calls across all steps
	TotalToolsLatency time.Duration // Total time spent on tool execution across all steps
	TotalTokenUsage   TokenUsage    // Total token usage across all steps
}

// ExecutionStep tracks what happened during a single turn.
type ExecutionStep struct {
	AgentName       string
	Action          string
	Input           interface{}
	Output          interface{}
	Error           string
	Duration        time.Duration // Total step duration (LLM + tools)
	LLMLatency      time.Duration // Time spent on LLM call
	ToolsLatency    time.Duration // Time spent on tool execution (sum of all tools)
	Timestamp       time.Time
	TokenUsage      *TokenUsage // Token usage for LLM call in this step
	ToolCalls       []ToolCall
	StateDelta      map[string]interface{}
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

// CompletionRequest encapsulates all parameters for LLM completion.
// This provides a clean, extensible way to pass parameters to model providers.
type CompletionRequest struct {
	Prompt       string                 // User input prompt
	Files        []FileInput            // Optional multimodal inputs (images, PDFs, etc.)
	Tools        []Tool                 // Optional function calling tools
	History      []Message              // Conversation history
	OutputSchema map[string]interface{} // Optional JSON schema for structured output (nil = unstructured)
	Temperature  *float32               // Optional sampling temperature (nil = use provider default)
	MaxTokens    *int                   // Optional max completion tokens (nil = use provider default)
}

// ModelProvider interfaces with LLM backends.
type ModelProvider interface {
	Complete(ctx context.Context, req *CompletionRequest) (*ModelResponse, error)
}

// TokenUsage tracks token consumption for LLM calls (model-agnostic).
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ModelResponse is the response from an LLM.
type ModelResponse struct {
	Content   string
	ToolCalls []ToolCall
	Reasoning string
	Finished  bool
	Usage     *TokenUsage // Token usage metadata (provider-dependent)
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
