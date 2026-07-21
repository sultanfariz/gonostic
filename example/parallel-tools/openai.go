package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/sultanfariz/gonostic/pkg/agent"
)

// ============================================================
// OPENAI MODEL PROVIDER
// ============================================================

type OpenAIModelProvider struct {
	apiKey  string
	baseURL string
	model   string
	timeout time.Duration
}

type OpenAIConfig struct {
	APIKey  string
	Model   string
	BaseURL string
	Timeout time.Duration
}

func NewOpenAIProvider(cfg OpenAIConfig) *OpenAIModelProvider {
	if cfg.Model == "" {
		cfg.Model = "gpt-4o"
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	return &OpenAIModelProvider{
		apiKey:  cfg.APIKey,
		baseURL: cfg.BaseURL,
		model:   cfg.Model,
		timeout: cfg.Timeout,
	}
}

type openaiMessage struct {
	Role      string            `json:"role"`
	Content   interface{}       `json:"content"`
	ToolCalls []openaiToolCall  `json:"tool_calls,omitempty"`
}

type contentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

type imageURL struct {
	URL string `json:"url"`
}

type openaiTool struct {
	Type     string              `json:"type"`
	Function openaiToolFunction  `json:"function"`
}

type openaiToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type openaiToolCall struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function openaiToolFunctionCall `json:"function"`
}

type openaiToolFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openaiRequest struct {
	Model      string         `json:"model"`
	Messages   []openaiMessage `json:"messages"`
	Tools      []openaiTool    `json:"tools,omitempty"`
	ToolChoice interface{}     `json:"tool_choice,omitempty"`
	MaxTokens  int             `json:"max_tokens,omitempty"`
}

type openaiResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []openaiChoice     `json:"choices"`
	Usage   openaiUsage        `json:"usage"`
	Error   *openaiErrorResponse `json:"error,omitempty"`
}

type openaiChoice struct {
	Index        int           `json:"index"`
	Message      openaiMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type openaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openaiErrorResponse struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

func (m *OpenAIModelProvider) Complete(ctx context.Context, req *agent.CompletionRequest) (*agent.ModelResponse, error) {
	messages := make([]openaiMessage, 0, len(req.History)+1)

	for _, msg := range req.History {
		openaiMsg := openaiMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}

		if len(msg.Parts) > 0 || len(req.Files) > 0 {
			contentParts := []contentPart{{Type: "text", Text: msg.Content}}

			for _, part := range msg.Parts {
				if part.Type == "image" && part.Data != nil {
					if imgBytes, ok := part.Data.([]byte); ok {
						base64Img := base64.StdEncoding.EncodeToString(imgBytes)
						mimeType := detectMimeType(part.Type, imgBytes)
						contentParts = append(contentParts, contentPart{
							Type: "image_url",
							ImageURL: &imageURL{
								URL: fmt.Sprintf("data:%s;base64,%s", mimeType, base64Img),
							},
						})
					}
				}
			}

			if msg.Role == "user" && len(req.Files) > 0 {
				for _, file := range req.Files {
					if strings.HasPrefix(file.Type, "image/") {
						base64Img := base64.StdEncoding.EncodeToString(file.Content)
						contentParts = append(contentParts, contentPart{
							Type: "image_url",
							ImageURL: &imageURL{
								URL: fmt.Sprintf("data:%s;base64,%s", file.Type, base64Img),
							},
						})
					}
				}
			}

			openaiMsg.Content = contentParts
		}

		messages = append(messages, openaiMsg)
	}

	openaiTools := make([]openaiTool, 0, len(req.Tools))
	for _, tool := range req.Tools {
		openaiTools = append(openaiTools, openaiTool{
			Type: "function",
			Function: openaiToolFunction{
				Name:        tool.Name(),
				Description: tool.Description(),
				Parameters:  tool.Schema().(map[string]interface{}),
			},
		})
	}

	reqBody := openaiRequest{
		Model:    m.model,
		Messages: messages,
		MaxTokens: 4096,
	}

	if len(openaiTools) > 0 {
		reqBody.Tools = openaiTools
		reqBody.ToolChoice = "auto"
	}

	reqJSON, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.baseURL+"/chat/completions", bytes.NewReader(reqJSON))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+m.apiKey)

	client := &http.Client{Timeout: m.timeout}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	var openaiResp openaiResponse
	if err := json.Unmarshal(body, &openaiResp); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	if openaiResp.Error != nil {
		return nil, fmt.Errorf("openai error: %s", openaiResp.Error.Message)
	}

	if len(openaiResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	choice := openaiResp.Choices[0]

	toolCalls := make([]agent.ToolCall, 0)
	for _, tc := range choice.Message.ToolCalls {
		var args map[string]interface{}
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			return nil, fmt.Errorf("unmarshal tool args: %w", err)
		}

		toolCalls = append(toolCalls, agent.ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: args,
		})
	}

	var content string
	if c, ok := choice.Message.Content.(string); ok {
		content = c
	}

	return &agent.ModelResponse{
		Content:   content,
		ToolCalls: toolCalls,
		Finished:  choice.FinishReason == "stop",
	}, nil
}

func detectMimeType(fileType string, data []byte) string {
	if fileType != "" {
		return fileType
	}
	if len(data) < 4 {
		return "application/octet-stream"
	}
	if data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47 {
		return "image/png"
	}
	if data[0] == 0xFF && data[1] == 0xD8 {
		return "image/jpeg"
	}
	if data[0] == 0x47 && data[1] == 0x49 {
		return "image/gif"
	}
	if data[0] == 0x52 && data[1] == 0x49 && data[2] == 0x46 && data[3] == 0x46 {
		return "image/webp"
	}
	return "application/octet-stream"
}
