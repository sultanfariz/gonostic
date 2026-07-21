package main

// HOW TO RUN:
// 1. From the gonostic directory:
//    go run example/parallel-tools/main.go
//
// 2. Set your API keys:
//    export OPENAI_API_KEY=sk-your-key
//    export BRAVE_API_KEY=BS-your-key
//
// 3. Or build and run:
//    go build -o parallel-tools example/parallel-tools/main.go
//    ./parallel-tools
//
// This example demonstrates PARALLEL tool calls - LLM can call multiple tools
// simultaneously in a single response. All tools are executed in the same turn.

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/sultanfariz/gonostic/pkg/agent"
)

// ============================================================
// PINECONE SEARCH TOOL (mock: returns canned data, wire in a real client)
// ============================================================

type PineconeSearchTool struct {
	apiKey string
}

func NewPineconeSearchTool(apiKey string) *PineconeSearchTool {
	return &PineconeSearchTool{apiKey: apiKey}
}

func (t *PineconeSearchTool) Name() string {
	return "pinecone_search"
}

func (t *PineconeSearchTool) Description() string {
	return "Search vector database using Pinecone for semantic search"
}

func (t *PineconeSearchTool) Schema() interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "Search query for semantic search",
			},
		},
		"required": []string{"query"},
	}
}

func (t *PineconeSearchTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	query, _ := args["query"].(string)
	return map[string]interface{}{
		"source":  "pinecone",
		"query":   query,
		"results": []string{"Result 1 from Pinecone", "Result 2 from Pinecone"},
	}, nil
}

// ============================================================
// SERPAPI TOOL (mock: returns canned data, wire in a real client)
// ============================================================

type SerpAPITool struct {
	apiKey string
}

func NewSerpAPITool(apiKey string) *SerpAPITool {
	return &SerpAPITool{apiKey: apiKey}
}

func (t *SerpAPITool) Name() string {
	return "serpapi_crawl"
}

func (t *SerpAPITool) Description() string {
	return "Crawl web pages using SerpAPI"
}

func (t *SerpAPITool) Schema() interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"url": map[string]interface{}{
				"type":        "string",
				"description": "URL to crawl",
			},
		},
		"required": []string{"url"},
	}
}

func (t *SerpAPITool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	url, _ := args["url"].(string)
	return map[string]interface{}{
		"source":  "serpapi",
		"url":     url,
		"content": "Crawled content from " + url,
	}, nil
}

// ============================================================
// MAIN
// ============================================================

func main() {
	// Get API keys from environment variables
	openaiAPIKey := os.Getenv("OPENAI_API_KEY")
	braveAPIKey := os.Getenv("BRAVE_API_KEY")

	if openaiAPIKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}
	if braveAPIKey == "" {
		log.Fatal("BRAVE_API_KEY environment variable is required")
	}

	// Create OpenAI model provider (local implementation in openai.go)
	modelProvider := NewOpenAIProvider(OpenAIConfig{
		APIKey: openaiAPIKey,
		Model:  "gpt-4o",
	})

	// Create tools using local implementations (brave_search.go)
	braveTool := NewBraveSearchTool(BraveSearchConfig{
		APIKey: braveAPIKey,
	})
	pineconeTool := NewPineconeSearchTool("pinecone-key")
	serpTool := NewSerpAPITool("serpapi-key")

	tools := []agent.Tool{braveTool, pineconeTool, serpTool}

	// MaxTurns=10, but the agent will finish early when done
	// This proves maxTurns is a MAXIMUM, not an exact count
	searchAgent := agent.NewLLMAgent(agent.LLMAgentConfig{
		Name:        "parallel_search_agent",
		Description: "Agent that searches multiple sources in parallel",
		Prompt: `You are a research assistant. You have access to:
- brave_search: for web search
- pinecone_search: for vector database search
- serpapi_crawl: for crawling web pages

You can call multiple tools at once to gather information in parallel.
Summarize findings from all sources.`,
		Model:    modelProvider,
		Tools:    tools,
		MaxTurns: agent.MaxTurnsConfig{Limit: 10},
	})

	task := &agent.Task{
		ID:    "task-1",
		Input: "Find comprehensive information about Go programming best practices",
		State: make(map[string]interface{}),
	}

	fmt.Println("=== Starting Parallel Search Agent ===")
	fmt.Println("MaxTurns set to: 10")
	fmt.Println("The agent can call multiple tools in parallel within a single turn")
	fmt.Println()

	start := time.Now()
	result, err := searchAgent.Execute(context.Background(), task)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("\n=== Execution Complete ===")
	fmt.Printf("Total Duration: %v\n", time.Since(start))
	fmt.Printf("Success: %v\n", result.Success)
	fmt.Printf("Actual Turns Used: %d (out of MaxTurns=10)\n", len(result.Steps))
	fmt.Printf("\nFinal Output:\n%s\n", result.Output)

	fmt.Println("\n=== Step-by-Step Breakdown ===")
	for i, step := range result.Steps {
		fmt.Printf("Turn %d: %s (%v)\n", i+1, step.Action, step.Duration)
		if len(step.ToolCalls) > 0 {
			fmt.Printf("  - Tools called in parallel: %d\n", len(step.ToolCalls))
			for _, tc := range step.ToolCalls {
				fmt.Printf("    • %s: %v\n", tc.Name, tc.Arguments)
			}
		}
	}

	fmt.Println("\n=== Key Takeaways ===")
	fmt.Println("1. OpenAI GPT-4o can return multiple tool calls in a single response")
	fmt.Println("2. All tools in a single response are executed in parallel")
	fmt.Println("3. Agent finishes when LLM stops requesting tools (early termination)")

	// Output results as JSON
	jsonOutput, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		log.Fatal("Failed to marshal result to JSON:", err)
	}
	fmt.Println(string(jsonOutput))
}
