package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/sultanfariz/gonostic/pkg/agent"
)

// ProductInfo represents the structured output we expect from the agent
type ProductInfo struct {
	Name        string   `json:"name"`
	Category    string   `json:"category"`
	Price       float64  `json:"price"`
	Currency    string   `json:"currency"`
	Brand       string   `json:"brand"`
	Features    []string `json:"features"`
	Description string   `json:"description"`
	InStock     bool     `json:"in_stock"`
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// Create OpenAI provider with structured output support
	provider := NewOpenAIProvider(OpenAIConfig{
		APIKey: apiKey,
		Model:  "gpt-4o-2024-08-06", // Structured output requires specific models
	})

	// Define JSON schema for structured product information
	productSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name": map[string]interface{}{
				"type":        "string",
				"description": "The product name",
			},
			"category": map[string]interface{}{
				"type":        "string",
				"description": "The product category (e.g., Electronics, Clothing, Books)",
			},
			"price": map[string]interface{}{
				"type":        "number",
				"description": "The product price as a number",
			},
			"currency": map[string]interface{}{
				"type":        "string",
				"description": "The currency code (e.g., USD, EUR, GBP)",
			},
			"brand": map[string]interface{}{
				"type":        "string",
				"description": "The brand or manufacturer name",
			},
			"features": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "string",
				},
				"description": "List of key product features",
			},
			"description": map[string]interface{}{
				"type":        "string",
				"description": "A concise product description",
			},
			"in_stock": map[string]interface{}{
				"type":        "boolean",
				"description": "Whether the product is in stock",
			},
		},
		"required": []string{
			"name",
			"category",
			"price",
			"currency",
			"brand",
			"features",
			"description",
			"in_stock",
		},
		"additionalProperties": false,
	}

	// Create LLM agent with structured output
	productExtractor := agent.NewLLMAgent(agent.LLMAgentConfig{
		Name: "ProductExtractor",
		Prompt: `You are a product information extraction assistant. Extract structured product information from the given text.
Make sure to identify:
- Product name
- Category
- Price and currency
- Brand/manufacturer
- Key features
- Brief description
- Stock availability

If information is not explicitly stated, make reasonable inferences based on context.`,
		OutputSchema: productSchema,
		Model:        provider,
		MaxTurns: agent.MaxTurnsConfig{Limit: 1}, // Structured output typically needs only one turn
	})

	// Example product descriptions to extract
	examples := []string{
		`Check out the new iPhone 15 Pro from Apple! This amazing smartphone features
a titanium design, A17 Pro chip, and a stunning 6.1-inch Super Retina XDR display.
It comes with a pro camera system with 48MP main lens. Available now for $999 USD.
Limited stock available!`,

		`Sony WH-1000XM5 wireless noise-cancelling headphones - the best in class!
Features industry-leading noise cancellation, 30-hour battery life, and premium sound quality.
Comfortable over-ear design perfect for travel. Price: €399. Currently in stock.`,

		`The Complete Works of William Shakespeare - Hardcover Collector's Edition by Barnes & Noble.
This beautiful leather-bound edition includes all 37 plays, 154 sonnets, and narrative poems.
Gold-embossed cover with ribbon bookmark. Perfect gift for literature lovers.
Only $45.99 with free shipping. Last few copies remaining!`,
	}

	fmt.Println("=== Structured Output Example: Product Information Extractor ===")
	fmt.Println()

	for i, description := range examples {
		fmt.Printf("Example %d:\n", i+1)
		fmt.Printf("Input: %s\n\n", description)

		// Create task
		task := &agent.Task{
			ID:    fmt.Sprintf("product-%d", i+1),
			Input: description,
			Config: &agent.ExecutionConfig{
				MaxIterations: 1,
			},
			StartedAt: time.Now(),
		}

		// Execute agent
		result, err := productExtractor.Execute(context.Background(), task)
		if err != nil {
			log.Printf("Error executing agent: %v\n", err)
			continue
		}

		if !result.Success {
			log.Printf("Agent execution failed: %s\n", result.Error)
			continue
		}

		// Parse the structured output
		var product ProductInfo
		if err := json.Unmarshal([]byte(result.Output.(string)), &product); err != nil {
			log.Printf("Error parsing product info: %v\n", err)
			log.Printf("Raw output: %s\n", result.Output)
			continue
		}

		// Display structured output
		fmt.Println("Extracted Product Information:")
		fmt.Printf("  Name:        %s\n", product.Name)
		fmt.Printf("  Brand:       %s\n", product.Brand)
		fmt.Printf("  Category:    %s\n", product.Category)
		fmt.Printf("  Price:       %.2f %s\n", product.Price, product.Currency)
		fmt.Printf("  In Stock:    %v\n", product.InStock)
		fmt.Printf("  Description: %s\n", product.Description)
		fmt.Printf("  Features:\n")
		for _, feature := range product.Features {
			fmt.Printf("    - %s\n", feature)
		}

		// Display metrics
		if result.TotalTokenUsage.TotalTokens > 0 {
			fmt.Printf("\nMetrics:\n")
			fmt.Printf("  Tokens Used: %d (prompt: %d, completion: %d)\n",
				result.TotalTokenUsage.TotalTokens,
				result.TotalTokenUsage.PromptTokens,
				result.TotalTokenUsage.CompletionTokens)
			fmt.Printf("  Latency:     %v\n", result.TotalLLMLatency)
		}

		fmt.Println("\n" + strings.Repeat("-", 80) + "\n")
	}

	fmt.Println("✓ All examples completed successfully!")
}
