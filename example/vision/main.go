package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/sultanfariz/gonostic/pkg/agent"
)

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// Create OpenAI provider with vision model
	provider := NewOpenAIProvider(OpenAIConfig{
		APIKey: apiKey,
		Model:  "gpt-4o", // Vision-capable model
	})

	fmt.Println("=== Multimodal Vision Agent Example ===")
	fmt.Println()

	// Fetch the image from the URL
	imageURL := "https://upload.wikimedia.org/wikipedia/commons/8/80/Go-Logo_Blue.png"
	fmt.Printf("Fetching image from: %s\n", imageURL)

	imageData, mimeType, err := fetchImage(imageURL)
	if err != nil {
		log.Fatalf("Failed to fetch image: %v", err)
	}

	fmt.Printf("Image loaded successfully (%d bytes, type: %s)\n\n", len(imageData), mimeType)

	// Create FileInput for the image
	imageFile := agent.FileInput{
		Name:    "input_image.png",
		Type:    mimeType,
		Content: imageData,
		URI:     imageURL,
	}

	// Example 1: General image description
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("Example 1: General Image Description")
	fmt.Println(strings.Repeat("=", 80))

	descriptionAgent := agent.NewLLMAgent(agent.LLMAgentConfig{
		Name: "ImageDescriber",
		Prompt: `You are an expert image analysis assistant. Provide a detailed description of the image,
including what you see, the overall composition, colors, and any notable elements or details.`,
		Model:    provider,
		MaxTurns: agent.MaxTurnsConfig{Limit: 1},
	})

	result := executeVisionTask(descriptionAgent, "Describe this image in detail.", imageFile)
	fmt.Printf("Description:\n%s\n\n", result)

	// Example 2: Object detection and counting
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("Example 2: Object Detection and Analysis")
	fmt.Println(strings.Repeat("=", 80))

	objectDetectionAgent := agent.NewLLMAgent(agent.LLMAgentConfig{
		Name: "ObjectDetector",
		Prompt: `You are an object detection specialist. Analyze the image and:
1. List all objects you can identify
2. Count similar objects if there are multiple
3. Describe their positions and relationships
4. Note any text or labels visible in the image`,
		Model:    provider,
		MaxTurns: agent.MaxTurnsConfig{Limit: 1},
	})

	result = executeVisionTask(objectDetectionAgent, "Identify and list all objects in this image.", imageFile)
	fmt.Printf("Object Analysis:\n%s\n\n", result)

	// Example 3: Color and style analysis
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("Example 3: Color and Style Analysis")
	fmt.Println(strings.Repeat("=", 80))

	styleAgent := agent.NewLLMAgent(agent.LLMAgentConfig{
		Name: "StyleAnalyzer",
		Prompt: `You are a visual design expert. Analyze the image's:
1. Color palette and dominant colors
2. Visual style (modern, vintage, minimalist, etc.)
3. Composition and layout
4. Typography if any text is present
5. Overall aesthetic and mood`,
		Model:    provider,
		MaxTurns: agent.MaxTurnsConfig{Limit: 1},
	})

	result = executeVisionTask(styleAgent, "Analyze the visual style and design of this image.", imageFile)
	fmt.Printf("Style Analysis:\n%s\n\n", result)

	// Example 4: Contextual understanding
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("Example 4: Contextual Understanding")
	fmt.Println(strings.Repeat("=", 80))

	contextAgent := agent.NewLLMAgent(agent.LLMAgentConfig{
		Name: "ContextAnalyzer",
		Prompt: `You are an expert at understanding context and purpose. Analyze the image and determine:
1. What is the likely purpose or use case of this image?
2. What context or setting does it belong to?
3. Who might be the intended audience?
4. What message or information is it trying to convey?`,
		Model:    provider,
		MaxTurns: agent.MaxTurnsConfig{Limit: 1},
	})

	result = executeVisionTask(contextAgent, "What is the purpose and context of this image?", imageFile)
	fmt.Printf("Context Analysis:\n%s\n\n", result)

	// Example 5: Multi-turn conversation with the image
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("Example 5: Multi-turn Conversation About the Image")
	fmt.Println(strings.Repeat("=", 80))

	conversationalAgent := agent.NewLLMAgent(agent.LLMAgentConfig{
		Name:     "VisionConversationalist",
		Prompt:   "You are a helpful assistant that can see and discuss images with users.",
		Model:    provider,
		MaxTurns: agent.MaxTurnsConfig{Limit: 5},
	})

	questions := []string{
		"What is the main subject of this image?",
		"Can you describe any text or symbols you see?",
		"What emotions or feelings does this image evoke?",
	}

	for i, question := range questions {
		fmt.Printf("Question %d: %s\n", i+1, question)
		result = executeVisionTask(conversationalAgent, question, imageFile)
		fmt.Printf("Answer: %s\n\n", result)
	}

	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("✓ All vision examples completed successfully!")
}

// executeVisionTask is a helper function to execute a vision task and handle errors
func executeVisionTask(ag agent.Agent, prompt string, imageFile agent.FileInput) string {
	task := &agent.Task{
		ID:    fmt.Sprintf("vision-task-%d", time.Now().Unix()),
		Input: prompt,
		Files: []agent.FileInput{imageFile},
		Config: &agent.ExecutionConfig{
			MaxIterations: 1,
		},
		StartedAt: time.Now(),
	}

	result, err := ag.Execute(context.Background(), task)
	if err != nil {
		log.Printf("Error executing agent: %v\n", err)
		return fmt.Sprintf("Error: %v", err)
	}

	if !result.Success {
		log.Printf("Agent execution failed: %s\n", result.Error)
		return fmt.Sprintf("Failed: %s", result.Error)
	}

	// Display metrics if available
	if result.TotalTokenUsage.TotalTokens > 0 {
		fmt.Printf("\n[Metrics] Tokens: %d | Latency: %v\n",
			result.TotalTokenUsage.TotalTokens,
			result.TotalLLMLatency)
	}

	return result.Output.(string)
}

// fetchImage downloads an image from a URL and returns the image data and MIME type
func fetchImage(url string) ([]byte, string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, "", fmt.Errorf("failed to fetch image: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("HTTP error: %d %s", resp.StatusCode, resp.Status)
	}

	imageData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", fmt.Errorf("failed to read image data: %w", err)
	}

	// Get MIME type from Content-Type header or detect from data
	mimeType := resp.Header.Get("Content-Type")
	if mimeType == "" {
		mimeType = detectImageMimeType(imageData)
	}

	return imageData, mimeType, nil
}

// detectImageMimeType detects the MIME type of an image from its binary data
func detectImageMimeType(data []byte) string {
	if len(data) < 4 {
		return "application/octet-stream"
	}

	// PNG: 89 50 4E 47
	if data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47 {
		return "image/png"
	}

	// JPEG: FF D8
	if data[0] == 0xFF && data[1] == 0xD8 {
		return "image/jpeg"
	}

	// GIF: 47 49
	if data[0] == 0x47 && data[1] == 0x49 {
		return "image/gif"
	}

	// WebP: 52 49 46 46
	if data[0] == 0x52 && data[1] == 0x49 && data[2] == 0x46 && data[3] == 0x46 {
		return "image/webp"
	}

	return "application/octet-stream"
}
