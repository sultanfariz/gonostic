# Vision Agent Example

This example demonstrates the **multimodal capabilities** of the gonostic framework, specifically how to use images as input to AI agents for visual understanding and analysis.

## Features

This example showcases five different use cases of vision-enabled agents:

1. **General Image Description** - Detailed analysis of image content, composition, and colors
2. **Object Detection and Analysis** - Identifying and counting objects, analyzing their positions
3. **Color and Style Analysis** - Analyzing visual design, color palette, and aesthetic
4. **Contextual Understanding** - Understanding purpose, audience, and message
5. **Multi-turn Conversation** - Interactive Q&A about the image

## Requirements

- Go 1.19 or later
- OpenAI API key (for GPT-4 with vision capabilities)

## Setup

1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. Build the example:
   ```bash
   go build -o vision .
   ```

3. Run the example:
   ```bash
   ./vision
   ```

## How It Works

### Fetching Images

The example fetches an image from a URL and converts it to a `FileInput`:

```go
imageData, mimeType, err := fetchImage(imageURL)
imageFile := agent.FileInput{
    Name:    "input_image.png",
    Type:    mimeType,
    Content: imageData,
    URI:     imageURL,
}
```

### Creating Vision-Enabled Agents

Vision agents are created using `NewLLMAgent` with a vision-capable model provider:

```go
visionAgent := agent.NewLLMAgent(agent.LLMAgentConfig{
    Name:   "ImageAnalyzer",
    Prompt: "You are an expert image analysis assistant...",
    Model:  provider, // OpenAI GPT-4o with vision
    MaxTurns: 1,
})
```

### Executing Vision Tasks

Pass the image through the `Files` field in the task:

```go
task := &agent.Task{
    Input: "Describe this image",
    Files: []agent.FileInput{imageFile},
    Config: &agent.ExecutionConfig{
        MaxIterations: 1,
    },
}

result, err := visionAgent.Execute(ctx, task)
```

## Customization

You can easily adapt this example for your own use cases:

- **Change the image source**: Modify `imageURL` to analyze different images
- **Add more prompts**: Create specialized agents for specific analysis tasks
- **Use local images**: Read from local files instead of URLs
- **Combine with tools**: Add tool-calling capabilities to vision agents
- **Multiple images**: Pass multiple images in the `Files` array

## Example Output

```
=== Multimodal Vision Agent Example ===

Fetching image from: https://upload.wikimedia.org/wikipedia/commons/8/80/Go-Logo_Blue.png
Image loaded successfully (45231 bytes, type: image/png)

================================================================================
Example 1: General Image Description
================================================================================

[Metrics] Tokens: 1234 | Latency: 2.5s
Description:
The image shows a modern mobile app interface with a clean, minimalist design...

================================================================================
Example 2: Object Detection and Analysis
================================================================================
...
```

## Supported Image Formats

- PNG (image/png)
- JPEG (image/jpeg)
- GIF (image/gif)
- WebP (image/webp)

## Notes

- The example uses GPT-4o which has native vision capabilities
- Token usage may be higher for vision tasks compared to text-only tasks
- Image size affects processing time and token consumption
- The framework automatically handles base64 encoding for the API
