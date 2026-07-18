# Structured Output Example

This example demonstrates using JSON schema to enforce structured output from LLM agents using the gonostic framework.

## Overview

The example implements a **Product Information Extractor** that takes unstructured product descriptions and returns structured JSON data conforming to a predefined schema.

## Features

- **JSON Schema Enforcement**: Output is guaranteed to match the defined schema
- **Structured Data Extraction**: Extracts product name, category, price, brand, features, and more
- **OpenAI Integration**: Uses OpenAI's structured output API with `gpt-4o-2024-08-06`
- **Token Usage Tracking**: Displays metrics for each extraction

## Prerequisites

- Go 1.21 or later
- OpenAI API key with access to structured output models

## Setup

1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. Build the example:
   ```bash
   go build -o structured-output.exe ./example/structured-output
   ```

3. Run the example:
   ```bash
   ./structured-output.exe
   ```

## How It Works

### 1. Define Output Schema

The JSON schema defines the structure of the expected output:

```go
productSchema := map[string]interface{}{
    "type": "object",
    "properties": map[string]interface{}{
        "name":        map[string]interface{}{"type": "string"},
        "category":    map[string]interface{}{"type": "string"},
        "price":       map[string]interface{}{"type": "number"},
        "currency":    map[string]interface{}{"type": "string"},
        "brand":       map[string]interface{}{"type": "string"},
        "features":    map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "string"}},
        "description": map[string]interface{}{"type": "string"},
        "in_stock":    map[string]interface{}{"type": "boolean"},
    },
    "required": []string{"name", "category", "price", "currency", "brand", "features", "description", "in_stock"},
}
```

### 2. Create Agent with Schema

Pass the schema to the agent configuration:

```go
productExtractor := agent.NewLLMAgent(agent.LLMAgentConfig{
    Name:         "ProductExtractor",
    Prompt:       "Extract structured product information...",
    OutputSchema: productSchema,
    Model:        provider, // Must implement StructuredModelProvider
    MaxTurns:     1,
})
```

### 3. Execute and Parse

The agent returns JSON that matches the schema:

```go
result, err := productExtractor.Execute(ctx, task)
var product ProductInfo
json.Unmarshal([]byte(result.Output.(string)), &product)
```

## Implementation Details

### OpenAI Provider Extensions

The `OpenAIModelProvider` implements both:
- `ModelProvider` - Standard completion interface
- `StructuredModelProvider` - Structured output with JSON schema

```go
type StructuredModelProvider interface {
    ModelProvider
    CompleteWithSchema(ctx context.Context, prompt string, files []FileInput,
        tools []Tool, history []Message, schema map[string]interface{}) (*ModelResponse, error)
}
```

### Behavior

- **With Schema**: Uses OpenAI's `response_format` with `json_schema` type
- **Without Schema**: Falls back to regular completion (backward compatible)

## Example Output

```
Example 1:
Input: Check out the new iPhone 15 Pro from Apple! ...

Extracted Product Information:
  Name:        iPhone 15 Pro
  Brand:       Apple
  Category:    Electronics
  Price:       999.00 USD
  In Stock:    true
  Description: Premium smartphone with titanium design and A17 Pro chip
  Features:
    - Titanium design
    - A17 Pro chip
    - 6.1-inch Super Retina XDR display
    - 48MP pro camera system

Metrics:
  Tokens Used: 345 (prompt: 234, completion: 111)
  Latency:     1.2s
```

## Use Cases

- **E-commerce**: Extract product data from listings
- **Content Processing**: Structure unstructured text
- **Data Normalization**: Convert free-form input to standardized formats
- **Form Filling**: Extract information for structured forms
- **API Integration**: Generate API-ready payloads from natural language

## Notes

- Requires OpenAI models with structured output support (e.g., `gpt-4o-2024-08-06`)
- Schema must be valid JSON Schema with `strict: true` mode
- All required fields must be present in the schema
- Works with any use case where structured output is needed
