package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// ============================================================
// BRAVE SEARCH TOOL
// ============================================================

type BraveSearchTool struct {
	apiKey   string
	endpoint string
	client   *http.Client
}

type BraveSearchConfig struct {
	APIKey   string
	Endpoint string
	Timeout  time.Duration
}

func NewBraveSearchTool(cfg BraveSearchConfig) *BraveSearchTool {
	if cfg.Endpoint == "" {
		cfg.Endpoint = "https://api.search.brave.com/res/v1/web/search"
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 30 * time.Second
	}
	return &BraveSearchTool{
		apiKey:   cfg.APIKey,
		endpoint: cfg.Endpoint,
		client:   &http.Client{Timeout: cfg.Timeout},
	}
}

type WebSearchParams struct {
	Q               string `json:"q"`
	Count           int    `json:"count"`
	Offset          int    `json:"offset"`
	Safesearch      string `json:"safesearch,omitempty"`
	Freshness       string `json:"freshness,omitempty"`
	TextDecorations bool   `json:"text_decorations,omitempty"`
	Spellcheck      bool   `json:"spellcheck,omitempty"`
	SearchLang      string `json:"search_lang,omitempty"`
	ResultFilter    string `json:"result_filter,omitempty"`
}

type SearchResponse struct {
	Query struct {
		Original   string `json:"original"`
		Normalized string `json:"normalized"`
		Display    string `json:"display"`
	} `json:"query"`
	Type    string         `json:"type"`
	Web     WebResults     `json:"web"`
	News    []NewsResult   `json:"news,omitempty"`
	Images  []ImageResult  `json:"images,omitempty"`
	Videos  []VideoResult  `json:"videos,omitempty"`
}

type WebResults struct {
	Type           string      `json:"type"`
	Results        []WebResult `json:"results"`
	FamilyFriendly bool        `json:"family_friendly"`
}

type WebResult struct {
	Title           string            `json:"title"`
	URL             string            `json:"url"`
	IsSourceLocal   bool              `json:"is_source_local"`
	IsSourceBoth    bool              `json:"is_source_both"`
	Description     string            `json:"description"`
	PageAge         string            `json:"page_age,omitempty"`
	Language        string            `json:"language,omitempty"`
	FamilyFriendly  bool              `json:"family_friendly"`
	Type            string            `json:"type"`
	Subtype         string            `json:"subtype,omitempty"`
	IsLive          bool              `json:"is_live"`
	MetaURL         MetaURL           `json:"meta_url"`
	Age             string            `json:"age,omitempty"`
	ExtraSnippets   []string          `json:"extra_snippets,omitempty"`
}

type MetaURL struct {
	Scheme   string `json:"scheme"`
	Netloc   string `json:"netloc"`
	Hostname string `json:"hostname"`
	Favicon  string `json:"favicon"`
	Path     string `json:"path"`
}

type NewsResult struct {
	Title         string    `json:"title"`
	URL           string    `json:"url"`
	Description   string    `json:"description"`
	PublishedTime time.Time `json:"published_time,omitempty"`
	Source        string    `json:"source,omitempty"`
}

type ImageResult struct {
	Title      string `json:"title"`
	URL        string `json:"url"`
	Properties struct {
		Width  int `json:"width"`
		Height int `json:"height"`
	} `json:"properties"`
}

type VideoResult struct {
	Title     string `json:"title"`
	URL       string `json:"url"`
	Duration  int    `json:"duration,omitempty"`
	Thumbnail string `json:"thumbnail,omitempty"`
}

func (t *BraveSearchTool) Name() string {
	return "brave_search"
}

func (t *BraveSearchTool) Description() string {
	return "Search the web using Brave Search API. Returns web results, news, images, and videos."
}

func (t *BraveSearchTool) Schema() interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "Search query string",
			},
			"count": map[string]interface{}{
				"type":        "integer",
				"description": "Number of results to return (1-20, default 10)",
				"default":     10,
			},
			"result_filter": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"web", "news", "images", "videos"},
				"description": "Filter results by type (default: web)",
			},
			"freshness": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"pw", "pd", "pm", "py"},
				"description": "Time filter: pw=past week, pd=past day, pm=past month, py=past year",
			},
		},
		"required": []string{"query"},
	}
}

func (t *BraveSearchTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	query, ok := args["query"].(string)
	if !ok {
		return nil, fmt.Errorf("query is required and must be a string")
	}

	params := WebSearchParams{
		Q:     query,
		Count: 10,
	}

	if count, ok := args["count"].(float64); ok {
		params.Count = int(count)
	}

	if filter, ok := args["result_filter"].(string); ok {
		params.ResultFilter = filter
	}

	if freshness, ok := args["freshness"].(string); ok {
		params.Freshness = freshness
	}

	return t.Search(ctx, params)
}

func (t *BraveSearchTool) Search(ctx context.Context, req WebSearchParams) (SearchResponse, error) {
	values := make(map[string]string)
	values["q"] = req.Q
	if req.Count > 0 {
		values["count"] = fmt.Sprintf("%d", req.Count)
	}
	if req.Offset > 0 {
		values["offset"] = fmt.Sprintf("%d", req.Offset)
	}
	if req.Safesearch != "" {
		values["safesearch"] = req.Safesearch
	}
	if req.Freshness != "" {
		values["freshness"] = req.Freshness
	}
	if req.ResultFilter != "" {
		values["result_filter"] = req.ResultFilter
	}
	if req.SearchLang != "" {
		values["search_lang"] = req.SearchLang
	}

	urlStr := t.endpoint + "?q=" + url.QueryEscape(req.Q)
	for k, v := range values {
		if k != "q" {
			urlStr += "&" + k + "=" + url.QueryEscape(v)
		}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "GET", urlStr, nil)
	if err != nil {
		return SearchResponse{}, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Accept", "application/json")
	httpReq.Header.Set("X-Subscription-Token", t.apiKey)

	resp, err := t.client.Do(httpReq)
	if err != nil {
		return SearchResponse{}, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return SearchResponse{}, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return SearchResponse{}, fmt.Errorf("brave api error: status %d, body: %s", resp.StatusCode, string(body))
	}

	var searchResp SearchResponse
	if err := json.Unmarshal(body, &searchResp); err != nil {
		return SearchResponse{}, fmt.Errorf("unmarshal response: %w", err)
	}

	return searchResp, nil
}
