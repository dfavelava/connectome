package managers

import (
	"context"
	"fmt"

	"github.com/ollama/ollama/api"
)

const EMBEDDING_MODEL = "nomic-embed-text"

// nomic-embed-text is trained with task prefixes that tell it whether a text
// is something to be retrieved or a query retrieving it; embedding both sides
// with their prefix measurably improves retrieval over raw text (issue #9).
// Changing either prefix changes every stored vector, so existing indexes
// must be rebuilt with cmd/reindex.
const (
	DocumentPrefix = "search_document: "
	QueryPrefix    = "search_query: "
)

type OllamaManager struct {
	ollamaClient *api.Client
}

func NewOllamaManager() *OllamaManager {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil
	}
	return &OllamaManager{ollamaClient: client}
}

// Embed embeds input as-is, with no task prefix.
func (o *OllamaManager) Embed(ctx context.Context, input string) ([]float32, error) {
	embeddings, err := o.embed(ctx, input, 1)
	if err != nil {
		return nil, err
	}
	return embeddings[0], nil
}

// EmbedDocuments embeds each input as a document to be stored and retrieved,
// in a single multi-input request, returning one embedding per input in order.
func (o *OllamaManager) EmbedDocuments(ctx context.Context, inputs []string) ([][]float32, error) {
	if len(inputs) == 0 {
		return nil, nil
	}
	prefixed := make([]string, len(inputs))
	for i, input := range inputs {
		prefixed[i] = DocumentPrefix + input
	}
	return o.embed(ctx, prefixed, len(inputs))
}

// EmbedQuery embeds input as a search query against stored documents.
func (o *OllamaManager) EmbedQuery(ctx context.Context, input string) ([]float32, error) {
	return o.Embed(ctx, QueryPrefix+input)
}

// embed sends input (a string or []string) to Ollama's /api/embed and checks
// it got back the want embeddings the input implies.
func (o *OllamaManager) embed(ctx context.Context, input any, want int) ([][]float32, error) {
	body := api.EmbedRequest{Model: EMBEDDING_MODEL, Input: input}

	res, err := o.ollamaClient.Embed(ctx, &body)
	if err != nil {
		return nil, err
	}
	if len(res.Embeddings) != want {
		return nil, fmt.Errorf("embed: expected %d embeddings, got %d", want, len(res.Embeddings))
	}
	return res.Embeddings, nil
}
