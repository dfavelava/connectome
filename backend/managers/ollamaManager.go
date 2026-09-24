package managers

import (
	"context"

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
func (o *OllamaManager) Embed(input string) ([]float32, error) {
	body := api.EmbedRequest{Model: EMBEDDING_MODEL, Input: input}

	res, err := o.ollamaClient.Embed(context.TODO(), &body)
	if err != nil {
		return nil, err
	}
	return res.Embeddings[0], nil
}

// EmbedDocument embeds input as a document to be stored and retrieved.
func (o *OllamaManager) EmbedDocument(input string) ([]float32, error) {
	return o.Embed(DocumentPrefix + input)
}

// EmbedQuery embeds input as a search query against stored documents.
func (o *OllamaManager) EmbedQuery(input string) ([]float32, error) {
	return o.Embed(QueryPrefix + input)
}
