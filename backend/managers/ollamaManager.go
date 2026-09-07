package managers

import (
	"context"

	"github.com/ollama/ollama/api"
)

const EMBEDDING_MODEL = "nomic-embed-text"

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

func (o *OllamaManager) Embed(input string) ([]float32, error) {
	body := api.EmbedRequest{Model: EMBEDDING_MODEL, Input: input}

	res, err := o.ollamaClient.Embed(context.TODO(), &body)
	if err != nil {
		return nil, err
	}
	return res.Embeddings[0], nil
}
