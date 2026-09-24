package managers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ollama/ollama/api"
)

// TestEmbedPrefixes checks each embed variant sends the right task prefix to
// Ollama, against a fake /api/embed that records the input it received.
func TestEmbedPrefixes(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req api.EmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode embed request: %v", err)
		}
		got, _ = req.Input.(string)
		json.NewEncoder(w).Encode(api.EmbedResponse{Model: req.Model, Embeddings: [][]float32{{1}}})
	}))
	t.Cleanup(srv.Close)
	t.Setenv("OLLAMA_HOST", srv.URL)

	o := NewOllamaManager()
	cases := []struct {
		name  string
		embed func(string) ([]float32, error)
		want  string
	}{
		{"raw", o.Embed, "tea"},
		{"document", o.EmbedDocument, "search_document: tea"},
		{"query", o.EmbedQuery, "search_query: tea"},
	}
	for _, tc := range cases {
		if _, err := tc.embed("tea"); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if got != tc.want {
			t.Fatalf("%s: expected input %q, got %q", tc.name, tc.want, got)
		}
	}
}
