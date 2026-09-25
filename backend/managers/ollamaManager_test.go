package managers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// fakeOllama serves a fake /api/embed that records the input it received
// and returns one single-component embedding per input.
func fakeOllama(t *testing.T, got *any) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req api.EmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode embed request: %v", err)
		}
		*got = req.Input
		n := 1
		if inputs, ok := req.Input.([]any); ok {
			n = len(inputs)
		}
		embeddings := make([][]float32, n)
		for i := range embeddings {
			embeddings[i] = []float32{float32(i)}
		}
		json.NewEncoder(w).Encode(api.EmbedResponse{Model: req.Model, Embeddings: embeddings})
	}))
	t.Cleanup(srv.Close)
	t.Setenv("OLLAMA_HOST", srv.URL)
}

// TestEmbedPrefixes checks each embed variant sends the right task prefix to
// Ollama, against a fake /api/embed that records the input it received.
func TestEmbedPrefixes(t *testing.T) {
	var got any
	fakeOllama(t, &got)

	o := NewOllamaManager()
	ctx := context.Background()
	cases := []struct {
		name  string
		embed func(context.Context, string) ([]float32, error)
		want  string
	}{
		{"raw", o.Embed, "tea"},
		{"query", o.EmbedQuery, "search_query: tea"},
	}
	for _, tc := range cases {
		if _, err := tc.embed(ctx, "tea"); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if got != tc.want {
			t.Fatalf("%s: expected input %q, got %v", tc.name, tc.want, got)
		}
	}
}

// TestEmbedDocumentsBatches checks a memory's chunks go to Ollama as one
// multi-input request, each with the document prefix, and come back in order.
func TestEmbedDocumentsBatches(t *testing.T) {
	var got any
	fakeOllama(t, &got)

	embeddings, err := NewOllamaManager().EmbedDocuments(context.Background(), []string{"tea", "coffee"})
	if err != nil {
		t.Fatalf("embed documents: %v", err)
	}
	want := []any{"search_document: tea", "search_document: coffee"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("expected input %v, got %v", want, got)
	}
	if len(embeddings) != 2 || embeddings[0][0] != 0 || embeddings[1][0] != 1 {
		t.Fatalf("expected one embedding per input in order, got %v", embeddings)
	}
}

// TestEmbedHonorsContext checks a cancelled context aborts an in-flight
// embed instead of waiting on Ollama.
func TestEmbedHonorsContext(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })
	t.Setenv("OLLAMA_HOST", srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := NewOllamaManager().EmbedQuery(ctx, "tea")
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected context deadline error, got %v", err)
	}
}
