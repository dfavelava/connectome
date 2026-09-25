package resources

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"connectome-dev-service/managers"
)

// gatedEmbedder blocks every EmbedDocuments call until release is closed or
// the call's context is cancelled, tracking how many calls are in flight at
// once and how many were ever started.
type gatedEmbedder struct {
	release chan struct{}

	mu          sync.Mutex
	inFlight    int
	maxInFlight int
	started     int
	cancelled   int
}

func newGatedEmbedder() *gatedEmbedder {
	return &gatedEmbedder{release: make(chan struct{})}
}

func (g *gatedEmbedder) EmbedDocuments(ctx context.Context, inputs []string) ([][]float32, error) {
	g.mu.Lock()
	g.started++
	g.inFlight++
	g.maxInFlight = max(g.maxInFlight, g.inFlight)
	g.mu.Unlock()

	defer func() {
		g.mu.Lock()
		g.inFlight--
		g.mu.Unlock()
	}()

	select {
	case <-g.release:
		return fakeEmbedder{}.EmbedDocuments(ctx, inputs)
	case <-ctx.Done():
		g.mu.Lock()
		g.cancelled++
		g.mu.Unlock()
		return nil, ctx.Err()
	}
}

func (g *gatedEmbedder) EmbedQuery(ctx context.Context, input string) ([]float32, error) {
	return fakeEmbedder{}.EmbedQuery(ctx, input)
}

func (g *gatedEmbedder) snapshot() (inFlight, maxInFlight, started, cancelled int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.inFlight, g.maxInFlight, g.started, g.cancelled
}

// waitFor polls cond until it holds or a deadline passes.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// newGatedResource builds a memory resource over a per-test local blob store
// with INDEX_CONCURRENCY set to limit.
func newGatedResource(t *testing.T, limit int, embedder Embedder) (*MemoryResourceImpl, *fakeIndexer) {
	t.Helper()

	home := t.TempDir()
	if err := os.MkdirAll(filepath.Join(home, ".connectome"), 0o755); err != nil {
		t.Fatalf("create connectome dir: %v", err)
	}
	t.Setenv("HOME", home)
	t.Setenv("MEMORY_MANAGER", "local")
	t.Setenv("apikey", testToken)
	t.Setenv("INDEX_CONCURRENCY", fmt.Sprint(limit))

	indexer := newFakeIndexer()
	return NewMemoryResource(managers.NewMemoryManagerFromEnv(), embedder, indexer), indexer
}

func memoryParts(prefix string, n int) []filePart {
	parts := make([]filePart, n)
	for i := range parts {
		parts[i] = filePart{
			name:    fmt.Sprintf("mem_%s%d.md", prefix, i),
			content: memoryDocument("note", fmt.Sprintf("memory %s%d", prefix, i), nil),
		}
	}
	return parts
}

// TestBatchWriteBoundsInFlightEmbeds sends two concurrent batch writes of
// several memories each and checks the number of embed calls in flight never
// exceeds INDEX_CONCURRENCY, across both requests, while every memory still
// ends up indexed.
func TestBatchWriteBoundsInFlightEmbeds(t *testing.T) {
	const limit = 2
	embedder := newGatedEmbedder()
	resource, indexer := newGatedResource(t, limit, embedder)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST("/batch", resource.batchWrite)
	srv := httptest.NewServer(r)
	t.Cleanup(srv.Close)

	batches := [][]filePart{memoryParts("a", 5), memoryParts("b", 5)}

	var wg sync.WaitGroup
	statuses := make([]int, len(batches))
	for i, parts := range batches {
		wg.Add(1)
		go func() {
			defer wg.Done()
			body, contentType := multipartBody(t, parts)
			resp, _ := doRequest(t, http.MethodPost, srv.URL+"/batch", body, map[string]string{"Content-Type": contentType})
			statuses[i] = resp.StatusCode
		}()
	}

	waitFor(t, "embeds to fill the limit", func() bool {
		inFlight, _, _, _ := embedder.snapshot()
		return inFlight == limit
	})
	// Give any embed that would overshoot the limit time to start.
	time.Sleep(50 * time.Millisecond)
	close(embedder.release)
	wg.Wait()

	_, maxInFlight, started, _ := embedder.snapshot()
	if maxInFlight > limit {
		t.Fatalf("expected at most %d embeds in flight, saw %d", limit, maxInFlight)
	}
	if started != 10 {
		t.Fatalf("expected one embed call per memory (10), got %d", started)
	}
	for i, status := range statuses {
		if status != http.StatusOK {
			t.Fatalf("batch %d: expected 200, got %d", i, status)
		}
	}
	for _, parts := range batches {
		for _, p := range parts {
			if rows := indexer.rowsFor(p.name); len(rows) != 1 {
				t.Fatalf("expected %s indexed with 1 row, got %d", p.name, len(rows))
			}
		}
	}
}

// TestBatchWriteCancelStopsPendingEmbeds cancels a batch write while its
// first embed is in flight and checks that embed is cancelled and the
// memories still waiting for a slot are never embedded.
func TestBatchWriteCancelStopsPendingEmbeds(t *testing.T) {
	embedder := newGatedEmbedder()
	t.Cleanup(func() { close(embedder.release) })
	resource, indexer := newGatedResource(t, 1, embedder)

	parts := memoryParts("c", 4)
	body, contentType := multipartBody(t, parts)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req := httptest.NewRequestWithContext(ctx, http.MethodPost, "/batch", body)
	req.Header.Set("Content-Type", contentType)

	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = req

	done := make(chan struct{})
	go func() {
		defer close(done)
		resource.batchWrite(c)
	}()

	waitFor(t, "the first embed to start", func() bool {
		inFlight, _, _, _ := embedder.snapshot()
		return inFlight == 1
	})
	cancel()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("batch write did not return after its request was cancelled")
	}

	_, _, started, cancelled := embedder.snapshot()
	if started != 1 {
		t.Fatalf("expected only the in-flight embed to have started, got %d", started)
	}
	if cancelled != 1 {
		t.Fatalf("expected the in-flight embed to be cancelled, got %d cancellations", cancelled)
	}
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500 for a cancelled batch, got %d", w.Code)
	}
	for _, p := range parts {
		if rows := indexer.rowsFor(p.name); len(rows) != 0 {
			t.Fatalf("expected %s left unindexed, got %d rows", p.name, len(rows))
		}
	}
}

func TestIndexConcurrencyFromEnv(t *testing.T) {
	cases := map[string]int{
		"":     defaultIndexConcurrency,
		"8":    8,
		" 3 ":  3,
		"0":    defaultIndexConcurrency,
		"-1":   defaultIndexConcurrency,
		"many": defaultIndexConcurrency,
	}
	for raw, want := range cases {
		t.Setenv("INDEX_CONCURRENCY", raw)
		if got := indexConcurrencyFromEnv(); got != want {
			t.Fatalf("INDEX_CONCURRENCY=%q: expected %d, got %d", raw, want, got)
		}
	}
}
