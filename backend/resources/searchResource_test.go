package resources

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"gopkg.in/yaml.v3"

	"daybid-dev-service/daos"
	"daybid-dev-service/managers"
)

// fakeSearchIndex stands in for *daos.EmbeddingsDao in tests: it returns
// pre-seeded hits instead of talking to a real Postgres, and records the
// filters/k it was called with so tests can assert on them.
type fakeSearchIndex struct {
	hits []daos.SearchHit

	gotK       int
	gotFilters daos.SearchFilters
}

func (f *fakeSearchIndex) Search(_ context.Context, _ string, _ []float32, k int, filters daos.SearchFilters) ([]daos.SearchHit, error) {
	f.gotK = k
	f.gotFilters = filters
	return f.hits, nil
}

// newSearchTestServer wires the search route onto a fresh gin engine backed
// by the local filesystem manager rooted at a per-test temp directory, and
// seeds memory files directly onto disk (standing in for what indexMemory
// would have embedded on write) so the fake index's hits can resolve to real
// content for snippet/hydrate extraction.
func newSearchTestServer(t *testing.T, hits []daos.SearchHit, seed map[string]string) (*httptest.Server, *fakeSearchIndex) {
	t.Helper()

	home := t.TempDir()
	connectomeDir := filepath.Join(home, ".connectome")
	if err := os.MkdirAll(connectomeDir, 0o755); err != nil {
		t.Fatalf("create connectome dir: %v", err)
	}
	for key, content := range seed {
		if err := os.WriteFile(filepath.Join(connectomeDir, key), []byte(content), 0o644); err != nil {
			t.Fatalf("seed %s: %v", key, err)
		}
	}

	t.Setenv("HOME", home)
	t.Setenv("MEMORY_MANAGER", "local")
	t.Setenv("apikey", testToken)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	index := &fakeSearchIndex{hits: hits}
	InitSearchResource(r.Group("/api/connectome"), managers.NewMemoryManagerFromEnv(), fakeEmbedder{}, index)

	srv := httptest.NewServer(r)
	t.Cleanup(srv.Close)

	return srv, index
}

func TestSearchRequiresBearerToken(t *testing.T) {
	srv, _ := newSearchTestServer(t, nil, nil)

	req, err := http.NewRequest(http.MethodPost, srv.URL+"/api/connectome/memory/search", strings.NewReader(`{"query":"tea"}`))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("do request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401 with no Authorization header, got %d", resp.StatusCode)
	}
}

func TestSearchRejectsEmptyQuery(t *testing.T) {
	srv, _ := newSearchTestServer(t, nil, nil)

	resp, payload := doRequest(t, http.MethodPost, srv.URL+"/api/connectome/memory/search", strings.NewReader(`{"query":"  "}`), map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 for blank query, got %d (%s)", resp.StatusCode, payload)
	}
}

func TestSearchReturnsRankedSnippets(t *testing.T) {
	const key = "mem_tea.md"

	// Build a body long enough to split into exactly two chunks (chunk size
	// 512 words, overlap 50, so chunk 0 covers words [0,512) and chunk 1
	// covers [462,700)), with a marker word placed so it lands in only one
	// chunk and well within the snippet's truncation window.
	words := make([]string, 700)
	for i := range words {
		words[i] = "f"
	}
	words[0] = "ALPHA"   // chunk 0 only
	words[520] = "OMEGA" // chunk 1 only, near its start so truncation keeps it
	content := memoryDocument("preference", strings.Join(words, " "), []string{"david"})

	srv, index := newSearchTestServer(t, []daos.SearchHit{
		{MemoryKey: key, ChunkIndex: 1, Type: "preference", Score: 0.8},
	}, map[string]string{key: content})

	resp, payload := doRequest(t, http.MethodPost, srv.URL+"/api/connectome/memory/search",
		strings.NewReader(`{"query":"what does David drink","k":3,"filters":{"type":"preference","entity":"david"}}`),
		map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", resp.StatusCode, payload)
	}

	if index.gotK != 3 {
		t.Fatalf("expected k=3 forwarded to the index, got %d", index.gotK)
	}
	if index.gotFilters.Type == nil || *index.gotFilters.Type != "preference" {
		t.Fatalf("expected type filter 'preference' forwarded, got %v", index.gotFilters.Type)
	}
	if index.gotFilters.Entity == nil || *index.gotFilters.Entity != "david" {
		t.Fatalf("expected entity filter 'david' forwarded, got %v", index.gotFilters.Entity)
	}

	var decoded struct {
		Results []SearchResult `json:"results"`
	}
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("decode %q: %v", payload, err)
	}
	if len(decoded.Results) != 1 {
		t.Fatalf("expected 1 result, got %d: %+v", len(decoded.Results), decoded.Results)
	}
	result := decoded.Results[0]
	if result.Key != key {
		t.Fatalf("expected key %s, got %s", key, result.Key)
	}
	if result.Type != "preference" {
		t.Fatalf("expected type preference, got %s", result.Type)
	}
	if result.Score != 0.8 {
		t.Fatalf("expected score to pass through the hit's blended score 0.8, got %v", result.Score)
	}
	if result.Content != "" {
		t.Fatalf("expected no content without hydrate, got %q", result.Content)
	}
	if !strings.Contains(result.Snippet, "OMEGA") {
		t.Fatalf("expected chunk 1's snippet to contain its marker word OMEGA, got %q", result.Snippet)
	}
	if strings.Contains(result.Snippet, "ALPHA") {
		t.Fatalf("expected chunk 1's snippet to exclude chunk 0's marker word ALPHA, got %q", result.Snippet)
	}
}

func TestSearchHydrateReturnsFullContent(t *testing.T) {
	const key = "mem_short.md"
	// An explicit (even empty) acl means hydration has nothing to resolve,
	// so the stored document round-trips byte-for-byte - see
	// TestSearchHydrateResolvesACLForPreOneOneMemory for the defaulting case.
	content := "---\n" +
		"type: note\n" +
		"created_at: \"2024-01-01T00:00:00Z\"\n" +
		"entities: []\n" +
		"acl: []\n" +
		"---\nshort body here\n"

	srv, _ := newSearchTestServer(t, []daos.SearchHit{
		{MemoryKey: key, ChunkIndex: 0, Type: "note", Score: 0.9},
	}, map[string]string{key: content})

	resp, payload := doRequest(t, http.MethodPost, srv.URL+"/api/connectome/memory/search",
		strings.NewReader(`{"query":"anything","hydrate":true}`),
		map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", resp.StatusCode, payload)
	}

	var decoded struct {
		Results []SearchResult `json:"results"`
	}
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("decode %q: %v", payload, err)
	}
	if len(decoded.Results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(decoded.Results))
	}
	if decoded.Results[0].Content != content {
		t.Fatalf("expected hydrated content to equal stored document, got %q", decoded.Results[0].Content)
	}
	if decoded.Results[0].Snippet != "" {
		t.Fatalf("expected no snippet when hydrate is set, got %q", decoded.Results[0].Snippet)
	}
}

func TestSearchHydrateResolvesACLForPreOneOneMemory(t *testing.T) {
	const key = "mem_pre11.md"
	// A pre-1.1 memory: no acl key at all.
	content := memoryDocument("note", "short body here", nil)

	t.Setenv("DEFAULT_ACL", "GM")
	srv, _ := newSearchTestServer(t, []daos.SearchHit{
		{MemoryKey: key, ChunkIndex: 0, Type: "note", Score: 0.9},
	}, map[string]string{key: content})

	resp, payload := doRequest(t, http.MethodPost, srv.URL+"/api/connectome/memory/search",
		strings.NewReader(`{"query":"anything","hydrate":true}`),
		map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", resp.StatusCode, payload)
	}

	var decoded struct {
		Results []SearchResult `json:"results"`
	}
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("decode %q: %v", payload, err)
	}
	if len(decoded.Results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(decoded.Results))
	}

	raw, _, ok := splitFrontmatter(decoded.Results[0].Content)
	if !ok {
		t.Fatalf("expected hydrated content to still have parseable frontmatter, got %q", decoded.Results[0].Content)
	}
	var fm fullMemoryFrontmatter
	if err := yaml.Unmarshal([]byte(raw), &fm); err != nil {
		t.Fatalf("unmarshal hydrated frontmatter: %v", err)
	}
	if fm.ACL == nil || len(*fm.ACL) != 1 || (*fm.ACL)[0] != "GM" {
		t.Fatalf("expected hydrated content to resolve acl to the configured DEFAULT_ACL [GM], got %v", fm.ACL)
	}
}

func TestSearchDefaultsKAndReturnsEmptyResultsWhenNoHits(t *testing.T) {
	srv, index := newSearchTestServer(t, nil, nil)

	resp, payload := doRequest(t, http.MethodPost, srv.URL+"/api/connectome/memory/search",
		strings.NewReader(`{"query":"nothing indexed yet"}`),
		map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	if index.gotK != defaultSearchK {
		t.Fatalf("expected default k=%d, got %d", defaultSearchK, index.gotK)
	}

	var decoded struct {
		Results []SearchResult `json:"results"`
	}
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("decode %q: %v", payload, err)
	}
	if len(decoded.Results) != 0 {
		t.Fatalf("expected no results, got %+v", decoded.Results)
	}
}

func TestSearchResolvesACLScopeFromAsEntitysMemberOf(t *testing.T) {
	srv, index := newSearchTestServer(t, nil, map[string]string{
		"ent_alice.json": `{"id":"alice","name":"Alice","member_of":["Party A","Adventurers"]}`,
	})

	resp, payload := doRequest(t, http.MethodPost, srv.URL+"/api/connectome/memory/search",
		strings.NewReader(`{"query":"anything","as":"alice"}`),
		map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", resp.StatusCode, payload)
	}

	got := index.gotFilters.ACLScope
	if len(got) != 3 || got[0] != "alice" || got[1] != "Party A" || got[2] != "Adventurers" {
		t.Fatalf("expected ACLScope [alice, Party A, Adventurers], got %v", got)
	}
}

func TestSearchWithoutAsAppliesNoACLScope(t *testing.T) {
	srv, index := newSearchTestServer(t, nil, nil)

	resp, payload := doRequest(t, http.MethodPost, srv.URL+"/api/connectome/memory/search",
		strings.NewReader(`{"query":"anything"}`),
		map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", resp.StatusCode, payload)
	}

	if index.gotFilters.ACLScope != nil {
		t.Fatalf("expected nil ACLScope with no `as`, got %v", index.gotFilters.ACLScope)
	}
}

func TestSearchAsWithoutEntityRecordScopesToJustThatID(t *testing.T) {
	srv, index := newSearchTestServer(t, nil, nil)

	resp, payload := doRequest(t, http.MethodPost, srv.URL+"/api/connectome/memory/search",
		strings.NewReader(`{"query":"anything","as":"ghost"}`),
		map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", resp.StatusCode, payload)
	}

	got := index.gotFilters.ACLScope
	if len(got) != 1 || got[0] != "ghost" {
		t.Fatalf("expected ACLScope [ghost] when the entity record doesn't exist, got %v", got)
	}
}

func TestSearchClampsKToMax(t *testing.T) {
	srv, index := newSearchTestServer(t, nil, nil)

	_, payload := doRequest(t, http.MethodPost, srv.URL+"/api/connectome/memory/search",
		strings.NewReader(`{"query":"broad query","k":10000}`),
		map[string]string{"Content-Type": "application/json"})
	if index.gotK != maxSearchK {
		t.Fatalf("expected k clamped to %d, got %d (%s)", maxSearchK, index.gotK, payload)
	}
}
