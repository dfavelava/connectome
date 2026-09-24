package resources

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"connectome-dev-service/daos"
	"connectome-dev-service/managers"
)

const testToken = "e2e-test-token"

// fakeEmbedder stands in for *managers.OllamaManager in tests: it returns a
// deterministic, cheap embedding without needing a real Ollama server.
type fakeEmbedder struct{}

func (fakeEmbedder) Embed(input string) ([]float32, error) {
	return []float32{float32(len(input))}, nil
}

// fakeIndexer stands in for *daos.EmbeddingsDao in tests: it records rows
// in memory instead of talking to a real Postgres, and counts inserts so
// tests can assert whether a re-index happened.
type fakeIndexer struct {
	mu      sync.Mutex
	rows    map[string][]daos.EmbeddingRow
	inserts int
}

func newFakeIndexer() *fakeIndexer {
	return &fakeIndexer{rows: make(map[string][]daos.EmbeddingRow)}
}

func (f *fakeIndexer) InsertEmbeddings(_ context.Context, memoryKey string, rows []daos.EmbeddingRow) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.inserts++
	f.rows[memoryKey] = append(f.rows[memoryKey], rows...)
	return nil
}

func (f *fakeIndexer) DeleteEmbeddingsForKey(_ context.Context, memoryKey string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.rows, memoryKey)
	return nil
}

// DeleteEmbeddingsForTome removes every row whose TomeID matches tomeID,
// mirroring the real `DELETE FROM embeddings WHERE tome_id = $1`.
func (f *fakeIndexer) DeleteEmbeddingsForTome(_ context.Context, tomeID string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	for key, rows := range f.rows {
		kept := rows[:0]
		for _, row := range rows {
			if row.TomeID != tomeID {
				kept = append(kept, row)
			}
		}
		if len(kept) == 0 {
			delete(f.rows, key)
		} else {
			f.rows[key] = kept
		}
	}
	return nil
}

func (f *fakeIndexer) rowsFor(key string) []daos.EmbeddingRow {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]daos.EmbeddingRow{}, f.rows[key]...)
}

func (f *fakeIndexer) insertCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.inserts
}

// newTestServer wires the memory routes onto a fresh gin engine backed by the
// local filesystem manager rooted at a per-test temp directory, and returns an
// httptest server, that directory's .connectome path, and the fake indexer
// standing in for the embeddings table so tests can assert on it.
func newTestServer(t *testing.T) (*httptest.Server, string, *fakeIndexer) {
	t.Helper()

	home := t.TempDir()
	connectomeDir := filepath.Join(home, ".connectome")
	if err := os.MkdirAll(connectomeDir, 0o755); err != nil {
		t.Fatalf("create connectome dir: %v", err)
	}

	t.Setenv("HOME", home)
	t.Setenv("MEMORY_MANAGER", "local")
	t.Setenv("apikey", testToken)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	indexer := newFakeIndexer()
	InitMemoryResource(r.Group("/api/connectome"), managers.NewMemoryManagerFromEnv(), fakeEmbedder{}, indexer)

	srv := httptest.NewServer(r)
	t.Cleanup(srv.Close)

	return srv, connectomeDir, indexer
}

type filePart struct {
	name    string
	content string
}

func multipartBody(t *testing.T, parts []filePart) (*bytes.Buffer, string) {
	t.Helper()

	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for _, p := range parts {
		fw, err := w.CreateFormFile("file", p.name)
		if err != nil {
			t.Fatalf("create form file %s: %v", p.name, err)
		}
		if _, err := io.WriteString(fw, p.content); err != nil {
			t.Fatalf("write form file %s: %v", p.name, err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	return &buf, w.FormDataContentType()
}

func doRequest(t *testing.T, method, url string, body io.Reader, headers map[string]string) (*http.Response, []byte) {
	t.Helper()

	req, err := http.NewRequest(method, url, body)
	if err != nil {
		t.Fatalf("new request %s %s: %v", method, url, err)
	}
	req.Header.Set("Authorization", "Bearer "+testToken)
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("do request %s %s: %v", method, url, err)
	}
	defer resp.Body.Close()

	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body %s %s: %v", method, url, err)
	}
	return resp, payload
}

func decodeJSON(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	var out map[string]any
	if err := json.Unmarshal(payload, &out); err != nil {
		t.Fatalf("decode json %q: %v", string(payload), err)
	}
	return out
}

func TestMemoryRoutesRequireBearerToken(t *testing.T) {
	srv, _, _ := newTestServer(t)

	cases := []struct {
		name   string
		header string
	}{
		{name: "missing header", header: ""},
		{name: "wrong scheme", header: "Token " + testToken},
		{name: "bad key", header: "Bearer wrong-token"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req, err := http.NewRequest(http.MethodGet, srv.URL+"/api/connectome/memory/list", nil)
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			if tc.header != "" {
				req.Header.Set("Authorization", tc.header)
			}

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatalf("do request: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusUnauthorized {
				t.Fatalf("expected 401, got %d", resp.StatusCode)
			}
		})
	}
}

func TestMemoryWriteReadListDeleteRoundTrip(t *testing.T) {
	srv, connectomeDir, _ := newTestServer(t)
	base := srv.URL + "/api/connectome/memory"

	const key = "mem_roundtrip.md"
	const content = "David prefers tea over coffee."

	// write
	body, contentType := multipartBody(t, []filePart{{name: key, content: content}})
	resp, payload := doRequest(t, http.MethodPost, base+"/", body, map[string]string{"Content-Type": contentType})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("write: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	if got := decodeJSON(t, payload)["message"]; got != "success" {
		t.Fatalf("write: expected message success, got %v", got)
	}

	onDisk, err := os.ReadFile(filepath.Join(connectomeDir, key))
	if err != nil {
		t.Fatalf("read written file: %v", err)
	}
	if string(onDisk) != content {
		t.Fatalf("written file mismatch: got %q", string(onDisk))
	}

	// read
	resp, payload = doRequest(t, http.MethodGet, base+"/?key="+key, nil, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("read: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	if got := decodeJSON(t, payload)["content"]; got != content {
		t.Fatalf("read: expected content %q, got %v", content, got)
	}

	// list
	resp, payload = doRequest(t, http.MethodGet, base+"/list", nil, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("list: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	var listResult struct {
		Contents []struct {
			Key     string  `json:"Key"`
			Preview *string `json:"Preview"`
		} `json:"Contents"`
	}
	if err := json.Unmarshal(payload, &listResult); err != nil {
		t.Fatalf("list: decode %q: %v", payload, err)
	}
	if len(listResult.Contents) != 1 || listResult.Contents[0].Key != key {
		t.Fatalf("list: expected [%s], got %+v", key, listResult.Contents)
	}
	if listResult.Contents[0].Preview == nil || *listResult.Contents[0].Preview != content {
		t.Fatalf("list: expected preview %q, got %v", content, listResult.Contents[0].Preview)
	}

	// delete
	resp, payload = doRequest(t, http.MethodDelete, base+"/", strings.NewReader(`{"key":"`+key+`"}`), map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("delete: expected 204, got %d (%s)", resp.StatusCode, payload)
	}
	if _, err := os.Stat(filepath.Join(connectomeDir, key)); !os.IsNotExist(err) {
		t.Fatalf("delete: expected file gone, stat err = %v", err)
	}

	// read after delete fails
	resp, _ = doRequest(t, http.MethodGet, base+"/?key="+key, nil, nil)
	if resp.StatusCode == http.StatusOK {
		t.Fatalf("read after delete: expected error status, got 200")
	}
}

func TestMemoryReadRejectsMissingKey(t *testing.T) {
	srv, _, _ := newTestServer(t)

	resp, _ := doRequest(t, http.MethodGet, srv.URL+"/api/connectome/memory/", nil, nil)
	if resp.StatusCode == http.StatusOK {
		t.Fatalf("expected error status for missing key, got 200")
	}
}

func TestMemoryWriteResolvesACLFromFrontmatterOrDefault(t *testing.T) {
	srv, _, indexer := newTestServer(t)
	base := srv.URL + "/api/connectome/memory"

	// An explicit acl in the frontmatter is stored as-is.
	const explicitKey = "mem_acl_explicit.md"
	explicitDoc := "---\n" +
		"type: note\n" +
		"created_at: \"2024-01-01T00:00:00Z\"\n" +
		"entities: []\n" +
		"acl: [\"GM\"]\n" +
		"---\nbody\n"
	body, contentType := multipartBody(t, []filePart{{name: explicitKey, content: explicitDoc}})
	resp, payload := doRequest(t, http.MethodPost, base+"/", body, map[string]string{"Content-Type": contentType})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("write: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	rows := indexer.rowsFor(explicitKey)
	if len(rows) != 1 || len(rows[0].ACL) != 1 || rows[0].ACL[0] != "GM" {
		t.Fatalf("expected acl [GM], got %+v", rows)
	}

	// A memory whose frontmatter omits acl entirely picks up DEFAULT_ACL.
	t.Setenv("DEFAULT_ACL", "players")
	const defaultedKey = "mem_acl_default.md"
	body, contentType = multipartBody(t, []filePart{{name: defaultedKey, content: memoryDocument("note", "body", nil)}})
	resp, payload = doRequest(t, http.MethodPost, base+"/", body, map[string]string{"Content-Type": contentType})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("write: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	rows = indexer.rowsFor(defaultedKey)
	if len(rows) != 1 || len(rows[0].ACL) != 1 || rows[0].ACL[0] != "players" {
		t.Fatalf("expected default acl [players], got %+v", rows)
	}
}

// TestMemoryReadHydratesACLForPreOneOneMemory exercises get_memory's read
// path (GET /memory/): a memory written before acl existed has no acl key
// on disk, and the response should resolve it through this instance's
// DEFAULT_ACL, without rewriting the stored blob.
func TestMemoryReadHydratesACLForPreOneOneMemory(t *testing.T) {
	srv, connectomeDir, _ := newTestServer(t)
	base := srv.URL + "/api/connectome/memory"

	const key = "mem_pre11.md"
	doc := memoryDocument("note", "body", nil) // no acl key at all

	body, contentType := multipartBody(t, []filePart{{name: key, content: doc}})
	resp, payload := doRequest(t, http.MethodPost, base+"/", body, map[string]string{"Content-Type": contentType})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("write: expected 200, got %d (%s)", resp.StatusCode, payload)
	}

	t.Run("DEFAULT_ACL set resolves the read to that default", func(t *testing.T) {
		t.Setenv("DEFAULT_ACL", "GM")
		resp, payload := doRequest(t, http.MethodGet, base+"/?key="+key, nil, nil)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("read: expected 200, got %d (%s)", resp.StatusCode, payload)
		}
		content, _ := decodeJSON(t, payload)["content"].(string)
		fm := parseHydratedFrontmatter(t, content)
		if fm.ACL == nil || len(*fm.ACL) != 1 || (*fm.ACL)[0] != "GM" {
			t.Fatalf("expected hydrated read to resolve acl to [GM], got %v", fm.ACL)
		}
	})

	t.Run("no DEFAULT_ACL leaves the read unrestricted", func(t *testing.T) {
		resp, payload := doRequest(t, http.MethodGet, base+"/?key="+key, nil, nil)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("read: expected 200, got %d (%s)", resp.StatusCode, payload)
		}
		content, _ := decodeJSON(t, payload)["content"].(string)
		fm := parseHydratedFrontmatter(t, content)
		if fm.ACL == nil || len(*fm.ACL) != 0 {
			t.Fatalf("expected hydrated read to stay unrestricted ([]), got %v", fm.ACL)
		}
	})

	// The stored blob itself is never rewritten by a read.
	onDisk, err := os.ReadFile(filepath.Join(connectomeDir, key))
	if err != nil {
		t.Fatalf("read stored file: %v", err)
	}
	if string(onDisk) != doc {
		t.Fatalf("expected the stored blob to be untouched by hydration, got %q", string(onDisk))
	}
}

// TestMemoryWriteIndexesOccurredAt covers occurred_at's three states: absent
// (nil on the row), present and valid (parsed onto the row), and present but
// malformed (the write is rejected rather than silently dropping or
// misinterpreting it as created_at).
func TestMemoryWriteIndexesOccurredAt(t *testing.T) {
	srv, _, indexer := newTestServer(t)
	base := srv.URL + "/api/connectome/memory"

	t.Run("absent occurred_at leaves the row's OccurredAt nil", func(t *testing.T) {
		const key = "mem_occurred_absent.md"
		body, contentType := multipartBody(t, []filePart{{name: key, content: memoryDocument("note", "body", nil)}})
		resp, payload := doRequest(t, http.MethodPost, base+"/", body, map[string]string{"Content-Type": contentType})
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("write: expected 200, got %d (%s)", resp.StatusCode, payload)
		}
		rows := indexer.rowsFor(key)
		if len(rows) != 1 || rows[0].OccurredAt != nil {
			t.Fatalf("expected nil OccurredAt, got %+v", rows)
		}
	})

	t.Run("valid occurred_at is parsed onto the row", func(t *testing.T) {
		const key = "mem_occurred_valid.md"
		doc := "---\n" +
			"type: event\n" +
			"created_at: \"2024-01-01T00:00:00Z\"\n" +
			"occurred_at: \"2023-06-15T09:30:00Z\"\n" +
			"entities: []\n" +
			"---\nbody\n"
		body, contentType := multipartBody(t, []filePart{{name: key, content: doc}})
		resp, payload := doRequest(t, http.MethodPost, base+"/", body, map[string]string{"Content-Type": contentType})
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("write: expected 200, got %d (%s)", resp.StatusCode, payload)
		}
		rows := indexer.rowsFor(key)
		want := time.Date(2023, 6, 15, 9, 30, 0, 0, time.UTC)
		if len(rows) != 1 || rows[0].OccurredAt == nil || !rows[0].OccurredAt.Equal(want) {
			t.Fatalf("expected OccurredAt %v, got %+v", want, rows)
		}
	})

	t.Run("malformed occurred_at rejects the write", func(t *testing.T) {
		const key = "mem_occurred_bad.md"
		doc := "---\n" +
			"type: event\n" +
			"created_at: \"2024-01-01T00:00:00Z\"\n" +
			"occurred_at: \"not-a-date\"\n" +
			"entities: []\n" +
			"---\nbody\n"
		body, contentType := multipartBody(t, []filePart{{name: key, content: doc}})
		resp, payload := doRequest(t, http.MethodPost, base+"/", body, map[string]string{"Content-Type": contentType})
		if resp.StatusCode == http.StatusOK {
			t.Fatalf("expected write to reject malformed occurred_at, got 200 (%s)", payload)
		}
		if len(indexer.rowsFor(key)) != 0 {
			t.Fatalf("expected no rows indexed for a rejected write")
		}
	})
}

func TestMemoryBatchWriteAndBatchRead(t *testing.T) {
	srv, connectomeDir, indexer := newTestServer(t)
	base := srv.URL + "/api/connectome/memory"

	parts := []filePart{
		{name: "mem_a.md", content: "first note"},
		{name: "ent_ada.json", content: `{"id":"ada","memory_ids":["mem_a.md"]}`},
	}

	// batch write
	body, contentType := multipartBody(t, parts)
	resp, payload := doRequest(t, http.MethodPost, base+"/batch", body, map[string]string{"Content-Type": contentType})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("batch write: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	for _, p := range parts {
		onDisk, err := os.ReadFile(filepath.Join(connectomeDir, p.name))
		if err != nil {
			t.Fatalf("batch write: read %s: %v", p.name, err)
		}
		if string(onDisk) != p.content {
			t.Fatalf("batch write: %s mismatch: got %q", p.name, string(onDisk))
		}
		// Neither file carries valid memory frontmatter (mem_a.md is a plain
		// string, ent_ada.json is an entity record), so neither is indexed.
		if rows := indexer.rowsFor(p.name); len(rows) != 0 {
			t.Fatalf("batch write: expected %s to be left unindexed, got %d rows", p.name, len(rows))
		}
	}

	// batch read, all present
	resp, payload = doRequest(t, http.MethodPost, base+"/batch/read", strings.NewReader(`{"keys":["mem_a.md","ent_ada.json"]}`), map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("batch read: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	contents, ok := decodeJSON(t, payload)["contents"].(map[string]any)
	if !ok {
		t.Fatalf("batch read: missing contents in %s", payload)
	}
	if contents["mem_a.md"] != "first note" || contents["ent_ada.json"] != parts[1].content {
		t.Fatalf("batch read: unexpected contents %v", contents)
	}

	// batch read with a missing key reports a partial result
	resp, payload = doRequest(t, http.MethodPost, base+"/batch/read", strings.NewReader(`{"keys":["mem_a.md","missing.md"]}`), map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusMultiStatus {
		t.Fatalf("batch read partial: expected 207, got %d (%s)", resp.StatusCode, payload)
	}
	decoded := decodeJSON(t, payload)
	gotContents, _ := decoded["contents"].(map[string]any)
	if gotContents["mem_a.md"] != "first note" {
		t.Fatalf("batch read partial: expected mem_a.md content, got %v", decoded["contents"])
	}
	gotErrors, _ := decoded["errors"].(map[string]any)
	if _, exists := gotErrors["missing.md"]; !exists {
		t.Fatalf("batch read partial: expected error for missing.md, got %v", decoded["errors"])
	}
}

func TestMemoryBatchReadRejectsEmptyKeys(t *testing.T) {
	srv, _, _ := newTestServer(t)

	resp, payload := doRequest(t, http.MethodPost, srv.URL+"/api/connectome/memory/batch/read", strings.NewReader(`{"keys":[]}`), map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d (%s)", resp.StatusCode, payload)
	}
}

// memoryDocument builds a minimal connectome memory document with the
// frontmatter shape connectomemcp's format_memory writes.
func memoryDocument(memType, body string, entities []string) string {
	entitiesYAML := "[]"
	if len(entities) > 0 {
		var quoted []string
		for _, e := range entities {
			quoted = append(quoted, `"`+e+`"`)
		}
		entitiesYAML = "[" + strings.Join(quoted, ", ") + "]"
	}
	return "---\n" +
		"type: " + memType + "\n" +
		"created_at: \"2024-01-01T00:00:00Z\"\n" +
		"entities: " + entitiesYAML + "\n" +
		"---\n" + body + "\n"
}

func TestMemoryWriteIndexesRewriteSupersedesDeleteRemoves(t *testing.T) {
	srv, _, indexer := newTestServer(t)
	base := srv.URL + "/api/connectome/memory"

	const key = "mem_indexed.md"
	longBody := strings.Repeat("word ", 600) // > memoryChunkWords, so it splits into multiple chunks

	// write: a long memory produces multiple chunk rows, each carrying type/entities.
	body, contentType := multipartBody(t, []filePart{{name: key, content: memoryDocument("fact", longBody, []string{"ada"})}})
	resp, payload := doRequest(t, http.MethodPost, base+"/", body, map[string]string{"Content-Type": contentType})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("write: expected 200, got %d (%s)", resp.StatusCode, payload)
	}

	rows := indexer.rowsFor(key)
	if len(rows) < 2 {
		t.Fatalf("expected multiple chunk rows for a long memory, got %d", len(rows))
	}
	for i, row := range rows {
		if row.ChunkIndex != i {
			t.Fatalf("expected chunk index %d, got %d", i, row.ChunkIndex)
		}
		if row.Type != "fact" {
			t.Fatalf("expected type fact, got %q", row.Type)
		}
		if len(row.EntityIDs) != 1 || row.EntityIDs[0] != "ada" {
			t.Fatalf("expected entity [ada], got %v", row.EntityIDs)
		}
		if row.Model != managers.EMBEDDING_MODEL {
			t.Fatalf("expected model %s, got %s", managers.EMBEDDING_MODEL, row.Model)
		}
		if row.Dim == 0 || len(row.Embedding) != row.Dim {
			t.Fatalf("expected a non-empty embedding matching dim, got %v (dim %d)", row.Embedding, row.Dim)
		}
		if strings.TrimSpace(row.ChunkText) == "" {
			t.Fatalf("expected chunk text to be populated for full-text search, got %q", row.ChunkText)
		}
	}

	// rewrite: a short memory at the same key supersedes the old rows entirely.
	shortBody, contentType := multipartBody(t, []filePart{{name: key, content: memoryDocument("note", "short body", nil)}})
	resp, payload = doRequest(t, http.MethodPost, base+"/", shortBody, map[string]string{"Content-Type": contentType})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("rewrite: expected 200, got %d (%s)", resp.StatusCode, payload)
	}

	rows = indexer.rowsFor(key)
	if len(rows) != 1 {
		t.Fatalf("expected rewrite to supersede down to 1 row, got %d", len(rows))
	}
	if rows[0].Type != "note" {
		t.Fatalf("expected rewritten type note, got %q", rows[0].Type)
	}

	// delete: removes the memory's rows entirely.
	resp, payload = doRequest(t, http.MethodDelete, base+"/", strings.NewReader(`{"key":"`+key+`"}`), map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("delete: expected 204, got %d (%s)", resp.StatusCode, payload)
	}
	if rows := indexer.rowsFor(key); len(rows) != 0 {
		t.Fatalf("expected delete to remove rows, got %d", len(rows))
	}
}

// memoryDocumentWithRelationships builds a connectome memory document whose
// frontmatter carries the given raw YAML relationships block, matching the
// shape connectomemcp's format_memory writes once Relationship gains kind and
// superseded_by.
func memoryDocumentWithRelationships(relationshipsYAML string) string {
	return "---\n" +
		"version: connectome/memory/0.1\n" +
		"id: mem_rel.md\n" +
		"type: fact\n" +
		"created_at: \"2024-01-01T00:00:00Z\"\n" +
		"source:\n  type: mcp\n  created_at: \"2024-01-01T00:00:00Z\"\n" +
		"entities: [\"david\", \"tea\"]\n" +
		"relationships:\n" + relationshipsYAML +
		"---\n" +
		"David likes tea.\n"
}

func TestSupersedeRelationshipPatchesWithoutReindexing(t *testing.T) {
	srv, connectomeDir, indexer := newTestServer(t)
	base := srv.URL + "/api/connectome/memory"

	const key = "mem_rel.md"
	doc := memoryDocumentWithRelationships(
		"  - subjectEntityId: david\n    predicate: likes\n    objectEntityId: tea\n    kind: fact\n    superseded_by: null\n",
	)

	body, contentType := multipartBody(t, []filePart{{name: key, content: doc}})
	resp, payload := doRequest(t, http.MethodPost, base+"/", body, map[string]string{"Content-Type": contentType})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("write: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	insertsAfterWrite := indexer.insertCount()
	if insertsAfterWrite == 0 {
		t.Fatalf("expected the initial write to index the memory")
	}

	patchBody := `{"key":"mem_rel.md","subjectEntityId":"david","predicate":"likes","objectEntityId":"tea","superseded_by":"mem_correction.md"}`
	resp, payload = doRequest(t, http.MethodPatch, base+"/relationship", strings.NewReader(patchBody), map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("supersede: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	if got := decodeJSON(t, payload)["message"]; got != "success" {
		t.Fatalf("supersede: expected message success, got %v", got)
	}

	if got := indexer.insertCount(); got != insertsAfterWrite {
		t.Fatalf("expected supersede to never call the embedder, insert count went from %d to %d", insertsAfterWrite, got)
	}

	onDisk, err := os.ReadFile(filepath.Join(connectomeDir, key))
	if err != nil {
		t.Fatalf("read patched file: %v", err)
	}
	if !strings.Contains(string(onDisk), "superseded_by: mem_correction.md") {
		t.Fatalf("expected patched file to carry the new superseded_by, got:\n%s", onDisk)
	}
	if !strings.Contains(string(onDisk), "David likes tea.") {
		t.Fatalf("expected body to survive the patch untouched, got:\n%s", onDisk)
	}
}

func TestSupersedeRelationshipReturnsNotFoundForMissingMemory(t *testing.T) {
	srv, _, _ := newTestServer(t)
	base := srv.URL + "/api/connectome/memory"

	patchBody := `{"key":"mem_missing.md","subjectEntityId":"david","predicate":"likes","objectEntityId":"tea","superseded_by":"mem_correction.md"}`
	resp, payload := doRequest(t, http.MethodPatch, base+"/relationship", strings.NewReader(patchBody), map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404 for a missing memory, got %d (%s)", resp.StatusCode, payload)
	}
}

func TestSupersedeRelationshipReturnsNotFoundForMissingRelationship(t *testing.T) {
	srv, _, _ := newTestServer(t)
	base := srv.URL + "/api/connectome/memory"

	const key = "mem_rel.md"
	doc := memoryDocumentWithRelationships(
		"  - subjectEntityId: david\n    predicate: likes\n    objectEntityId: tea\n    kind: fact\n    superseded_by: null\n",
	)
	body, contentType := multipartBody(t, []filePart{{name: key, content: doc}})
	resp, payload := doRequest(t, http.MethodPost, base+"/", body, map[string]string{"Content-Type": contentType})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("write: expected 200, got %d (%s)", resp.StatusCode, payload)
	}

	patchBody := `{"key":"mem_rel.md","subjectEntityId":"david","predicate":"dislikes","objectEntityId":"tea","superseded_by":"mem_correction.md"}`
	resp, payload = doRequest(t, http.MethodPatch, base+"/relationship", strings.NewReader(patchBody), map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404 for a non-matching relationship, got %d (%s)", resp.StatusCode, payload)
	}
}

func TestSupersedeRelationshipRejectsMissingRequiredFields(t *testing.T) {
	srv, _, _ := newTestServer(t)
	base := srv.URL + "/api/connectome/memory"

	resp, payload := doRequest(t, http.MethodPatch, base+"/relationship", strings.NewReader(`{"key":"mem_rel.md"}`), map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 when subjectEntityId/predicate are missing, got %d (%s)", resp.StatusCode, payload)
	}
}
