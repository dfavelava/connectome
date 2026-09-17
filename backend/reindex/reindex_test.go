package reindex

import (
	"context"
	"errors"
	"mime/multipart"
	"testing"

	"daybid-dev-service/daos"
	"daybid-dev-service/managers"
)

// fakeManager stands in for a MemoryManager backed by a real blob store: its
// ListObjects/GetObject are driven entirely by the maps below, so tests can
// simulate a listed key whose read then fails without touching a filesystem.
type fakeManager struct {
	objects map[string]string
	unread  map[string]bool
	listErr error
}

func (m *fakeManager) GetObject(key string) (string, error) {
	if m.unread[key] {
		return "", errors.New("boom")
	}
	content, ok := m.objects[key]
	if !ok {
		return "", managers.ErrNotFound
	}
	return content, nil
}

func (m *fakeManager) PutObject(string, multipart.File) error { return nil }
func (m *fakeManager) DeleteObject(string) error              { return nil }
func (m *fakeManager) DeleteObjectsWithPrefix(string) error   { return nil }

func (m *fakeManager) ListObjects() (*managers.MemoryListResult, error) {
	if m.listErr != nil {
		return nil, m.listErr
	}
	items := make([]managers.MemoryListItem, 0, len(m.objects)+len(m.unread))
	for key := range m.objects {
		items = append(items, managers.MemoryListItem{Key: key})
	}
	for key := range m.unread {
		items = append(items, managers.MemoryListItem{Key: key})
	}
	return &managers.MemoryListResult{Contents: items}, nil
}

// fakeEmbedder stands in for *managers.OllamaManager: a cheap, deterministic
// embedding plus a call counter so dry-run tests can assert it was skipped.
type fakeEmbedder struct{ calls int }

func (f *fakeEmbedder) Embed(input string) ([]float32, error) {
	f.calls++
	return []float32{float32(len(input))}, nil
}

// fakeStore stands in for *daos.EmbeddingsDao: an in-memory table that
// mirrors truncate/delete/insert semantics without a real Postgres.
type fakeStore struct {
	rows      map[string][]daos.EmbeddingRow
	truncated bool
}

func newFakeStore() *fakeStore { return &fakeStore{rows: make(map[string][]daos.EmbeddingRow)} }

func (s *fakeStore) InsertEmbeddings(_ context.Context, key string, rows []daos.EmbeddingRow) error {
	s.rows[key] = append(s.rows[key], rows...)
	return nil
}

func (s *fakeStore) DeleteEmbeddingsForKey(_ context.Context, key string) error {
	delete(s.rows, key)
	return nil
}

func (s *fakeStore) TruncateEmbeddings(context.Context) error {
	s.truncated = true
	s.rows = make(map[string][]daos.EmbeddingRow)
	return nil
}

const validMemory = "---\n" +
	"type: fact\n" +
	"created_at: \"2024-03-05T12:00:00Z\"\n" +
	"entities: [\"ada\"]\n" +
	"---\n" +
	"David prefers tea over coffee.\n"

func TestRunRebuildsFromBlobStore(t *testing.T) {
	manager := &fakeManager{objects: map[string]string{
		"mem_a.md":     validMemory,
		"mem_b.md":     validMemory,
		"ent_ada.json": `{"id":"ada"}`,
	}}
	embedder := &fakeEmbedder{}
	store := newFakeStore()
	// A stale row for a key no longer in the blob store must not survive a
	// real (non-dry-run) rebuild - that's the whole point of truncating.
	store.rows["mem_deleted.md"] = []daos.EmbeddingRow{{ChunkIndex: 0}}

	result, err := Run(context.Background(), manager, embedder, store, false)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	want := Result{Total: 3, Indexed: 2, Skipped: 1, Failed: 0}
	if result != want {
		t.Fatalf("expected %+v, got %+v", want, result)
	}
	if !store.truncated {
		t.Fatalf("expected the embeddings table to be truncated before rebuilding")
	}
	if _, ok := store.rows["mem_deleted.md"]; ok {
		t.Fatalf("expected stale rows for a key no longer in the blob store to be gone")
	}
	if len(store.rows["mem_a.md"]) == 0 || len(store.rows["mem_b.md"]) == 0 {
		t.Fatalf("expected rows for both indexable memories, got %+v", store.rows)
	}
	if embedder.calls == 0 {
		t.Fatalf("expected the embedder to be called for indexable memories")
	}
}

func TestRunIsIdempotent(t *testing.T) {
	manager := &fakeManager{objects: map[string]string{"mem_a.md": validMemory}}
	embedder := &fakeEmbedder{}
	store := newFakeStore()

	first, err := Run(context.Background(), manager, embedder, store, false)
	if err != nil {
		t.Fatalf("first run: %v", err)
	}
	firstRows := len(store.rows["mem_a.md"])

	second, err := Run(context.Background(), manager, embedder, store, false)
	if err != nil {
		t.Fatalf("second run: %v", err)
	}

	if first != second {
		t.Fatalf("expected identical results across runs, got %+v then %+v", first, second)
	}
	if got := len(store.rows["mem_a.md"]); got != firstRows {
		t.Fatalf("expected rebuilding twice to leave the same row count, got %d then %d", firstRows, got)
	}
}

func TestRunDryRunLeavesTableAndEmbedderUntouched(t *testing.T) {
	manager := &fakeManager{objects: map[string]string{
		"mem_a.md":     validMemory,
		"ent_ada.json": `{"id":"ada"}`,
	}}
	embedder := &fakeEmbedder{}
	store := newFakeStore()
	store.rows["mem_a.md"] = []daos.EmbeddingRow{{ChunkIndex: 0}}

	result, err := Run(context.Background(), manager, embedder, store, true)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	want := Result{Total: 2, Indexed: 1, Skipped: 1, Failed: 0}
	if result != want {
		t.Fatalf("expected %+v, got %+v", want, result)
	}
	if store.truncated {
		t.Fatalf("expected dry-run not to truncate the embeddings table")
	}
	if embedder.calls != 0 {
		t.Fatalf("expected dry-run not to call the embedder, got %d calls", embedder.calls)
	}
	if len(store.rows["mem_a.md"]) != 1 {
		t.Fatalf("expected dry-run to leave existing rows untouched, got %+v", store.rows["mem_a.md"])
	}
}

func TestRunCountsReadFailures(t *testing.T) {
	manager := &fakeManager{
		objects: map[string]string{"mem_a.md": validMemory},
		unread:  map[string]bool{"mem_broken.md": true},
	}
	embedder := &fakeEmbedder{}
	store := newFakeStore()

	result, err := Run(context.Background(), manager, embedder, store, false)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	want := Result{Total: 2, Indexed: 1, Skipped: 0, Failed: 1}
	if result != want {
		t.Fatalf("expected %+v, got %+v", want, result)
	}
}

// preOneOneMemory is a pre-1.1-shaped memory document: no acl key (acl
// didn't exist yet) and a relationship with no kind key (kind didn't exist
// yet either).
const preOneOneMemory = "---\n" +
	"type: fact\n" +
	"created_at: \"2024-03-05T12:00:00Z\"\n" +
	"entities: [\"david\", \"gm\"]\n" +
	"relationships:\n" +
	"  - subjectEntityId: david\n" +
	"    predicate: reports_to\n" +
	"    objectEntityId: gm\n" +
	"---\n" +
	"David reports to the GM.\n"

// TestRunResolvesACLFromFrontmatterOrDefaultForPreOneOneMemories asserts the
// second half of "default at read time, no frontmatter migration": a
// reindex backfills the embeddings table's acl column for a memory that
// predates acl entirely, through this instance's configured DEFAULT_ACL,
// without rewriting the memory's stored blob (Run never calls PutObject).
func TestRunResolvesACLFromFrontmatterOrDefaultForPreOneOneMemories(t *testing.T) {
	t.Run("DEFAULT_ACL set backfills that default, not world-readable", func(t *testing.T) {
		t.Setenv("DEFAULT_ACL", "GM")
		manager := &fakeManager{objects: map[string]string{"mem_pre11.md": preOneOneMemory}}
		store := newFakeStore()

		if _, err := Run(context.Background(), manager, &fakeEmbedder{}, store, false); err != nil {
			t.Fatalf("Run: %v", err)
		}

		rows := store.rows["mem_pre11.md"]
		if len(rows) == 0 || len(rows[0].ACL) != 1 || rows[0].ACL[0] != "GM" {
			t.Fatalf("expected reindexed acl [GM], got %+v", rows)
		}
	})

	t.Run("DEFAULT_ACL unset stays unrestricted rather than silently GM-only", func(t *testing.T) {
		manager := &fakeManager{objects: map[string]string{"mem_pre11.md": preOneOneMemory}}
		store := newFakeStore()

		if _, err := Run(context.Background(), manager, &fakeEmbedder{}, store, false); err != nil {
			t.Fatalf("Run: %v", err)
		}

		rows := store.rows["mem_pre11.md"]
		if len(rows) == 0 || len(rows[0].ACL) != 0 {
			t.Fatalf("expected reindexed acl [] (unrestricted), got %+v", rows)
		}
	})
}

func TestRunPropagatesListError(t *testing.T) {
	manager := &fakeManager{listErr: errors.New("boom")}
	embedder := &fakeEmbedder{}
	store := newFakeStore()

	if _, err := Run(context.Background(), manager, embedder, store, false); err == nil {
		t.Fatalf("expected an error when ListObjects fails")
	}
}
