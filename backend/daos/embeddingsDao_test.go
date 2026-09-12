package daos

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	pgxvec "github.com/pgvector/pgvector-go/pgx"
)

// testPool connects to a real Postgres for integration-testing the DAO
// against actual pgvector behavior. It skips the test (rather than failing
// the whole run) when no Postgres is reachable, so `go test ./...` stays
// usable without docker compose running. CI provides a matching service
// for the go job (see .github/workflows/ci.yml) so this always runs there.
func testPool(t *testing.T) *pgxpool.Pool {
	t.Helper()

	config, err := pgxpool.ParseConfig(testConnString())
	if err != nil {
		t.Fatalf("parse test postgres config: %v", err)
	}
	config.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		return pgxvec.RegisterTypes(ctx, conn)
	}

	pool, err := pgxpool.NewWithConfig(context.Background(), config)
	if err != nil {
		t.Skipf("postgres not available: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		t.Skipf("postgres not reachable: %v", err)
	}
	t.Cleanup(pool.Close)

	if _, err := pool.Exec(context.Background(), `CREATE EXTENSION IF NOT EXISTS vector`); err != nil {
		t.Fatalf("create vector extension: %v", err)
	}
	if _, err := pool.Exec(context.Background(), `
		CREATE TABLE IF NOT EXISTS embeddings (
			id BIGSERIAL PRIMARY KEY,
			memory_key TEXT NOT NULL,
			chunk_index INT NOT NULL DEFAULT 0,
			embedding VECTOR(768) NOT NULL,
			model TEXT NOT NULL,
			dim INT NOT NULL,
			type TEXT NOT NULL CHECK (type IN ('note', 'fact', 'preference', 'event')),
			entity_ids TEXT[] NOT NULL DEFAULT '{}',
			created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
			UNIQUE (memory_key, chunk_index)
		)`); err != nil {
		t.Fatalf("create embeddings table: %v", err)
	}

	return pool
}

func testConnString() string {
	user := envOrDefault("POSTGRES_USER", "postgres")
	password := envOrDefault("POSTGRES_PASSWORD", "fake-pass")
	host := envOrDefault("POSTGRES_HOST", "localhost")
	port := envOrDefault("POSTGRES_PORT", "5432")
	database := envOrDefault("POSTGRES_DB", "postgres")
	return fmt.Sprintf("postgresql://%s:%s@%s:%s/%s", user, password, host, port, database)
}

func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func unitVector(dim, hotIndex int) []float32 {
	v := make([]float32, dim)
	v[hotIndex] = 1
	return v
}

func TestEmbeddingsDaoInsertNearestNeighborsAndDelete(t *testing.T) {
	pool := testPool(t)
	dao := NewEmbeddingsDao(pool)
	ctx := context.Background()

	key := "mem_test_" + uuid.NewString() + ".md"
	t.Cleanup(func() { _ = dao.DeleteEmbeddingsForKey(context.Background(), key) })

	rows := []EmbeddingRow{
		{ChunkIndex: 0, Embedding: unitVector(768, 0), Model: "nomic-embed-text", Dim: 768, Type: "fact", EntityIDs: []string{"ada"}, CreatedAt: time.Now().UTC()},
		{ChunkIndex: 1, Embedding: unitVector(768, 1), Model: "nomic-embed-text", Dim: 768, Type: "fact", EntityIDs: []string{"ada"}, CreatedAt: time.Now().UTC()},
	}

	if err := dao.InsertEmbeddings(ctx, key, rows); err != nil {
		t.Fatalf("insert embeddings: %v", err)
	}

	// A query aligned with chunk 0's embedding should find chunk 0 as its
	// closest match among this key's rows.
	neighbors, err := dao.NearestNeighbors(ctx, unitVector(768, 0), 50)
	if err != nil {
		t.Fatalf("nearest neighbors: %v", err)
	}
	found := false
	for _, n := range neighbors {
		if n.MemoryKey != key {
			continue
		}
		found = true
		if n.ChunkIndex != 0 {
			t.Fatalf("expected closest chunk for %s to be chunk 0, got chunk %d", key, n.ChunkIndex)
		}
		break
	}
	if !found {
		t.Fatalf("expected %s to appear among nearest neighbors, got %+v", key, neighbors)
	}

	if err := dao.DeleteEmbeddingsForKey(ctx, key); err != nil {
		t.Fatalf("delete embeddings: %v", err)
	}

	neighbors, err = dao.NearestNeighbors(ctx, unitVector(768, 0), 50)
	if err != nil {
		t.Fatalf("nearest neighbors after delete: %v", err)
	}
	for _, n := range neighbors {
		if n.MemoryKey == key {
			t.Fatalf("expected no rows for %s after delete, found %+v", key, n)
		}
	}
}

// blendVector returns a unit vector combining dims 0 and 1 so its cosine
// distance to unitVector(dim, 0) is predictable and strictly between the
// identical (distance 0) and orthogonal (distance 1) cases.
func blendVector(dim int) []float32 {
	v := make([]float32, dim)
	v[0] = 0.8
	v[1] = 0.6
	return v
}

func strPtr(s string) *string { return &s }

func TestEmbeddingsDaoSearchFiltersAndCollapsesPerKey(t *testing.T) {
	pool := testPool(t)
	dao := NewEmbeddingsDao(pool)
	ctx := context.Background()

	suffix := uuid.NewString()
	closeKey := "mem_close_" + suffix + ".md"
	midKey := "mem_mid_" + suffix + ".md"
	farKey := "mem_far_" + suffix + ".md"
	t.Cleanup(func() {
		for _, key := range []string{closeKey, midKey, farKey} {
			_ = dao.DeleteEmbeddingsForKey(context.Background(), key)
		}
	})

	now := time.Now().UTC()
	yesterday := now.Add(-24 * time.Hour)

	// closeKey's chunk 0 is an exact match for the query; its chunk 1 is
	// orthogonal (far), so a correct collapse must surface chunk 0.
	if err := dao.InsertEmbeddings(ctx, closeKey, []EmbeddingRow{
		{ChunkIndex: 0, Embedding: unitVector(768, 0), Model: "nomic-embed-text", Dim: 768, Type: "fact", EntityIDs: []string{"ada"}, CreatedAt: now},
		{ChunkIndex: 1, Embedding: unitVector(768, 1), Model: "nomic-embed-text", Dim: 768, Type: "fact", EntityIDs: []string{"ada"}, CreatedAt: now},
	}); err != nil {
		t.Fatalf("insert closeKey: %v", err)
	}
	if err := dao.InsertEmbeddings(ctx, midKey, []EmbeddingRow{
		{ChunkIndex: 0, Embedding: blendVector(768), Model: "nomic-embed-text", Dim: 768, Type: "note", EntityIDs: []string{"grace"}, CreatedAt: yesterday},
	}); err != nil {
		t.Fatalf("insert midKey: %v", err)
	}
	if err := dao.InsertEmbeddings(ctx, farKey, []EmbeddingRow{
		{ChunkIndex: 0, Embedding: unitVector(768, 1), Model: "nomic-embed-text", Dim: 768, Type: "fact", EntityIDs: []string{"ada"}, CreatedAt: now},
	}); err != nil {
		t.Fatalf("insert farKey: %v", err)
	}

	query := unitVector(768, 0)

	t.Run("no filters ranks by distance and collapses to the best chunk", func(t *testing.T) {
		hits, err := dao.Search(ctx, query, 10, SearchFilters{})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		byKey := make(map[string]SearchHit, len(hits))
		var order []string
		for _, h := range hits {
			if h.MemoryKey == closeKey || h.MemoryKey == midKey || h.MemoryKey == farKey {
				byKey[h.MemoryKey] = h
				order = append(order, h.MemoryKey)
			}
		}
		if len(order) != 3 {
			t.Fatalf("expected 3 of our keys among hits, got %v", order)
		}
		if order[0] != closeKey || order[1] != midKey || order[2] != farKey {
			t.Fatalf("expected order [close, mid, far], got %v", order)
		}
		if got := byKey[closeKey].ChunkIndex; got != 0 {
			t.Fatalf("expected closeKey to collapse to its closer chunk 0, got chunk %d", got)
		}
	})

	t.Run("type filter excludes other types", func(t *testing.T) {
		hits, err := dao.Search(ctx, query, 10, SearchFilters{Type: strPtr("fact")})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		for _, h := range hits {
			if h.MemoryKey == midKey {
				t.Fatalf("expected midKey (type note) excluded by type=fact filter, got %+v", h)
			}
		}
	})

	t.Run("entity filter matches only memories with that entity", func(t *testing.T) {
		hits, err := dao.Search(ctx, query, 10, SearchFilters{Entity: strPtr("grace")})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		var keys []string
		for _, h := range hits {
			if h.MemoryKey == closeKey || h.MemoryKey == midKey || h.MemoryKey == farKey {
				keys = append(keys, h.MemoryKey)
			}
		}
		if len(keys) != 1 || keys[0] != midKey {
			t.Fatalf("expected only midKey for entity=grace, got %v", keys)
		}
	})

	t.Run("since/until filter by created_at", func(t *testing.T) {
		since := now.Add(-1 * time.Hour)
		hits, err := dao.Search(ctx, query, 10, SearchFilters{Since: &since})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		for _, h := range hits {
			if h.MemoryKey == midKey {
				t.Fatalf("expected midKey (created yesterday) excluded by since filter, got %+v", h)
			}
		}

		until := yesterday.Add(1 * time.Hour)
		hits, err = dao.Search(ctx, query, 10, SearchFilters{Until: &until})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		var keys []string
		for _, h := range hits {
			if h.MemoryKey == closeKey || h.MemoryKey == midKey || h.MemoryKey == farKey {
				keys = append(keys, h.MemoryKey)
			}
		}
		if len(keys) != 1 || keys[0] != midKey {
			t.Fatalf("expected only midKey for until=yesterday+1h, got %v", keys)
		}
	})

	t.Run("k limits the number of results", func(t *testing.T) {
		hits, err := dao.Search(ctx, query, 1, SearchFilters{Entity: strPtr("ada")})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(hits) != 1 {
			t.Fatalf("expected exactly 1 hit for k=1, got %d", len(hits))
		}
		if hits[0].MemoryKey != closeKey {
			t.Fatalf("expected closeKey as the single closest hit, got %s", hits[0].MemoryKey)
		}
	})
}

func TestEmbeddingsDaoInsertEmbeddingsNoopOnEmptyRows(t *testing.T) {
	pool := testPool(t)
	dao := NewEmbeddingsDao(pool)

	if err := dao.InsertEmbeddings(context.Background(), "mem_unused.md", nil); err != nil {
		t.Fatalf("expected inserting zero rows to be a no-op, got %v", err)
	}
}
