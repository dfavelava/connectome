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

func TestEmbeddingsDaoInsertEmbeddingsNoopOnEmptyRows(t *testing.T) {
	pool := testPool(t)
	dao := NewEmbeddingsDao(pool)

	if err := dao.InsertEmbeddings(context.Background(), "mem_unused.md", nil); err != nil {
		t.Fatalf("expected inserting zero rows to be a no-op, got %v", err)
	}
}
