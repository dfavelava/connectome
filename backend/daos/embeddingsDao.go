package daos

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
)

// EmbeddingRow is one chunk's embedding, ready to be stored for a memory key.
// It mirrors the non-key columns of the embeddings table (see
// backend/migrations/0001_create_embeddings.up.sql).
type EmbeddingRow struct {
	ChunkIndex int
	Embedding  []float32
	Model      string
	Dim        int
	Type       string
	EntityIDs  []string
	CreatedAt  time.Time
}

// NearestNeighbor is one hit from a nearest-neighbour search over embeddings.
// Distance is cosine distance (smaller is closer; 0 is identical).
type NearestNeighbor struct {
	MemoryKey  string
	ChunkIndex int
	Distance   float64
}

type EmbeddingsDao struct {
	pool *pgxpool.Pool
}

func NewEmbeddingsDao(pool *pgxpool.Pool) *EmbeddingsDao {
	return &EmbeddingsDao{pool: pool}
}

// InsertEmbeddings stores one row per chunk under memoryKey. It does not
// remove any existing rows for the key - callers that are rewriting a memory
// should call DeleteEmbeddingsForKey first so the write supersedes them.
func (dao *EmbeddingsDao) InsertEmbeddings(ctx context.Context, memoryKey string, rows []EmbeddingRow) error {
	if len(rows) == 0 {
		return nil
	}

	batch := &pgx.Batch{}
	for _, row := range rows {
		batch.Queue(
			`INSERT INTO embeddings (memory_key, chunk_index, embedding, model, dim, type, entity_ids, created_at)
			 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
			memoryKey, row.ChunkIndex, pgvector.NewVector(row.Embedding), row.Model, row.Dim, row.Type, row.EntityIDs, row.CreatedAt,
		)
	}

	results := dao.pool.SendBatch(ctx, batch)
	defer results.Close()

	for _, row := range rows {
		if _, err := results.Exec(); err != nil {
			return fmt.Errorf("insert embedding for %s chunk %d: %w", memoryKey, row.ChunkIndex, err)
		}
	}
	return nil
}

// DeleteEmbeddingsForKey removes every embedding row stored under memoryKey.
func (dao *EmbeddingsDao) DeleteEmbeddingsForKey(ctx context.Context, memoryKey string) error {
	if _, err := dao.pool.Exec(ctx, `DELETE FROM embeddings WHERE memory_key = $1`, memoryKey); err != nil {
		return fmt.Errorf("delete embeddings for %s: %w", memoryKey, err)
	}
	return nil
}

// NearestNeighbors returns the k chunks closest to query by cosine distance.
func (dao *EmbeddingsDao) NearestNeighbors(ctx context.Context, query []float32, k int) ([]NearestNeighbor, error) {
	rows, err := dao.pool.Query(ctx,
		`SELECT memory_key, chunk_index, embedding <=> $1 AS distance
		 FROM embeddings
		 ORDER BY embedding <=> $1
		 LIMIT $2`,
		pgvector.NewVector(query), k,
	)
	if err != nil {
		return nil, fmt.Errorf("nearest neighbors: %w", err)
	}
	defer rows.Close()

	var neighbors []NearestNeighbor
	for rows.Next() {
		var n NearestNeighbor
		if err := rows.Scan(&n.MemoryKey, &n.ChunkIndex, &n.Distance); err != nil {
			return nil, fmt.Errorf("scan nearest neighbor: %w", err)
		}
		neighbors = append(neighbors, n)
	}
	return neighbors, rows.Err()
}

// SearchFilters narrows a Search to a subset of embeddings. A nil field is
// not applied.
type SearchFilters struct {
	Type   *string
	Entity *string
	Since  *time.Time
	Until  *time.Time
}

// SearchHit is one memory's closest-matching chunk from a filtered nearest-
// neighbour search, collapsed to at most one hit per memory key. Distance is
// cosine distance (smaller is closer; 0 is identical).
type SearchHit struct {
	MemoryKey  string
	ChunkIndex int
	Type       string
	Distance   float64
}

// Search returns up to k memory keys ranked by similarity to query, after
// applying filters and collapsing each key down to its single
// closest-matching chunk. Collapsing happens on the database side (DISTINCT
// ON) rather than in Go so filtering and ranking stay consistent with what
// the LIMIT actually returns.
func (dao *EmbeddingsDao) Search(ctx context.Context, query []float32, k int, filters SearchFilters) ([]SearchHit, error) {
	rows, err := dao.pool.Query(ctx,
		`SELECT memory_key, chunk_index, type, distance
		 FROM (
			SELECT DISTINCT ON (memory_key)
				memory_key, chunk_index, type,
				embedding <=> $1 AS distance
			FROM embeddings
			WHERE ($2::text IS NULL OR type = $2)
			  AND ($3::text IS NULL OR $3 = ANY(entity_ids))
			  AND ($4::timestamptz IS NULL OR created_at >= $4)
			  AND ($5::timestamptz IS NULL OR created_at <= $5)
			ORDER BY memory_key, distance
		 ) collapsed
		 ORDER BY distance
		 LIMIT $6`,
		pgvector.NewVector(query), filters.Type, filters.Entity, filters.Since, filters.Until, k,
	)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}
	defer rows.Close()

	var hits []SearchHit
	for rows.Next() {
		var h SearchHit
		if err := rows.Scan(&h.MemoryKey, &h.ChunkIndex, &h.Type, &h.Distance); err != nil {
			return nil, fmt.Errorf("scan search hit: %w", err)
		}
		hits = append(hits, h)
	}
	return hits, rows.Err()
}
