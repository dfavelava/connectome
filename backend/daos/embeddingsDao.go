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
