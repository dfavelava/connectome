package daos

import (
	"context"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
)

// EmbeddingRow is one chunk's embedding, ready to be stored for a memory key.
// It mirrors the non-key columns of the embeddings table (see
// backend/migrations/0001_create_embeddings.up.sql and
// backend/migrations/0002_add_search_vector.up.sql).
type EmbeddingRow struct {
	ChunkIndex int
	Embedding  []float32
	ChunkText  string
	Model      string
	Dim        int
	Type       string
	EntityIDs  []string
	ACL        []string
	TomeID     string
	CreatedAt  time.Time
	// OccurredAt is when the memory's described event happened, distinct from
	// CreatedAt (when it was written). Nil stores NULL: event time unknown.
	OccurredAt *time.Time
}

// NearestNeighbor is one hit from a nearest-neighbour search over embeddings.
// Distance is cosine distance (smaller is closer; 0 is identical).
type NearestNeighbor struct {
	MemoryKey  string
	ChunkIndex int
	Distance   float64
}

type EmbeddingsDao struct {
	pool    *pgxpool.Pool
	weights HybridWeights
}

func NewEmbeddingsDao(pool *pgxpool.Pool) *EmbeddingsDao {
	return &EmbeddingsDao{pool: pool, weights: HybridWeightsFromEnv()}
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
			`INSERT INTO embeddings (memory_key, chunk_index, embedding, chunk_text, model, dim, type, entity_ids, acl, tome_id, created_at, occurred_at)
			 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)`,
			memoryKey, row.ChunkIndex, pgvector.NewVector(row.Embedding), row.ChunkText, row.Model, row.Dim, row.Type, row.EntityIDs, row.ACL, row.TomeID, row.CreatedAt, row.OccurredAt,
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

// DeleteEmbeddingsForTome removes every embedding row stored under tomeID.
func (dao *EmbeddingsDao) DeleteEmbeddingsForTome(ctx context.Context, tomeID string) error {
	if _, err := dao.pool.Exec(ctx, `DELETE FROM embeddings WHERE tome_id = $1`, tomeID); err != nil {
		return fmt.Errorf("delete embeddings for tome %s: %w", tomeID, err)
	}
	return nil
}

// TruncateEmbeddings removes every row from the embeddings table. Used by
// cmd/reindex to rebuild the index from scratch, so a rebuild is never left
// holding both old and re-derived rows for the same key.
func (dao *EmbeddingsDao) TruncateEmbeddings(ctx context.Context) error {
	if _, err := dao.pool.Exec(ctx, `TRUNCATE TABLE embeddings`); err != nil {
		return fmt.Errorf("truncate embeddings: %w", err)
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
	// Since and Until bound created_at (when the memory was written).
	Since *time.Time
	Until *time.Time

	// OccurredSince and OccurredUntil bound occurred_at (when the described
	// event happened). A row with no occurred_at never satisfies either
	// bound - it is excluded rather than falling back to created_at, which
	// would conflate the two timestamps.
	OccurredSince *time.Time
	OccurredUntil *time.Time

	// ACLScope, when non-nil, restricts results to embeddings whose acl
	// column is empty (unrestricted) or overlaps this set of entity/group
	// ids - see SearchResourceImpl.resolveACLScope, which resolves it from
	// an `as` id. A nil ACLScope applies no acl filtering at all.
	ACLScope []string

	// TomeID restricts results to embeddings rows with this exact tome_id,
	// always applied (unlike the other filters above, which skip filtering
	// when unset). The zero value "" is resources.DefaultTome, matching
	// today's unscoped rows - so a caller that never sets TomeID searches
	// the default tome, same as before tomes existed.
	TomeID string
}

// SearchHit is one memory's best-ranked chunk from a hybrid (vector +
// full-text) search, collapsed to at most one hit per memory key. Score is
// the blended reciprocal-rank-fusion score described on HybridWeights -
// higher is better, with no fixed upper bound.
type SearchHit struct {
	MemoryKey  string
	ChunkIndex int
	Type       string
	Score      float64
}

// HybridWeights controls how vector similarity and full-text relevance are
// blended when Search ranks results, using reciprocal-rank fusion (RRF):
// each signal contributes weight/(RRFRankConstant+rank) to a chunk's combined
// score, where rank is that chunk's 1-based position within that signal's own
// candidate list (a chunk absent from a signal's list simply gets 0 from it,
// rather than being penalized). RRF blends rank positions instead of raw
// distance/ts_rank values, which live on unrelated scales and would need
// normalization to combine directly.
type HybridWeights struct {
	Vector          float64
	Text            float64
	RRFRankConstant float64
}

const (
	defaultVectorWeight    = 0.6
	defaultTextWeight      = 0.4
	defaultRRFRankConstant = 60.0

	// searchCandidateMultiplier and minSearchCandidates size the pool of
	// per-signal candidates Search fuses before collapsing to k memory keys:
	// large enough that a memory ranked outside the top k on one signal but
	// well inside it on the other still gets picked up by both queries.
	searchCandidateMultiplier = 8
	minSearchCandidates       = 40
)

// DefaultHybridWeights are used when EmbeddingsDao is constructed without an
// env override. See HybridWeightsFromEnv and backend/.env.example for how to
// change these.
func DefaultHybridWeights() HybridWeights {
	return HybridWeights{Vector: defaultVectorWeight, Text: defaultTextWeight, RRFRankConstant: defaultRRFRankConstant}
}

// HybridWeightsFromEnv reads SEARCH_VECTOR_WEIGHT, SEARCH_TEXT_WEIGHT, and
// SEARCH_RRF_K, falling back to DefaultHybridWeights for any unset or
// unparsable value.
func HybridWeightsFromEnv() HybridWeights {
	weights := DefaultHybridWeights()
	if v, ok := floatEnv("SEARCH_VECTOR_WEIGHT"); ok {
		weights.Vector = v
	}
	if v, ok := floatEnv("SEARCH_TEXT_WEIGHT"); ok {
		weights.Text = v
	}
	if v, ok := floatEnv("SEARCH_RRF_K"); ok {
		weights.RRFRankConstant = v
	}
	return weights
}

func floatEnv(key string) (float64, bool) {
	raw := os.Getenv(key)
	if raw == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

// chunkRef identifies one chunk row along with its type. Type is included
// (rather than tracked separately) because it's an attribute of the row
// itself, so it's identical no matter which signal's query found the chunk.
type chunkRef struct {
	MemoryKey  string
	ChunkIndex int
	Type       string
}

// candidateFilterSQL is shared by the vector and full-text candidate queries
// below (each binds it starting at $2, with $1 reserved for its own ranking
// expression, $6 for the LIMIT, $7 for ACLScope, $8 for TomeID, and $9/$10 for
// OccurredSince/OccurredUntil). A NULL $7 (ACLScope nil - no `as` given)
// applies no acl filtering; otherwise a row passes when its acl is empty
// (unrestricted) or overlaps $7. $8 is always compared exactly, since TomeID
// is never nil - see SearchFilters.TomeID. A NULL $9/$10 applies no bound; a
// non-NULL one compares occurred_at, which is NULL for a memory with no
// occurred_at, so that row fails the comparison and is excluded.
const candidateFilterSQL = `
	  ($2::text IS NULL OR type = $2)
	  AND ($3::text IS NULL OR $3 = ANY(entity_ids))
	  AND ($4::timestamptz IS NULL OR created_at >= $4)
	  AND ($5::timestamptz IS NULL OR created_at <= $5)
	  AND ($7::text[] IS NULL OR cardinality(acl) = 0 OR acl && $7::text[])
	  AND tome_id = $8
	  AND ($9::timestamptz IS NULL OR occurred_at >= $9)
	  AND ($10::timestamptz IS NULL OR occurred_at <= $10)`

// vectorCandidates returns up to limit chunks ordered by ascending cosine
// distance to query, at chunk granularity (not collapsed per memory key).
func (dao *EmbeddingsDao) vectorCandidates(ctx context.Context, query []float32, limit int, filters SearchFilters) ([]chunkRef, error) {
	rows, err := dao.pool.Query(ctx,
		`SELECT memory_key, chunk_index, type
		 FROM embeddings
		 WHERE`+candidateFilterSQL+`
		 ORDER BY embedding <=> $1
		 LIMIT $6`,
		pgvector.NewVector(query), filters.Type, filters.Entity, filters.Since, filters.Until, limit, nilIfEmpty(filters.ACLScope), filters.TomeID, filters.OccurredSince, filters.OccurredUntil,
	)
	if err != nil {
		return nil, fmt.Errorf("vector candidates: %w", err)
	}
	return scanCandidates(rows)
}

// textCandidates returns up to limit chunks whose chunk_text matches
// queryText, ordered by descending full-text rank, at chunk granularity. An
// empty or all-stopword queryText matches nothing and is not an error.
func (dao *EmbeddingsDao) textCandidates(ctx context.Context, queryText string, limit int, filters SearchFilters) ([]chunkRef, error) {
	if strings.TrimSpace(queryText) == "" {
		return nil, nil
	}

	rows, err := dao.pool.Query(ctx,
		`SELECT memory_key, chunk_index, type
		 FROM embeddings, plainto_tsquery('english', $1) AS query
		 WHERE search_vector @@ query
		   AND`+candidateFilterSQL+`
		 ORDER BY ts_rank(search_vector, query) DESC
		 LIMIT $6`,
		queryText, filters.Type, filters.Entity, filters.Since, filters.Until, limit, nilIfEmpty(filters.ACLScope), filters.TomeID, filters.OccurredSince, filters.OccurredUntil,
	)
	if err != nil {
		return nil, fmt.Errorf("text candidates: %w", err)
	}
	return scanCandidates(rows)
}

// nilIfEmpty maps an empty or nil ACLScope to a nil slice, so pgx encodes it
// as SQL NULL - candidateFilterSQL's "$7::text[] IS NULL" branch then reads
// as "no `as` given" rather than "acl scope of zero ids" (which would match
// nothing).
func nilIfEmpty(scope []string) []string {
	if len(scope) == 0 {
		return nil
	}
	return scope
}

// scanCandidates drains rows of (memory_key, chunk_index, type) into ordered
// chunkRefs.
func scanCandidates(rows pgx.Rows) ([]chunkRef, error) {
	defer rows.Close()

	var refs []chunkRef
	for rows.Next() {
		var ref chunkRef
		if err := rows.Scan(&ref.MemoryKey, &ref.ChunkIndex, &ref.Type); err != nil {
			return nil, fmt.Errorf("scan candidate: %w", err)
		}
		refs = append(refs, ref)
	}
	return refs, rows.Err()
}

// Search returns up to k memory keys ranked by a blend of vector similarity
// and full-text relevance to query (see HybridWeights), after applying
// filters and collapsing each key down to its single best-scoring chunk.
// Fusing two independently-ranked candidate lists needs per-chunk rank
// bookkeeping that doesn't fit cleanly in one SQL query, so the blend and the
// final per-key collapse both happen here in Go rather than via SQL's
// DISTINCT ON as the old vector-only Search did.
func (dao *EmbeddingsDao) Search(ctx context.Context, queryText string, queryEmbedding []float32, k int, filters SearchFilters) ([]SearchHit, error) {
	pool := k * searchCandidateMultiplier
	if pool < minSearchCandidates {
		pool = minSearchCandidates
	}

	vectorHits, err := dao.vectorCandidates(ctx, queryEmbedding, pool, filters)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}
	textHits, err := dao.textCandidates(ctx, queryText, pool, filters)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}

	return dao.fuse(vectorHits, textHits, k), nil
}

// fuse blends vector- and text-ranked candidate lists via reciprocal-rank
// fusion and collapses the result to at most one (best-scoring) hit per
// memory key, sorted by descending score.
func (dao *EmbeddingsDao) fuse(vectorHits, textHits []chunkRef, k int) []SearchHit {
	type scored struct {
		chunkRef
		score float64
	}

	scores := make(map[chunkRef]*scored)
	add := func(ref chunkRef, rank int, weight float64) {
		s, ok := scores[ref]
		if !ok {
			s = &scored{chunkRef: ref}
			scores[ref] = s
		}
		s.score += weight / (dao.weights.RRFRankConstant + float64(rank))
	}
	for i, ref := range vectorHits {
		add(ref, i+1, dao.weights.Vector)
	}
	for i, ref := range textHits {
		add(ref, i+1, dao.weights.Text)
	}

	bestPerKey := make(map[string]*scored)
	for _, s := range scores {
		current, ok := bestPerKey[s.MemoryKey]
		if !ok || s.score > current.score || (s.score == current.score && s.ChunkIndex < current.ChunkIndex) {
			bestPerKey[s.MemoryKey] = s
		}
	}

	hits := make([]SearchHit, 0, len(bestPerKey))
	for key, s := range bestPerKey {
		hits = append(hits, SearchHit{MemoryKey: key, ChunkIndex: s.ChunkIndex, Type: s.Type, Score: s.score})
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Score != hits[j].Score {
			return hits[i].Score > hits[j].Score
		}
		return hits[i].MemoryKey < hits[j].MemoryKey
	})
	if len(hits) > k {
		hits = hits[:k]
	}
	return hits
}
