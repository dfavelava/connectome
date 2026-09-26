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
	pool      *pgxpool.Pool
	weights   HybridWeights
	textQuery TextQueryMode
	bm25      BM25Params
}

func NewEmbeddingsDao(pool *pgxpool.Pool) *EmbeddingsDao {
	return &EmbeddingsDao{pool: pool, weights: HybridWeightsFromEnv(), textQuery: TextQueryModeFromEnv(), bm25: BM25ParamsFromEnv()}
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

// TextQueryMode selects how Search's full-text leg turns the query text into
// a tsquery. Question-shaped queries rarely contain every one of their terms
// in a short memory, so the AND-of-terms modes often match nothing and leave
// ranking to the vector leg alone - see issue #8.
type TextQueryMode string

const (
	// TextQueryPlain is plainto_tsquery: every non-stopword term must match
	// (AND). The default.
	TextQueryPlain TextQueryMode = "plain"
	// TextQueryWebsearch is websearch_to_tsquery: AND of terms like plain,
	// but honoring "quoted phrases", OR, and -negation in the query text.
	TextQueryWebsearch TextQueryMode = "websearch"
	// TextQueryOr matches a chunk containing any of plainto_tsquery's terms
	// (its & operators rewritten to |), ranked by ts_rank, which favors
	// chunks matching more of them.
	TextQueryOr TextQueryMode = "or"
	// TextQueryAndOr matches like TextQueryOr but ranks every chunk that
	// satisfies the full AND query above the OR-only matches.
	TextQueryAndOr TextQueryMode = "and_or"
	// TextQueryBM25 matches a chunk containing any of the query's lexemes,
	// like TextQueryOr, but ranks by Okapi BM25 instead of ts_rank: each
	// matched lexeme is weighted by its inverse document frequency within
	// the searched tome, so a match on a term most chunks contain (a
	// speaker's name, a month) counts for little next to a rare one. See
	// issue #20 and BM25Params.
	TextQueryBM25 TextQueryMode = "bm25"
	// TextQueryRareOr drops the query lexemes found in more than
	// BM25Params.MaxDF of the searched tome's chunks, then matches and
	// ranks like TextQueryOr over the rest - a cheap approximation of
	// TextQueryBM25's IDF weighting. A query with only common lexemes
	// matches nothing.
	TextQueryRareOr TextQueryMode = "rare_or"
)

// TextQueryModeFromEnv reads SEARCH_TEXT_QUERY, falling back to
// TextQueryPlain when it is unset or not a known mode.
func TextQueryModeFromEnv() TextQueryMode {
	switch mode := TextQueryMode(os.Getenv("SEARCH_TEXT_QUERY")); mode {
	case TextQueryPlain, TextQueryWebsearch, TextQueryOr, TextQueryAndOr, TextQueryBM25, TextQueryRareOr:
		return mode
	default:
		return TextQueryPlain
	}
}

// BM25Params tunes TextQueryBM25's scoring and TextQueryRareOr's cutoff. A
// chunk scores the sum, over each query lexeme it contains, of
//
//	idf * tf*(K1+1) / (tf + K1*(1 - B + B*len/avglen))
//
// where idf = ln(1 + (N - df + 0.5)/(df + 0.5)), N is the number of chunks in
// the searched tome, df how many of them contain the lexeme, tf how often the
// chunk contains it, and len / avglen the chunk's and the tome's mean count
// of distinct lexemes. K1 = 0 drops term frequency and length entirely,
// leaving a plain sum of IDF over the matched lexemes.
type BM25Params struct {
	K1 float64
	B  float64
	// MaxDF is TextQueryRareOr's cutoff: the largest fraction (0-1] of the
	// tome's chunks a query lexeme may appear in and still be searched. A
	// lexeme found in a single chunk is always kept, so a small tome, where
	// any one chunk is a large fraction, still has searchable terms.
	MaxDF float64
}

const (
	defaultBM25K1    = 1.2
	defaultBM25B     = 0.75
	defaultTextMaxDF = 0.05
)

// DefaultBM25Params are used when EmbeddingsDao is constructed without an
// env override. See BM25ParamsFromEnv and backend/.env.example.
func DefaultBM25Params() BM25Params {
	return BM25Params{K1: defaultBM25K1, B: defaultBM25B, MaxDF: defaultTextMaxDF}
}

// BM25ParamsFromEnv reads SEARCH_BM25_K1, SEARCH_BM25_B, and
// SEARCH_TEXT_MAX_DF, falling back to DefaultBM25Params for any unset or
// unparsable value.
func BM25ParamsFromEnv() BM25Params {
	params := DefaultBM25Params()
	if v, ok := floatEnv("SEARCH_BM25_K1"); ok {
		params.K1 = v
	}
	if v, ok := floatEnv("SEARCH_BM25_B"); ok {
		params.B = v
	}
	if v, ok := floatEnv("SEARCH_TEXT_MAX_DF"); ok {
		params.MaxDF = v
	}
	return params
}

// textQuerySQL returns the FROM-clause items for mode, binding the query
// text as $1 and exposing the tsquery a chunk must match as "query", plus
// the ORDER BY clause ranking the matches. Rewriting " & " to " | " in plainto_tsquery's text form leaves
// phrase operators (<->, from hyphenated words) intact. The IDF modes
// (TextQueryBM25, TextQueryRareOr) are built from lexemeStatsSQL instead.
func textQuerySQL(mode TextQueryMode) (from, orderBy string) {
	const orQuery = `(SELECT replace(plainto_tsquery('english', $1)::text, ' & ', ' | ')::tsquery AS query) AS or_query`
	switch mode {
	case TextQueryWebsearch:
		return `websearch_to_tsquery('english', $1) AS query`, `ts_rank(search_vector, query) DESC`
	case TextQueryOr:
		return orQuery, `ts_rank(search_vector, query) DESC`
	case TextQueryAndOr:
		return orQuery + `, plainto_tsquery('english', $1) AS and_query`,
			`(search_vector @@ and_query) DESC, ts_rank(search_vector, query) DESC`
	default:
		return `plainto_tsquery('english', $1) AS query`, `ts_rank(search_vector, query) DESC`
	}
}

// lexemeStatsSQL is the WITH clause shared by the IDF modes. "stats" holds
// the searched tome's ($8) chunk count and mean distinct-lexeme count;
// "terms" holds each lexeme of the query text ($1) that occurs in the tome,
// with its document frequency and IDF, and "matched" ORs those lexemes into
// the tsquery a candidate must match. Statistics come from the whole tome
// rather than the filtered rows, so one tome's vocabulary never skews
// another's ranking and a filter doesn't change what counts as rare.
//
// Each lexeme is turned back into a tsquery with the 'simple' config, which
// only lowercases, so an already-stemmed lexeme matches itself.
const lexemeStatsSQL = `WITH stats AS (
	  SELECT count(*)::float8 AS n, coalesce(avg(length(search_vector)), 0)::float8 AS avglen
	  FROM embeddings WHERE tome_id = $8
	), terms AS (
	  SELECT t.lexeme, t.query, ln(1 + (stats.n - d.df + 0.5) / (d.df + 0.5)) AS idf
	  FROM (SELECT lexeme, plainto_tsquery('simple', lexeme) AS query FROM unnest(to_tsvector('english', $1))) t
	  CROSS JOIN stats
	  CROSS JOIN LATERAL (
	    SELECT count(*)::float8 AS df FROM embeddings WHERE tome_id = $8 AND search_vector @@ t.query
	  ) d
	  WHERE d.df > 0 AND %s
	), matched AS (
	  SELECT string_agg(query::text, ' | ')::tsquery AS query FROM terms
	)`

// bm25ScoreSQL scores the embeddings row aliased e per BM25Params, with
// K1 bound as $11 and B as $12.
const bm25ScoreSQL = `(
	  SELECT sum(terms.idf * tf.tf * ($11::float8 + 1) / (tf.tf + $11::float8 * (1 - $12::float8 + $12::float8 * length(e.search_vector) / nullif(stats.avglen, 0))))
	  FROM unnest(e.search_vector) v
	  JOIN terms ON terms.lexeme = v.lexeme
	  CROSS JOIN LATERAL (SELECT coalesce(array_length(v.positions, 1), 1)::float8 AS tf) tf
	)`

// textCandidatesSQL returns the full text-candidates query for mode, and any
// arguments it binds beyond the shared $1-$10.
func (dao *EmbeddingsDao) textCandidatesSQL(mode TextQueryMode) (string, []any) {
	switch mode {
	case TextQueryBM25:
		return fmt.Sprintf(lexemeStatsSQL, "true") + `
		 SELECT memory_key, chunk_index, type
		 FROM embeddings e, matched, stats
		 WHERE e.search_vector @@ matched.query
		   AND` + candidateFilterSQL + `
		 ORDER BY ` + bm25ScoreSQL + ` DESC NULLS LAST, memory_key, chunk_index
		 LIMIT $6`, []any{dao.bm25.K1, dao.bm25.B}
	case TextQueryRareOr:
		return fmt.Sprintf(lexemeStatsSQL, "d.df <= greatest(1, $11::float8 * stats.n)") + `
		 SELECT memory_key, chunk_index, type
		 FROM embeddings e, matched
		 WHERE e.search_vector @@ matched.query
		   AND` + candidateFilterSQL + `
		 ORDER BY ts_rank(e.search_vector, matched.query) DESC NULLS LAST, memory_key, chunk_index
		 LIMIT $6`, []any{dao.bm25.MaxDF}
	default:
		from, orderBy := textQuerySQL(mode)
		return `SELECT memory_key, chunk_index, type
		 FROM embeddings, ` + from + `
		 WHERE search_vector @@ query
		   AND` + candidateFilterSQL + `
		 ORDER BY ` + orderBy + `
		 LIMIT $6`, nil
	}
}

// textCandidates returns up to limit chunks whose chunk_text matches
// queryText (as interpreted by the dao's TextQueryMode), ordered by
// descending full-text rank, at chunk granularity. An empty or all-stopword
// queryText matches nothing and is not an error.
func (dao *EmbeddingsDao) textCandidates(ctx context.Context, queryText string, limit int, filters SearchFilters) ([]chunkRef, error) {
	if strings.TrimSpace(queryText) == "" {
		return nil, nil
	}

	query, extraArgs := dao.textCandidatesSQL(dao.textQuery)
	args := append([]any{queryText, filters.Type, filters.Entity, filters.Since, filters.Until, limit, nilIfEmpty(filters.ACLScope), filters.TomeID, filters.OccurredSince, filters.OccurredUntil}, extraArgs...)
	rows, err := dao.pool.Query(ctx, query, args...)
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
