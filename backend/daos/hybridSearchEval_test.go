package daos

import (
	"context"
	"testing"
	"time"

	"github.com/google/uuid"
)

// axisVector returns a vector that gets monotonically closer (by cosine
// distance) to e0 (the hot-at-index-0 unit vector) as lean goes from 0 to 1,
// by trading off an orthogonal component at index 1. lean=1 is e0 itself
// (distance 0 from the query built the same way); lean=0 is orthogonal to it
// (distance 1). Used to seed this eval set's memories at a precise,
// reproducible vector rank against each other.
func axisVector(dim int, lean float32) []float32 {
	v := make([]float32, dim)
	v[0] = lean
	v[1] = 1 - lean*lean
	return v
}

var allTextQueryModes = []TextQueryMode{TextQueryPlain, TextQueryWebsearch, TextQueryOr, TextQueryAndOr, TextQueryBM25, TextQueryRareOr}

// TestHybridSearchEvalSet is a small, fixed regression set for the ranking
// behavior issue #18 exists to fix: vector-only search misses on names and
// rare exact terms because their embeddings can be a mediocre ("lukewarm")
// match even when the term itself is an exact, unambiguous hit. Each case
// pins a query against the same four seeded memories and asserts which one
// hybrid search must surface first under each TextQueryMode - if a future
// change to the blend regresses ranking quality, one of these cases should
// fail.
//
// The seeded memories' embeddings are constructed (not real model output) so
// each one's vector rank against the shared query embedding is exact and
// reproducible:
//
//	rank 1 (closest)  vecTop     - "The quick brown fox jumps over the lazy dog."
//	rank 2            vecSecond  - "A completely unrelated sentence about weather patterns."
//	rank 3            vecMid     - "David mentioned the code name Xyzzyquark during standup."
//	rank 4 (farthest) vecFar     - "Ada asked about the project Velvetrix release date."
func TestHybridSearchEvalSet(t *testing.T) {
	pool := testPool(t)
	dao := NewEmbeddingsDao(pool)
	ctx := context.Background()

	suffix := uuid.NewString()
	// A tome of its own, so the IDF modes' per-tome lexeme statistics come
	// from these four memories alone and not from whatever else the test
	// database holds.
	tome := "eval-" + suffix
	vecTop := "mem_eval_vectop_" + suffix + ".md"
	vecSecond := "mem_eval_vecsecond_" + suffix + ".md"
	vecMid := "mem_eval_vecmid_" + suffix + ".md"
	vecFar := "mem_eval_vecfar_" + suffix + ".md"
	t.Cleanup(func() { _ = dao.DeleteEmbeddingsForTome(context.Background(), tome) })

	now := time.Now().UTC()
	seed := func(key, text string, lean float32) {
		t.Helper()
		if err := dao.InsertEmbeddings(ctx, key, []EmbeddingRow{
			{ChunkIndex: 0, Embedding: axisVector(768, lean), ChunkText: text, Model: "nomic-embed-text", Dim: 768, Type: "note", EntityIDs: []string{"eval"}, ACL: []string{}, TomeID: tome, CreatedAt: now},
		}); err != nil {
			t.Fatalf("seed %s: %v", key, err)
		}
	}
	seed(vecTop, "The quick brown fox jumps over the lazy dog.", 1.0)
	seed(vecSecond, "A completely unrelated sentence about weather patterns.", 0.95)
	seed(vecMid, "David mentioned the code name Xyzzyquark during standup.", 0.5)
	seed(vecFar, "Ada asked about the project Velvetrix release date.", 0.0)

	query := axisVector(768, 1.0) // identical direction to vecTop: exact match on the vector axis alone

	// expectedTop is the top hit under every TextQueryMode unless
	// expectedTopByMode overrides it for a mode.
	cases := []struct {
		name              string
		query             string
		expectedTop       string
		expectedTopByMode map[TextQueryMode]string
	}{
		{
			// vecMid's vector rank is only 3rd of 4, but "Xyzzyquark" appears
			// nowhere else - the keyword signal must be enough to promote it
			// above vecTop and vecSecond, which rank higher on vector alone.
			name:        "rare exact term rescues a lukewarm vector match",
			query:       "Xyzzyquark",
			expectedTop: vecMid,
		},
		{
			// Same shape, different memory: vecFar has the worst possible
			// vector rank (orthogonal to the query) yet must still win when
			// it holds the one exact keyword hit.
			name:        "rare name match wins even from the back of the vector ranking",
			query:       "Velvetrix",
			expectedTop: vecFar,
		},
		{
			// vecSecond already ranks 2nd on vectors; adding a keyword match
			// it uniquely owns should make it the clear top result.
			name:        "keyword match reinforces an already-strong vector rank",
			query:       "weather",
			expectedTop: vecSecond,
		},
		{
			// No seeded memory contains this word, so full-text contributes
			// nothing for anyone: pure vector ranking must still surface the
			// closest embedding, confirming the blend doesn't regress plain
			// semantic search when there's no keyword signal to add.
			name:        "no keyword signal falls back to vector ranking",
			query:       "giraffe",
			expectedTop: vecTop,
		},
		{
			// A question-shaped query (issue #8): vecFar holds "Ada",
			// "asked", and "Velvetrix" but not "launch". The AND-of-terms
			// modes need every term, so the keyword signal goes silent and
			// vector ranking wins; the OR modes still credit the partial
			// match and rescue vecFar.
			name:        "question with a term the memory lacks",
			query:       "When did Ada ask about the Velvetrix launch?",
			expectedTop: vecFar,
			expectedTopByMode: map[TextQueryMode]string{
				TextQueryPlain:     vecTop,
				TextQueryWebsearch: vecTop,
			},
		},
	}

	for _, mode := range allTextQueryModes {
		modeDao := &EmbeddingsDao{pool: pool, weights: DefaultHybridWeights(), textQuery: mode, bm25: DefaultBM25Params()}
		for _, tc := range cases {
			t.Run(string(mode)+"/"+tc.name, func(t *testing.T) {
				expectedTop := tc.expectedTop
				if override, ok := tc.expectedTopByMode[mode]; ok {
					expectedTop = override
				}
				hits, err := modeDao.Search(ctx, tc.query, query, 10, SearchFilters{Entity: strPtr("eval"), TomeID: tome})
				if err != nil {
					t.Fatalf("search: %v", err)
				}
				if len(hits) == 0 {
					t.Fatalf("expected at least one eval hit for query %q, got none", tc.query)
				}
				if hits[0].MemoryKey != expectedTop {
					t.Fatalf("query %q: expected top hit %s, got %s (all hits: %+v)", tc.query, expectedTop, hits[0].MemoryKey, hits)
				}
			})
		}
	}
}

// TestHybridSearchCommonTermMustNotOutrankRare pins issue #20's failure mode:
// a query term nearly every memory contains (a speaker's name) must not let
// those memories outrank the one memory holding the query's rare term.
// ts_rank has no IDF, so under the OR modes the name's repeated occurrences
// outscore a single rare match; the IDF modes must rank the rare match first.
//
// Vector ranks, closest first: commonTop, rare, then the other common
// memories. commonTop's text is the longest of the name-only memories, so
// BM25's length normalization ranks it below them - it keeps only its vector
// lead, which the rare memory's text lead has to overcome.
func TestHybridSearchCommonTermMustNotOutrankRare(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	tome := "eval-common-" + uuid.NewString()
	dao := NewEmbeddingsDao(pool)
	t.Cleanup(func() { _ = dao.DeleteEmbeddingsForTome(context.Background(), tome) })

	now := time.Now().UTC()
	seed := func(key, text string, lean float32) string {
		t.Helper()
		key = key + "_" + tome + ".md"
		if err := dao.InsertEmbeddings(ctx, key, []EmbeddingRow{
			{ChunkIndex: 0, Embedding: axisVector(768, lean), ChunkText: text, Model: "nomic-embed-text", Dim: 768, Type: "note", EntityIDs: []string{}, ACL: []string{}, TomeID: tome, CreatedAt: now},
		}); err != nil {
			t.Fatalf("seed %s: %v", key, err)
		}
		return key
	}
	commonTop := seed("mem_common_top", "Caroline said Caroline would call Caroline's mom about the weekend plans and the new car.", 1.0)
	rare := seed("mem_rare", "Melanie said the pottery was fun.", 0.9)
	seed("mem_common_a", "Caroline met Caroline's friend Caroline.", 0.5)
	seed("mem_common_b", "Caroline told Caroline's dad Caroline.", 0.3)
	seed("mem_common_c", "Caroline asked Caroline's boss Caroline.", 0.1)

	// "try" is in no memory, so the AND modes match nothing and ranking falls
	// to the vectors, where commonTop leads.
	expectedTop := map[TextQueryMode]string{
		TextQueryPlain:     commonTop,
		TextQueryWebsearch: commonTop,
		TextQueryOr:        commonTop,
		TextQueryAndOr:     commonTop,
		TextQueryBM25:      rare,
		TextQueryRareOr:    rare,
	}
	for _, mode := range allTextQueryModes {
		t.Run(string(mode), func(t *testing.T) {
			modeDao := &EmbeddingsDao{pool: pool, weights: DefaultHybridWeights(), textQuery: mode, bm25: DefaultBM25Params()}
			hits, err := modeDao.Search(ctx, "When did Caroline try pottery?", axisVector(768, 1.0), 10, SearchFilters{TomeID: tome})
			if err != nil {
				t.Fatalf("search: %v", err)
			}
			if len(hits) == 0 || hits[0].MemoryKey != expectedTop[mode] {
				t.Fatalf("expected top hit %s, got %+v", expectedTop[mode], hits)
			}
		})
	}
}
