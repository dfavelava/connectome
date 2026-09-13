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

// TestHybridSearchEvalSet is a small, fixed regression set for the ranking
// behavior issue #18 exists to fix: vector-only search misses on names and
// rare exact terms because their embeddings can be a mediocre ("lukewarm")
// match even when the term itself is an exact, unambiguous hit. Each case
// pins a query against the same four seeded memories and asserts which one
// hybrid search must surface first - if a future change to the blend
// regresses ranking quality, one of these cases should fail.
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
	vecTop := "mem_eval_vectop_" + suffix + ".md"
	vecSecond := "mem_eval_vecsecond_" + suffix + ".md"
	vecMid := "mem_eval_vecmid_" + suffix + ".md"
	vecFar := "mem_eval_vecfar_" + suffix + ".md"
	t.Cleanup(func() {
		for _, key := range []string{vecTop, vecSecond, vecMid, vecFar} {
			_ = dao.DeleteEmbeddingsForKey(context.Background(), key)
		}
	})

	now := time.Now().UTC()
	seed := func(key, text string, lean float32) {
		t.Helper()
		if err := dao.InsertEmbeddings(ctx, key, []EmbeddingRow{
			{ChunkIndex: 0, Embedding: axisVector(768, lean), ChunkText: text, Model: "nomic-embed-text", Dim: 768, Type: "note", EntityIDs: []string{"eval"}, CreatedAt: now},
		}); err != nil {
			t.Fatalf("seed %s: %v", key, err)
		}
	}
	seed(vecTop, "The quick brown fox jumps over the lazy dog.", 1.0)
	seed(vecSecond, "A completely unrelated sentence about weather patterns.", 0.95)
	seed(vecMid, "David mentioned the code name Xyzzyquark during standup.", 0.5)
	seed(vecFar, "Ada asked about the project Velvetrix release date.", 0.0)

	query := axisVector(768, 1.0) // identical direction to vecTop: exact match on the vector axis alone

	cases := []struct {
		name        string
		query       string
		expectedTop string
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
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hits, err := dao.Search(ctx, tc.query, query, 10, SearchFilters{Entity: strPtr("eval")})
			if err != nil {
				t.Fatalf("search: %v", err)
			}
			if len(hits) == 0 {
				t.Fatalf("expected at least one eval hit for query %q, got none", tc.query)
			}
			if hits[0].MemoryKey != tc.expectedTop {
				t.Fatalf("query %q: expected top hit %s, got %s (all hits: %+v)", tc.query, tc.expectedTop, hits[0].MemoryKey, hits)
			}
		})
	}
}
