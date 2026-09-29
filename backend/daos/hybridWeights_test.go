package daos

import (
	"sync"
	"testing"
)

func TestHybridWeightsFromEnvDefaultsAndOverrides(t *testing.T) {
	if got := HybridWeightsFromEnv(); got != DefaultHybridWeights() {
		t.Fatalf("expected defaults with no env vars set, got %+v", got)
	}

	t.Setenv("SEARCH_VECTOR_WEIGHT", "0.9")
	t.Setenv("SEARCH_TEXT_WEIGHT", "0.1")
	t.Setenv("SEARCH_RRF_K", "10")

	got := HybridWeightsFromEnv()
	want := HybridWeights{Vector: 0.9, Text: 0.1, RRFRankConstant: 10}
	if got != want {
		t.Fatalf("expected overrides %+v, got %+v", want, got)
	}
}

func TestHybridWeightsFromEnvIgnoresUnparsableValues(t *testing.T) {
	t.Setenv("SEARCH_VECTOR_WEIGHT", "not-a-number")

	if got := HybridWeightsFromEnv(); got.Vector != defaultVectorWeight {
		t.Fatalf("expected unparsable value to fall back to the default, got %+v", got)
	}
}

func TestTextQueryModeFromEnv(t *testing.T) {
	if got := TextQueryModeFromEnv(); got != TextQueryPlain {
		t.Fatalf("expected plain with SEARCH_TEXT_QUERY unset, got %q", got)
	}

	t.Setenv("SEARCH_TEXT_QUERY", "or")
	if got := TextQueryModeFromEnv(); got != TextQueryOr {
		t.Fatalf("expected or, got %q", got)
	}

	t.Setenv("SEARCH_TEXT_QUERY", "bm25")
	if got := TextQueryModeFromEnv(); got != TextQueryBM25 {
		t.Fatalf("expected bm25, got %q", got)
	}

	t.Setenv("SEARCH_TEXT_QUERY", "fuzzy")
	if got := TextQueryModeFromEnv(); got != TextQueryPlain {
		t.Fatalf("expected an unknown mode to fall back to plain, got %q", got)
	}
}

func TestBM25ParamsFromEnv(t *testing.T) {
	want := DefaultBM25Params()
	if got := BM25ParamsFromEnv(); got != want {
		t.Fatalf("expected defaults %+v with no env vars set, got %+v", want, got)
	}

	t.Setenv("SEARCH_BM25_K1", "0")
	t.Setenv("SEARCH_BM25_B", "not-a-number")
	t.Setenv("SEARCH_TEXT_MAX_DF", "0.1")
	want = BM25Params{K1: 0, B: defaultBM25B, MaxDF: 0.1}
	if got := BM25ParamsFromEnv(); got != want {
		t.Fatalf("expected %+v, got %+v", want, got)
	}
}

// TestFuseWeightsAreConfigurable exercises fuse (the in-memory RRF blend)
// directly, without a database, to pin down that a weight of zero for one
// signal makes it stop influencing ranking entirely - the mechanism the
// hybrid weights knobs described in backend/.env.example rely on.
func TestFuseWeightsAreConfigurable(t *testing.T) {
	// vector ranks textFavorite first, keyFavorite second; text ranks
	// keyFavorite first, textFavorite unranked (absent).
	vectorHits := []chunkRef{
		{MemoryKey: "textFavorite", ChunkIndex: 0, Type: "note"},
		{MemoryKey: "keyFavorite", ChunkIndex: 0, Type: "note"},
	}
	textHits := []chunkRef{
		{MemoryKey: "keyFavorite", ChunkIndex: 0, Type: "note"},
	}

	t.Run("text weight zero collapses to pure vector ranking", func(t *testing.T) {
		hits := fuse(vectorHits, textHits, 10, HybridWeights{Vector: 1, Text: 0, RRFRankConstant: 60})
		if len(hits) < 1 || hits[0].MemoryKey != "textFavorite" {
			t.Fatalf("expected vector's top hit to win with text weight 0, got %+v", hits)
		}
	})

	t.Run("vector weight zero collapses to pure text ranking", func(t *testing.T) {
		hits := fuse(vectorHits, textHits, 10, HybridWeights{Vector: 0, Text: 1, RRFRankConstant: 60})
		if len(hits) < 1 || hits[0].MemoryKey != "keyFavorite" {
			t.Fatalf("expected text's top hit to win with vector weight 0, got %+v", hits)
		}
	})
}

func TestRankingValidate(t *testing.T) {
	if err := DefaultRanking().Validate(); err != nil {
		t.Fatalf("default ranking should be valid: %v", err)
	}
	cases := map[string]func(*Ranking){
		"negative weight":    func(r *Ranking) { r.Weights.Vector = -1 },
		"both weights zero":  func(r *Ranking) { r.Weights.Vector, r.Weights.Text = 0, 0 },
		"zero rrf_k":         func(r *Ranking) { r.Weights.RRFRankConstant = 0 },
		"unknown text_query": func(r *Ranking) { r.TextQuery = "fuzzy" },
		"zero text_max_df":   func(r *Ranking) { r.BM25.MaxDF = 0 },
		"text_max_df over 1": func(r *Ranking) { r.BM25.MaxDF = 1.5 },
		"negative bm25_k1":   func(r *Ranking) { r.BM25.K1 = -1 },
		"bm25_b over 1":      func(r *Ranking) { r.BM25.B = 2 },
	}
	for name, mutate := range cases {
		r := DefaultRanking()
		mutate(&r)
		if err := r.Validate(); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	r := DefaultRanking()
	r.Weights.Text = 0
	r.BM25.MaxDF = 1
	if err := r.Validate(); err != nil {
		t.Errorf("one zero weight and max_df 1 are valid: %v", err)
	}
}

// TestFuseConcurrentRankingsAreIndependent runs fuse with opposite weights
// from many goroutines: the ranking is an argument, not dao state, so neither
// setting may leak into the other. Run with -race.
func TestFuseConcurrentRankingsAreIndependent(t *testing.T) {
	vectorHits := []chunkRef{{MemoryKey: "v", Type: "note"}, {MemoryKey: "t", Type: "note"}}
	textHits := []chunkRef{{MemoryKey: "t", Type: "note"}}
	vectorOnly := HybridWeights{Vector: 1, Text: 0, RRFRankConstant: 60}
	textOnly := HybridWeights{Vector: 0, Text: 1, RRFRankConstant: 60}

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			if hits := fuse(vectorHits, textHits, 10, vectorOnly); hits[0].MemoryKey != "v" {
				t.Errorf("vector-only ranking led with %s", hits[0].MemoryKey)
			}
		}()
		go func() {
			defer wg.Done()
			if hits := fuse(vectorHits, textHits, 10, textOnly); hits[0].MemoryKey != "t" {
				t.Errorf("text-only ranking led with %s", hits[0].MemoryKey)
			}
		}()
	}
	wg.Wait()
}
