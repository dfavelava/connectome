package daos

import "testing"

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
		dao := &EmbeddingsDao{weights: HybridWeights{Vector: 1, Text: 0, RRFRankConstant: 60}}
		hits := dao.fuse(vectorHits, textHits, 10)
		if len(hits) < 1 || hits[0].MemoryKey != "textFavorite" {
			t.Fatalf("expected vector's top hit to win with text weight 0, got %+v", hits)
		}
	})

	t.Run("vector weight zero collapses to pure text ranking", func(t *testing.T) {
		dao := &EmbeddingsDao{weights: HybridWeights{Vector: 0, Text: 1, RRFRankConstant: 60}}
		hits := dao.fuse(vectorHits, textHits, 10)
		if len(hits) < 1 || hits[0].MemoryKey != "keyFavorite" {
			t.Fatalf("expected text's top hit to win with vector weight 0, got %+v", hits)
		}
	})
}
