package resources

import (
	"context"
	"errors"
	"mime/multipart"
	"testing"

	"daybid-dev-service/managers"
)

func TestTomeScopedKeyReturnsKeyUnchangedForDefaultTome(t *testing.T) {
	if got := TomeScopedKey(DefaultTome, "mem_abc.md"); got != "mem_abc.md" {
		t.Fatalf("expected default tome to leave key unchanged, got %q", got)
	}
	if got := TomeScopedKey(DefaultTome, "ent_alice.json"); got != "ent_alice.json" {
		t.Fatalf("expected default tome to leave key unchanged, got %q", got)
	}
}

func TestTomeScopedKeyPrefixesNonDefaultTome(t *testing.T) {
	if got := TomeScopedKey("west-marches", "mem_abc.md"); got != "tomes/west-marches/mem_abc.md" {
		t.Fatalf("expected tomes/west-marches/mem_abc.md, got %q", got)
	}
	if got := TomeScopedKey("west-marches", "ent_alice.json"); got != "tomes/west-marches/ent_alice.json" {
		t.Fatalf("expected tomes/west-marches/ent_alice.json, got %q", got)
	}
}

func TestCheckTomeDestroyAllowedRefusesDefaultTomeEvenWithConfirm(t *testing.T) {
	if err := checkTomeDestroyAllowed(DefaultTome, false); !errors.Is(err, ErrDestroyDefaultTome) {
		t.Fatalf("expected ErrDestroyDefaultTome without confirm, got %v", err)
	}
	if err := checkTomeDestroyAllowed(DefaultTome, true); !errors.Is(err, ErrDestroyDefaultTome) {
		t.Fatalf("expected ErrDestroyDefaultTome even with confirm=true, got %v", err)
	}
}

func TestCheckTomeDestroyAllowedAllowsTestConventionPrefixesWithoutConfirm(t *testing.T) {
	for _, tome := range []string{"temp-abc", "test-abc", "temp-", "test-"} {
		if err := checkTomeDestroyAllowed(tome, false); err != nil {
			t.Fatalf("expected %q to be destroyable without confirm, got %v", tome, err)
		}
	}
}

func TestCheckTomeDestroyAllowedRequiresConfirmOutsideTestConvention(t *testing.T) {
	if err := checkTomeDestroyAllowed("west-marches", false); !errors.Is(err, ErrDestroyNeedsConfirm) {
		t.Fatalf("expected ErrDestroyNeedsConfirm without confirm, got %v", err)
	}
	if err := checkTomeDestroyAllowed("west-marches", true); err != nil {
		t.Fatalf("expected confirm=true to override the guard, got %v", err)
	}
}

// fakeTomeManager is a minimal managers.MemoryManager stand-in that only
// tracks DeleteObjectsWithPrefix calls - DestroyTome doesn't touch any other
// method.
type fakeTomeManager struct {
	deletedPrefixes []string
}

func (f *fakeTomeManager) GetObject(string) (string, error)       { return "", nil }
func (f *fakeTomeManager) PutObject(string, multipart.File) error { return nil }
func (f *fakeTomeManager) DeleteObject(string) error              { return nil }
func (f *fakeTomeManager) ListObjects(string) (*managers.MemoryListResult, error) {
	return &managers.MemoryListResult{}, nil
}
func (f *fakeTomeManager) DeleteObjectsWithPrefix(prefix string) error {
	f.deletedPrefixes = append(f.deletedPrefixes, prefix)
	return nil
}

// fakeTomeEmbeddingsDeleter tracks DeleteEmbeddingsForTome calls.
type fakeTomeEmbeddingsDeleter struct {
	deletedTomes []string
}

func (f *fakeTomeEmbeddingsDeleter) DeleteEmbeddingsForTome(_ context.Context, tomeID string) error {
	f.deletedTomes = append(f.deletedTomes, tomeID)
	return nil
}

func TestDestroyTomeDeletesBlobsAndEmbeddingsForAllowedTome(t *testing.T) {
	manager := &fakeTomeManager{}
	embeddings := &fakeTomeEmbeddingsDeleter{}

	if err := DestroyTome(context.Background(), manager, embeddings, "temp-scratch", false); err != nil {
		t.Fatalf("DestroyTome: %v", err)
	}

	if len(manager.deletedPrefixes) != 1 || manager.deletedPrefixes[0] != "tomes/temp-scratch/" {
		t.Fatalf("expected blobs deleted under tomes/temp-scratch/, got %v", manager.deletedPrefixes)
	}
	if len(embeddings.deletedTomes) != 1 || embeddings.deletedTomes[0] != "temp-scratch" {
		t.Fatalf("expected embeddings deleted for temp-scratch, got %v", embeddings.deletedTomes)
	}
}

func TestDestroyTomeRefusesWithoutTouchingBlobsOrEmbeddings(t *testing.T) {
	manager := &fakeTomeManager{}
	embeddings := &fakeTomeEmbeddingsDeleter{}

	err := DestroyTome(context.Background(), manager, embeddings, "west-marches", false)
	if !errors.Is(err, ErrDestroyNeedsConfirm) {
		t.Fatalf("expected ErrDestroyNeedsConfirm, got %v", err)
	}
	if len(manager.deletedPrefixes) != 0 {
		t.Fatalf("expected no blob deletion for a refused destroy, got %v", manager.deletedPrefixes)
	}
	if len(embeddings.deletedTomes) != 0 {
		t.Fatalf("expected no embeddings deletion for a refused destroy, got %v", embeddings.deletedTomes)
	}
}

func TestDestroyTomeRefusesDefaultTomeEvenWithConfirm(t *testing.T) {
	manager := &fakeTomeManager{}
	embeddings := &fakeTomeEmbeddingsDeleter{}

	err := DestroyTome(context.Background(), manager, embeddings, DefaultTome, true)
	if !errors.Is(err, ErrDestroyDefaultTome) {
		t.Fatalf("expected ErrDestroyDefaultTome, got %v", err)
	}
	if len(manager.deletedPrefixes) != 0 || len(embeddings.deletedTomes) != 0 {
		t.Fatalf("expected no side effects for a refused destroy, got prefixes=%v tomes=%v", manager.deletedPrefixes, embeddings.deletedTomes)
	}
}

func TestDestroyTomeAllowsNonConventionTomeWithConfirm(t *testing.T) {
	manager := &fakeTomeManager{}
	embeddings := &fakeTomeEmbeddingsDeleter{}

	if err := DestroyTome(context.Background(), manager, embeddings, "west-marches", true); err != nil {
		t.Fatalf("DestroyTome: %v", err)
	}
	if len(manager.deletedPrefixes) != 1 || manager.deletedPrefixes[0] != "tomes/west-marches/" {
		t.Fatalf("expected blobs deleted under tomes/west-marches/, got %v", manager.deletedPrefixes)
	}
}
