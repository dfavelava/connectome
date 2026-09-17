package resources

import (
	"context"
	"errors"
	"mime/multipart"
	"strings"
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

// fakeListManager is a minimal managers.MemoryManager stand-in whose
// ListObjects filters a fixed key set by raw string prefix, the same way
// S3's ListObjectsV2 and localFsManager's ListObjects both behave - so
// ListTome's own filtering can be tested without a real backing store.
type fakeListManager struct {
	keys []string
}

func (f *fakeListManager) GetObject(string) (string, error)       { return "", nil }
func (f *fakeListManager) PutObject(string, multipart.File) error { return nil }
func (f *fakeListManager) DeleteObject(string) error              { return nil }
func (f *fakeListManager) DeleteObjectsWithPrefix(string) error   { return nil }
func (f *fakeListManager) ListObjects(prefix string) (*managers.MemoryListResult, error) {
	contents := []managers.MemoryListItem{}
	for _, key := range f.keys {
		if strings.HasPrefix(key, prefix) {
			contents = append(contents, managers.MemoryListItem{Key: key})
		}
	}
	return &managers.MemoryListResult{Contents: contents}, nil
}

func TestListTomeScopesToTomePrefix(t *testing.T) {
	manager := &fakeListManager{keys: []string{
		"mem_a.md",
		"ent_ada.json",
		"tomes/west-marches/mem_b.md",
		"tomes/other-tome/mem_c.md",
	}}

	result, err := ListTome(manager, "west-marches")
	if err != nil {
		t.Fatalf("ListTome: %v", err)
	}

	if len(result.Contents) != 1 || result.Contents[0].Key != "tomes/west-marches/mem_b.md" {
		t.Fatalf("expected only tomes/west-marches/mem_b.md, got %+v", result.Contents)
	}
}

func TestListTomeDefaultTomeExcludesOtherTomesKeys(t *testing.T) {
	manager := &fakeListManager{keys: []string{
		"mem_a.md",
		"ent_ada.json",
		"tomes/west-marches/mem_b.md",
		"tomes/other-tome/mem_c.md",
	}}

	result, err := ListTome(manager, DefaultTome)
	if err != nil {
		t.Fatalf("ListTome: %v", err)
	}

	got := make(map[string]bool, len(result.Contents))
	for _, item := range result.Contents {
		got[item.Key] = true
	}

	if len(got) != 2 || !got["mem_a.md"] || !got["ent_ada.json"] {
		t.Fatalf("expected only the default tome's unprefixed keys, got %+v", result.Contents)
	}
}
