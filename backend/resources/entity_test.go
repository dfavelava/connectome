package resources

import (
	"os"
	"path/filepath"
	"testing"

	"connectome-dev-service/managers"
)

func newLocalManager(t *testing.T, seed map[string]string) managers.MemoryManager {
	t.Helper()

	home := t.TempDir()
	connectomeDir := filepath.Join(home, ".connectome")
	if err := os.MkdirAll(connectomeDir, 0o755); err != nil {
		t.Fatalf("create connectome dir: %v", err)
	}
	for key, content := range seed {
		if err := os.WriteFile(filepath.Join(connectomeDir, key), []byte(content), 0o644); err != nil {
			t.Fatalf("seed %s: %v", key, err)
		}
	}

	t.Setenv("HOME", home)
	t.Setenv("MEMORY_MANAGER", "local")
	return managers.NewMemoryManagerFromEnv()
}

func TestResolveACLScopeIncludesOwnIDAndMemberOf(t *testing.T) {
	manager := newLocalManager(t, map[string]string{
		"ent_alice.json": `{"id":"alice","name":"Alice","member_of":["Party A","Adventurers"]}`,
	})
	resource := NewSearchResource(manager, fakeEmbedder{}, nil)

	scope := resource.resolveACLScope("alice")
	if len(scope) != 3 || scope[0] != "alice" || scope[1] != "Party A" || scope[2] != "Adventurers" {
		t.Fatalf("expected [alice, Party A, Adventurers], got %v", scope)
	}
}

func TestResolveACLScopeDedupesMemberOf(t *testing.T) {
	manager := newLocalManager(t, map[string]string{
		"ent_alice.json": `{"id":"alice","member_of":["Party A","Party A"]}`,
	})
	resource := NewSearchResource(manager, fakeEmbedder{}, nil)

	scope := resource.resolveACLScope("alice")
	if len(scope) != 2 || scope[0] != "alice" || scope[1] != "Party A" {
		t.Fatalf("expected [alice, Party A] with the duplicate collapsed, got %v", scope)
	}
}

func TestResolveACLScopeWithoutEntityRecordIsJustTheID(t *testing.T) {
	manager := newLocalManager(t, nil)
	resource := NewSearchResource(manager, fakeEmbedder{}, nil)

	scope := resource.resolveACLScope("ghost")
	if len(scope) != 1 || scope[0] != "ghost" {
		t.Fatalf("expected [ghost], got %v", scope)
	}
}

func TestResolveACLScopeWithNoMemberOfFieldIsJustTheID(t *testing.T) {
	manager := newLocalManager(t, map[string]string{
		"ent_bob.json": `{"id":"bob","name":"Bob"}`,
	})
	resource := NewSearchResource(manager, fakeEmbedder{}, nil)

	scope := resource.resolveACLScope("bob")
	if len(scope) != 1 || scope[0] != "bob" {
		t.Fatalf("expected [bob], got %v", scope)
	}
}
