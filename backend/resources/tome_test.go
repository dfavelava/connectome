package resources

import "testing"

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
