package resources

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func parseHydratedFrontmatter(t *testing.T, content string) fullMemoryFrontmatter {
	t.Helper()

	raw, _, ok := splitFrontmatter(content)
	if !ok {
		t.Fatalf("expected hydrated content to still have parseable frontmatter, got %q", content)
	}
	var fm fullMemoryFrontmatter
	if err := yaml.Unmarshal([]byte(raw), &fm); err != nil {
		t.Fatalf("unmarshal hydrated frontmatter: %v", err)
	}
	return fm
}

func TestHydrateMemoryDocumentResolvesMissingACLToDefault(t *testing.T) {
	t.Setenv("DEFAULT_ACL", "players")
	doc := memoryDocument("note", "body", nil) // no acl key at all

	fm := parseHydratedFrontmatter(t, HydrateMemoryDocument(doc))
	if fm.ACL == nil || len(*fm.ACL) != 1 || (*fm.ACL)[0] != "players" {
		t.Fatalf("expected resolved acl [players], got %v", fm.ACL)
	}
}

func TestHydrateMemoryDocumentResolvesMissingACLToUnrestrictedWhenNoDefault(t *testing.T) {
	doc := memoryDocument("note", "body", nil) // no acl key, no DEFAULT_ACL set

	fm := parseHydratedFrontmatter(t, HydrateMemoryDocument(doc))
	if fm.ACL == nil || len(*fm.ACL) != 0 {
		t.Fatalf("expected resolved acl [] (unrestricted), got %v", fm.ACL)
	}
}

func TestHydrateMemoryDocumentLeavesExplicitACLUntouched(t *testing.T) {
	t.Setenv("DEFAULT_ACL", "players")
	doc := "---\n" +
		"type: note\n" +
		"created_at: \"2024-01-01T00:00:00Z\"\n" +
		"entities: []\n" +
		"acl: [\"GM\"]\n" +
		"---\nbody\n"

	fm := parseHydratedFrontmatter(t, HydrateMemoryDocument(doc))
	if fm.ACL == nil || len(*fm.ACL) != 1 || (*fm.ACL)[0] != "GM" {
		t.Fatalf("expected explicit acl [GM] to survive hydration untouched, got %v", fm.ACL)
	}
}

func TestHydrateMemoryDocumentBackfillsMissingRelationshipKind(t *testing.T) {
	// A relationship written before `kind` existed has no kind key at all.
	doc := memoryDocumentWithRelationships("  - subjectEntityId: david\n    predicate: likes\n    objectEntityId: tea\n")

	fm := parseHydratedFrontmatter(t, HydrateMemoryDocument(doc))
	if len(fm.Relationships) != 1 || fm.Relationships[0].Kind != defaultRelationshipKind {
		t.Fatalf("expected backfilled kind %q, got %+v", defaultRelationshipKind, fm.Relationships)
	}
}

func TestHydrateMemoryDocumentLeavesExplicitRelationshipKindUntouched(t *testing.T) {
	doc := memoryDocumentWithRelationships("  - subjectEntityId: david\n    predicate: likes\n    objectEntityId: tea\n    kind: rumor\n")

	fm := parseHydratedFrontmatter(t, HydrateMemoryDocument(doc))
	if len(fm.Relationships) != 1 || fm.Relationships[0].Kind != "rumor" {
		t.Fatalf("expected explicit kind rumor to survive hydration untouched, got %+v", fm.Relationships)
	}
}

func TestHydrateMemoryDocumentPreservesDerivedFrom(t *testing.T) {
	doc := "---\n" +
		"version: connectome/memory/0.1\n" +
		"id: mem_facet.md\n" +
		"type: note\n" +
		"created_at: \"2024-01-01T00:00:00Z\"\n" +
		"source:\n  type: mcp\n  created_at: \"2024-01-01T00:00:00Z\"\n" +
		"entities: []\n" +
		"relationships: []\n" +
		"derived_from: mem_root.md\n" +
		"---\nbody\n"

	fm := parseHydratedFrontmatter(t, HydrateMemoryDocument(doc))
	if fm.DerivedFrom == nil || *fm.DerivedFrom != "mem_root.md" {
		t.Fatalf("expected derived_from to survive hydration, got %v", fm.DerivedFrom)
	}
}

func TestHydrateMemoryDocumentIsNoOpWhenAlreadyCompliant(t *testing.T) {
	doc := "---\n" +
		"type: note\n" +
		"created_at: \"2024-01-01T00:00:00Z\"\n" +
		"entities: []\n" +
		"acl: []\n" +
		"---\nbody\n"

	if got := HydrateMemoryDocument(doc); got != doc {
		t.Fatalf("expected already-compliant content to round-trip byte-for-byte, got %q", got)
	}
}

func TestHydrateMemoryDocumentLeavesNonMemoryContentUnchanged(t *testing.T) {
	cases := []string{
		`{"id":"ada","memory_ids":["mem_a.md"]}`, // entity record, no frontmatter
		"plain string with no frontmatter",
		"---\nnot: memory\n---\nbody\n", // frontmatter with no valid type
	}
	for _, content := range cases {
		if got := HydrateMemoryDocument(content); got != content {
			t.Fatalf("expected non-memory content %q to pass through unchanged, got %q", content, got)
		}
	}
}

func TestHydrateMemoryDocumentBackfillsBothACLAndRelationshipKindTogether(t *testing.T) {
	t.Setenv("DEFAULT_ACL", "GM")
	doc := "---\n" +
		"version: connectome/memory/0.1\n" +
		"id: mem_rel.md\n" +
		"type: fact\n" +
		"created_at: \"2024-01-01T00:00:00Z\"\n" +
		"source:\n  type: mcp\n  created_at: \"2024-01-01T00:00:00Z\"\n" +
		"entities: [\"david\", \"gm\"]\n" +
		"relationships:\n  - subjectEntityId: david\n    predicate: reports_to\n    objectEntityId: gm\n" +
		"---\nDavid reports to the GM.\n"

	hydrated := HydrateMemoryDocument(doc)
	fm := parseHydratedFrontmatter(t, hydrated)
	if fm.ACL == nil || len(*fm.ACL) != 1 || (*fm.ACL)[0] != "GM" {
		t.Fatalf("expected resolved acl [GM], got %v", fm.ACL)
	}
	if len(fm.Relationships) != 1 || fm.Relationships[0].Kind != defaultRelationshipKind {
		t.Fatalf("expected backfilled relationship kind %q, got %+v", defaultRelationshipKind, fm.Relationships)
	}
	_, body, _ := splitFrontmatter(hydrated)
	if strings.TrimSpace(body) != "David reports to the GM." {
		t.Fatalf("expected body untouched, got %q", body)
	}
}
