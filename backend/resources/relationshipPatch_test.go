package resources

import (
	"errors"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func ptr(s string) *string { return &s }

func relationshipMemoryDocument(relationshipsYAML string) string {
	return "---\n" +
		"version: connectome/memory/0.1\n" +
		"id: mem_rel.md\n" +
		"type: fact\n" +
		"created_at: \"2024-01-01T00:00:00Z\"\n" +
		"source:\n  type: mcp\n  created_at: \"2024-01-01T00:00:00Z\"\n" +
		"entities: [\"david\", \"grace\"]\n" +
		"relationships:\n" + relationshipsYAML +
		"acl: [\"GM\"]\n" +
		"---\n" +
		"David and Grace both like tea.\n"
}

func TestPatchRelationshipSupersededBySetsMatchingEntryOnly(t *testing.T) {
	doc := relationshipMemoryDocument(
		"  - subjectEntityId: david\n    predicate: likes\n    objectEntityId: tea\n    kind: fact\n    superseded_by: null\n" +
			"  - subjectEntityId: grace\n    predicate: likes\n    objectEntityId: tea\n    kind: fact\n    superseded_by: null\n",
	)

	patched, err := PatchRelationshipSupersededBy(doc, "david", "likes", ptr("tea"), ptr("mem_correction.md"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	fm, body, ok := splitFrontmatter(patched)
	if !ok {
		t.Fatalf("expected patched document to still have parseable frontmatter, got %q", patched)
	}
	if strings.TrimSpace(body) != "David and Grace both like tea." {
		t.Fatalf("expected body untouched, got %q", body)
	}

	var out fullMemoryFrontmatter
	if err := yaml.Unmarshal([]byte(fm), &out); err != nil {
		t.Fatalf("unmarshal patched frontmatter: %v", err)
	}

	if len(out.Relationships) != 2 {
		t.Fatalf("expected 2 relationships, got %d", len(out.Relationships))
	}
	if out.Relationships[0].SupersededBy == nil || *out.Relationships[0].SupersededBy != "mem_correction.md" {
		t.Fatalf("expected david/likes/tea superseded_by mem_correction.md, got %v", out.Relationships[0].SupersededBy)
	}
	if out.Relationships[1].SupersededBy != nil {
		t.Fatalf("expected grace/likes/tea to be untouched, got %v", out.Relationships[1].SupersededBy)
	}

	// Untouched fields round-trip unchanged.
	if out.ID != "mem_rel.md" || out.Version != "connectome/memory/0.1" || out.Type != "fact" {
		t.Fatalf("expected id/version/type preserved, got %+v", out)
	}
	if out.ACL == nil || len(*out.ACL) != 1 || (*out.ACL)[0] != "GM" {
		t.Fatalf("expected acl [GM] preserved, got %v", out.ACL)
	}
	if len(out.Entities) != 2 || out.Entities[0] != "david" || out.Entities[1] != "grace" {
		t.Fatalf("expected entities preserved, got %v", out.Entities)
	}
}

func TestPatchRelationshipSupersededByClearsWithNil(t *testing.T) {
	doc := relationshipMemoryDocument(
		"  - subjectEntityId: david\n    predicate: likes\n    objectEntityId: tea\n    kind: fact\n    superseded_by: mem_old.md\n",
	)

	patched, err := PatchRelationshipSupersededBy(doc, "david", "likes", ptr("tea"), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	fm, _, _ := splitFrontmatter(patched)
	var out fullMemoryFrontmatter
	if err := yaml.Unmarshal([]byte(fm), &out); err != nil {
		t.Fatalf("unmarshal patched frontmatter: %v", err)
	}
	if out.Relationships[0].SupersededBy != nil {
		t.Fatalf("expected superseded_by cleared, got %v", *out.Relationships[0].SupersededBy)
	}
}

func TestPatchRelationshipSupersededByMatchesNilObjectEntityID(t *testing.T) {
	doc := relationshipMemoryDocument(
		"  - subjectEntityId: david\n    predicate: is_tired\n    objectEntityId: null\n    kind: fact\n    superseded_by: null\n",
	)

	patched, err := PatchRelationshipSupersededBy(doc, "david", "is_tired", nil, ptr("mem_correction.md"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	fm, _, _ := splitFrontmatter(patched)
	var out fullMemoryFrontmatter
	if err := yaml.Unmarshal([]byte(fm), &out); err != nil {
		t.Fatalf("unmarshal patched frontmatter: %v", err)
	}
	if out.Relationships[0].SupersededBy == nil || *out.Relationships[0].SupersededBy != "mem_correction.md" {
		t.Fatalf("expected match on nil objectEntityId, got %v", out.Relationships[0].SupersededBy)
	}
}

func TestPatchRelationshipSupersededByBackfillsMissingKind(t *testing.T) {
	// A relationship written before `kind` existed has no kind key at all.
	doc := relationshipMemoryDocument(
		"  - subjectEntityId: david\n    predicate: likes\n    objectEntityId: tea\n",
	)

	patched, err := PatchRelationshipSupersededBy(doc, "david", "likes", ptr("tea"), ptr("mem_correction.md"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	fm, _, _ := splitFrontmatter(patched)
	var out fullMemoryFrontmatter
	if err := yaml.Unmarshal([]byte(fm), &out); err != nil {
		t.Fatalf("unmarshal patched frontmatter: %v", err)
	}
	if out.Relationships[0].Kind != defaultRelationshipKind {
		t.Fatalf("expected backfilled kind %q, got %q", defaultRelationshipKind, out.Relationships[0].Kind)
	}
}

func TestPatchRelationshipSupersededByReturnsErrorWhenNoMatch(t *testing.T) {
	doc := relationshipMemoryDocument(
		"  - subjectEntityId: david\n    predicate: likes\n    objectEntityId: tea\n    kind: fact\n    superseded_by: null\n",
	)

	_, err := PatchRelationshipSupersededBy(doc, "david", "dislikes", ptr("tea"), ptr("mem_correction.md"))
	if !errors.Is(err, ErrRelationshipNotFound) {
		t.Fatalf("expected ErrRelationshipNotFound, got %v", err)
	}
}

func TestPatchRelationshipSupersededByRejectsUnparseableContent(t *testing.T) {
	if _, err := PatchRelationshipSupersededBy("no frontmatter here", "david", "likes", nil, nil); err == nil {
		t.Fatalf("expected error for content with no frontmatter")
	}
}
