package resources

import (
	"errors"
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"
)

// ErrRelationshipNotFound is returned by PatchRelationshipSupersededBy when no
// relationship in the document's frontmatter matches the given
// subject/predicate/object. Callers can detect it with errors.Is.
var ErrRelationshipNotFound = errors.New("relationship not found")

// defaultRelationshipKind mirrors DEFAULT_RELATIONSHIP_KIND in
// daybidMCP/src/daybidmcp/server.py, used to backfill relationships written
// before the `kind` field existed.
const defaultRelationshipKind = "fact"

// relationshipDocument mirrors one entry of daybidmcp's Relationship model as
// written into memory frontmatter (see format_memory in
// daybidMCP/src/daybidmcp/server.py). Field order matches the Python
// model_dump so a patched document looks the same as a freshly written one.
type relationshipDocument struct {
	SubjectEntityID string  `yaml:"subjectEntityId"`
	Predicate       string  `yaml:"predicate"`
	ObjectEntityID  *string `yaml:"objectEntityId"`
	Kind            string  `yaml:"kind"`
	SupersededBy    *string `yaml:"superseded_by"`
}

// memorySourceDocument mirrors MemorySource in server.py.
type memorySourceDocument struct {
	Type      string `yaml:"type"`
	CreatedAt string `yaml:"created_at"`
}

// fullMemoryFrontmatter is the complete connectome memory frontmatter shape
// (see MemoryMetadata in server.py). Unlike memoryFrontmatter - which only
// extracts what indexing needs - this round-trips every field so a patch can
// rewrite the document without dropping anything.
type fullMemoryFrontmatter struct {
	Version       string                 `yaml:"version"`
	ID            string                 `yaml:"id"`
	Type          string                 `yaml:"type"`
	CreatedAt     string                 `yaml:"created_at"`
	Source        memorySourceDocument   `yaml:"source"`
	Entities      []string               `yaml:"entities"`
	Relationships []relationshipDocument `yaml:"relationships"`
	// ACL is a pointer with omitempty so a document that never had the key
	// (nil) round-trips without one, matching format_memory's behavior of
	// omitting "acl" entirely rather than writing "acl: null".
	ACL *[]string `yaml:"acl,omitempty"`
}

// relationshipMatches reports whether rel names the same
// subject/predicate/object as the patch target. A nil objectEntityId only
// matches another nil objectEntityId.
func relationshipMatches(rel relationshipDocument, subjectEntityID, predicate string, objectEntityID *string) bool {
	if rel.SubjectEntityID != subjectEntityID || rel.Predicate != predicate {
		return false
	}
	if (rel.ObjectEntityID == nil) != (objectEntityID == nil) {
		return false
	}
	return rel.ObjectEntityID == nil || *rel.ObjectEntityID == *objectEntityID
}

// PatchRelationshipSupersededBy sets superseded_by on the relationship entry
// matching (subjectEntityId, predicate, objectEntityId) within a memory
// document's frontmatter - pass nil to clear a prior supersession. Content,
// entities, acl, and every other relationship are left untouched, and the
// body is never parsed or rewritten, so callers can persist the result
// without re-embedding.
func PatchRelationshipSupersededBy(content, subjectEntityID, predicate string, objectEntityID, supersededBy *string) (string, error) {
	raw, body, ok := splitFrontmatter(content)
	if !ok {
		return "", fmt.Errorf("content has no parseable memory frontmatter")
	}

	var fm fullMemoryFrontmatter
	if err := yaml.Unmarshal([]byte(raw), &fm); err != nil {
		return "", fmt.Errorf("parse frontmatter: %w", err)
	}

	found := false
	for i := range fm.Relationships {
		// Backfill relationships written before `kind` existed.
		if fm.Relationships[i].Kind == "" {
			fm.Relationships[i].Kind = defaultRelationshipKind
		}
		if relationshipMatches(fm.Relationships[i], subjectEntityID, predicate, objectEntityID) {
			fm.Relationships[i].SupersededBy = supersededBy
			found = true
		}
	}
	if !found {
		return "", ErrRelationshipNotFound
	}

	patchedYAML, err := yaml.Marshal(fm)
	if err != nil {
		return "", fmt.Errorf("marshal frontmatter: %w", err)
	}

	return "---\n" + strings.TrimRight(string(patchedYAML), "\n") + "\n---\n" + body, nil
}
