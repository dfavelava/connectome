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
// connectomeMCP/src/connectomemcp/server.py, used to backfill relationships written
// before the `kind` field existed.
const defaultRelationshipKind = "fact"

// relationshipDocument mirrors one entry of connectomemcp's Relationship model as
// written into memory frontmatter (see format_memory in
// connectomeMCP/src/connectomemcp/server.py). Field order matches the Python
// model_dump so a patched document looks the same as a freshly written one.
type relationshipDocument struct {
	SubjectEntityID string  `yaml:"subjectEntityId"`
	Predicate       string  `yaml:"predicate"`
	ObjectEntityID  *string `yaml:"objectEntityId"`
	Kind            string  `yaml:"kind"`
	SupersededBy    *string `yaml:"superseded_by"`
}

// fullMemoryFrontmatter is the complete connectome memory frontmatter shape
// (see MemoryMetadata in server.py). Unlike memoryFrontmatter - which only
// extracts what indexing needs - this round-trips every field so a patch can
// rewrite the document without dropping anything.
type fullMemoryFrontmatter struct {
	Version   string `yaml:"version"`
	ID        string `yaml:"id"`
	Type      string `yaml:"type"`
	CreatedAt string `yaml:"created_at"`
	// OccurredAt is omitempty so a document that never had the key (the
	// common case - it's optional) round-trips without one, rather than
	// gaining an `occurred_at: ""` line. Preserving it matters: hydrate and
	// relationship patches rewrite frontmatter through this struct, and would
	// otherwise drop the field.
	OccurredAt string `yaml:"occurred_at,omitempty"`
	// Source is the 0.1 provenance block. It is only ever read: upgrade
	// folds it into Generated and clears it, so a rewritten document never
	// carries one.
	Source *memorySourceDocument `yaml:"source,omitempty"`
	// Generated, Verified and Sources are the 0.2 provenance fields - see
	// memoryProvenance.go.
	Generated     *actorStamp            `yaml:"generated,omitempty"`
	Verified      verifiedList           `yaml:"verified,omitempty"`
	Sources       []sourceDocument       `yaml:"sources,omitempty"`
	Entities      []string               `yaml:"entities"`
	Relationships []relationshipDocument `yaml:"relationships"`
	// ACL is a pointer with omitempty so a document that never had the key
	// (nil) round-trips without one, matching format_memory's behavior of
	// omitting "acl" entirely rather than writing "acl: null".
	ACL *[]string `yaml:"acl,omitempty"`
	// DerivedFrom mirrors MemoryMetadata.derived_from in server.py. Unlike
	// acl, format_memory always includes this key (as null for a root
	// memory), so it round-trips without omitempty.
	DerivedFrom *string `yaml:"derived_from"`
	// Extra holds every key this struct doesn't declare, so a rewrite keeps
	// fields added by a newer schema or another producer instead of silently
	// dropping them (OKF requires consumers to preserve unrecognized keys).
	Extra map[string]yaml.Node `yaml:",inline"`
}

// parseFullMemoryFrontmatter splits a memory document and decodes its whole
// frontmatter for a rewrite.
func parseFullMemoryFrontmatter(content string) (fm fullMemoryFrontmatter, body string, err error) {
	raw, body, ok := splitFrontmatter(content)
	if !ok {
		return fm, "", fmt.Errorf("content has no parseable memory frontmatter")
	}
	if err := yaml.Unmarshal([]byte(raw), &fm); err != nil {
		return fm, "", fmt.Errorf("parse frontmatter: %w", err)
	}
	return fm, body, nil
}

// render reassembles a memory document from fm and an untouched body.
func (fm fullMemoryFrontmatter) render(body string) (string, error) {
	out, err := yaml.Marshal(fm)
	if err != nil {
		return "", fmt.Errorf("marshal frontmatter: %w", err)
	}
	return "---\n" + strings.TrimRight(string(out), "\n") + "\n---\n" + body, nil
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
// without re-embedding. A 0.1 document comes back as 0.2 (see upgrade).
func PatchRelationshipSupersededBy(content, subjectEntityID, predicate string, objectEntityID, supersededBy *string) (string, error) {
	fm, body, err := parseFullMemoryFrontmatter(content)
	if err != nil {
		return "", err
	}
	fm.upgrade()

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

	return fm.render(body)
}
