package resources

import (
	"strings"

	"gopkg.in/yaml.v3"
)

// HydrateMemoryDocument resolves read-time defaults for a memory document
// written before a field existed, for read-path consumers - get_memory and
// hydrated recall results - that expect a document to carry every field its
// current schema defines. A frontmatter with no acl key picks up this
// instance's configured DEFAULT_ACL (see ResolveACL in acl.go); a
// relationship written before `kind` existed defaults its kind to
// defaultRelationshipKind (see relationshipPatch.go). This is
// "default at read time, no frontmatter migration": the blob store is never
// rewritten, and a document that already carries every field it needs is
// returned byte-for-byte unchanged rather than reformatted. Content with no
// parseable memory frontmatter (e.g. an ent_*.json entity record) is
// likewise returned unchanged.
func HydrateMemoryDocument(content string) string {
	raw, body, ok := splitFrontmatter(content)
	if !ok {
		return content
	}

	var fm fullMemoryFrontmatter
	if err := yaml.Unmarshal([]byte(raw), &fm); err != nil || !validMemoryTypes[fm.Type] {
		return content
	}

	changed := false
	if fm.ACL == nil {
		acl := ResolveACL(fm.ACL)
		fm.ACL = &acl
		changed = true
	}
	for i := range fm.Relationships {
		if fm.Relationships[i].Kind == "" {
			fm.Relationships[i].Kind = defaultRelationshipKind
			changed = true
		}
	}
	if !changed {
		return content
	}

	hydratedYAML, err := yaml.Marshal(fm)
	if err != nil {
		return content
	}

	return "---\n" + strings.TrimRight(string(hydratedYAML), "\n") + "\n---\n" + body
}
