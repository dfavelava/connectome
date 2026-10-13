package resources

import (
	"fmt"
	"time"

	"gopkg.in/yaml.v3"
)

// memorySchemaVersion mirrors MEMORY_SCHEMA_VERSION in
// connectomeMCP/src/connectomemcp/server.py. legacyMemorySchemaVersion is the
// only older version upgrade knows how to read.
const (
	memorySchemaVersion       = "connectome/memory/0.2"
	legacyMemorySchemaVersion = "connectome/memory/0.1"
)

// memorySourceDocument is the 0.1 `source` block that `generated` replaced.
type memorySourceDocument struct {
	Type      string `yaml:"type"`
	CreatedAt string `yaml:"created_at"`
}

// actorStamp is OKF's {by, at} shape, used for `generated` and each
// `verified` entry. By follows the actor convention: `<producer>/<version>`
// for agents and tools, `human:<id>` for people, `process:<id>` for jobs.
type actorStamp struct {
	By    string               `yaml:"by"`
	At    string               `yaml:"at"`
	Extra map[string]yaml.Node `yaml:",inline"`
}

// verifiedList is the `verified` field: who confirmed the memory's content
// and when. OKF allows a bare mapping in place of a one-element list, so it
// decodes either; it always encodes as a list.
type verifiedList []actorStamp

func (v *verifiedList) UnmarshalYAML(node *yaml.Node) error {
	if node.Kind == yaml.MappingNode {
		var one actorStamp
		if err := node.Decode(&one); err != nil {
			return err
		}
		*v = verifiedList{one}
		return nil
	}
	var many []actorStamp
	if err := node.Decode(&many); err != nil {
		return err
	}
	*v = many
	return nil
}

// sourceDocument is one `sources` entry: where a memory's content was
// extracted from. Resource is a memory key or a URL.
type sourceDocument struct {
	ID       string               `yaml:"id,omitempty"`
	Resource string               `yaml:"resource,omitempty"`
	Title    string               `yaml:"title,omitempty"`
	Extra    map[string]yaml.Node `yaml:",inline"`
}

// upgrade turns a 0.1 frontmatter into 0.2 in memory: `generated` is derived
// from `source` (by = source.type, at = source.created_at) and `source` is
// dropped. It reports whether anything changed. Documents at any other
// version - including ones with no version key at all - are left alone. The
// blob store is never migrated; a document is only persisted as 0.2 the next
// time something rewrites it.
func (fm *fullMemoryFrontmatter) upgrade() bool {
	if fm.Version != legacyMemorySchemaVersion {
		return false
	}
	fm.Version = memorySchemaVersion
	if fm.Generated == nil && fm.Source != nil {
		at := fm.Source.CreatedAt
		if at == "" {
			at = fm.CreatedAt
		}
		fm.Generated = &actorStamp{By: fm.Source.Type, At: at}
	}
	fm.Source = nil
	return true
}

// AppendMemoryVerified appends a {by, at} entry to a memory document's
// `verified` list. Like PatchRelationshipSupersededBy it only rewrites the
// frontmatter - the body is untouched, so callers can persist the result
// without re-embedding - and a 0.1 document comes back as 0.2.
func AppendMemoryVerified(content, by string, at time.Time) (string, error) {
	fm, body, err := parseFullMemoryFrontmatter(content)
	if err != nil {
		return "", err
	}
	if !validMemoryTypes[fm.Type] {
		return "", fmt.Errorf("content is not a memory document")
	}
	fm.upgrade()
	fm.Verified = append(fm.Verified, actorStamp{By: by, At: at.UTC().Format(time.RFC3339)})
	return fm.render(body)
}
