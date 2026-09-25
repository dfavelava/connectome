package resources

import (
	"fmt"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// validMemoryTypes mirrors the CHECK constraint on embeddings.type in
// backend/migrations/0001_create_embeddings.up.sql.
var validMemoryTypes = map[string]bool{
	"note":       true,
	"fact":       true,
	"preference": true,
	"event":      true,
}

// memoryFrontmatter is the subset of the connectome memory YAML frontmatter
// (see connectomeMCP/src/connectomemcp/server.py's format_memory) needed to index a
// memory.
type memoryFrontmatter struct {
	Type      string `yaml:"type"`
	CreatedAt string `yaml:"created_at"`
	// OccurredAt is when the described event happened (RFC3339), as opposed
	// to CreatedAt, which is when the memory was written. Empty means
	// unknown - see occurredAt.
	OccurredAt string   `yaml:"occurred_at"`
	Entities   []string `yaml:"entities"`
	// ACL is a pointer so an omitted key (nil) is distinguishable from an
	// explicit empty list - see ResolveACL in acl.go.
	ACL *[]string `yaml:"acl"`
}

// splitFrontmatter splits a memory file's raw content into its raw YAML
// frontmatter block and body text. ok is false for content with no
// "---"-delimited frontmatter at all - notably the plain-JSON entity records
// stored under the same key namespace.
func splitFrontmatter(content string) (frontmatterYAML, body string, ok bool) {
	const delim = "---"
	if !strings.HasPrefix(content, delim+"\n") {
		return "", "", false
	}

	rest := content[len(delim)+1:]
	end := strings.Index(rest, "\n"+delim)
	if end == -1 {
		return "", "", false
	}

	frontmatterYAML = rest[:end]
	body = strings.TrimPrefix(rest[end+len(delim)+1:], "\n")
	return frontmatterYAML, body, true
}

// ParseMemoryDocument splits a memory file's raw content into its frontmatter
// and body. ok is false for content with no valid, indexable frontmatter -
// notably the plain-JSON entity records stored under the same key namespace,
// which have no frontmatter at all.
func ParseMemoryDocument(content string) (frontmatter memoryFrontmatter, body string, ok bool) {
	raw, body, ok := splitFrontmatter(content)
	if !ok {
		return memoryFrontmatter{}, "", false
	}

	var fm memoryFrontmatter
	if err := yaml.Unmarshal([]byte(raw), &fm); err != nil {
		return memoryFrontmatter{}, "", false
	}
	if !validMemoryTypes[fm.Type] {
		return memoryFrontmatter{}, "", false
	}

	return fm, body, true
}

func (fm memoryFrontmatter) createdAtOrNow() time.Time {
	if t, err := time.Parse(time.RFC3339, fm.CreatedAt); err == nil {
		return t
	}
	return time.Now().UTC()
}

// occurredAt returns the memory's occurred_at as a time, or nil when the key
// is absent (event time unknown). Unlike createdAtOrNow it never falls back
// to a default: a present-but-malformed value is an error rather than being
// silently replaced, since inventing an event time would reintroduce the
// created_at/occurred_at conflation this field exists to avoid.
func (fm memoryFrontmatter) occurredAt() (*time.Time, error) {
	if fm.OccurredAt == "" {
		return nil, nil
	}
	t, err := time.Parse(time.RFC3339, fm.OccurredAt)
	if err != nil {
		return nil, fmt.Errorf("invalid occurred_at %q (want RFC3339): %w", fm.OccurredAt, err)
	}
	return &t, nil
}

// ValidateMemoryDocument reports a client error in a memory document that
// would otherwise only surface when IndexMemory runs, after the blob is
// stored: today, a present-but-malformed occurred_at. Content with no valid
// memory frontmatter (e.g. an ent_*.json entity record) always passes.
func ValidateMemoryDocument(content string) error {
	fm, _, ok := ParseMemoryDocument(content)
	if !ok {
		return nil
	}
	_, err := fm.occurredAt()
	return err
}

// ChunkWords splits text into ~chunkSize-word chunks with overlap words of
// context repeated between consecutive chunks. Word count stands in for a
// token count here since no tokenizer is wired up for nomic-embed-text.
func ChunkWords(text string, chunkSize, overlap int) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}
	if chunkSize <= 0 {
		chunkSize = len(words)
	}
	if overlap < 0 || overlap >= chunkSize {
		overlap = 0
	}

	step := chunkSize - overlap
	chunks := make([]string, 0, len(words)/step+1)
	for start := 0; start < len(words); start += step {
		end := min(start+chunkSize, len(words))
		chunks = append(chunks, strings.Join(words[start:end], " "))
		if end == len(words) {
			break
		}
	}
	return chunks
}
