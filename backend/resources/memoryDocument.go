package resources

import (
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
// (see daybidMCP/src/daybidmcp/server.py's format_memory) needed to index a
// memory.
type memoryFrontmatter struct {
	Type      string   `yaml:"type"`
	CreatedAt string   `yaml:"created_at"`
	Entities  []string `yaml:"entities"`
}

// parseMemoryDocument splits a memory file's raw content into its frontmatter
// and body. ok is false for content with no valid, indexable frontmatter -
// notably the plain-JSON entity records stored under the same key namespace,
// which have no frontmatter at all.
func parseMemoryDocument(content string) (frontmatter memoryFrontmatter, body string, ok bool) {
	const delim = "---"
	if !strings.HasPrefix(content, delim+"\n") {
		return memoryFrontmatter{}, "", false
	}

	rest := content[len(delim)+1:]
	end := strings.Index(rest, "\n"+delim)
	if end == -1 {
		return memoryFrontmatter{}, "", false
	}

	var fm memoryFrontmatter
	if err := yaml.Unmarshal([]byte(rest[:end]), &fm); err != nil {
		return memoryFrontmatter{}, "", false
	}
	if !validMemoryTypes[fm.Type] {
		return memoryFrontmatter{}, "", false
	}

	body = strings.TrimPrefix(rest[end+len(delim)+1:], "\n")
	return fm, body, true
}

func (fm memoryFrontmatter) createdAtOrNow() time.Time {
	if t, err := time.Parse(time.RFC3339, fm.CreatedAt); err == nil {
		return t
	}
	return time.Now().UTC()
}

// chunkWords splits text into ~chunkSize-word chunks with overlap words of
// context repeated between consecutive chunks. Word count stands in for a
// token count here since no tokenizer is wired up for nomic-embed-text.
func chunkWords(text string, chunkSize, overlap int) []string {
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
