package resources

import (
	"strings"
	"testing"
	"time"
)

func TestParseMemoryDocumentExtractsFrontmatterAndBody(t *testing.T) {
	content := "---\n" +
		"type: preference\n" +
		"created_at: \"2024-03-05T12:00:00Z\"\n" +
		"entities: [\"ada\", \"grace\"]\n" +
		"---\n" +
		"David prefers tea over coffee.\n"

	fm, body, ok := parseMemoryDocument(content)
	if !ok {
		t.Fatalf("expected ok=true for a valid memory document")
	}
	if fm.Type != "preference" {
		t.Fatalf("expected type preference, got %q", fm.Type)
	}
	if len(fm.Entities) != 2 || fm.Entities[0] != "ada" || fm.Entities[1] != "grace" {
		t.Fatalf("expected entities [ada grace], got %v", fm.Entities)
	}
	if strings.TrimSpace(body) != "David prefers tea over coffee." {
		t.Fatalf("expected trimmed body %q, got %q", "David prefers tea over coffee.", body)
	}

	wantCreatedAt := time.Date(2024, 3, 5, 12, 0, 0, 0, time.UTC)
	if got := fm.createdAtOrNow(); !got.Equal(wantCreatedAt) {
		t.Fatalf("expected created_at %v, got %v", wantCreatedAt, got)
	}
}

func TestParseMemoryDocumentRejectsNonMemoryContent(t *testing.T) {
	cases := map[string]string{
		"plain text":        "just a note with no frontmatter",
		"entity json":       `{"id":"ada","memory_ids":["mem_a.md"]}`,
		"unknown type":      "---\ntype: unknown\n---\nbody\n",
		"unterminated yaml": "---\ntype: note\nbody without a closing marker",
	}

	for name, content := range cases {
		t.Run(name, func(t *testing.T) {
			if _, _, ok := parseMemoryDocument(content); ok {
				t.Fatalf("expected ok=false for %s", name)
			}
		})
	}
}

func TestFrontmatterCreatedAtFallsBackToNowWhenUnparseable(t *testing.T) {
	fm := memoryFrontmatter{CreatedAt: "not-a-timestamp"}
	before := time.Now().UTC()
	got := fm.createdAtOrNow()
	after := time.Now().UTC()

	if got.Before(before) || got.After(after) {
		t.Fatalf("expected createdAtOrNow to fall back to now(), got %v not within [%v, %v]", got, before, after)
	}
}

func TestChunkWordsSplitsWithOverlap(t *testing.T) {
	words := make([]string, 20)
	for i := range words {
		words[i] = "w"
	}
	text := strings.Join(words, " ")

	chunks := chunkWords(text, 10, 2)
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks (10, 10 w/ 2 overlap, remainder), got %d: %v", len(chunks), chunks)
	}
	for i, c := range chunks {
		n := len(strings.Fields(c))
		if n == 0 {
			t.Fatalf("chunk %d unexpectedly empty", i)
		}
	}
}

func TestChunkWordsSingleChunkWhenShort(t *testing.T) {
	chunks := chunkWords("short body here", 512, 50)
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk for short text, got %d: %v", len(chunks), chunks)
	}
	if chunks[0] != "short body here" {
		t.Fatalf("expected chunk to equal input, got %q", chunks[0])
	}
}

func TestChunkWordsEmptyTextProducesNoChunks(t *testing.T) {
	if chunks := chunkWords("   \n\t  ", 512, 50); len(chunks) != 0 {
		t.Fatalf("expected no chunks for blank text, got %v", chunks)
	}
}
