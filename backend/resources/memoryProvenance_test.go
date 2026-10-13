package resources

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

const legacyMemoryDocument = "---\n" +
	"version: connectome/memory/0.1\n" +
	"id: mem_old.md\n" +
	"type: fact\n" +
	"created_at: \"2024-01-01T00:00:00Z\"\n" +
	"source:\n  type: mcp\n  created_at: \"2024-01-01T00:00:00Z\"\n" +
	"entities: [\"david\", \"tea\"]\n" +
	"relationships:\n" +
	"  - subjectEntityId: david\n    predicate: likes\n    objectEntityId: tea\n    kind: fact\n    superseded_by: null\n" +
	"acl: []\n" +
	"derived_from: null\n" +
	"---\n" +
	"David likes tea.\n"

// provenanceMemoryDocument is a 0.2 document carrying every provenance field
// plus a key (x_campaign) no connectome struct declares.
const provenanceMemoryDocument = "---\n" +
	"version: connectome/memory/0.2\n" +
	"id: mem_new.md\n" +
	"type: fact\n" +
	"created_at: \"2026-10-06T18:00:00Z\"\n" +
	"generated:\n  by: process:session-pipeline\n  at: \"2026-10-06T18:00:00Z\"\n" +
	"verified:\n  - by: human:discord-1234\n    at: \"2026-10-06T19:00:00Z\"\n" +
	"sources:\n  - id: chunk-12\n    resource: mem_5f2c.md\n    title: Session 14 transcript, 00:42-00:47\n    locator: turns 3-9\n" +
	"entities: [\"david\", \"tea\"]\n" +
	"relationships:\n" +
	"  - subjectEntityId: david\n    predicate: likes\n    objectEntityId: tea\n    kind: fact\n    superseded_by: null\n" +
	"acl: []\n" +
	"derived_from: null\n" +
	"x_campaign:\n  name: west-marches\n  session: 14\n" +
	"---\n" +
	"David likes tea.\n"

// frontmatterMap decodes a document's frontmatter generically, so a test can
// see keys fullMemoryFrontmatter doesn't declare.
func frontmatterMap(t *testing.T, content string) map[string]any {
	t.Helper()
	raw, _, ok := splitFrontmatter(content)
	if !ok {
		t.Fatalf("expected parseable frontmatter, got %q", content)
	}
	var out map[string]any
	if err := yaml.Unmarshal([]byte(raw), &out); err != nil {
		t.Fatalf("unmarshal frontmatter: %v", err)
	}
	return out
}

func TestHydrateMemoryDocumentUpgradesLegacySourceToGenerated(t *testing.T) {
	hydrated := HydrateMemoryDocument(legacyMemoryDocument)

	fm := frontmatterMap(t, hydrated)
	if fm["version"] != memorySchemaVersion {
		t.Fatalf("expected version %q, got %v", memorySchemaVersion, fm["version"])
	}
	generated, _ := fm["generated"].(map[string]any)
	if generated["by"] != "mcp" || generated["at"] != "2024-01-01T00:00:00Z" {
		t.Fatalf("expected generated derived from source, got %v", fm["generated"])
	}
	for _, key := range []string{"source", "verified", "sources"} {
		if _, present := fm[key]; present {
			t.Fatalf("expected no %q key on an upgraded 0.1 document, got %v", key, fm[key])
		}
	}
	if fm["created_at"] != "2024-01-01T00:00:00Z" {
		t.Fatalf("expected created_at untouched, got %v", fm["created_at"])
	}
}

func TestHydrateMemoryDocumentLeavesCurrentDocumentByteForByte(t *testing.T) {
	if got := HydrateMemoryDocument(provenanceMemoryDocument); got != provenanceMemoryDocument {
		t.Fatalf("expected a compliant 0.2 document to round-trip byte-for-byte, got %q", got)
	}
}

func TestParseMemoryDocumentIndexesLegacyAndCurrentAlike(t *testing.T) {
	for _, doc := range []string{legacyMemoryDocument, provenanceMemoryDocument} {
		fm, body, ok := ParseMemoryDocument(doc)
		if !ok {
			t.Fatalf("expected document to be indexable: %q", doc)
		}
		if fm.Type != "fact" || len(fm.Entities) != 2 || strings.TrimSpace(body) != "David likes tea." {
			t.Fatalf("unexpected parse result %+v / %q", fm, body)
		}
	}
}

func TestPatchRelationshipSupersededByPreservesProvenanceAndUnknownKeys(t *testing.T) {
	patched, err := PatchRelationshipSupersededBy(provenanceMemoryDocument, "david", "likes", ptr("tea"), ptr("mem_correction.md"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	before := frontmatterMap(t, provenanceMemoryDocument)
	after := frontmatterMap(t, patched)
	for _, key := range []string{"version", "generated", "verified", "sources", "x_campaign"} {
		want, _ := yaml.Marshal(before[key])
		got, _ := yaml.Marshal(after[key])
		if string(want) != string(got) {
			t.Fatalf("expected %q preserved across a patch\nwant: %s\ngot:  %s", key, want, got)
		}
	}
	// The unknown key nested inside a sources entry survives too.
	if !strings.Contains(patched, "locator: turns 3-9") {
		t.Fatalf("expected unknown sources key preserved, got:\n%s", patched)
	}
	if !strings.Contains(patched, "superseded_by: mem_correction.md") {
		t.Fatalf("expected the patch applied, got:\n%s", patched)
	}
}

func TestPatchRelationshipSupersededByWritesLegacyDocumentAsCurrent(t *testing.T) {
	patched, err := PatchRelationshipSupersededBy(legacyMemoryDocument, "david", "likes", ptr("tea"), ptr("mem_correction.md"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	fm := frontmatterMap(t, patched)
	generated, _ := fm["generated"].(map[string]any)
	if fm["version"] != memorySchemaVersion || generated["by"] != "mcp" {
		t.Fatalf("expected a 0.2 document with generated.by mcp, got %v", fm)
	}
	if _, present := fm["source"]; present {
		t.Fatalf("expected source dropped on rewrite, got %v", fm["source"])
	}
}

func TestVerifiedAcceptsBareMapping(t *testing.T) {
	doc := strings.Replace(provenanceMemoryDocument,
		"verified:\n  - by: human:discord-1234\n    at: \"2026-10-06T19:00:00Z\"\n",
		"verified:\n  by: human:discord-1234\n  at: \"2026-10-06T19:00:00Z\"\n", 1)

	patched, err := AppendMemoryVerified(doc, "human:discord-5678", time.Date(2026, 10, 7, 12, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	verified, _ := frontmatterMap(t, patched)["verified"].([]any)
	if len(verified) != 2 {
		t.Fatalf("expected the bare mapping kept plus one appended entry, got %v", verified)
	}
	first, _ := verified[0].(map[string]any)
	second, _ := verified[1].(map[string]any)
	if first["by"] != "human:discord-1234" || second["by"] != "human:discord-5678" || second["at"] != "2026-10-07T12:00:00Z" {
		t.Fatalf("unexpected verified entries %v", verified)
	}
}

func TestAppendMemoryVerifiedRejectsNonMemoryContent(t *testing.T) {
	for _, content := range []string{`{"id":"ada"}`, "---\nnot: memory\n---\nbody\n"} {
		if _, err := AppendMemoryVerified(content, "human:gm", time.Now()); err == nil {
			t.Fatalf("expected an error for non-memory content %q", content)
		}
	}
}

func TestVerifyAppendsWithoutReindexing(t *testing.T) {
	srv, connectomeDir, indexer := newTestServer(t)
	base := srv.URL + "/api/connectome/memory"

	const key = "mem_old.md"
	body, contentType := multipartBody(t, []filePart{{name: key, content: legacyMemoryDocument}})
	resp, payload := doRequest(t, http.MethodPost, base+"/", body, map[string]string{"Content-Type": contentType})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("write: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	insertsAfterWrite := indexer.insertCount()

	jsonHeaders := map[string]string{"Content-Type": "application/json"}
	for _, patchBody := range []string{
		`{"key":"mem_old.md","by":"human:discord-1234","at":"2026-10-06T19:00:00Z"}`,
		`{"key":"mem_old.md","by":"human:discord-5678"}`,
	} {
		resp, payload = doRequest(t, http.MethodPatch, base+"/verified", strings.NewReader(patchBody), jsonHeaders)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("verify: expected 200, got %d (%s)", resp.StatusCode, payload)
		}
	}

	if got := indexer.insertCount(); got != insertsAfterWrite {
		t.Fatalf("expected verify to never call the embedder, insert count went from %d to %d", insertsAfterWrite, got)
	}

	onDisk, err := os.ReadFile(filepath.Join(connectomeDir, key))
	if err != nil {
		t.Fatalf("read patched file: %v", err)
	}
	fm := frontmatterMap(t, string(onDisk))
	verified, _ := fm["verified"].([]any)
	if len(verified) != 2 {
		t.Fatalf("expected 2 verified entries, got %v", fm["verified"])
	}
	first, _ := verified[0].(map[string]any)
	second, _ := verified[1].(map[string]any)
	if first["by"] != "human:discord-1234" || first["at"] != "2026-10-06T19:00:00Z" {
		t.Fatalf("unexpected first verified entry %v", first)
	}
	if at, _ := second["at"].(string); second["by"] != "human:discord-5678" || at == "" {
		t.Fatalf("expected second entry stamped with a default at, got %v", second)
	}
	if fm["version"] != memorySchemaVersion {
		t.Fatalf("expected the rewritten document to be %q, got %v", memorySchemaVersion, fm["version"])
	}
	if !strings.Contains(string(onDisk), "David likes tea.") {
		t.Fatalf("expected body to survive the patch, got:\n%s", onDisk)
	}
}

func TestVerifyRejectsBadRequests(t *testing.T) {
	srv, _, _ := newTestServer(t)
	url := srv.URL + "/api/connectome/memory/verified"
	jsonHeaders := map[string]string{"Content-Type": "application/json"}

	cases := map[string]int{
		`{"key":"mem_old.md"}`: http.StatusBadRequest,
		`{"by":"human:gm"}`:    http.StatusBadRequest,
		`{"key":"mem_old.md","by":"human:gm","at":"yesterday"}`: http.StatusBadRequest,
		`{"key":"mem_missing.md","by":"human:gm"}`:              http.StatusNotFound,
	}
	for body, want := range cases {
		resp, payload := doRequest(t, http.MethodPatch, url, strings.NewReader(body), jsonHeaders)
		if resp.StatusCode != want {
			t.Fatalf("%s: expected %d, got %d (%s)", body, want, resp.StatusCode, payload)
		}
	}
}
