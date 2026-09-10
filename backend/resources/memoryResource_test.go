package resources

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

const testToken = "e2e-test-token"

// newTestServer wires the memory routes onto a fresh gin engine backed by the
// local filesystem manager rooted at a per-test temp directory, and returns an
// httptest server plus that directory's .connectome path.
func newTestServer(t *testing.T) (*httptest.Server, string) {
	t.Helper()

	home := t.TempDir()
	connectomeDir := filepath.Join(home, ".connectome")
	if err := os.MkdirAll(connectomeDir, 0o755); err != nil {
		t.Fatalf("create connectome dir: %v", err)
	}

	t.Setenv("HOME", home)
	t.Setenv("MEMORY_MANAGER", "local")
	t.Setenv("apikey", testToken)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	InitMemoryResource(r.Group("/api/connectome"))

	srv := httptest.NewServer(r)
	t.Cleanup(srv.Close)

	return srv, connectomeDir
}

type filePart struct {
	name    string
	content string
}

func multipartBody(t *testing.T, parts []filePart) (*bytes.Buffer, string) {
	t.Helper()

	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for _, p := range parts {
		fw, err := w.CreateFormFile("file", p.name)
		if err != nil {
			t.Fatalf("create form file %s: %v", p.name, err)
		}
		if _, err := io.WriteString(fw, p.content); err != nil {
			t.Fatalf("write form file %s: %v", p.name, err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	return &buf, w.FormDataContentType()
}

func doRequest(t *testing.T, method, url string, body io.Reader, headers map[string]string) (*http.Response, []byte) {
	t.Helper()

	req, err := http.NewRequest(method, url, body)
	if err != nil {
		t.Fatalf("new request %s %s: %v", method, url, err)
	}
	req.Header.Set("Authorization", "Bearer "+testToken)
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("do request %s %s: %v", method, url, err)
	}
	defer resp.Body.Close()

	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body %s %s: %v", method, url, err)
	}
	return resp, payload
}

func decodeJSON(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	var out map[string]any
	if err := json.Unmarshal(payload, &out); err != nil {
		t.Fatalf("decode json %q: %v", string(payload), err)
	}
	return out
}

func TestMemoryRoutesRequireBearerToken(t *testing.T) {
	srv, _ := newTestServer(t)

	cases := []struct {
		name   string
		header string
	}{
		{name: "missing header", header: ""},
		{name: "wrong scheme", header: "Token " + testToken},
		{name: "bad key", header: "Bearer wrong-token"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req, err := http.NewRequest(http.MethodGet, srv.URL+"/api/connectome/memory/list", nil)
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			if tc.header != "" {
				req.Header.Set("Authorization", tc.header)
			}

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatalf("do request: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusUnauthorized {
				t.Fatalf("expected 401, got %d", resp.StatusCode)
			}
		})
	}
}

func TestMemoryWriteReadListDeleteRoundTrip(t *testing.T) {
	srv, connectomeDir := newTestServer(t)
	base := srv.URL + "/api/connectome/memory"

	const key = "mem_roundtrip.md"
	const content = "David prefers tea over coffee."

	// write
	body, contentType := multipartBody(t, []filePart{{name: key, content: content}})
	resp, payload := doRequest(t, http.MethodPost, base+"/", body, map[string]string{"Content-Type": contentType})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("write: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	if got := decodeJSON(t, payload)["message"]; got != "success" {
		t.Fatalf("write: expected message success, got %v", got)
	}

	onDisk, err := os.ReadFile(filepath.Join(connectomeDir, key))
	if err != nil {
		t.Fatalf("read written file: %v", err)
	}
	if string(onDisk) != content {
		t.Fatalf("written file mismatch: got %q", string(onDisk))
	}

	// read
	resp, payload = doRequest(t, http.MethodGet, base+"/?key="+key, nil, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("read: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	if got := decodeJSON(t, payload)["content"]; got != content {
		t.Fatalf("read: expected content %q, got %v", content, got)
	}

	// list
	resp, payload = doRequest(t, http.MethodGet, base+"/list", nil, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("list: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	var listResult struct {
		Contents []struct {
			Key     string  `json:"Key"`
			Preview *string `json:"Preview"`
		} `json:"Contents"`
	}
	if err := json.Unmarshal(payload, &listResult); err != nil {
		t.Fatalf("list: decode %q: %v", payload, err)
	}
	if len(listResult.Contents) != 1 || listResult.Contents[0].Key != key {
		t.Fatalf("list: expected [%s], got %+v", key, listResult.Contents)
	}
	if listResult.Contents[0].Preview == nil || *listResult.Contents[0].Preview != content {
		t.Fatalf("list: expected preview %q, got %v", content, listResult.Contents[0].Preview)
	}

	// delete
	resp, payload = doRequest(t, http.MethodDelete, base+"/", strings.NewReader(`{"key":"`+key+`"}`), map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("delete: expected 204, got %d (%s)", resp.StatusCode, payload)
	}
	if _, err := os.Stat(filepath.Join(connectomeDir, key)); !os.IsNotExist(err) {
		t.Fatalf("delete: expected file gone, stat err = %v", err)
	}

	// read after delete fails
	resp, _ = doRequest(t, http.MethodGet, base+"/?key="+key, nil, nil)
	if resp.StatusCode == http.StatusOK {
		t.Fatalf("read after delete: expected error status, got 200")
	}
}

func TestMemoryReadRejectsMissingKey(t *testing.T) {
	srv, _ := newTestServer(t)

	resp, _ := doRequest(t, http.MethodGet, srv.URL+"/api/connectome/memory/", nil, nil)
	if resp.StatusCode == http.StatusOK {
		t.Fatalf("expected error status for missing key, got 200")
	}
}

func TestMemoryBatchWriteAndBatchRead(t *testing.T) {
	srv, connectomeDir := newTestServer(t)
	base := srv.URL + "/api/connectome/memory"

	parts := []filePart{
		{name: "mem_a.md", content: "first note"},
		{name: "ent_ada.json", content: `{"id":"ada","memory_ids":["mem_a.md"]}`},
	}

	// batch write
	body, contentType := multipartBody(t, parts)
	resp, payload := doRequest(t, http.MethodPost, base+"/batch", body, map[string]string{"Content-Type": contentType})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("batch write: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	for _, p := range parts {
		onDisk, err := os.ReadFile(filepath.Join(connectomeDir, p.name))
		if err != nil {
			t.Fatalf("batch write: read %s: %v", p.name, err)
		}
		if string(onDisk) != p.content {
			t.Fatalf("batch write: %s mismatch: got %q", p.name, string(onDisk))
		}
	}

	// batch read, all present
	resp, payload = doRequest(t, http.MethodPost, base+"/batch/read", strings.NewReader(`{"keys":["mem_a.md","ent_ada.json"]}`), map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("batch read: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	contents, ok := decodeJSON(t, payload)["contents"].(map[string]any)
	if !ok {
		t.Fatalf("batch read: missing contents in %s", payload)
	}
	if contents["mem_a.md"] != "first note" || contents["ent_ada.json"] != parts[1].content {
		t.Fatalf("batch read: unexpected contents %v", contents)
	}

	// batch read with a missing key reports a partial result
	resp, payload = doRequest(t, http.MethodPost, base+"/batch/read", strings.NewReader(`{"keys":["mem_a.md","missing.md"]}`), map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusMultiStatus {
		t.Fatalf("batch read partial: expected 207, got %d (%s)", resp.StatusCode, payload)
	}
	decoded := decodeJSON(t, payload)
	gotContents, _ := decoded["contents"].(map[string]any)
	if gotContents["mem_a.md"] != "first note" {
		t.Fatalf("batch read partial: expected mem_a.md content, got %v", decoded["contents"])
	}
	gotErrors, _ := decoded["errors"].(map[string]any)
	if _, exists := gotErrors["missing.md"]; !exists {
		t.Fatalf("batch read partial: expected error for missing.md, got %v", decoded["errors"])
	}
}

func TestMemoryBatchReadRejectsEmptyKeys(t *testing.T) {
	srv, _ := newTestServer(t)

	resp, payload := doRequest(t, http.MethodPost, srv.URL+"/api/connectome/memory/batch/read", strings.NewReader(`{"keys":[]}`), map[string]string{"Content-Type": "application/json"})
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d (%s)", resp.StatusCode, payload)
	}
}
