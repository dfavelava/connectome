package resources

import (
	"bytes"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/gin-gonic/gin"

	"connectome-dev-service/managers"
)

// multipartBodyWithTome is multipartBody plus a "tome" form field, since
// memoryResource's write/batchWrite handlers read the tome to scope a write
// to via c.PostForm/c.MultipartForm, not a query parameter.
func multipartBodyWithTome(t *testing.T, tome string, parts []filePart) (*bytes.Buffer, string) {
	t.Helper()

	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	if err := w.WriteField("tome", tome); err != nil {
		t.Fatalf("write tome field: %v", err)
	}
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

// newTomeTestServer wires the tome and memory routes onto a fresh gin engine
// backed by the local filesystem manager rooted at a per-test temp
// directory, mirroring newTestServer in memoryResource_test.go. The memory
// routes are included so tests can write a tome-scoped blob before
// destroying it.
func newTomeTestServer(t *testing.T) (*httptest.Server, string, *fakeIndexer) {
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
	manager := managers.NewMemoryManagerFromEnv()
	embeddings := newFakeIndexer()
	group := r.Group("/api/connectome")
	InitTomeResource(group, manager, embeddings)
	InitMemoryResource(group, manager, fakeEmbedder{}, embeddings)

	srv := httptest.NewServer(r)
	t.Cleanup(srv.Close)

	return srv, connectomeDir, embeddings
}

func TestDestroyTomeRouteRemovesBlobsAndEmbeddingsForTempTome(t *testing.T) {
	srv, connectomeDir, embeddings := newTomeTestServer(t)

	const tome = "temp-scratch"
	const key = "mem_a.md"

	body, contentType := multipartBodyWithTome(t, tome, []filePart{{name: key, content: memoryDocument("note", "scratch note", nil)}})
	resp, payload := doRequest(t, http.MethodPost, srv.URL+"/api/connectome/memory/", body, map[string]string{"Content-Type": contentType})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("write: expected 200, got %d (%s)", resp.StatusCode, payload)
	}

	scopedPath := filepath.Join(connectomeDir, "tomes", tome, key)
	if _, err := os.Stat(scopedPath); err != nil {
		t.Fatalf("expected scoped blob to exist before destroy: %v", err)
	}
	scopedKey := "tomes/" + tome + "/" + key
	if rows := embeddings.rowsFor(scopedKey); len(rows) == 0 {
		t.Fatalf("expected embeddings rows to exist before destroy")
	}

	resp, payload = doRequest(t, http.MethodDelete, srv.URL+"/api/connectome/tome/"+tome, nil, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("destroy: expected 200, got %d (%s)", resp.StatusCode, payload)
	}

	if _, err := os.Stat(filepath.Join(connectomeDir, "tomes", tome)); !os.IsNotExist(err) {
		t.Fatalf("expected tomes/%s directory to be gone, stat err = %v", tome, err)
	}
	if rows := embeddings.rowsFor(scopedKey); len(rows) != 0 {
		t.Fatalf("expected embeddings rows to be gone after destroy, got %+v", rows)
	}
}

func TestDestroyTomeRouteRefusesNonConventionTomeWithoutConfirmThenAllowsWithIt(t *testing.T) {
	srv, connectomeDir, _ := newTomeTestServer(t)

	const tome = "west-marches"
	const key = "mem_a.md"

	body, contentType := multipartBodyWithTome(t, tome, []filePart{{name: key, content: memoryDocument("note", "campaign note", nil)}})
	resp, payload := doRequest(t, http.MethodPost, srv.URL+"/api/connectome/memory/", body, map[string]string{"Content-Type": contentType})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("write: expected 200, got %d (%s)", resp.StatusCode, payload)
	}

	resp, payload = doRequest(t, http.MethodDelete, srv.URL+"/api/connectome/tome/"+tome, nil, nil)
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("destroy without confirm: expected 403, got %d (%s)", resp.StatusCode, payload)
	}

	scopedPath := filepath.Join(connectomeDir, "tomes", tome, key)
	if _, err := os.Stat(scopedPath); err != nil {
		t.Fatalf("expected blob to survive a refused destroy: %v", err)
	}

	resp, payload = doRequest(t, http.MethodDelete, srv.URL+"/api/connectome/tome/"+tome+"?confirm=true", nil, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("destroy with confirm: expected 200, got %d (%s)", resp.StatusCode, payload)
	}
	if _, err := os.Stat(filepath.Join(connectomeDir, "tomes", tome)); !os.IsNotExist(err) {
		t.Fatalf("expected tomes/%s directory to be gone after confirm, stat err = %v", tome, err)
	}
}

func TestDestroyTomeRouteRequiresBearerToken(t *testing.T) {
	srv, _, _ := newTomeTestServer(t)

	req, err := http.NewRequest(http.MethodDelete, srv.URL+"/api/connectome/tome/temp-anything", nil)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("do request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", resp.StatusCode)
	}
}
