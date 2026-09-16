package resources

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/gin-gonic/gin"

	"daybid-dev-service/managers"
)

// newEntityTestServer wires just the /entity routes onto a fresh gin engine
// backed by the local filesystem manager rooted at a per-test temp
// directory, mirroring newTestServer in memoryResource_test.go.
func newEntityTestServer(t *testing.T) *httptest.Server {
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
	InitEntityResource(r.Group("/api/connectome"), managers.NewMemoryManagerFromEnv())

	srv := httptest.NewServer(r)
	t.Cleanup(srv.Close)
	return srv
}

func postRelationship(t *testing.T, srv *httptest.Server, body map[string]any) (*http.Response, EntityRelationshipResponse) {
	t.Helper()

	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	resp, raw := doRequest(t, http.MethodPost, srv.URL+"/api/connectome/entity/relationship", bytes.NewReader(payload), map[string]string{
		"Content-Type": "application/json",
	})

	var decoded EntityRelationshipResponse
	if resp.StatusCode == http.StatusOK {
		if err := json.Unmarshal(raw, &decoded); err != nil {
			t.Fatalf("decode response %q: %v", string(raw), err)
		}
	}
	return resp, decoded
}

func TestEntityRelationshipRouteRequiresBearerToken(t *testing.T) {
	srv := newEntityTestServer(t)

	req, err := http.NewRequest(http.MethodPost, srv.URL+"/api/connectome/entity/relationship", bytes.NewReader([]byte(`{}`)))
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

func TestEntityRelationshipRouteUpsertsMemberOf(t *testing.T) {
	srv := newEntityTestServer(t)

	resp, decoded := postRelationship(t, srv, map[string]any{
		"subjectEntityId": "alice",
		"predicate":       "member_of",
		"objectEntityId":  "Party A",
	})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if decoded.Subject.ID != "alice" || len(decoded.Subject.MemberOf) != 1 || decoded.Subject.MemberOf[0] != "Party A" {
		t.Fatalf("expected alice member_of [Party A], got %+v", decoded.Subject)
	}
	if decoded.Object == nil || decoded.Object.ID != "Party A" {
		t.Fatalf("expected Party A stub in response, got %+v", decoded.Object)
	}
}

func TestEntityRelationshipRouteSetsSubjectKindAndMeta(t *testing.T) {
	srv := newEntityTestServer(t)

	resp, decoded := postRelationship(t, srv, map[string]any{
		"subjectEntityId": "thorin",
		"predicate":       "member_of",
		"objectEntityId":  "discord-1-characters",
		"subjectKind":     "character",
		"subjectMeta":     map[string]any{"owner": "discord-1"},
	})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if decoded.Subject.Kind == nil || *decoded.Subject.Kind != "character" {
		t.Fatalf("expected subject kind character, got %+v", decoded.Subject)
	}
	if decoded.Subject.Meta["owner"] != "discord-1" {
		t.Fatalf("expected subject meta.owner discord-1, got %+v", decoded.Subject.Meta)
	}
}

func TestEntityRelationshipRouteDefaultsAndValidatesKind(t *testing.T) {
	srv := newEntityTestServer(t)

	resp, _ := postRelationship(t, srv, map[string]any{
		"subjectEntityId": "alice",
		"predicate":       "likes",
		"objectEntityId":  "tea",
		"kind":            "not-a-real-kind",
	})
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 for invalid kind, got %d", resp.StatusCode)
	}
}

func TestEntityRelationshipRouteRequiresSubjectAndPredicate(t *testing.T) {
	srv := newEntityTestServer(t)

	resp, _ := postRelationship(t, srv, map[string]any{
		"predicate":      "likes",
		"objectEntityId": "tea",
	})
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 for missing subjectEntityId, got %d", resp.StatusCode)
	}
}
