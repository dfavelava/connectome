package managers

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLocalFsManagerListObjectsIncludesPreview(t *testing.T) {
	tempHome := t.TempDir()
	t.Setenv("HOME", tempHome)

	connectomeDir := filepath.Join(tempHome, ".connectome")
	if err := os.MkdirAll(connectomeDir, 0o755); err != nil {
		t.Fatalf("create connectome dir: %v", err)
	}

	content := strings.Repeat("abc123", 200)
	filePath := filepath.Join(connectomeDir, "note.txt")
	if err := os.WriteFile(filePath, []byte(content), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	manager := NewLocalFsManager()
	result, err := manager.ListObjects()
	if err != nil {
		t.Fatalf("list objects: %v", err)
	}

	if len(result.Contents) != 1 {
		t.Fatalf("expected 1 object, got %d", len(result.Contents))
	}

	item := result.Contents[0]
	if item.Key != "note.txt" {
		t.Fatalf("expected key note.txt, got %q", item.Key)
	}

	expectedPreview := content
	if len(expectedPreview) > previewLength {
		expectedPreview = expectedPreview[:previewLength]
	}

	if item.Preview == nil {
		t.Fatal("expected preview to be present")
	}
	if *item.Preview != expectedPreview {
		t.Fatalf("unexpected preview length/content: got %d bytes", len(*item.Preview))
	}
}
