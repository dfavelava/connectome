package managers

import (
	"errors"
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
	result, err := manager.ListObjects("")
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

func TestLocalFsManagerListObjectsFiltersByPrefix(t *testing.T) {
	tempHome := t.TempDir()
	t.Setenv("HOME", tempHome)

	connectomeDir := filepath.Join(tempHome, ".connectome")
	scopedDir := filepath.Join(connectomeDir, "tomes", "west-marches")
	if err := os.MkdirAll(scopedDir, 0o755); err != nil {
		t.Fatalf("create scoped dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(scopedDir, "mem_a.md"), []byte("scoped"), 0o644); err != nil {
		t.Fatalf("write scoped file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(connectomeDir, "mem_unscoped.md"), []byte("unscoped"), 0o644); err != nil {
		t.Fatalf("write unscoped file: %v", err)
	}

	manager := NewLocalFsManager()
	result, err := manager.ListObjects("tomes/west-marches/")
	if err != nil {
		t.Fatalf("list objects: %v", err)
	}

	if len(result.Contents) != 1 {
		t.Fatalf("expected 1 object, got %+v", result.Contents)
	}
	if result.Contents[0].Key != "tomes/west-marches/mem_a.md" {
		t.Fatalf("expected tomes/west-marches/mem_a.md, got %q", result.Contents[0].Key)
	}
}

func TestLocalFsManagerDeleteObjectsWithPrefixRemovesOnlyMatchingBlobs(t *testing.T) {
	tempHome := t.TempDir()
	t.Setenv("HOME", tempHome)

	connectomeDir := filepath.Join(tempHome, ".connectome")
	scopedDir := filepath.Join(connectomeDir, "tomes", "temp-scratch")
	if err := os.MkdirAll(scopedDir, 0o755); err != nil {
		t.Fatalf("create scoped dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(scopedDir, "mem_a.md"), []byte("scoped"), 0o644); err != nil {
		t.Fatalf("write scoped file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(connectomeDir, "mem_unscoped.md"), []byte("unscoped"), 0o644); err != nil {
		t.Fatalf("write unscoped file: %v", err)
	}

	manager := NewLocalFsManager()
	if err := manager.DeleteObjectsWithPrefix("tomes/temp-scratch/"); err != nil {
		t.Fatalf("delete objects with prefix: %v", err)
	}

	if _, err := os.Stat(filepath.Join(connectomeDir, "tomes", "temp-scratch")); !os.IsNotExist(err) {
		t.Fatalf("expected scoped directory to be gone, stat err = %v", err)
	}
	if _, err := os.Stat(filepath.Join(connectomeDir, "mem_unscoped.md")); err != nil {
		t.Fatalf("expected unscoped file to survive: %v", err)
	}
}

func TestLocalFsManagerDeleteObjectsWithPrefixIsNoopForMissingPrefix(t *testing.T) {
	tempHome := t.TempDir()
	t.Setenv("HOME", tempHome)

	connectomeDir := filepath.Join(tempHome, ".connectome")
	if err := os.MkdirAll(connectomeDir, 0o755); err != nil {
		t.Fatalf("create connectome dir: %v", err)
	}

	manager := NewLocalFsManager()
	if err := manager.DeleteObjectsWithPrefix("tomes/never-existed/"); err != nil {
		t.Fatalf("expected no error deleting a never-existed prefix, got %v", err)
	}
}

func TestLocalFsManagerGetObjectMissingReturnsErrNotFound(t *testing.T) {
	tempHome := t.TempDir()
	t.Setenv("HOME", tempHome)

	connectomeDir := filepath.Join(tempHome, ".connectome")
	if err := os.MkdirAll(connectomeDir, 0o755); err != nil {
		t.Fatalf("create connectome dir: %v", err)
	}

	manager := NewLocalFsManager()
	_, err := manager.GetObject("does-not-exist.txt")
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}
}
