package managers

import (
	"errors"
	"io"
	"io/fs"
	"mime/multipart"
	"os"
	"path/filepath"
	"strings"
)

const previewLength = 1024

type LocalFsManagerImpl struct {
	basePath string
	fsys     fs.FS
}

func NewLocalFsManager() *LocalFsManagerImpl {
	basePath := os.ExpandEnv("$HOME/.connectome")
	fsys := os.DirFS(basePath)

	return &LocalFsManagerImpl{basePath: basePath, fsys: fsys}
}

func InitLocalFsManager() *LocalFsManagerImpl {
	return NewLocalFsManager()
}

func (l *LocalFsManagerImpl) GetPreview(path string) (string, error) {
	file, err := l.fsys.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()

	preview, err := io.ReadAll(io.LimitReader(file, previewLength))
	if err != nil {
		return "", err
	}

	return string(preview), nil
}

func (l *LocalFsManagerImpl) GetObject(path string) (string, error) {
	data, err := fs.ReadFile(l.fsys, path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return "", ErrNotFound
		}
		return "", err
	}
	return string(data), nil
}

func (l *LocalFsManagerImpl) PutObject(path string, file multipart.File) error {
	fullPath := filepath.Join(l.basePath, path)
	if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
		return err
	}

	out, err := os.Create(fullPath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, file)
	if err != nil {
		return err
	}
	return nil
}

func (l *LocalFsManagerImpl) DeleteObject(path string) error {
	fullPath := filepath.Join(l.basePath, path)
	return os.Remove(fullPath)
}

// DeleteObjectsWithPrefix removes every file under prefix. A tome's blobs
// all live under one tomes/<id>/ directory, so this is just an
// os.RemoveAll of that directory - a missing directory is not an error.
func (l *LocalFsManagerImpl) DeleteObjectsWithPrefix(prefix string) error {
	return os.RemoveAll(filepath.Join(l.basePath, prefix))
}

// ListObjects lists every object whose key starts with prefix, mirroring
// S3's raw string-prefix semantics: it walks the whole tree and filters each
// entry's key, rather than assuming prefix aligns to a directory boundary.
func (l *LocalFsManagerImpl) ListObjects(prefix string) (*MemoryListResult, error) {
	contents := []MemoryListItem{}

	err := filepath.WalkDir(l.basePath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}

		relativePath, err := filepath.Rel(l.basePath, path)
		if err != nil {
			return err
		}

		key := filepath.ToSlash(relativePath)
		if !strings.HasPrefix(key, prefix) {
			return nil
		}

		preview, err := l.GetPreview(key)
		if err != nil {
			return err
		}

		contents = append(contents, MemoryListItem{Key: key, Preview: &preview})
		return nil
	})
	if err != nil {
		if os.IsNotExist(err) {
			return &MemoryListResult{Contents: contents}, nil
		}
		return nil, err
	}

	return &MemoryListResult{Contents: contents}, nil
}
