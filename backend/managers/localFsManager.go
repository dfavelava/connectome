package managers

import (
	"io"
	"io/fs"
	"mime/multipart"
	"os"
	"path/filepath"
)

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

func (l *LocalFsManagerImpl) GetObject(path string) (string, error) {
	data, err := fs.ReadFile(l.fsys, path)
	if err != nil {
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

func (l *LocalFsManagerImpl) ListObjects() (*MemoryListResult, error) {
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

		contents = append(contents, MemoryListItem{Key: filepath.ToSlash(relativePath)})
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
