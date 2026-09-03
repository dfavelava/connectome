package managers

import "mime/multipart"

type MemoryListItem struct {
	Key string `json:"Key"`
}

type MemoryListResult struct {
	Contents []MemoryListItem `json:"Contents"`
}

type MemoryManager interface {
	GetObject(key string) (string, error)
	PutObject(key string, file multipart.File) error
	DeleteObject(key string) error
	ListObjects() (*MemoryListResult, error)
}
