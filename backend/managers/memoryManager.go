package managers

import "mime/multipart"

type MemoryListItem struct {
	Key     string  `json:"Key"`
	Preview *string `json:"Preview"`
}

type MemoryListResult struct {
	Contents []MemoryListItem `json:"Contents"`
}

type MemoryManager interface {
	GetObject(key string) (string, error)
	PutObject(key string, file multipart.File) error
	DeleteObject(key string) error
	// ListObjects lists every object whose key starts with prefix. An empty
	// prefix lists everything.
	ListObjects(prefix string) (*MemoryListResult, error)

	// DeleteObjectsWithPrefix deletes every object whose key starts with
	// prefix. Deleting a prefix with no matching objects is not an error.
	DeleteObjectsWithPrefix(prefix string) error
}
