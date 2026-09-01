package managers

import "mime/multipart"

type MemoryManager interface {
	GetObject(key string) (string, error)
	PutObject(key string, file multipart.File) error
}
