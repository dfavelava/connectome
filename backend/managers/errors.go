package managers

import "errors"

// ErrNotFound is returned by MemoryManager implementations when the requested
// key does not exist. Callers can detect it with errors.Is.
var ErrNotFound = errors.New("memory not found")
