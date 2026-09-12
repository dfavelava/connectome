package managers

import (
	"fmt"
	"os"
	"strings"
)

type memoryManagerType string

const (
	memoryManagerTypeS3    memoryManagerType = "s3"
	memoryManagerTypeLocal memoryManagerType = "local"
)

// NewMemoryManagerFromEnv builds the MemoryManager configured via the
// MEMORY_MANAGER env var (defaulting to s3). Callers should build one
// instance and share it, rather than each deriving their own from env -
// InitS3Manager in particular resolves AWS credentials on construction,
// which is wasted work (and, on failure, a wasted log.Fatal) to repeat.
func NewMemoryManagerFromEnv() MemoryManager {
	managerType := memoryManagerType(strings.ToLower(os.Getenv("MEMORY_MANAGER")))
	if managerType == "" {
		managerType = memoryManagerTypeS3
	}

	switch managerType {
	case memoryManagerTypeLocal:
		return InitLocalFsManager()
	case memoryManagerTypeS3:
		return InitS3Manager()
	default:
		panic(fmt.Sprintf("unsupported MEMORY_MANAGER %q", managerType))
	}
}
