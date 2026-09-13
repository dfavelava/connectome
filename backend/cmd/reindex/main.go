// Command reindex rebuilds the embeddings table from the memory blob store,
// proving the blob store is the source of truth for search: drop the
// embeddings table and `go run ./cmd/reindex` fully reconstructs it, no
// other input needed.
//
// By default it truncates the embeddings table and rebuilds it from
// scratch. Pass -dry-run to report what would happen without touching the
// table or calling the embedder.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"

	"daybid-dev-service/daos"
	"daybid-dev-service/managers"
	"daybid-dev-service/reindex"

	"github.com/joho/godotenv"
)

func main() {
	dryRun := flag.Bool("dry-run", false, "report counts without truncating or rewriting the embeddings table")
	flag.Parse()

	if err := godotenv.Load(); err != nil {
		log.Printf("Warning: .env file not loaded: %v", err)
	}

	postgresManager := managers.NewPostgresManager()
	embeddingsDao := daos.NewEmbeddingsDao(postgresManager.Pool)
	ollamaManager := managers.NewOllamaManager()
	memoryManager := managers.NewMemoryManagerFromEnv()

	result, err := reindex.Run(context.Background(), memoryManager, ollamaManager, embeddingsDao, *dryRun)
	if err != nil {
		log.Fatalf("reindex: %v", err)
	}

	if *dryRun {
		fmt.Printf("reindex (dry run): %s\n", result)
	} else {
		fmt.Printf("reindex: %s\n", result)
	}

	if result.Failed > 0 {
		log.Fatalf("reindex: %d memories failed", result.Failed)
	}
}
