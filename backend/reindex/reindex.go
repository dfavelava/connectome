// Package reindex rebuilds the embeddings table from the memory blob store,
// so the blob store (not the table) is the source of truth for search: drop
// embeddings, run Run, and search is fully reconstructed from mem_*.md
// content alone.
package reindex

import (
	"context"
	"fmt"
	"log"

	"daybid-dev-service/managers"
	"daybid-dev-service/resources"
)

// EmbeddingsStore is the subset of *daos.EmbeddingsDao that Run needs: the
// same indexing operations resources.MemoryResourceImpl uses, plus the
// ability to clear the table before rebuilding it.
type EmbeddingsStore interface {
	resources.EmbeddingsIndexer
	TruncateEmbeddings(ctx context.Context) error
}

// Result summarizes one Run over the memory blob store.
type Result struct {
	Total   int
	Indexed int
	Skipped int
	Failed  int
}

func (r Result) String() string {
	return fmt.Sprintf("%d memories: %d indexed, %d skipped (no frontmatter), %d failed", r.Total, r.Indexed, r.Skipped, r.Failed)
}

// Run walks every object in manager.ListObjects and, for each one that
// parses as a memory document (resources.ParseMemoryDocument), re-chunks and
// re-embeds it via resources.MemoryResourceImpl.IndexMemory, superseding any
// existing rows for that key. Content with no valid memory frontmatter
// (e.g. an ent_*.json entity record) is counted as skipped, not failed.
//
// Unless dryRun is set, the embeddings table is truncated first, so a
// completed Run leaves the table containing exactly the rows derivable from
// the current blob store - safe to run repeatedly, and safe to run against a
// table that already has rows in it. With dryRun, the table is left
// untouched and the embedder is never called; Result reports what a real run
// would do.
func Run(ctx context.Context, manager managers.MemoryManager, embedder resources.Embedder, store EmbeddingsStore, dryRun bool) (Result, error) {
	list, err := manager.ListObjects()
	if err != nil {
		return Result{}, fmt.Errorf("list objects: %w", err)
	}

	result := Result{Total: len(list.Contents)}

	if !dryRun {
		if err := store.TruncateEmbeddings(ctx); err != nil {
			return Result{}, fmt.Errorf("truncate embeddings: %w", err)
		}
	}

	memoryResource := resources.NewMemoryResource(manager, embedder, store)

	for _, item := range list.Contents {
		content, err := manager.GetObject(item.Key)
		if err != nil {
			log.Printf("reindex: read %s: %v", item.Key, err)
			result.Failed++
			continue
		}

		if _, _, ok := resources.ParseMemoryDocument(content); !ok {
			result.Skipped++
			continue
		}

		if dryRun {
			result.Indexed++
			continue
		}

		if err := memoryResource.IndexMemory(ctx, item.Key, content); err != nil {
			log.Printf("reindex: index %s: %v", item.Key, err)
			result.Failed++
			continue
		}
		result.Indexed++
	}

	return result, nil
}
