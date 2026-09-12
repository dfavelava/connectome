package resources

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"daybid-dev-service/daos"
	"daybid-dev-service/managers"
	"daybid-dev-service/middleware"
)

const (
	defaultSearchK = 5
	maxSearchK     = 50
	snippetChars   = 400
)

// SearchIndex is the subset of *daos.EmbeddingsDao the search resource needs
// to run a filtered nearest-neighbour search.
type SearchIndex interface {
	Search(ctx context.Context, query []float32, k int, filters daos.SearchFilters) ([]daos.SearchHit, error)
}

type SearchResourceImpl struct {
	manager  managers.MemoryManager
	embedder Embedder
	index    SearchIndex
}

type SearchFiltersRequest struct {
	Type   *string    `json:"type,omitempty"`
	Entity *string    `json:"entity,omitempty"`
	Since  *time.Time `json:"since,omitempty"`
	Until  *time.Time `json:"until,omitempty"`
}

type SearchRequest struct {
	Query   string                `json:"query"`
	K       int                   `json:"k,omitempty"`
	Filters *SearchFiltersRequest `json:"filters,omitempty"`
	Hydrate bool                  `json:"hydrate,omitempty"`
}

type SearchResult struct {
	Key     string  `json:"key"`
	Score   float64 `json:"score"`
	Type    string  `json:"type"`
	Snippet string  `json:"snippet,omitempty"`
	Content string  `json:"content,omitempty"`
}

func NewSearchResource(manager managers.MemoryManager, embedder Embedder, index SearchIndex) *SearchResourceImpl {
	return &SearchResourceImpl{
		manager:  manager,
		embedder: embedder,
		index:    index,
	}
}

func InitSearchResource(r *gin.RouterGroup, manager managers.MemoryManager, embedder Embedder, index SearchIndex) {
	resource := NewSearchResource(manager, embedder, index)

	group := r.Group("/memory", middleware.AuthMiddleware())
	group.POST("/search", resource.search)
}

func (resource *SearchResourceImpl) search(c *gin.Context) {
	var req SearchRequest
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if strings.TrimSpace(req.Query) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "query must not be empty"})
		return
	}

	k := req.K
	if k <= 0 {
		k = defaultSearchK
	}
	if k > maxSearchK {
		k = maxSearchK
	}

	queryEmbedding, err := resource.embedder.Embed(req.Query)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("embed query: %v", err)})
		return
	}

	var filters daos.SearchFilters
	if req.Filters != nil {
		filters = daos.SearchFilters{
			Type:   req.Filters.Type,
			Entity: req.Filters.Entity,
			Since:  req.Filters.Since,
			Until:  req.Filters.Until,
		}
	}

	hits, err := resource.index.Search(c.Request.Context(), queryEmbedding, k, filters)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("search: %v", err)})
		return
	}

	results := resource.snippetResults(hits)
	if req.Hydrate {
		results = resource.hydrateResults(hits)
	}

	c.JSON(http.StatusOK, gin.H{"results": results})
}

func (resource *SearchResourceImpl) hydrateResults(hits []daos.SearchHit) []SearchResult {
	results := make([]SearchResult, 0, len(hits))
	for _, hit := range hits {
		result := SearchResult{Key: hit.MemoryKey, Score: scoreFromDistance(hit.Distance), Type: hit.Type}
		if content, err := resource.manager.GetObject(hit.MemoryKey); err == nil {
			result.Content = content
		}
		results = append(results, result)
	}
	return results
}

func (resource *SearchResourceImpl) snippetResults(hits []daos.SearchHit) []SearchResult {
	results := make([]SearchResult, 0, len(hits))
	for _, hit := range hits {
		result := SearchResult{Key: hit.MemoryKey, Score: scoreFromDistance(hit.Distance), Type: hit.Type}
		if content, err := resource.manager.GetObject(hit.MemoryKey); err == nil {
			result.Snippet = snippetForChunk(content, hit.ChunkIndex)
		}
		results = append(results, result)
	}
	return results
}

// snippetForChunk re-derives the chunk at chunkIndex from a memory's current
// content instead of storing chunk text in the embeddings table, keeping the
// blob store the single source of truth for memory bodies (see
// backend/resources/memoryDocument.go's chunkWords, used at index time). If
// the memory has been rewritten since it was indexed, chunkIndex may no
// longer line up exactly; out-of-range indexes fall back to the first chunk
// rather than failing the whole result.
func snippetForChunk(content string, chunkIndex int) string {
	_, body, ok := parseMemoryDocument(content)
	if !ok {
		body = content
	}

	chunks := chunkWords(body, memoryChunkWords, memoryChunkOverlapWords)
	if len(chunks) == 0 {
		return ""
	}
	if chunkIndex < 0 || chunkIndex >= len(chunks) {
		chunkIndex = 0
	}

	snippet := chunks[chunkIndex]
	if len(snippet) > snippetChars {
		snippet = strings.TrimSpace(snippet[:snippetChars]) + "…"
	}
	return snippet
}

// scoreFromDistance turns cosine distance (0 = identical, larger = further)
// into a similarity score (higher = better match) for API consumers.
func scoreFromDistance(distance float64) float64 {
	return 1 - distance
}
