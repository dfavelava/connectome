package resources

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"connectome-dev-service/daos"
	"connectome-dev-service/managers"
	"connectome-dev-service/middleware"
)

const (
	defaultSearchK = 5
	maxSearchK     = 50
	snippetChars   = 400
)

// SearchIndex is the subset of *daos.EmbeddingsDao the search resource needs
// to run a filtered hybrid (vector + full-text) search.
type SearchIndex interface {
	Search(ctx context.Context, queryText string, queryEmbedding []float32, k int, filters daos.SearchFilters) ([]daos.SearchHit, error)
}

type SearchResourceImpl struct {
	manager  managers.MemoryManager
	embedder Embedder
	index    SearchIndex
}

type SearchFiltersRequest struct {
	Type   *string `json:"type,omitempty"`
	Entity *string `json:"entity,omitempty"`
	// Since and Until bound created_at - when the memory was written.
	Since *time.Time `json:"since,omitempty"`
	Until *time.Time `json:"until,omitempty"`
	// OccurredSince and OccurredUntil bound occurred_at - when the described
	// event happened. Memories with no occurred_at are excluded whenever
	// either is set.
	OccurredSince *time.Time `json:"occurred_since,omitempty"`
	OccurredUntil *time.Time `json:"occurred_until,omitempty"`
	// Tome restricts results to this tome id. Omitted/nil searches
	// DefaultTome, matching search behavior from before tomes existed.
	Tome *string `json:"tome,omitempty"`
}

type SearchRequest struct {
	Query   string                `json:"query"`
	K       int                   `json:"k,omitempty"`
	Filters *SearchFiltersRequest `json:"filters,omitempty"`
	Hydrate bool                  `json:"hydrate,omitempty"`
	// As is an entity id to scope results to: only memories whose acl is
	// empty (unrestricted) or overlaps that entity's own id or one level of
	// its member_of groups are returned. Omitted/nil applies no acl
	// filtering. See SearchResourceImpl.resolveACLScope.
	As *string `json:"as,omitempty"`
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

	queryEmbedding, err := resource.embedder.EmbedQuery(req.Query)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("embed query: %v", err)})
		return
	}

	filters := daos.SearchFilters{TomeID: DefaultTome}
	if req.Filters != nil {
		filters.Type = req.Filters.Type
		filters.Entity = req.Filters.Entity
		filters.Since = req.Filters.Since
		filters.Until = req.Filters.Until
		filters.OccurredSince = req.Filters.OccurredSince
		filters.OccurredUntil = req.Filters.OccurredUntil
		if req.Filters.Tome != nil {
			filters.TomeID = *req.Filters.Tome
		}
	}
	if req.As != nil {
		filters.ACLScope = resource.resolveACLScope(*req.As)
	}

	hits, err := resource.index.Search(c.Request.Context(), req.Query, queryEmbedding, k, filters)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("search: %v", err)})
		return
	}

	results := resource.snippetResults(hits)
	if req.Hydrate {
		results = resource.hydrateResults(hits)
	}
	// hit.MemoryKey is the tome-scoped blob key; return the bare key so a
	// result can be passed straight back to read/delete with the same tome.
	for i := range results {
		results[i].Key = TomeUnscopedKey(filters.TomeID, results[i].Key)
	}

	c.JSON(http.StatusOK, gin.H{"results": results})
}

func (resource *SearchResourceImpl) hydrateResults(hits []daos.SearchHit) []SearchResult {
	results := make([]SearchResult, 0, len(hits))
	for _, hit := range hits {
		result := SearchResult{Key: hit.MemoryKey, Score: hit.Score, Type: hit.Type}
		if content, err := resource.manager.GetObject(hit.MemoryKey); err == nil {
			result.Content = HydrateMemoryDocument(content)
		}
		results = append(results, result)
	}
	return results
}

func (resource *SearchResourceImpl) snippetResults(hits []daos.SearchHit) []SearchResult {
	results := make([]SearchResult, 0, len(hits))
	for _, hit := range hits {
		result := SearchResult{Key: hit.MemoryKey, Score: hit.Score, Type: hit.Type}
		if content, err := resource.manager.GetObject(hit.MemoryKey); err == nil {
			result.Snippet = snippetForChunk(content, hit.ChunkIndex)
		}
		results = append(results, result)
	}
	return results
}

// snippetForChunk re-derives the chunk at chunkIndex from a memory's current
// content rather than reading the embeddings table's chunk_text column
// (there only to feed full-text search - see backend/daos/embeddingsDao.go),
// keeping the blob store the single source of truth for memory bodies (see
// backend/resources/memoryDocument.go's ChunkWords, used at index time). If
// the memory has been rewritten since it was indexed, chunkIndex may no
// longer line up exactly; out-of-range indexes fall back to the first chunk
// rather than failing the whole result.
func snippetForChunk(content string, chunkIndex int) string {
	_, body, ok := ParseMemoryDocument(content)
	if !ok {
		body = content
	}

	chunks := ChunkWords(body, memoryChunkWords, memoryChunkOverlapWords)
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
