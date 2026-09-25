package resources

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"

	"connectome-dev-service/middleware"

	"github.com/gin-gonic/gin"

	"connectome-dev-service/daos"
	"connectome-dev-service/managers"
)

// memoryChunkWords and memoryChunkOverlapWords size the chunks that get
// embedded per memory: ~512 tokens with a small overlap so a chunk boundary
// doesn't sever context needed to make sense of either side of it.
const (
	memoryChunkWords        = 512
	memoryChunkOverlapWords = 50
)

// defaultIndexConcurrency caps how many memories are embedded at once when
// INDEX_CONCURRENCY is unset or invalid.
const defaultIndexConcurrency = 4

// indexConcurrencyFromEnv reads INDEX_CONCURRENCY, the most memories a
// MemoryResourceImpl embeds at once (and the most files one batch write
// processes at once), falling back to defaultIndexConcurrency when it is
// unset or not a positive integer.
func indexConcurrencyFromEnv() int {
	n, err := strconv.Atoi(strings.TrimSpace(os.Getenv("INDEX_CONCURRENCY")))
	if err != nil || n <= 0 {
		return defaultIndexConcurrency
	}
	return n
}

// Embedder produces vector embeddings for text. Stored chunks and search
// queries are embedded differently (see managers.DocumentPrefix), so callers
// must pick the side they're on. EmbedDocuments embeds all of a memory's
// chunks in one call, returning one embedding per input in order. Both stop
// when ctx is cancelled. Satisfied by *managers.OllamaManager.
type Embedder interface {
	EmbedDocuments(ctx context.Context, inputs []string) ([][]float32, error)
	EmbedQuery(ctx context.Context, input string) ([]float32, error)
}

// EmbeddingsIndexer is the subset of *daos.EmbeddingsDao that memoryResource
// needs to keep the embeddings table in sync with the memory blob store.
type EmbeddingsIndexer interface {
	InsertEmbeddings(ctx context.Context, memoryKey string, rows []daos.EmbeddingRow) error
	DeleteEmbeddingsForKey(ctx context.Context, memoryKey string) error
}

type MemoryResourceImpl struct {
	manager    managers.MemoryManager
	embedder   Embedder
	embeddings EmbeddingsIndexer

	// indexConcurrency bounds how many files one batchWrite processes at
	// once; embedSlots (sized the same) bounds how many IndexMemory embed
	// calls are in flight across every request this resource serves.
	indexConcurrency int
	embedSlots       chan struct{}
}

type BatchReadError struct {
	Key   string
	Error string
}

type BatchReadRequest struct {
	Keys []string `json:"keys"`
	Tome string   `json:"tome"`
}

type DeleteMemoryRequest struct {
	Key  string `json:"key"`
	Tome string `json:"tome"`
}

// SupersedeRelationshipRequest identifies one relationship entry on an
// existing memory (by subject/predicate/object) and the superseded_by value
// to set on it. ObjectEntityID and SupersededBy are pointers so a JSON null
// or an omitted key both decode to nil - "no object" and "clear the prior
// supersession" respectively.
type SupersedeRelationshipRequest struct {
	Key             string  `json:"key"`
	Tome            string  `json:"tome"`
	SubjectEntityID string  `json:"subjectEntityId"`
	Predicate       string  `json:"predicate"`
	ObjectEntityID  *string `json:"objectEntityId"`
	SupersededBy    *string `json:"superseded_by"`
}

// memoryFileReader adapts an in-memory byte slice to multipart.File so a
// patched document can be written back through the same
// MemoryManager.PutObject used by uploads, without a temp file.
type memoryFileReader struct {
	*bytes.Reader
}

func (memoryFileReader) Close() error { return nil }

func newMemoryFile(content []byte) multipart.File {
	return memoryFileReader{bytes.NewReader(content)}
}

func NewMemoryResource(manager managers.MemoryManager, embedder Embedder, embeddings EmbeddingsIndexer) *MemoryResourceImpl {
	concurrency := indexConcurrencyFromEnv()
	return &MemoryResourceImpl{
		manager:          manager,
		embedder:         embedder,
		embeddings:       embeddings,
		indexConcurrency: concurrency,
		embedSlots:       make(chan struct{}, concurrency),
	}
}

func InitMemoryResource(r *gin.RouterGroup, manager managers.MemoryManager, embedder Embedder, embeddings EmbeddingsIndexer) {
	resource := NewMemoryResource(manager, embedder, embeddings)

	group := r.Group("/memory")
	group.Use(middleware.AuthMiddleware())
	group.POST("/", resource.write)
	group.POST("/batch", resource.batchWrite)
	group.POST("/batch/read", resource.batchRead)
	group.GET("/", resource.read)
	group.DELETE("/", resource.delete)
	group.GET("/list", resource.list)
	group.PATCH("/relationship", resource.supersedeRelationship)
}

func (resource *MemoryResourceImpl) read(c *gin.Context) {
	key := c.Query("key")

	if key == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "memory key not specified"})
		return
	}

	content, err := resource.manager.GetObject(TomeScopedKey(c.Query("tome"), key))
	if err != nil {
		if errors.Is(err, managers.ErrNotFound) {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("memory %q not found", key)})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, gin.H{"content": HydrateMemoryDocument(content)})
}

func (resource *MemoryResourceImpl) batchRead(c *gin.Context) {
	var request BatchReadRequest
	if err := c.BindJSON(&request); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	if len(request.Keys) == 0 {
		c.JSON(400, gin.H{"error": "no keys provided"})
		return
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	contents := make(map[string]string, len(request.Keys))
	errCh := make(chan BatchReadError, len(request.Keys))

	for _, key := range request.Keys {
		wg.Add(1)
		go func(key string) {
			defer wg.Done()

			content, err := resource.manager.GetObject(TomeScopedKey(request.Tome, key))
			if err != nil {
				errCh <- BatchReadError{Key: key, Error: fmt.Sprintf("read %s: %v", key, err)}
				return
			}

			mu.Lock()
			contents[key] = content
			mu.Unlock()
		}(key)
	}

	wg.Wait()
	close(errCh)

	errors := make(map[string]string)
	for batchErr := range errCh {
		if batchErr.Error != "" {
			errors[batchErr.Key] = batchErr.Error
		}
	}

	if len(errors) > 0 {
		c.JSON(207, gin.H{
			"contents": contents,
			"errors":   errors,
		})
		return
	}

	c.JSON(200, gin.H{"contents": contents})
}

// IndexMemory keeps the embeddings table in sync with one written memory
// key: it parses the frontmatter written by connectomemcp's format_memory, chunks
// the body, embeds each chunk, and supersedes any prior rows for the key.
// Content with no valid memory frontmatter (e.g. an ent_*.json entity
// record) is left unindexed. It is also the entry point cmd/reindex uses to
// rebuild the embeddings table from the blob store.
func (resource *MemoryResourceImpl) IndexMemory(ctx context.Context, key, content, tome string) error {
	fm, body, ok := ParseMemoryDocument(content)
	if !ok {
		return nil
	}

	occurredAt, err := fm.occurredAt()
	if err != nil {
		return fmt.Errorf("index %s: %w", key, err)
	}

	chunks := ChunkWords(body, memoryChunkWords, memoryChunkOverlapWords)

	embeddings, err := resource.embedDocuments(ctx, chunks)
	if err != nil {
		return fmt.Errorf("embed %s: %w", key, err)
	}
	if len(embeddings) != len(chunks) {
		return fmt.Errorf("embed %s: expected %d embeddings, got %d", key, len(chunks), len(embeddings))
	}

	rows := make([]daos.EmbeddingRow, len(chunks))
	createdAt := fm.createdAtOrNow()
	acl := ResolveACL(fm.ACL)
	for i, chunk := range chunks {
		embedding := embeddings[i]
		rows[i] = daos.EmbeddingRow{
			ChunkIndex: i,
			Embedding:  embedding,
			ChunkText:  chunk,
			Model:      managers.EMBEDDING_MODEL,
			Dim:        len(embedding),
			Type:       fm.Type,
			EntityIDs:  fm.Entities,
			ACL:        acl,
			TomeID:     tome,
			CreatedAt:  createdAt,
			OccurredAt: occurredAt,
		}
	}

	if err := resource.embeddings.DeleteEmbeddingsForKey(ctx, key); err != nil {
		return fmt.Errorf("supersede embeddings for %s: %w", key, err)
	}
	return resource.embeddings.InsertEmbeddings(ctx, key, rows)
}

// embedDocuments embeds chunks once one of the resource's embed slots is
// free, so no more than indexConcurrency embed calls are ever in flight. It
// gives up without embedding if ctx is cancelled while waiting.
func (resource *MemoryResourceImpl) embedDocuments(ctx context.Context, chunks []string) ([][]float32, error) {
	if len(chunks) == 0 {
		return nil, nil
	}
	select {
	case resource.embedSlots <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	defer func() { <-resource.embedSlots }()
	return resource.embedder.EmbedDocuments(ctx, chunks)
}

func (resource *MemoryResourceImpl) write(c *gin.Context) {
	fileHeader, err := c.FormFile("file")
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	file, err := fileHeader.Open()
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	tome := c.PostForm("tome")
	scopedKey := TomeScopedKey(tome, fileHeader.Filename)

	if err := resource.manager.PutObject(scopedKey, file); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	if err := resource.IndexMemory(c.Request.Context(), scopedKey, string(content), tome); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, gin.H{"message": "success"})
}

func (resource *MemoryResourceImpl) batchWrite(c *gin.Context) {
	form, err := c.MultipartForm()
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	fileHeaders, ok := form.File["file"]
	if !ok || len(fileHeaders) == 0 {
		c.JSON(400, gin.H{"error": "no files provided"})
		return
	}

	tome := c.PostForm("tome")
	ctx := c.Request.Context()

	var wg sync.WaitGroup
	errCh := make(chan error, len(fileHeaders))
	workers := make(chan struct{}, resource.indexConcurrency)

	for _, fileHeader := range fileHeaders {
		wg.Add(1)
		go func(fileHeader *multipart.FileHeader) {
			defer wg.Done()

			select {
			case workers <- struct{}{}:
			case <-ctx.Done():
				errCh <- fmt.Errorf("%s: %w", fileHeader.Filename, ctx.Err())
				return
			}
			defer func() { <-workers }()

			file, err := fileHeader.Open()
			if err != nil {
				errCh <- fmt.Errorf("open %s: %w", fileHeader.Filename, err)
				return
			}
			defer file.Close()

			content, err := io.ReadAll(file)
			if err != nil {
				errCh <- fmt.Errorf("read %s: %w", fileHeader.Filename, err)
				return
			}
			if _, err := file.Seek(0, io.SeekStart); err != nil {
				errCh <- fmt.Errorf("seek %s: %w", fileHeader.Filename, err)
				return
			}

			scopedKey := TomeScopedKey(tome, fileHeader.Filename)

			if err := resource.manager.PutObject(scopedKey, file); err != nil {
				errCh <- fmt.Errorf("upload %s: %w", fileHeader.Filename, err)
				return
			}

			if err := resource.IndexMemory(ctx, scopedKey, string(content), tome); err != nil {
				errCh <- fmt.Errorf("index %s: %w", fileHeader.Filename, err)
			}
		}(fileHeader)
	}

	wg.Wait()
	close(errCh)

	var errors []string
	for err := range errCh {
		if err != nil {
			errors = append(errors, err.Error())
		}
	}

	if len(errors) > 0 {
		c.JSON(500, gin.H{"error": fmt.Sprintf("batch upload failed: %s", strings.Join(errors, "; "))})
		return
	}

	c.JSON(200, gin.H{"message": "success"})
}

func (resource *MemoryResourceImpl) list(c *gin.Context) {
	res, err := ListTome(resource.manager, c.Query("tome"))

	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, res)
}

// supersedeRelationship patches superseded_by on one relationship entry of an
// existing memory's frontmatter and writes the result straight back to the
// blob store. It deliberately skips IndexMemory: superseded_by is metadata
// that lives in the frontmatter, not the embedded content, so there is
// nothing here for the embedder to re-run.
func (resource *MemoryResourceImpl) supersedeRelationship(c *gin.Context) {
	var req SupersedeRelationshipRequest
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if req.Key == "" || req.SubjectEntityID == "" || req.Predicate == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "key, subjectEntityId, and predicate are required"})
		return
	}

	scopedKey := TomeScopedKey(req.Tome, req.Key)

	content, err := resource.manager.GetObject(scopedKey)
	if err != nil {
		if errors.Is(err, managers.ErrNotFound) {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("memory %q not found", req.Key)})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	patched, err := PatchRelationshipSupersededBy(content, req.SubjectEntityID, req.Predicate, req.ObjectEntityID, req.SupersededBy)
	if err != nil {
		if errors.Is(err, ErrRelationshipNotFound) {
			c.JSON(http.StatusNotFound, gin.H{"error": "no matching relationship found on this memory"})
			return
		}
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := resource.manager.PutObject(scopedKey, newMemoryFile([]byte(patched))); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "success", "key": req.Key})
}

func (resource *MemoryResourceImpl) delete(c *gin.Context) {
	var body DeleteMemoryRequest
	if err := c.BindJSON(&body); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	scopedKey := TomeScopedKey(body.Tome, body.Key)

	if err := resource.manager.DeleteObject(scopedKey); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	if err := resource.embeddings.DeleteEmbeddingsForKey(c.Request.Context(), scopedKey); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(204, gin.H{"message": "success"})
}
