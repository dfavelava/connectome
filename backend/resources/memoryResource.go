package resources

import (
	"context"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"sync"

	"daybid-dev-service/middleware"

	"github.com/gin-gonic/gin"

	"daybid-dev-service/daos"
	"daybid-dev-service/managers"
)

// memoryChunkWords and memoryChunkOverlapWords size the chunks that get
// embedded per memory: ~512 tokens with a small overlap so a chunk boundary
// doesn't sever context needed to make sense of either side of it.
const (
	memoryChunkWords        = 512
	memoryChunkOverlapWords = 50
)

// Embedder produces a vector embedding for a chunk of text. Satisfied by
// *managers.OllamaManager.
type Embedder interface {
	Embed(input string) ([]float32, error)
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
}

type BatchReadError struct {
	Key   string
	Error string
}

type BatchReadRequest struct {
	Keys []string `json:"keys"`
}

type DeleteMemoryRequest struct {
	Key string `json:"key"`
}

func NewMemoryResource(manager managers.MemoryManager, embedder Embedder, embeddings EmbeddingsIndexer) *MemoryResourceImpl {
	return &MemoryResourceImpl{
		manager:    manager,
		embedder:   embedder,
		embeddings: embeddings,
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
}

func (resource *MemoryResourceImpl) read(c *gin.Context) {
	key := c.Query("key")

	if key == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "memory key not specified"})
		return
	}

	content, err := resource.manager.GetObject(key)
	if err != nil {
		if errors.Is(err, managers.ErrNotFound) {
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("memory %q not found", key)})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, gin.H{"content": content})
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

			content, err := resource.manager.GetObject(key)
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

// indexMemory keeps the embeddings table in sync with one written memory
// key: it parses the frontmatter written by daybidmcp's format_memory, chunks
// the body, embeds each chunk, and supersedes any prior rows for the key.
// Content with no valid memory frontmatter (e.g. an ent_*.json entity
// record) is left unindexed.
func (resource *MemoryResourceImpl) indexMemory(ctx context.Context, key, content string) error {
	fm, body, ok := parseMemoryDocument(content)
	if !ok {
		return nil
	}

	chunks := chunkWords(body, memoryChunkWords, memoryChunkOverlapWords)

	rows := make([]daos.EmbeddingRow, len(chunks))
	createdAt := fm.createdAtOrNow()
	for i, chunk := range chunks {
		embedding, err := resource.embedder.Embed(chunk)
		if err != nil {
			return fmt.Errorf("embed chunk %d of %s: %w", i, key, err)
		}
		rows[i] = daos.EmbeddingRow{
			ChunkIndex: i,
			Embedding:  embedding,
			Model:      managers.EMBEDDING_MODEL,
			Dim:        len(embedding),
			Type:       fm.Type,
			EntityIDs:  fm.Entities,
			CreatedAt:  createdAt,
		}
	}

	if err := resource.embeddings.DeleteEmbeddingsForKey(ctx, key); err != nil {
		return fmt.Errorf("supersede embeddings for %s: %w", key, err)
	}
	return resource.embeddings.InsertEmbeddings(ctx, key, rows)
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

	if err := resource.manager.PutObject(fileHeader.Filename, file); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	if err := resource.indexMemory(c.Request.Context(), fileHeader.Filename, string(content)); err != nil {
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

	var wg sync.WaitGroup
	errCh := make(chan error, len(fileHeaders))

	for _, fileHeader := range fileHeaders {
		wg.Add(1)
		go func(fileHeader *multipart.FileHeader) {
			defer wg.Done()

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

			if err := resource.manager.PutObject(fileHeader.Filename, file); err != nil {
				errCh <- fmt.Errorf("upload %s: %w", fileHeader.Filename, err)
				return
			}

			if err := resource.indexMemory(c.Request.Context(), fileHeader.Filename, string(content)); err != nil {
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
	res, err := resource.manager.ListObjects()

	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, res)
}

func (resource *MemoryResourceImpl) delete(c *gin.Context) {
	var body DeleteMemoryRequest
	if err := c.BindJSON(&body); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	key := body.Key

	if err := resource.manager.DeleteObject(key); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	if err := resource.embeddings.DeleteEmbeddingsForKey(c.Request.Context(), key); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(204, gin.H{"message": "success"})
}
