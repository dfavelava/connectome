package resources

import (
	"fmt"
	"mime/multipart"
	"strings"
	"sync"

	"daybid-dev-service/middleware"

	"github.com/gin-gonic/gin"

	"daybid-dev-service/managers"
)

type ManagerType string

const (
	ManagerTypeS3    ManagerType = "s3"
	ManagerTypeLocal ManagerType = "local"
)

type MemoryResourceImpl struct {
	s3Manager  *managers.S3ManagerImpl
	managerMap map[ManagerType]managers.MemoryManager
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

func NewMemoryResource(r *gin.RouterGroup) *MemoryResourceImpl {
	s3Manager := managers.InitS3Manager()
	localManager := managers.InitLocalFsManager()
	managerMap := make(map[ManagerType]managers.MemoryManager)
	managerMap[ManagerTypeS3] = s3Manager
	managerMap[ManagerTypeLocal] = localManager
	return &MemoryResourceImpl{managerMap: managerMap}
}

func (resource *MemoryResourceImpl) GetManager() managers.MemoryManager {
	return resource.managerMap[ManagerTypeS3]
}

func InitMemoryResource(r *gin.RouterGroup) {
	resource := NewMemoryResource(r)

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
		c.JSON(500, gin.H{"error": "s3 key not specified"})
		return
	}

	content, err := resource.s3Manager.GetObject(key)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
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

			content, err := resource.s3Manager.GetObject(key)
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

	if err := resource.s3Manager.PutObject(fileHeader.Filename, file); err != nil {
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

			if err := resource.s3Manager.PutObject(fileHeader.Filename, file); err != nil {
				errCh <- fmt.Errorf("upload %s: %w", fileHeader.Filename, err)
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
	res, err := resource.s3Manager.ListObjects()

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

	_, err := resource.s3Manager.DeleteObject(key)

	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(204, gin.H{"message": "success"})
}
