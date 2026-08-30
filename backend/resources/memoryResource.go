package resources

import (
	"fmt"
	"io"
	"mime/multipart"
	"strings"
	"sync"

	"daybid-dev-service/middleware"

	"github.com/gin-gonic/gin"

	"daybid-dev-service/managers"
)

type MemoryResourceImpl struct {
	s3Manager *managers.S3ManagerImpl
}

type DeleteMemoryRequest struct {
	Key string `json:"key"`
}

func NewMemoryResource(r *gin.RouterGroup) *MemoryResourceImpl {
	s3Manager := managers.InitS3Manager()
	return &MemoryResourceImpl{s3Manager: s3Manager}
}

func InitMemoryResource(r *gin.RouterGroup) {
	resource := NewMemoryResource(r)

	group := r.Group("/memory")
	group.Use(middleware.AuthMiddleware())
	group.POST("/", resource.write)
	group.POST("/batch", resource.batchWrite)
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

	result, err := resource.s3Manager.GetObject(key)

	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	defer result.Body.Close()

	bodyBytes, err := io.ReadAll(result.Body)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
	}

	content := string(bodyBytes)

	c.JSON(200, gin.H{"content": content})
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

	_, err = resource.s3Manager.PutObject(fileHeader.Filename, file)

	if err != nil {
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

			if _, err := resource.s3Manager.PutObject(fileHeader.Filename, file); err != nil {
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
