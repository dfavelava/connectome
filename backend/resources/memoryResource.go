package resources

import (
	"fmt"

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
	group.POST("/", resource.write)
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

	var contentLength int64
	if result.ContentLength != nil {
		contentLength = *result.ContentLength
	}

	contentType := "application/octet-stream"
	if result.ContentType != nil {
		contentType = *result.ContentType
	}

	extraHeaders := map[string]string{
		"Content-Disposition": fmt.Sprintf(`attachment; filename="%s"`, key),
	}

	c.DataFromReader(200, contentLength, contentType, result.Body, extraHeaders)
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
