package resources

import (
	"context"
	"fmt"
	"log"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/gin-gonic/gin"
)

type MemoryResourceImpl struct {
	s3Client *s3.Client
}

type CreateMemoryRequest struct {
	Dir string `json:"dir"`
}

func NewMemoryResource(r *gin.Engine) *MemoryResourceImpl {
	return &MemoryResourceImpl{}
}

func InitMemoryResource(r *gin.Engine) {
	resource := NewMemoryResource(r)

	cfg, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		log.Fatal(err)
	}

	resource.s3Client = s3.NewFromConfig(cfg)

	group := r.Group("/memory")
	group.POST("/", resource.write)
	group.GET("/:key", resource.read)
}

func (resource *MemoryResourceImpl) read(c *gin.Context) {
	key := c.Param("key")

	result, err := resource.s3Client.GetObject(context.TODO(), &s3.GetObjectInput{
		Bucket: aws.String("daybid-dev"),
		Key:    aws.String(key),
	})

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

	_, err = resource.s3Client.PutObject(context.TODO(), &s3.PutObjectInput{
		Bucket: aws.String("daybid-dev"),
		Key:    aws.String(fileHeader.Filename),
		Body:   file,
	})

	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, gin.H{"message": "success"})
}
