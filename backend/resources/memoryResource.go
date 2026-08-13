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

type DeleteMemoryRequest struct {
	Key string `json:"key"`
}

func NewMemoryResource(r *gin.RouterGroup) *MemoryResourceImpl {
	return &MemoryResourceImpl{}
}

func InitMemoryResource(r *gin.RouterGroup) {
	resource := NewMemoryResource(r)

	cfg, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		log.Fatal(err)
	}

	resource.s3Client = s3.NewFromConfig(cfg)

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

func (resource *MemoryResourceImpl) list(c *gin.Context) {
	res, err := resource.s3Client.ListObjects(context.TODO(), &s3.ListObjectsInput{
		Bucket:    aws.String("daybid-dev"),
		Delimiter: aws.String("/"),
	})

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

	_, err := resource.s3Client.DeleteObject(context.TODO(), &s3.DeleteObjectInput{
		Bucket: aws.String("daybid-dev"),
		Key:    aws.String(key),
	})

	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(204, gin.H{"message": "success"})
}
