package resources

import (
	"bytes"
	"context"
	"daybid-dev-service/middleware"
	"log"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/gin-gonic/gin"
)

type AWSResourceImpl struct {
	s3Client *s3.Client
}

func NewAWSResource(r *gin.Engine) *AWSResourceImpl {
	return &AWSResourceImpl{}
}

func InitAWSResource(r *gin.Engine) {
	resource := NewAWSResource(r)

	cfg, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		log.Fatal(err)
	}

	resource.s3Client = s3.NewFromConfig(cfg)

	group := r.Group("/aws")
	group.Use(middleware.AuthMiddleware())
	group.GET("/s3", resource.ListBuckets)
	group.POST("/s3/upload", resource.TestUpload)
}

func (r *AWSResourceImpl) ListBuckets(ctx *gin.Context) {
	buckets, err := r.s3Client.ListBuckets(context.TODO(), &s3.ListBucketsInput{
		BucketRegion: aws.String("us-east-1"),
	})
	if err != nil {
		log.Fatal(err)
	}

	ctx.JSON(200, buckets)
}

func (r *AWSResourceImpl) TestUpload(ctx *gin.Context) {
	_, err := r.s3Client.PutObject(context.TODO(), &s3.PutObjectInput{
		Bucket:        aws.String("daybid-dev"),
		Key:           aws.String("test.txt"),
		Body:          bytes.NewReader([]byte{}),
		ContentLength: aws.Int64(0),
	})
	if err != nil {
		log.Fatal(err)
	}
	ctx.JSON(204, nil)
}
