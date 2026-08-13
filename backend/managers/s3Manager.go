package managers

import (
	"context"
	"log"
	"mime/multipart"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

type S3ManagerImpl struct {
	client *s3.Client
}

func NewS3Manager(cfg *aws.Config) *S3ManagerImpl {
	return &S3ManagerImpl{client: s3.NewFromConfig(*cfg)}
}

func InitS3Manager() *S3ManagerImpl {
	cfg, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		log.Fatal(err)
	}
	return NewS3Manager(&cfg)
}

func (manager *S3ManagerImpl) GetObject(key string) (*s3.GetObjectOutput, error) {
	return manager.client.GetObject(context.TODO(), &s3.GetObjectInput{
		Bucket: aws.String("daybid-dev"),
		Key:    aws.String(key),
	})
}

func (manager *S3ManagerImpl) PutObject(key string, file multipart.File) (*s3.PutObjectOutput, error) {
	return manager.client.PutObject(context.TODO(), &s3.PutObjectInput{
		Bucket: aws.String("daybid-dev"),
		Key:    aws.String(key),
		Body:   file,
	})
}

func (manager *S3ManagerImpl) DeleteObject(key string) (*s3.DeleteObjectOutput, error) {
	return manager.client.DeleteObject(context.TODO(), &s3.DeleteObjectInput{
		Bucket: aws.String("daybid-dev"),
		Key:    aws.String(key),
	})
}

func (manager *S3ManagerImpl) ListObjects() (*s3.ListObjectsOutput, error) {
	return manager.client.ListObjects(context.TODO(), &s3.ListObjectsInput{
		Bucket:    aws.String("daybid-dev"),
		Delimiter: aws.String("/"),
	})
}
