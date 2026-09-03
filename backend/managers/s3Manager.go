package managers

import (
	"context"
	"io"
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

func (manager *S3ManagerImpl) GetObject(key string) (string, error) {
	result, err := manager.client.GetObject(context.TODO(), &s3.GetObjectInput{
		Bucket: aws.String("daybid-dev"),
		Key:    aws.String(key),
	})
	if err != nil {
		return "", err
	}
	defer result.Body.Close()

	bodyBytes, err := io.ReadAll(result.Body)
	if err != nil {
		return "", err
	}

	content := string(bodyBytes)
	return content, nil
}

func (manager *S3ManagerImpl) PutObject(key string, file multipart.File) error {
	_, err := manager.client.PutObject(context.TODO(), &s3.PutObjectInput{
		Bucket: aws.String("daybid-dev"),
		Key:    aws.String(key),
		Body:   file,
	})
	return err
}

func (manager *S3ManagerImpl) DeleteObject(key string) error {
	_, err := manager.client.DeleteObject(context.TODO(), &s3.DeleteObjectInput{
		Bucket: aws.String("daybid-dev"),
		Key:    aws.String(key),
	})
	return err
}

func (manager *S3ManagerImpl) ListObjects() (*MemoryListResult, error) {
	result, err := manager.client.ListObjects(context.TODO(), &s3.ListObjectsInput{
		Bucket:    aws.String("daybid-dev"),
		Delimiter: aws.String("/"),
	})
	if err != nil {
		return nil, err
	}

	contents := make([]MemoryListItem, 0, len(result.Contents))
	for _, item := range result.Contents {
		if item.Key == nil {
			continue
		}
		contents = append(contents, MemoryListItem{Key: *item.Key})
	}

	return &MemoryListResult{Contents: contents}, nil
}
