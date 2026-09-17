package managers

import (
	"context"
	"errors"
	"io"
	"log"
	"mime/multipart"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
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
		var noSuchKey *types.NoSuchKey
		var notFound *types.NotFound
		if errors.As(err, &noSuchKey) || errors.As(err, &notFound) {
			return "", ErrNotFound
		}
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

// s3DeleteBatchSize is the max number of keys S3's DeleteObjects accepts in
// a single request.
const s3DeleteBatchSize = 1000

// DeleteObjectsWithPrefix deletes every object whose key starts with prefix,
// paging through ListObjectsV2 (no delimiter, so it recurses into every
// "directory") and batching deletes at s3DeleteBatchSize keys per request.
func (manager *S3ManagerImpl) DeleteObjectsWithPrefix(prefix string) error {
	var continuationToken *string
	for {
		page, err := manager.client.ListObjectsV2(context.TODO(), &s3.ListObjectsV2Input{
			Bucket:            aws.String("daybid-dev"),
			Prefix:            aws.String(prefix),
			ContinuationToken: continuationToken,
		})
		if err != nil {
			return err
		}

		ids := make([]types.ObjectIdentifier, 0, len(page.Contents))
		for _, item := range page.Contents {
			if item.Key == nil {
				continue
			}
			ids = append(ids, types.ObjectIdentifier{Key: item.Key})
		}

		for start := 0; start < len(ids); start += s3DeleteBatchSize {
			end := min(start+s3DeleteBatchSize, len(ids))
			if _, err := manager.client.DeleteObjects(context.TODO(), &s3.DeleteObjectsInput{
				Bucket: aws.String("daybid-dev"),
				Delete: &types.Delete{Objects: ids[start:end]},
			}); err != nil {
				return err
			}
		}

		if page.IsTruncated == nil || !*page.IsTruncated {
			return nil
		}
		continuationToken = page.NextContinuationToken
	}
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
