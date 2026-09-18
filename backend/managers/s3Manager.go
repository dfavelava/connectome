package managers

import (
	"context"
	"errors"
	"io"
	"log"
	"mime/multipart"
	"os"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
)

type S3ManagerImpl struct {
	client *s3.Client
	bucket string
}

func NewS3Manager(cfg *aws.Config, bucket string) *S3ManagerImpl {
	return &S3ManagerImpl{client: s3.NewFromConfig(*cfg), bucket: bucket}
}

// InitS3Manager builds an S3ManagerImpl from the ambient AWS config and the
// S3_BUCKET env var, failing fast (matching the AWS config-load check below)
// if the bucket isn't set - there's no sane default bucket to fall back to.
func InitS3Manager() *S3ManagerImpl {
	cfg, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		log.Fatal(err)
	}
	bucket := os.Getenv("S3_BUCKET")
	if bucket == "" {
		log.Fatal("S3_BUCKET must be set when MEMORY_MANAGER=s3")
	}
	return NewS3Manager(&cfg, bucket)
}

func (manager *S3ManagerImpl) GetObject(key string) (string, error) {
	result, err := manager.client.GetObject(context.TODO(), &s3.GetObjectInput{
		Bucket: aws.String(manager.bucket),
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
		Bucket: aws.String(manager.bucket),
		Key:    aws.String(key),
		Body:   file,
	})
	return err
}

func (manager *S3ManagerImpl) DeleteObject(key string) error {
	_, err := manager.client.DeleteObject(context.TODO(), &s3.DeleteObjectInput{
		Bucket: aws.String(manager.bucket),
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
			Bucket:            aws.String(manager.bucket),
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
				Bucket: aws.String(manager.bucket),
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

// ListObjects lists every object whose key starts with prefix, paging
// through ListObjectsV2 (no delimiter, so it recurses into every "directory"
// - matching DeleteObjectsWithPrefix's traversal) until IsTruncated is
// false.
func (manager *S3ManagerImpl) ListObjects(prefix string) (*MemoryListResult, error) {
	contents := []MemoryListItem{}

	var continuationToken *string
	for {
		page, err := manager.client.ListObjectsV2(context.TODO(), &s3.ListObjectsV2Input{
			Bucket:            aws.String(manager.bucket),
			Prefix:            aws.String(prefix),
			ContinuationToken: continuationToken,
		})
		if err != nil {
			return nil, err
		}

		for _, item := range page.Contents {
			if item.Key == nil {
				continue
			}
			contents = append(contents, MemoryListItem{Key: *item.Key})
		}

		if page.IsTruncated == nil || !*page.IsTruncated {
			break
		}
		continuationToken = page.NextContinuationToken
	}

	return &MemoryListResult{Contents: contents}, nil
}
