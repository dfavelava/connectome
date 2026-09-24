package resources

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"connectome-dev-service/managers"
)

// DefaultTome is the sentinel tome id for "no tome": today's blobs, stored
// unprefixed under their bare key, and embeddings rows with tome_id = ”.
// No caller can select a different tome yet - see TomeScopedKey.
const DefaultTome = ""

// tomeKeyPrefix is the blob key prefix every non-default tome's data lives
// under.
const tomeKeyPrefix = "tomes/"

// TomeScopedKey returns the blob store key that key should be read, written,
// or deleted under for tome. The default tome returns key unchanged, so
// today's blobs stay unprefixed and need no migration; any other tome scopes
// key under tomes/<tome>/, keeping each tome's blobs in their own namespace.
func TomeScopedKey(tome, key string) string {
	if tome == DefaultTome {
		return key
	}
	return tomeKeyPrefix + tome + "/" + key
}

// TomeUnscopedKey is TomeScopedKey's inverse: it strips tome's prefix off a
// blob store key, returning the bare key callers address memories by
// alongside a tome argument. Keys that aren't under tome's prefix (and every
// key for the default tome) come back unchanged.
func TomeUnscopedKey(tome, scopedKey string) string {
	return strings.TrimPrefix(scopedKey, tomeListPrefix(tome))
}

// tomeListPrefix returns the blob key prefix that scopes ListObjects to
// tome: empty for the default tome (ListObjects("") lists every tome's
// keys, so this alone isn't exclusive - see ListTome), or tomes/<tome>/
// otherwise, which is exact.
func tomeListPrefix(tome string) string {
	if tome == DefaultTome {
		return ""
	}
	return tomeKeyPrefix + tome + "/"
}

// ListTome lists manager's objects scoped to tome: exactly the ones under
// tomes/<tome>/ for a non-default tome, or every object that ISN'T under any
// tomes/ prefix for the default tome. The default case needs the extra
// filter because ListObjects("") returns every tome's keys, not just the
// unprefixed ones TomeScopedKey uses for DefaultTome. Keys come back bare
// (see TomeUnscopedKey), the same shape read/delete take with a tome.
func ListTome(manager managers.MemoryManager, tome string) (*managers.MemoryListResult, error) {
	result, err := manager.ListObjects(tomeListPrefix(tome))
	if err != nil {
		return nil, err
	}
	if tome != DefaultTome {
		contents := make([]managers.MemoryListItem, len(result.Contents))
		for i, item := range result.Contents {
			item.Key = TomeUnscopedKey(tome, item.Key)
			contents[i] = item
		}
		return &managers.MemoryListResult{Contents: contents}, nil
	}

	contents := make([]managers.MemoryListItem, 0, len(result.Contents))
	for _, item := range result.Contents {
		if !strings.HasPrefix(item.Key, tomeKeyPrefix) {
			contents = append(contents, item)
		}
	}
	return &managers.MemoryListResult{Contents: contents}, nil
}

// tomeTestConventionPrefixes are the tome id prefixes DestroyTome treats as
// obviously disposable, safe to destroy without a caller passing confirm.
var tomeTestConventionPrefixes = []string{"temp-", "test-"}

// ErrDestroyDefaultTome is returned by DestroyTome when asked to destroy
// DefaultTome. There is no override for this - the default tome holds
// today's unscoped data, so destroying it is never a "clean up a scratch
// tome" operation.
var ErrDestroyDefaultTome = errors.New("refusing to destroy the default tome")

// ErrDestroyNeedsConfirm is returned by DestroyTome when tome doesn't match
// an obvious test/temp naming convention and the caller didn't pass confirm.
// This is the guard called for in issue #66: a careless acceptance-test run
// that destroys tomes by id should not be able to take out a real tome (e.g.
// "west-marches") just because it forgot to check the id first.
var ErrDestroyNeedsConfirm = errors.New("destroying this tome requires confirm=true")

// TomeEmbeddingsDeleter is the subset of *daos.EmbeddingsDao that DestroyTome
// needs to remove a tome's embeddings rows.
type TomeEmbeddingsDeleter interface {
	DeleteEmbeddingsForTome(ctx context.Context, tomeID string) error
}

// isTomeTestConvention reports whether tome matches a naming convention
// (temp-/test- prefix) that marks it as obviously disposable.
func isTomeTestConvention(tome string) bool {
	for _, prefix := range tomeTestConventionPrefixes {
		if strings.HasPrefix(tome, prefix) {
			return true
		}
	}
	return false
}

// checkTomeDestroyAllowed enforces DestroyTome's guard: the default tome can
// never be destroyed, and any tome id outside the temp-/test- naming
// convention needs an explicit confirm=true.
func checkTomeDestroyAllowed(tome string, confirm bool) error {
	if tome == DefaultTome {
		return ErrDestroyDefaultTome
	}
	if isTomeTestConvention(tome) || confirm {
		return nil
	}
	return ErrDestroyNeedsConfirm
}

// DestroyTome deletes every blob stored under tome's prefix and every
// embeddings row with that tome_id. It refuses to run - returning
// ErrDestroyDefaultTome or ErrDestroyNeedsConfirm - unless
// checkTomeDestroyAllowed accepts (tome, confirm), so the guard lives here in
// the backend rather than depending on every caller remembering to check
// first.
func DestroyTome(ctx context.Context, manager managers.MemoryManager, embeddings TomeEmbeddingsDeleter, tome string, confirm bool) error {
	if err := checkTomeDestroyAllowed(tome, confirm); err != nil {
		return err
	}

	if err := manager.DeleteObjectsWithPrefix(tomeKeyPrefix + tome + "/"); err != nil {
		return fmt.Errorf("delete blobs for tome %s: %w", tome, err)
	}
	if err := embeddings.DeleteEmbeddingsForTome(ctx, tome); err != nil {
		return fmt.Errorf("delete embeddings for tome %s: %w", tome, err)
	}
	return nil
}
