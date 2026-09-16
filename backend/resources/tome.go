package resources

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
