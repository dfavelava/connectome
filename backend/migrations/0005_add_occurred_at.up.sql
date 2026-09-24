-- When the memory's described event happened, as opposed to created_at (when
-- the memory was written). Nullable: NULL means unknown, and an occurred_*
-- search filter excludes NULL rows rather than falling back to created_at, so
-- the two meanings never blur. This table is derived state - cmd/reindex
-- backfills the column from each memory's `occurred_at` frontmatter key.
ALTER TABLE embeddings ADD COLUMN occurred_at TIMESTAMPTZ;
