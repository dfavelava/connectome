ALTER TABLE embeddings ADD COLUMN chunk_text TEXT NOT NULL DEFAULT '';

-- Generated so it's always derived from chunk_text and never falls out of
-- sync with it (no separate write path to forget).
ALTER TABLE embeddings ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED;

CREATE INDEX embeddings_search_vector_gin_idx
    ON embeddings
    USING gin (search_vector);
