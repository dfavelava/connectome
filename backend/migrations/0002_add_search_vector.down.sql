DROP INDEX IF EXISTS embeddings_search_vector_gin_idx;
ALTER TABLE embeddings DROP COLUMN IF EXISTS search_vector;
ALTER TABLE embeddings DROP COLUMN IF EXISTS chunk_text;
