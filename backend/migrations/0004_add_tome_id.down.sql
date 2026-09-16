DROP INDEX IF EXISTS embeddings_tome_id_idx;
ALTER TABLE embeddings DROP COLUMN IF EXISTS tome_id;
