CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE embeddings (
    id BIGSERIAL PRIMARY KEY,
    memory_key TEXT NOT NULL,
    chunk_index INT NOT NULL DEFAULT 0,
    embedding VECTOR(768) NOT NULL,
    model TEXT NOT NULL,
    dim INT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('note', 'fact', 'preference', 'event')),
    entity_ids TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (memory_key, chunk_index)
);

CREATE INDEX embeddings_embedding_hnsw_idx
    ON embeddings
    USING hnsw (embedding vector_cosine_ops);
