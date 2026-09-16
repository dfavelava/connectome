-- Empty string is the default tome's sentinel, not NULL, so
-- "WHERE tome_id = $1" stays simple SQL with no null-handling special case
-- (same reasoning as ResolveACL returning [], not nil, for "unrestricted" -
-- see backend/resources/acl.go).
ALTER TABLE embeddings ADD COLUMN tome_id TEXT NOT NULL DEFAULT '';

CREATE INDEX embeddings_tome_id_idx
    ON embeddings (tome_id);
