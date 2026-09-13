-- No literal default here (e.g. '{"GM"}') - the schema stays instance-agnostic.
-- A write that omits acl has it resolved by the backend's DEFAULT_ACL config
-- before the row reaches this table (see backend/resources/acl.go).
ALTER TABLE embeddings ADD COLUMN acl TEXT[] NOT NULL DEFAULT '{}';
