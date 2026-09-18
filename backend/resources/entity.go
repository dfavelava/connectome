package resources

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"

	"connectome-dev-service/managers"
)

// memberOfPredicate mirrors MEMBER_OF_PREDICATE in
// connectomeMCP/src/connectomemcp/server.py: a relationship predicate that, rather
// than just recording the edge, also merges the object entity id onto the
// subject entity's member_of list.
const memberOfPredicate = "member_of"

// entityRecord is the subset of the JSON entity record (see ent_<id>.json,
// written under EntityWithMemories) that resolveACLScope needs.
type entityRecord struct {
	MemberOf []string `json:"member_of"`
}

// EntityWithMemories mirrors connectomemcp's EntityWithMemories model as written
// to ent_<id>.json (see format_entity in connectomeMCP/src/connectomemcp/server.py).
// Fields round-trip as null rather than being omitted when unset, matching
// pydantic's model_dump_json(indent=2) default of including every field.
type EntityWithMemories struct {
	ID        string         `json:"id"`
	Name      *string        `json:"name"`
	Kind      *string        `json:"kind"`
	Meta      map[string]any `json:"meta"`
	MemoryIDs []string       `json:"memory_ids"`
	MemberOf  []string       `json:"member_of"`
}

// entityKey returns the blob store key for an entity id under tome, matching
// the ent_<id>.json convention connectomemcp's remember writes entity records
// under.
func entityKey(tome, id string) string {
	return TomeScopedKey(tome, "ent_"+id+".json")
}

// resolveACLScope returns the set of entity/group ids whose presence in a
// memory's acl grants an `as` id access to it: the id itself, plus any group
// ids its entity record lists in member_of (set by remember's member_of
// predicate special-case - see 1.1B/1.1C). Only that one level is resolved,
// not the groups' own member_of, matching the "no recursion" scope this
// phase settled on. A missing or unparsable entity record (the id has never
// been remembered as an entity) degrades to just the id itself rather than
// failing the search.
func (resource *SearchResourceImpl) resolveACLScope(as string) []string {
	scope := []string{as}

	// Search/recall isn't tome-aware yet (see issue #64), so this always
	// resolves against the default tome for now.
	content, err := resource.manager.GetObject(entityKey(DefaultTome, as))
	if err != nil {
		return scope
	}

	var entity entityRecord
	if err := json.Unmarshal([]byte(content), &entity); err != nil {
		return scope
	}

	for _, group := range entity.MemberOf {
		if group != "" && !slices.Contains(scope, group) {
			scope = append(scope, group)
		}
	}
	return scope
}

// readEntity fetches and parses an ent_<id>.json record, reporting whether
// one already existed. A missing record degrades to a bare stub (mirrors
// stub_entities_for_relationships's Entity(id=entity_id)) rather than an
// error, since "no record yet" is the expected state for an id only ever
// seen as a relationship endpoint so far.
func readEntity(manager managers.MemoryManager, tome, id string) (EntityWithMemories, bool, error) {
	content, err := manager.GetObject(entityKey(tome, id))
	if err != nil {
		if errors.Is(err, managers.ErrNotFound) {
			return EntityWithMemories{ID: id}, false, nil
		}
		return EntityWithMemories{}, false, err
	}

	var entity EntityWithMemories
	if err := json.Unmarshal([]byte(content), &entity); err != nil {
		return EntityWithMemories{}, false, fmt.Errorf("parse entity record %s: %w", entityKey(tome, id), err)
	}
	entity.ID = id
	return entity, true, nil
}

func writeEntity(manager managers.MemoryManager, tome string, entity EntityWithMemories) error {
	content, err := json.MarshalIndent(entity, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal entity record %s: %w", entityKey(tome, entity.ID), err)
	}
	return manager.PutObject(entityKey(tome, entity.ID), newMemoryFile(content))
}

// mergeMemberOf appends any group ids not already present, preserving order
// and dropping duplicates - mirrors merge_member_of in
// connectomeMCP/src/connectomemcp/server.py.
func mergeMemberOf(existing, newGroupIDs []string) []string {
	memberOf := append([]string{}, existing...)
	for _, groupID := range newGroupIDs {
		if !slices.Contains(memberOf, groupID) {
			memberOf = append(memberOf, groupID)
		}
	}
	return memberOf
}

// mergeKind overwrites existing with new when new is given, otherwise leaves
// existing unchanged - mirrors merge_kind in connectomeMCP/src/connectomemcp/server.py.
func mergeKind(existing, new *string) *string {
	if new != nil {
		return new
	}
	return existing
}

// mergeMeta shallow-merges new over existing, with keys in new overriding
// same-named keys in existing - mirrors merge_meta in
// connectomeMCP/src/connectomemcp/server.py.
func mergeMeta(existing, new map[string]any) map[string]any {
	if existing == nil && new == nil {
		return nil
	}
	merged := make(map[string]any, len(existing)+len(new))
	for key, value := range existing {
		merged[key] = value
	}
	for key, value := range new {
		merged[key] = value
	}
	return merged
}

// UpsertEntityRelationship upserts stub ent_<id>.json records for the
// subject/object entities named by a relationship that don't have one yet
// (mirrors stub_entities_for_relationships), and - for the member_of
// predicate - merges the object entity id into the subject entity's
// member_of list (mirrors merge_member_of). It lets callers that only assert
// a relationship, rather than write a full memory document (e.g.
// discordbot), drive ent_<id>.json into the same state
// connectomemcp.server.remember would produce.
//
// subjectKind/subjectMeta optionally stamp the subject entity's kind/meta
// fields in the same call (mirrors the Entity.kind/Entity.meta merge that
// connectomemcp.server.remember applies via merge_kind/merge_meta) - Connectome
// itself has no opinion on what values callers use here; it just persists
// and merges whatever a caller (e.g. discordbot's /add-character) passes.
//
// A record is only written back when it's new or something about it
// actually changed; an already-existing, untouched entity is left alone.
func UpsertEntityRelationship(manager managers.MemoryManager, tome, subjectID, predicate string, objectID *string, subjectKind *string, subjectMeta map[string]any) (subject EntityWithMemories, object *EntityWithMemories, err error) {
	subject, subjectExisted, err := readEntity(manager, tome, subjectID)
	if err != nil {
		return EntityWithMemories{}, nil, fmt.Errorf("read subject entity: %w", err)
	}

	subjectChanged := !subjectExisted
	if predicate == memberOfPredicate && objectID != nil {
		merged := mergeMemberOf(subject.MemberOf, []string{*objectID})
		if len(merged) != len(subject.MemberOf) {
			subject.MemberOf = merged
			subjectChanged = true
		}
	}

	if subjectKind != nil {
		merged := mergeKind(subject.Kind, subjectKind)
		if !reflect.DeepEqual(subject.Kind, merged) {
			subject.Kind = merged
			subjectChanged = true
		}
	}
	if subjectMeta != nil {
		merged := mergeMeta(subject.Meta, subjectMeta)
		if !reflect.DeepEqual(subject.Meta, merged) {
			subject.Meta = merged
			subjectChanged = true
		}
	}

	if subjectChanged {
		if err := writeEntity(manager, tome, subject); err != nil {
			return EntityWithMemories{}, nil, fmt.Errorf("write subject entity: %w", err)
		}
	}

	if objectID == nil {
		return subject, nil, nil
	}

	objectEntity, objectExisted, err := readEntity(manager, tome, *objectID)
	if err != nil {
		return EntityWithMemories{}, nil, fmt.Errorf("read object entity: %w", err)
	}
	if !objectExisted {
		if err := writeEntity(manager, tome, objectEntity); err != nil {
			return EntityWithMemories{}, nil, fmt.Errorf("write object entity: %w", err)
		}
	}

	return subject, &objectEntity, nil
}
