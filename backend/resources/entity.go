package resources

import (
	"encoding/json"
	"slices"
)

// entityRecord is the subset of the JSON entity record (see
// daybidmcp's EntityWithMemories, written under ent_<id>.json) that
// resolveACLScope needs.
type entityRecord struct {
	MemberOf []string `json:"member_of"`
}

// entityKey returns the blob store key for an entity id, matching the
// ent_<id>.json convention daybidmcp's remember writes entity records under.
func entityKey(id string) string {
	return "ent_" + id + ".json"
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

	content, err := resource.manager.GetObject(entityKey(as))
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
