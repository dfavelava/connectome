package resources

import (
	"encoding/json"
	"testing"

	"daybid-dev-service/managers"
)

func readEntityJSON(t *testing.T, manager managers.MemoryManager, id string) EntityWithMemories {
	t.Helper()

	content, err := manager.GetObject(entityKey(id))
	if err != nil {
		t.Fatalf("read %s: %v", entityKey(id), err)
	}
	var entity EntityWithMemories
	if err := json.Unmarshal([]byte(content), &entity); err != nil {
		t.Fatalf("unmarshal %s: %v", entityKey(id), err)
	}
	return entity
}

func TestUpsertEntityRelationshipStubsBothEntitiesWhenNeitherExists(t *testing.T) {
	manager := newLocalManager(t, nil)

	subject, object, err := UpsertEntityRelationship(manager, "alice", "likes", ptr("tea"), nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if subject.ID != "alice" || len(subject.MemberOf) != 0 {
		t.Fatalf("expected bare subject stub, got %+v", subject)
	}
	if object == nil || object.ID != "tea" {
		t.Fatalf("expected bare object stub, got %+v", object)
	}

	if got := readEntityJSON(t, manager, "alice"); got.ID != "alice" {
		t.Fatalf("expected alice stub written, got %+v", got)
	}
	if got := readEntityJSON(t, manager, "tea"); got.ID != "tea" {
		t.Fatalf("expected tea stub written, got %+v", got)
	}
}

func TestUpsertEntityRelationshipDoesNotStubWithoutObjectEntityID(t *testing.T) {
	manager := newLocalManager(t, nil)

	subject, object, err := UpsertEntityRelationship(manager, "alice", "is_tired", nil, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if object != nil {
		t.Fatalf("expected no object entity, got %+v", object)
	}
	if subject.ID != "alice" {
		t.Fatalf("expected alice stub, got %+v", subject)
	}
}

func TestUpsertEntityRelationshipMergesMemberOfForMemberOfPredicate(t *testing.T) {
	manager := newLocalManager(t, map[string]string{
		"ent_alice.json": `{"id":"alice","name":"Alice","member_of":["Party A"]}`,
	})

	subject, object, err := UpsertEntityRelationship(manager, "alice", "member_of", ptr("Adventurers"), nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(subject.MemberOf) != 2 || subject.MemberOf[0] != "Party A" || subject.MemberOf[1] != "Adventurers" {
		t.Fatalf("expected member_of [Party A, Adventurers], got %v", subject.MemberOf)
	}
	if subject.Name == nil || *subject.Name != "Alice" {
		t.Fatalf("expected existing name preserved, got %v", subject.Name)
	}
	if object == nil || object.ID != "Adventurers" {
		t.Fatalf("expected Adventurers stub, got %+v", object)
	}

	got := readEntityJSON(t, manager, "alice")
	if len(got.MemberOf) != 2 || got.MemberOf[1] != "Adventurers" {
		t.Fatalf("expected member_of persisted, got %v", got.MemberOf)
	}
	if got.Name == nil || *got.Name != "Alice" {
		t.Fatalf("expected name preserved on disk, got %v", got.Name)
	}
}

func TestUpsertEntityRelationshipDedupesMemberOf(t *testing.T) {
	manager := newLocalManager(t, map[string]string{
		"ent_alice.json": `{"id":"alice","member_of":["Party A"]}`,
	})

	subject, _, err := UpsertEntityRelationship(manager, "alice", "member_of", ptr("Party A"), nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(subject.MemberOf) != 1 || subject.MemberOf[0] != "Party A" {
		t.Fatalf("expected member_of unchanged with duplicate collapsed, got %v", subject.MemberOf)
	}
}

func TestUpsertEntityRelationshipIgnoresMemberOfForOtherPredicates(t *testing.T) {
	manager := newLocalManager(t, map[string]string{
		"ent_alice.json": `{"id":"alice","member_of":["Party A"]}`,
	})

	subject, _, err := UpsertEntityRelationship(manager, "alice", "likes", ptr("tea"), nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(subject.MemberOf) != 1 || subject.MemberOf[0] != "Party A" {
		t.Fatalf("expected member_of untouched for non-member_of predicate, got %v", subject.MemberOf)
	}
}

func TestUpsertEntityRelationshipLeavesExistingObjectRecordUntouched(t *testing.T) {
	manager := newLocalManager(t, map[string]string{
		"ent_tea.json": `{"id":"tea","kind":"drink","member_of":["Beverages"]}`,
	})

	_, object, err := UpsertEntityRelationship(manager, "alice", "likes", ptr("tea"), nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if object == nil || object.Kind == nil || *object.Kind != "drink" {
		t.Fatalf("expected existing object record preserved, got %+v", object)
	}

	got := readEntityJSON(t, manager, "tea")
	if got.Kind == nil || *got.Kind != "drink" || len(got.MemberOf) != 1 || got.MemberOf[0] != "Beverages" {
		t.Fatalf("expected object record on disk unchanged, got %+v", got)
	}
}

func TestUpsertEntityRelationshipLeavesExistingUnaffectedSubjectUnwritten(t *testing.T) {
	manager := newLocalManager(t, map[string]string{
		"ent_alice.json": `{"id":"alice","name":"Alice"}`,
	})

	subject, _, err := UpsertEntityRelationship(manager, "alice", "likes", nil, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if subject.Name == nil || *subject.Name != "Alice" {
		t.Fatalf("expected existing subject returned untouched, got %+v", subject)
	}

	got := readEntityJSON(t, manager, "alice")
	if got.Name == nil || *got.Name != "Alice" {
		t.Fatalf("expected subject record on disk unchanged, got %+v", got)
	}
}

func TestUpsertEntityRelationshipSetsSubjectKindAndMetaOnNewEntity(t *testing.T) {
	manager := newLocalManager(t, nil)

	subject, _, err := UpsertEntityRelationship(manager, "thorin", "member_of", ptr("discord-1-characters"), ptr("character"), map[string]any{"owner": "discord-1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if subject.Kind == nil || *subject.Kind != "character" {
		t.Fatalf("expected kind character, got %+v", subject)
	}
	if subject.Meta["owner"] != "discord-1" {
		t.Fatalf("expected meta.owner discord-1, got %+v", subject.Meta)
	}

	got := readEntityJSON(t, manager, "thorin")
	if got.Kind == nil || *got.Kind != "character" || got.Meta["owner"] != "discord-1" {
		t.Fatalf("expected kind/meta persisted, got %+v", got)
	}
}

func TestUpsertEntityRelationshipOverwritesExistingKindWhenGiven(t *testing.T) {
	manager := newLocalManager(t, map[string]string{
		"ent_thorin.json": `{"id":"thorin","kind":"npc"}`,
	})

	subject, _, err := UpsertEntityRelationship(manager, "thorin", "likes", nil, ptr("character"), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if subject.Kind == nil || *subject.Kind != "character" {
		t.Fatalf("expected kind overwritten to character, got %+v", subject)
	}
}

func TestUpsertEntityRelationshipMergesMetaShallowPreservingOtherKeys(t *testing.T) {
	manager := newLocalManager(t, map[string]string{
		"ent_thorin.json": `{"id":"thorin","meta":{"owner":"discord-1","hp":10}}`,
	})

	subject, _, err := UpsertEntityRelationship(manager, "thorin", "likes", nil, nil, map[string]any{"hp": 8})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if subject.Meta["owner"] != "discord-1" || subject.Meta["hp"] != 8 {
		t.Fatalf("expected owner preserved and hp updated, got %+v", subject.Meta)
	}
}

func TestUpsertEntityRelationshipLeavesSubjectUnwrittenWhenKindMetaUnchanged(t *testing.T) {
	manager := newLocalManager(t, map[string]string{
		"ent_thorin.json": `{"id":"thorin","kind":"character","meta":{"owner":"discord-1"}}`,
	})

	subject, _, err := UpsertEntityRelationship(manager, "thorin", "likes", nil, ptr("character"), map[string]any{"owner": "discord-1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if subject.Kind == nil || *subject.Kind != "character" || subject.Meta["owner"] != "discord-1" {
		t.Fatalf("expected kind/meta unchanged, got %+v", subject)
	}
}
