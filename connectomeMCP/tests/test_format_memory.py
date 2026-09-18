import asyncio

import frontmatter

from connectomemcp import server
from connectomemcp.server import (
    DEFAULT_MEMORY_TYPE,
    DEFAULT_RELATIONSHIP_KIND,
    MEMBER_OF_PREDICATE,
    MEMORY_SCHEMA_VERSION,
    MEMORY_SOURCE_TYPE,
    MEMORY_TYPES,
    Entity,
    EntityWithMemories,
    Relationship,
    assert_member_of_relationships,
    format_memory,
    merge_kind,
    merge_meta,
    stub_entities_for_relationships,
)

CREATED_AT = "2026-09-09T00:00:00+00:00"


def test_format_memory_produces_parseable_frontmatter():
    document, payload = format_memory(
        "mem_abc.md",
        "David prefers tea over coffee.",
        [Entity(id="david", name="David")],
        [Relationship(subjectEntityId="david", predicate="prefers", objectEntityId="tea")],
        CREATED_AT,
        memory_type="preference",
    )

    # Fences at column 0, no leading blank line.
    assert document.startswith("---\n")

    post = frontmatter.loads(document)

    assert post.content.strip() == "David prefers tea over coffee."
    assert post["version"] == MEMORY_SCHEMA_VERSION
    assert post["id"] == "mem_abc.md"
    assert post["type"] == "preference"
    assert post["created_at"] == CREATED_AT
    assert post["entities"] == ["david"]
    assert post["source"] == {"type": MEMORY_SOURCE_TYPE, "created_at": CREATED_AT}
    assert post["relationships"] == [
        {
            "subjectEntityId": "david",
            "predicate": "prefers",
            "objectEntityId": "tea",
            "kind": DEFAULT_RELATIONSHIP_KIND,
            "superseded_by": None,
        }
    ]

    # The structured payload mirrors the frontmatter block.
    assert payload["id"] == "mem_abc.md"
    assert payload["content"] == "David prefers tea over coffee."
    assert payload["metadata"] == post.metadata


def test_format_memory_defaults_type_and_handles_no_entities():
    document, payload = format_memory("mem_xyz.md", "A bare note.", [], [], CREATED_AT)

    post = frontmatter.loads(document)

    assert post["type"] == DEFAULT_MEMORY_TYPE == "note"
    assert post["entities"] == []
    assert post["relationships"] == []
    assert post["source"]["type"] == MEMORY_SOURCE_TYPE
    assert payload["metadata"]["type"] == "note"


def test_default_memory_type_is_in_the_vocabulary():
    assert DEFAULT_MEMORY_TYPE in MEMORY_TYPES
    assert MEMORY_TYPES == ("note", "fact", "preference", "event")


def test_format_memory_preserves_multiline_body_including_triple_dash():
    body = "line one\n---\nline three\n"
    document, _ = format_memory("mem_multi.md", body, [], [], CREATED_AT)

    post = frontmatter.loads(document)

    assert post.content.strip("\n") == body.strip("\n")
    assert post["id"] == "mem_multi.md"


def test_format_memory_omits_acl_key_when_not_given():
    document, payload = format_memory("mem_no_acl.md", "body", [], [], CREATED_AT)

    post = frontmatter.loads(document)

    assert "acl" not in post.metadata
    assert "acl" not in payload["metadata"]


def test_format_memory_writes_explicit_acl():
    document, payload = format_memory(
        "mem_acl.md", "body", [], [], CREATED_AT, acl=["GM"]
    )

    post = frontmatter.loads(document)

    assert post["acl"] == ["GM"]
    assert payload["metadata"]["acl"] == ["GM"]


def test_format_memory_preserves_explicit_empty_acl():
    document, payload = format_memory(
        "mem_acl_empty.md", "body", [], [], CREATED_AT, acl=[]
    )

    post = frontmatter.loads(document)

    assert post["acl"] == []
    assert payload["metadata"]["acl"] == []


def test_format_memory_defaults_derived_from_to_none():
    document, payload = format_memory("mem_root.md", "body", [], [], CREATED_AT)

    post = frontmatter.loads(document)

    assert post["derived_from"] is None
    assert payload["metadata"]["derived_from"] is None


def test_format_memory_writes_derived_from():
    document, payload = format_memory(
        "mem_facet.md", "body", [], [], CREATED_AT, derived_from="mem_root.md"
    )

    post = frontmatter.loads(document)

    assert post["derived_from"] == "mem_root.md"
    assert payload["metadata"]["derived_from"] == "mem_root.md"


def test_relationship_defaults_kind_to_fact_and_superseded_by_to_none():
    relationship = Relationship(subjectEntityId="david", predicate="likes", objectEntityId="tea")

    assert relationship.kind == DEFAULT_RELATIONSHIP_KIND == "fact"
    assert relationship.superseded_by is None


def test_relationship_accepts_explicit_kind_and_superseded_by():
    relationship = Relationship(
        subjectEntityId="david",
        predicate="likes",
        objectEntityId="tea",
        kind="rumor",
        superseded_by="mem_correction.md",
    )

    assert relationship.kind == "rumor"
    assert relationship.superseded_by == "mem_correction.md"


def test_format_memory_round_trips_relationship_kind_and_superseded_by():
    document, payload = format_memory(
        "mem_kind.md",
        "body",
        [],
        [
            Relationship(
                subjectEntityId="david",
                predicate="suspects",
                objectEntityId="grace",
                kind="hypothesis",
                superseded_by="mem_confirmed.md",
            )
        ],
        CREATED_AT,
    )

    post = frontmatter.loads(document)

    assert post["relationships"] == [
        {
            "subjectEntityId": "david",
            "predicate": "suspects",
            "objectEntityId": "grace",
            "kind": "hypothesis",
            "superseded_by": "mem_confirmed.md",
        }
    ]
    assert payload["metadata"]["relationships"][0]["kind"] == "hypothesis"
    assert payload["metadata"]["relationships"][0]["superseded_by"] == "mem_confirmed.md"


def test_stub_entities_for_relationships_creates_bare_stubs_for_unlisted_ids():
    known = [Entity(id="david", name="David")]
    relationships = [
        Relationship(subjectEntityId="david", predicate="likes", objectEntityId="tea"),
        Relationship(subjectEntityId="grace", predicate="knows", objectEntityId="david"),
    ]

    stubs = stub_entities_for_relationships(known, relationships)

    assert [e.id for e in stubs] == ["tea", "grace"]
    assert all(e.name is None for e in stubs)


def test_stub_entities_for_relationships_dedupes_and_skips_known_and_none():
    known = [Entity(id="david")]
    relationships = [
        Relationship(subjectEntityId="tea", predicate="is_a", objectEntityId=None),
        Relationship(subjectEntityId="tea", predicate="is_a", objectEntityId="drink"),
        Relationship(subjectEntityId="david", predicate="likes", objectEntityId="tea"),
    ]

    stubs = stub_entities_for_relationships(known, relationships)

    assert [e.id for e in stubs] == ["tea", "drink"]


def test_stub_entities_for_relationships_empty_when_all_known():
    known = [Entity(id="david"), Entity(id="tea")]
    relationships = [Relationship(subjectEntityId="david", predicate="likes", objectEntityId="tea")]

    assert stub_entities_for_relationships(known, relationships) == []


def test_entity_with_memories_defaults_member_of_to_none():
    entity = EntityWithMemories(id="alice")

    assert entity.member_of is None


def test_entity_defaults_kind_and_meta_to_none():
    entity = Entity(id="alice")

    assert entity.kind is None
    assert entity.meta is None


def test_entity_accepts_free_form_kind_and_meta():
    entity = Entity(id="alice", kind="location", meta={"status": "scouted"})

    assert entity.kind == "location"
    assert entity.meta == {"status": "scouted"}


def test_merge_kind_prefers_new_value_and_falls_back_to_existing():
    assert merge_kind(None, "location") == "location"
    assert merge_kind("location", None) == "location"
    assert merge_kind("location", "region") == "region"


def test_merge_meta_shallow_merges_new_keys_over_existing():
    assert merge_meta(None, None) is None
    assert merge_meta(None, {"status": "scouted"}) == {"status": "scouted"}
    assert merge_meta({"status": "scouted"}, None) == {"status": "scouted"}
    assert merge_meta({"status": "scouted", "danger": "low"}, {"status": "cleared"}) == {
        "status": "cleared",
        "danger": "low",
    }


def test_format_memory_round_trips_member_of_relationship_like_any_other():
    document, payload = format_memory(
        "mem_membership.md",
        "Alice joins the party.",
        [],
        [Relationship(subjectEntityId="alice", predicate=MEMBER_OF_PREDICATE, objectEntityId="Party A")],
        CREATED_AT,
    )

    post = frontmatter.loads(document)

    assert post["relationships"] == [
        {
            "subjectEntityId": "alice",
            "predicate": MEMBER_OF_PREDICATE,
            "objectEntityId": "Party A",
            "kind": DEFAULT_RELATIONSHIP_KIND,
            "superseded_by": None,
        }
    ]
    assert payload["metadata"]["relationships"][0]["predicate"] == MEMBER_OF_PREDICATE


def test_assert_member_of_relationships_posts_only_member_of_with_an_object(monkeypatch):
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    async def fake_request(method, path, *, params=None, files=None, json_body=None):
        calls.append((method, path, json_body))
        return None

    monkeypatch.setattr(server, "request", fake_request)

    relationships = [
        Relationship(subjectEntityId="alice", predicate=MEMBER_OF_PREDICATE, objectEntityId="Party A"),
        Relationship(subjectEntityId="bob", predicate="likes", objectEntityId="tea"),
        Relationship(subjectEntityId="carol", predicate=MEMBER_OF_PREDICATE, objectEntityId=None),
    ]

    asyncio.run(assert_member_of_relationships(relationships))

    # Only the member_of relationship with a non-null object hits the shared
    # backend endpoint - "likes" and the object-less member_of are skipped,
    # mirroring the special-case the removed member_of_groups_by_subject used
    # to apply locally.
    assert calls == [
        (
            "POST",
            "/entity/relationship",
            {
                "subjectEntityId": "alice",
                "predicate": MEMBER_OF_PREDICATE,
                "objectEntityId": "Party A",
                "kind": DEFAULT_RELATIONSHIP_KIND,
            },
        )
    ]
