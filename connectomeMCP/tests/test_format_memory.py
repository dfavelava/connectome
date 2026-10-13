import asyncio

import frontmatter
import pytest

from connectomemcp import server
from connectomemcp.server import (
    DEFAULT_MEMORY_TYPE,
    DEFAULT_RELATIONSHIP_KIND,
    MEMBER_OF_PREDICATE,
    MEMORY_GENERATED_BY,
    MEMORY_SCHEMA_VERSION,
    MEMORY_TYPES,
    Entity,
    EntityWithMemories,
    MemoryMetadata,
    Relationship,
    Source,
    assert_member_of_relationships,
    format_memory,
    merge_kind,
    merge_meta,
    stub_entities_for_relationships,
    validate_occurred_at,
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
    assert post["generated"] == {"by": MEMORY_GENERATED_BY, "at": CREATED_AT}
    assert "source" not in post.metadata
    assert "verified" not in post.metadata
    assert "sources" not in post.metadata
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
    assert post["generated"]["by"] == MEMORY_GENERATED_BY
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


def test_format_memory_defaults_occurred_at_to_none():
    document, payload = format_memory("mem_no_occurred.md", "body", [], [], CREATED_AT)

    post = frontmatter.loads(document)

    assert post["occurred_at"] is None
    assert payload["metadata"]["occurred_at"] is None


def test_format_memory_writes_explicit_occurred_at():
    occurred_at = "2023-06-15T09:30:00+00:00"
    document, payload = format_memory(
        "mem_occurred.md", "body", [], [], CREATED_AT, occurred_at=occurred_at
    )

    post = frontmatter.loads(document)

    assert post["occurred_at"] == occurred_at
    assert post["created_at"] == CREATED_AT
    assert payload["metadata"]["occurred_at"] == occurred_at


def test_format_memory_rejects_malformed_occurred_at():
    with pytest.raises(ValueError, match="occurred_at"):
        format_memory("mem_bad_occurred.md", "body", [], [], CREATED_AT, occurred_at="not-a-date")


def test_validate_occurred_at_accepts_none_and_rfc3339():
    validate_occurred_at(None)
    validate_occurred_at("2023-06-15T09:30:00Z")


def test_validate_occurred_at_rejects_malformed_value():
    with pytest.raises(ValueError, match="occurred_at"):
        validate_occurred_at("not-a-date")


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


def test_schema_version_and_default_actor():
    assert MEMORY_SCHEMA_VERSION == "connectome/memory/0.2"
    assert MEMORY_GENERATED_BY.startswith("connectomemcp/")


def test_format_memory_writes_generated_by_override_and_sources():
    document, payload = format_memory(
        "mem_extracted.md",
        "body",
        [],
        [],
        CREATED_AT,
        generated_by="process:session-pipeline",
        sources=[
            Source(id="chunk-12", resource="mem_transcript.md", title="Session 14 transcript, 00:42-00:47"),
            Source(resource="https://example.test/notes"),
        ],
    )

    post = frontmatter.loads(document)

    assert post["generated"] == {"by": "process:session-pipeline", "at": CREATED_AT}
    # Unset source keys are omitted rather than written as null.
    assert post["sources"] == [
        {"id": "chunk-12", "resource": "mem_transcript.md", "title": "Session 14 transcript, 00:42-00:47"},
        {"resource": "https://example.test/notes"},
    ]
    assert payload["metadata"] == post.metadata


LEGACY_METADATA = {
    "version": "connectome/memory/0.1",
    "id": "mem_old.md",
    "type": "fact",
    "created_at": CREATED_AT,
    "source": {"type": "mcp", "created_at": CREATED_AT},
    "entities": ["david"],
    "relationships": [],
    "derived_from": None,
}


def test_legacy_metadata_reads_as_current_with_generated_from_source():
    metadata = MemoryMetadata.model_validate(LEGACY_METADATA)

    assert metadata.version == MEMORY_SCHEMA_VERSION
    assert metadata.generated.by == "mcp"
    assert metadata.generated.at == CREATED_AT
    assert metadata.verified is None
    assert metadata.sources is None
    assert "source" not in metadata.model_dump()
    # The input is not mutated.
    assert LEGACY_METADATA["version"] == "connectome/memory/0.1"
    assert "source" in LEGACY_METADATA


def test_metadata_accepts_bare_verified_mapping_and_keeps_unknown_keys():
    metadata = MemoryMetadata.model_validate(
        {
            "version": MEMORY_SCHEMA_VERSION,
            "id": "mem_new.md",
            "type": "fact",
            "created_at": CREATED_AT,
            "generated": {"by": "process:session-pipeline", "at": CREATED_AT},
            "verified": {"by": "human:discord-1234", "at": CREATED_AT},
            "x_campaign": {"name": "west-marches"},
        }
    )

    assert metadata.verified is not None
    assert [entry.by for entry in metadata.verified] == ["human:discord-1234"]
    assert metadata.model_dump()["x_campaign"] == {"name": "west-marches"}
