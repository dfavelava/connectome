import frontmatter

from daybidmcp.server import (
    DEFAULT_MEMORY_TYPE,
    DEFAULT_RELATIONSHIP_KIND,
    MEMORY_SCHEMA_VERSION,
    MEMORY_SOURCE_TYPE,
    MEMORY_TYPES,
    Entity,
    Relationship,
    format_memory,
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
