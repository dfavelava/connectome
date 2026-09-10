import frontmatter

from daybidmcp.server import (
    DEFAULT_MEMORY_TYPE,
    MEMORY_SCHEMA_VERSION,
    MEMORY_SOURCE_TYPE,
    MEMORY_TYPES,
    Entity,
    Relationship,
    format_memory,
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
        {"subjectEntityId": "david", "predicate": "prefers", "objectEntityId": "tea"}
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
