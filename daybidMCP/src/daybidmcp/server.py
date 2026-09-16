import json
import os
import uuid
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Literal, get_args

import httpx
import yaml
from dotenv import load_dotenv
from mcp.server import MCPServer
from pydantic import BaseModel, Field

DEFAULT_API_BASE_URL = "http://localhost:8080/api/connectome"
DEFAULT_TIMEOUT_SECONDS = 30.0
USER_AGENT = "connectome/0.1.0"
MEMORY_SCHEMA_VERSION = "connectome/memory/0.1"

# The kind of thing a memory records. Kept deliberately small; extend as real
# usage demands rather than guessing up front.
#   note       - freeform observation with no stronger structure
#   fact       - a discrete, durable statement of fact
#   preference - how the user wants things done
#   event      - something that happened at a point in time
MemoryType = Literal["note", "fact", "preference", "event"]
MEMORY_TYPES: tuple[str, ...] = get_args(MemoryType)
DEFAULT_MEMORY_TYPE: MemoryType = "note"

# Identifies where a memory came from. Every memory written through this server
# originates from an MCP client.
MEMORY_SOURCE_TYPE = "mcp"

# The truth-status of a relationship claim. Lives on the relationship entry,
# not the memory, so one memory can carry a fact and a rumor side by side.
#   fact       - a settled, current claim
#   hypothesis - a working guess, not yet confirmed
#   rumor      - reported but unverified, may turn out to be false
RelationshipKind = Literal["fact", "hypothesis", "rumor"]
DEFAULT_RELATIONSHIP_KIND: RelationshipKind = "fact"

# A relationship predicate with special meaning to `remember`: rather than
# just recording the edge, it also merges the object entity id onto the
# subject entity's member_of list (see EntityWithMemories.member_of), so
# recall's `as` scoping can resolve one level of group membership as an O(1)
# read of the entity record instead of a relationship-table scan.
MEMBER_OF_PREDICATE = "member_of"

_ = load_dotenv(Path(__file__).resolve().parents[2] / ".env")

mcp = MCPServer(
    "connectome",
    instructions="""Use this server proactively and liberally, not just when explicitly
asked to "remember" something. Call `remember` whenever you learn a durable fact,
a stated preference, a correction to how you should behave, or a notable event —
during any conversation, not only when told to. Prefer several small, well-typed
memories (note/fact/preference/event) over one large dump. Before writing, consider
calling `recall` or `get_memory` to check whether something similar already
exists, to avoid duplicates. Call `recall` with a question or topic to find
relevant memories ranked by relevance — prefer it over `browse_all` whenever you
have something specific in mind; fall back to `browse_all` only when you need
the full list of what's stored.""",
)

class Entity(BaseModel):
    """An entity mentioned in the memory content."""

    id: str = Field(description="A stable unique identifier for the entity, such as a slug, username, or system ID. Reuse the same ID across memories to link them together.")
    name: str | None = Field(default=None, description="A human-readable display name for the entity, if one is available.")
    kind: str | None = Field(default=None, description="A free-form, caller-defined tag for the entity's type, such as 'location' or 'person'. Connectome imposes no vocabulary, enum, or validation on this value - it's the calling application's convention to define. Overwrites any existing kind on this entity when given; omit to leave an existing kind unchanged.")
    meta: dict[str, object] | None = Field(default=None, description="An opaque, caller-defined JSON object for arbitrary structured data about the entity (e.g. {'status': 'scouted'}). Connectome does not interpret, validate, or enforce any schema on its contents. Shallow-merged into any existing meta on this entity, with keys given here overriding existing keys of the same name.")

class EntityWithMemories(Entity):
    memory_ids: list[str] | None = None
    member_of: list[str] | None = None

class Relationship(BaseModel):
    """A directed relationship between entities extracted from the memory.

    For a symmetric predicate (e.g. `adjacent_to`), write a single relationship
    entry at creation time rather than one in each direction - querying such
    predicates as an undirected graph edge is Phase 2's job, not this one's.
    """

    subjectEntityId: str = Field(description="The entity ID that acts as the subject or source of the relationship.")
    predicate: str = Field(description="The relationship label, action, or edge type connecting the subject to the object, such as 'works_with' or 'likes'.")
    objectEntityId: str | None = Field(default=None, description="The entity ID that acts as the object or target of the relationship. Leave null when the relationship has no explicit target entity.")
    kind: RelationshipKind = Field(default=DEFAULT_RELATIONSHIP_KIND, description="The truth-status of this claim: 'fact' for a settled claim, 'hypothesis' for an unconfirmed working guess, or 'rumor' for a reported but unverified claim.")
    superseded_by: str | None = Field(default=None, description="The id of the memory that supersedes/corrects this relationship claim, if any. Null means this relationship is current/active.")

class MemorySource(BaseModel):
    type: str
    created_at: str

class MemoryMetadata(BaseModel):
    version: str = MEMORY_SCHEMA_VERSION
    id: str | None = None
    type: str
    created_at: str

    source: MemorySource
    entities: list[str] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    acl: list[str] | None = None
    derived_from: str | None = None

class Memory(BaseModel):
    id: str
    content: str
    metadata: MemoryMetadata

def get_api_base_url() -> str:
    return os.getenv("DAYBID_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def get_api_key() -> str | None:
    return os.getenv("DAYBID_API_KEY") or os.getenv("apikey")


def get_headers() -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    api_key = get_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def build_url(path: str) -> str:
    return f"{get_api_base_url()}{path}"

def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4()}"

def format_memory(
    id: str,
    content: str,
    entities: list[Entity],
    relationships: list[Relationship],
    created_at: str,
    memory_type: str = DEFAULT_MEMORY_TYPE,
    acl: list[str] | None = None,
    derived_from: str | None = None,
) -> tuple[str, dict[str, object]]:
    metadata = MemoryMetadata(
        id=id,
        type=memory_type,
        created_at=created_at,
        source=MemorySource(type=MEMORY_SOURCE_TYPE, created_at=created_at),
        entities=[entity.id for entity in entities],
        relationships=relationships,
        acl=acl,
        derived_from=derived_from,
    )
    metadata_payload = metadata.model_dump(mode="json")
    if acl is None:
        # Omit the key entirely (rather than writing `acl: null`) so the Go
        # backend can tell "no acl given" apart from "explicitly cleared" and
        # apply its configured DEFAULT_ACL only in the former case.
        del metadata_payload["acl"]
    yaml_data = yaml.dump(metadata_payload, sort_keys=False).strip("\n")
    document = f"---\n{yaml_data}\n---\n{content}\n"
    return document, {
        "id": id,
        "content": content,
        "metadata": metadata_payload,
    }

def format_entity(entity: EntityWithMemories) -> str:
    return entity.model_dump_json(indent=2)


def merge_memory_ids(existing: list[str] | None, new_memory_id: str) -> list[str]:
    memory_ids = list(existing or [])
    if new_memory_id not in memory_ids:
        memory_ids.append(new_memory_id)
    return memory_ids


def merge_kind(existing: str | None, new: str | None) -> str | None:
    return new if new is not None else existing


def merge_meta(existing: dict[str, object] | None, new: dict[str, object] | None) -> dict[str, object] | None:
    if existing is None and new is None:
        return None
    return {**(existing or {}), **(new or {})}


def stub_entities_for_relationships(
    entities: list[Entity], relationships: list[Relationship]
) -> list[Entity]:
    """Bare stub entities for relationship endpoints not already in entities.

    A relationship's subjectEntityId/objectEntityId may name an id that
    wasn't passed in entities explicitly. Rather than erroring, such ids get
    a nameless stub Entity so an ent_*.json record still gets created for
    them and future memories can merge into it.
    """
    known_ids = {entity.id for entity in entities}
    stub_ids: list[str] = []
    seen: set[str] = set()
    for relationship in relationships:
        for entity_id in (relationship.subjectEntityId, relationship.objectEntityId):
            if entity_id is not None and entity_id not in known_ids and entity_id not in seen:
                seen.add(entity_id)
                stub_ids.append(entity_id)
    return [Entity(id=entity_id) for entity_id in stub_ids]

async def request(
    method: str,
    path: str,
    *,
    params: dict[str, str] | None = None,
    files: list[tuple[str, tuple[str, BytesIO, str]]] | dict[str, tuple[str, BytesIO, str]] | None = None,
    json_body: dict[str, object] | None = None,
) -> httpx.Response:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        response = await client.request(
            method,
            build_url(path),
            headers=get_headers(),
            params=params,
            files=files,
            json=json_body,
        )
        _ = response.raise_for_status()
        return response


async def batch_read(keys: list[str]) -> dict[str, str]:
    if not keys:
        return {}

    response = await request("POST", "/memory/batch/read", json_body={"keys": keys})
    payload = response.json()
    return payload.get("contents", {})


async def assert_member_of_relationships(relationships: list[Relationship]) -> None:
    """Merge each member_of relationship onto its subject entity's member_of
    list via the Go backend's shared /entity/relationship endpoint, rather
    than re-deriving that merge (dedupe-and-append) in Python - see issue #51.
    Keeping one implementation of the merge (backend/resources/entity.go's
    UpsertEntityRelationship) means daybidmcp and discordbot can't drift.

    Skips predicates other than member_of and relationships with no
    objectEntityId, mirroring the same special-case the removed
    member_of_groups_by_subject used to apply locally.
    """
    for relationship in relationships:
        if relationship.predicate != MEMBER_OF_PREDICATE or relationship.objectEntityId is None:
            continue
        _ = await request(
            "POST",
            "/entity/relationship",
            json_body={
                "subjectEntityId": relationship.subjectEntityId,
                "predicate": relationship.predicate,
                "objectEntityId": relationship.objectEntityId,
                "kind": relationship.kind,
            },
        )


@mcp.tool()
async def get_memory(key: str) -> str:
    """Fetch a stored memory document or entity record by key and return the backend JSON response."""
    response = await request("GET", "/memory/", params={"key": key})
    return response.text


@mcp.tool()
async def remember(
    content: str = Field(description="The raw memory content to store as the main document body."),
    entities: list[Entity] | None = Field(default=None, description="Entities explicitly mentioned in the memory. Each entity should use a stable ID so future memories can merge into the same entity record."),
    relationships: list[Relationship] | None = Field(default=None, description="Directed relationships between the provided entities. Use this to capture how entities are connected within the memory."),
    memory_type: MemoryType = Field(default=DEFAULT_MEMORY_TYPE, description="The kind of memory: 'note' for a freeform observation, 'fact' for a durable statement of fact, 'preference' for how the user wants things done, or 'event' for something that happened at a point in time."),
    acl: list[str] | None = Field(default=None, description="Access-control list (entity/group ids) restricting who can access this memory. Omit to apply this Connectome instance's configured default ACL policy (unrestricted if the instance has none configured)."),
    derived_from: str | None = Field(default=None, description="The id of another memory (e.g. 'mem_abc.md') this one is a facet of. A facet is an ordinary memory - stored, chunked, embedded, and ACL-filtered exactly like any other - that happens to record one entity's own version of the root memory's content. Use derived_from when the facet's *content* diverges from the root (a character's private take on a shared event, a rumor vs. the settled fact); if the audience is merely narrower but the content agrees with the root, just tighten the root memory's own acl instead of creating a facet."),
) -> str:
    """Create a memory document, merge its ID into related entity records, and return JSON with the memory key, structured memory payload, and entity keys."""
    # TODO: Check if memory already exists and update if found
    memory_id = f"{generate_id('mem')}.md"
    now = datetime.now(UTC)
    memory_entities = entities or []
    memory_relationships = relationships or []
    memory_entities = memory_entities + stub_entities_for_relationships(memory_entities, memory_relationships)

    memory, memory_payload = format_memory(
        memory_id,
        content,
        memory_entities,
        memory_relationships,
        now.isoformat(),
        memory_type,
        acl,
        derived_from,
    )

    # Merge member_of onto its subject entity record via the shared backend
    # endpoint before reading entities below, so existing_member_of already
    # reflects this memory's member_of relationships.
    await assert_member_of_relationships(memory_relationships)

    entity_keys = [f"ent_{entity.id}.json" for entity in memory_entities]
    existing_entity_contents = await batch_read(entity_keys)

    files: list[tuple[str, tuple[str, BytesIO, str]]] = [
        ("file", (memory_id, BytesIO(memory.encode("utf-8")), "text/plain; charset=utf-8"))
    ]

    for entity, entity_id in zip(memory_entities, entity_keys):
        existing_entity = existing_entity_contents.get(entity_id)
        existing_memory_ids: list[str] | None = None
        existing_member_of: list[str] | None = None
        existing_kind: str | None = None
        existing_meta: dict[str, object] | None = None
        if existing_entity:
            existing_parsed = EntityWithMemories.model_validate_json(existing_entity)
            existing_memory_ids = existing_parsed.memory_ids
            existing_member_of = existing_parsed.member_of
            existing_kind = existing_parsed.kind
            existing_meta = existing_parsed.meta

        entity_with_memories = EntityWithMemories(
            **entity.model_dump(exclude={"kind", "meta"}),
            memory_ids=merge_memory_ids(existing_memory_ids, memory_id),
            member_of=existing_member_of,
            kind=merge_kind(existing_kind, entity.kind),
            meta=merge_meta(existing_meta, entity.meta),
        )
        files.append(
            ("file", (entity_id, BytesIO(format_entity(entity_with_memories).encode("utf-8")), "application/json"))
        )

    _ = await request(
        "POST",
        "/memory/batch",
        files=files,
    )
    return json.dumps(
        {
            "message": "stored",
            "key": memory_id,
            "memory": memory_payload,
            "entity_keys": entity_keys,
        }
    )


@mcp.tool()
async def supersede_relationship(
    memory_id: str = Field(description="The memory key (e.g. 'mem_abc.md') whose relationship entry should be patched."),
    subjectEntityId: str = Field(description="The subjectEntityId of the relationship to patch."),
    predicate: str = Field(description="The predicate of the relationship to patch."),
    objectEntityId: str | None = Field(default=None, description="The objectEntityId of the relationship to patch. Must match exactly, including null for a relationship with no object."),
    superseded_by: str | None = Field(default=None, description="The id of the memory that supersedes/corrects this relationship claim. Pass null to clear a prior supersession and mark the relationship current again."),
) -> str:
    """Set (or clear) superseded_by on one relationship entry of an existing memory, in place, without touching its content or triggering re-embedding, and return the backend JSON response."""
    response = await request(
        "PATCH",
        "/memory/relationship",
        json_body={
            "key": memory_id,
            "subjectEntityId": subjectEntityId,
            "predicate": predicate,
            "objectEntityId": objectEntityId,
            "superseded_by": superseded_by,
        },
    )
    return response.text


@mcp.tool()
async def recall(
    query: str = Field(description="A natural-language question or topic to search memory for, e.g. 'what does David think about tabs vs spaces'."),
    k: int = Field(default=5, description="Maximum number of results to return."),
    memory_type: MemoryType | None = Field(default=None, description="Restrict results to this memory type."),
    entity: str | None = Field(default=None, description="Restrict results to memories mentioning this entity id."),
    since: str | None = Field(default=None, description="ISO-8601 timestamp; only include memories created at or after this time."),
    until: str | None = Field(default=None, description="ISO-8601 timestamp; only include memories created at or before this time."),
    hydrate: bool = Field(default=False, description="Return each result's full memory body instead of a short snippet."),
    as_: str | None = Field(default=None, validation_alias="as", description="Restrict results to memories visible to this entity id: its acl must be empty (unrestricted) or contain the id directly or a group it is member_of (one level, no recursion). Omit for unrestricted access."),
) -> str:
    """Search memory by semantic similarity to query and return ranked results as JSON, each with a key, score, type, and either a snippet or (with hydrate=True) the full memory body."""
    filters: dict[str, str] = {}
    if memory_type is not None:
        filters["type"] = memory_type
    if entity is not None:
        filters["entity"] = entity
    if since is not None:
        filters["since"] = since
    if until is not None:
        filters["until"] = until

    body: dict[str, object] = {"query": query, "k": k, "hydrate": hydrate}
    if filters:
        body["filters"] = filters
    if as_ is not None:
        body["as"] = as_

    response = await request("POST", "/memory/search", json_body=body)
    return response.text


@mcp.tool()
async def browse_all() -> str:
    """List all stored memory and entity keys and return them as JSON with optional previews."""
    response = await request("GET", "/memory/list")
    payload = response.json()
    keys = [
        {"key": item["Key"], "preview": item.get("Preview")}
        for item in payload.get("Contents", [])
        if "Key" in item
    ]
    return json.dumps({"keys": keys})


@mcp.tool()
async def forget(key: str) -> str:
    """Delete a stored memory or entity record by key and return JSON confirming the deletion."""
    _ = await request("DELETE", "/memory/", json_body={"key": key})
    return json.dumps({"message": "deleted", "key": key})


def main() -> None:
    mcp.run(transport="stdio")
