"""Minimal async HTTP client for the Connectome backend's /api/connectome routes.

Talks to the Go backend directly over HTTP, independent of connectomemcp - no
import of the MCP package and no MCP/stdio transport in the path. This
intentionally re-derives just enough of connectomemcp.server's memory
frontmatter format (see connectomemcp.server.format_memory) to write a valid
memory. Entity-record merge logic (stub entities, member_of) lives server-side
in the Go backend's /entity/relationship endpoint - see UpsertEntityRelationship
in backend/resources/entity.go - rather than being re-implemented here, so any
number of clients share one primitive instead of drifting Python
implementations.

Callers are responsible for loading their own environment (e.g. via
python-dotenv) before constructing a client; this package does not look for a
.env file itself, since a relative path to one wouldn't generalize once this
is installed rather than vendored.
"""

import json
import os
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from io import BytesIO
from typing import Literal, NotRequired, TypedDict, get_args
from urllib.parse import quote

import httpx
import yaml

DEFAULT_API_BASE_URL = "http://localhost:8080/api/connectome"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_USER_AGENT = "connectomeclient/0.1.0"
# The default `generated.by` actor. Actors follow one convention:
# `<producer>/<version>` for agents and tools, `human:<id>` for people,
# `process:<id>` for jobs.
DEFAULT_GENERATED_BY = f"connectomeclient/{version('connectomeclient')}"
MEMORY_SCHEMA_VERSION = "connectome/memory/0.2"
# The only older schema version upgrade_memory_metadata knows how to read.
LEGACY_MEMORY_SCHEMA_VERSION = "connectome/memory/0.1"

MemoryType = Literal["note", "fact", "preference", "event"]
MEMORY_TYPES: tuple[str, ...] = get_args(MemoryType)
DEFAULT_MEMORY_TYPE: MemoryType = "note"

# Mirrors RelationshipKind/DEFAULT_RELATIONSHIP_KIND in connectomemcp.server: the
# truth-status of a relationship claim, distinct from MemoryType even though
# "fact" is a value both happen to share.
RelationshipKind = Literal["fact", "hypothesis", "rumor"]
DEFAULT_RELATIONSHIP_KIND: RelationshipKind = "fact"


class Relationship(TypedDict):
    """A directed relationship between entities, mirroring connectomemcp.server's
    Relationship model - see format_memory."""

    subjectEntityId: str
    predicate: str
    objectEntityId: NotRequired[str | None]
    kind: NotRequired[RelationshipKind]


class Source(TypedDict):
    """Where a memory's content was extracted from, mirroring
    connectomemcp.server's Source model (OKF's `sources` shape)."""

    id: NotRequired[str]
    resource: NotRequired[str]
    title: NotRequired[str]


def upgrade_memory_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Return memory frontmatter in the current schema's shape.

    Mirrors connectomemcp.server.upgrade_memory_metadata: a 0.1 document
    gains `generated` derived from its `source` block (by = source.type, at =
    source.created_at) and loses `source`; a `verified` written as a bare
    mapping becomes a one-element list. The backend already does this for
    get_memory and hydrated recall, so this is for frontmatter read some
    other way, such as straight from the blob store.
    """
    upgraded = dict(metadata)
    if isinstance(upgraded.get("verified"), dict):
        upgraded["verified"] = [upgraded["verified"]]
    if upgraded.get("version") != LEGACY_MEMORY_SCHEMA_VERSION:
        return upgraded
    upgraded["version"] = MEMORY_SCHEMA_VERSION
    source = upgraded.pop("source", None)
    if "generated" not in upgraded and isinstance(source, dict):
        upgraded["generated"] = {
            "by": source.get("type"),
            "at": source.get("created_at") or upgraded.get("created_at"),
        }
    return upgraded


def entity_key(entity_id: str) -> str:
    """Blob store key for an entity id, matching the ent_<id>.json convention
    entityKey uses in backend/resources/entity.go."""
    return f"ent_{entity_id}.json"


def validate_occurred_at(occurred_at: str | None) -> None:
    """Raise ValueError if occurred_at is present but not RFC3339.

    Mirrors connectomemcp.server.validate_occurred_at. Unlike created_at
    (which the backend silently falls back to `now` for on a parse failure -
    see createdAtOrNow in backend/resources/memoryDocument.go), a malformed
    occurred_at is rejected outright here rather than silently dropped or
    reinterpreted as created_at.
    """
    if occurred_at is None:
        return
    try:
        _ = datetime.fromisoformat(occurred_at)
    except ValueError as exc:
        raise ValueError(f"invalid occurred_at {occurred_at!r} (want RFC3339): {exc}") from exc


class ConnectomeClient:
    """Minimal async HTTP client for the connectome backend's /api/connectome routes.

    See the module docstring for the broader design rationale.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        user_agent: str = DEFAULT_USER_AGENT,
        generated_by: str | None = None,
        source_type: str | None = None,
        cf_access_client_id: str | None = None,
        cf_access_client_secret: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("CONNECTOME_API_BASE_URL", DEFAULT_API_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.getenv("CONNECTOME_API_KEY") or os.getenv("DAYBID_API_KEY") or os.getenv("apikey")
        self.timeout = timeout
        self.user_agent = user_agent
        # Recorded as generated.by on every memory this client writes.
        # source_type is the 0.1 name for the same thing, kept so existing
        # callers keep working; generated_by wins when both are given.
        self.generated_by = generated_by or source_type or DEFAULT_GENERATED_BY
        # Cloudflare Access service-token credentials, for backends behind
        # Access; sent only when both halves are present.
        self.cf_access_client_id = cf_access_client_id or os.getenv("CF_ACCESS_CLIENT_ID")
        self.cf_access_client_secret = cf_access_client_secret or os.getenv("CF_ACCESS_CLIENT_SECRET")

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": self.user_agent}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.cf_access_client_id and self.cf_access_client_secret:
            headers["CF-Access-Client-Id"] = self.cf_access_client_id
            headers["CF-Access-Client-Secret"] = self.cf_access_client_secret
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        data: dict[str, str] | None = None,
        files: dict[str, tuple[str, BytesIO, str]] | None = None,
        json_body: dict[str, object] | None = None,
    ) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(
                method,
                self._url(path),
                headers=self._headers(),
                params=params,
                data=data,
                files=files,
                json=json_body,
            )
            _ = response.raise_for_status()
            return response

    async def remember(
        self,
        content: str,
        memory_type: MemoryType = DEFAULT_MEMORY_TYPE,
        entities: list[str] | None = None,
        relationships: list[Relationship] | None = None,
        acl: list[str] | None = None,
        tome: str | None = None,
        occurred_at: str | None = None,
        derived_from: str | None = None,
        generated_by: str | None = None,
        sources: list[Source] | None = None,
    ) -> dict[str, str]:
        """Write a memory document and return its key.

        tome scopes the memory to its own namespace; it must match the tome
        passed to any later get_memory/forget of this key. Omit for the
        default tome.

        occurred_at is an optional RFC3339 timestamp for when the memory's
        described event actually happened, distinct from created_at (always
        now). Omit when the event time is unknown or is simply now.

        derived_from is the key of another memory (e.g. 'mem_abc.md') this one
        is a facet of: one entity's own, diverging version of the root
        memory's content. It is not extraction provenance - use sources for
        that.

        sources records what the memory was extracted from, such as the
        transcript chunk a fact was drawn from: each entry has an optional id,
        resource (a memory key or URL) and title.

        generated_by overrides, for this memory only, the actor the client
        was constructed with: '<producer>/<version>' for an agent or tool,
        'human:<id>' for a person, 'process:<id>' for a job.

        occurred_at and derived_from are always written, as null when
        omitted, while sources is left out entirely when empty; key order
        matches connectomemcp.server.format_memory's front matter for the
        same arguments."""
        validate_occurred_at(occurred_at)
        memory_id = f"mem_{uuid.uuid4()}.md"
        now = datetime.now(UTC).isoformat()
        metadata: dict[str, object] = {
            "version": MEMORY_SCHEMA_VERSION,
            "id": memory_id,
            "type": memory_type,
            "created_at": now,
            "occurred_at": occurred_at,
            "generated": {"by": generated_by or self.generated_by, "at": now},
        }
        if sources:
            metadata["sources"] = [dict(source) for source in sources]
        metadata |= {
            "entities": list(entities or []),
            "relationships": [dict(relationship) for relationship in (relationships or [])],
        }
        if acl is not None:
            metadata["acl"] = acl
        metadata["derived_from"] = derived_from
        yaml_data = yaml.dump(metadata, sort_keys=False).strip("\n")
        document = f"---\n{yaml_data}\n---\n{content}\n"

        _ = await self._request(
            "POST",
            "/memory/",
            files={"file": (memory_id, BytesIO(document.encode("utf-8")), "text/plain; charset=utf-8")},
            data={"tome": tome} if tome is not None else None,
        )
        return {"key": memory_id}

    async def recall(
        self,
        query: str,
        k: int = 5,
        memory_type: MemoryType | None = None,
        entity: str | None = None,
        since: str | None = None,
        until: str | None = None,
        occurred_since: str | None = None,
        occurred_until: str | None = None,
        hydrate: bool = False,
        as_: str | None = None,
        tome: str | None = None,
        ranking: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Search memory by semantic similarity to query and return ranked results.

        ranking overrides the backend's ranking settings for this request only
        (vector_weight, text_weight, rrf_k, text_query, bm25_k1, bm25_b,
        text_max_df); omitted keys keep the backend's defaults. The response's
        "ranking" holds the settings that actually ran.

        tome restricts results to that tome; omit to search the default tome.

        occurred_since/occurred_until bound occurred_at (when the described
        event happened), distinct from since/until (which bound created_at).
        Memories with no occurred_at are excluded whenever either is set."""
        filters: dict[str, str] = {}
        if memory_type is not None:
            filters["type"] = memory_type
        if entity is not None:
            filters["entity"] = entity
        if since is not None:
            filters["since"] = since
        if until is not None:
            filters["until"] = until
        if occurred_since is not None:
            filters["occurred_since"] = occurred_since
        if occurred_until is not None:
            filters["occurred_until"] = occurred_until
        if tome is not None:
            filters["tome"] = tome

        body: dict[str, object] = {"query": query, "k": k, "hydrate": hydrate}
        if filters:
            body["filters"] = filters
        if as_ is not None:
            body["as"] = as_
        if ranking:
            body["ranking"] = ranking

        response = await self._request("POST", "/memory/search", json_body=body)
        return response.json()

    async def assert_relationship(
        self,
        subject_entity_id: str,
        predicate: str,
        object_entity_id: str | None = None,
        kind: str | None = None,
        subject_kind: str | None = None,
        subject_meta: dict[str, object] | None = None,
        tome: str | None = None,
    ) -> dict[str, object]:
        """Assert a relationship between two entities via the Go backend's shared
        merge endpoint, so ent_*.json state (stub entities, member_of) ends up the
        same as an equivalent connectomemcp.server.remember call would produce - see
        UpsertEntityRelationship in backend/resources/entity.go.

        subject_kind/subject_meta optionally stamp the subject entity's
        kind/meta fields in the same call (overwrite and shallow-merge
        respectively, mirroring connectomemcp.server's Entity.kind/Entity.meta)."""
        body: dict[str, object] = {"subjectEntityId": subject_entity_id, "predicate": predicate}
        if object_entity_id is not None:
            body["objectEntityId"] = object_entity_id
        if kind is not None:
            body["kind"] = kind
        if subject_kind is not None:
            body["subjectKind"] = subject_kind
        if subject_meta is not None:
            body["subjectMeta"] = subject_meta
        if tome is not None:
            body["tome"] = tome

        response = await self._request("POST", "/entity/relationship", json_body=body)
        return response.json()

    async def supersede_relationship(
        self,
        key: str,
        subject_entity_id: str,
        predicate: str,
        object_entity_id: str | None = None,
        superseded_by: str | None = None,
        tome: str | None = None,
    ) -> dict[str, object]:
        """Set (or clear) superseded_by on one relationship entry of an existing memory.

        Hits the Go backend's PATCH /memory/relationship route (see
        supersedeRelationship in backend/resources/memoryResource.go), mirroring
        connectomemcp.server.supersede_relationship. Pass superseded_by=None to
        clear a prior supersession and mark the relationship current again.
        """
        body: dict[str, object] = {
            "key": key,
            "subjectEntityId": subject_entity_id,
            "predicate": predicate,
            "objectEntityId": object_entity_id,
            "superseded_by": superseded_by,
        }
        if tome is not None:
            body["tome"] = tome
        response = await self._request("PATCH", "/memory/relationship", json_body=body)
        return response.json()

    async def verify(
        self,
        key: str,
        by: str,
        at: str | None = None,
        tome: str | None = None,
    ) -> dict[str, object]:
        """Record that an actor confirmed an existing memory's content.

        Appends {by, at} to the memory's verified list via the Go backend's
        PATCH /memory/verified route, in place and without re-embedding,
        mirroring connectomemcp.server.verify. by follows the actor
        convention ('human:<id>', 'process:<id>', '<producer>/<version>'); at
        is an RFC3339 timestamp and defaults to now.
        """
        body: dict[str, object] = {"key": key, "by": by}
        if at is not None:
            body["at"] = at
        if tome is not None:
            body["tome"] = tome
        response = await self._request("PATCH", "/memory/verified", json_body=body)
        return response.json()

    async def get_memory(self, key: str, tome: str | None = None) -> dict[str, object]:
        """Fetch a stored memory document or entity record by key.

        tome must match the tome the record was written with."""
        params = {"key": key}
        if tome is not None:
            params["tome"] = tome
        response = await self._request("GET", "/memory/", params=params)
        return response.json()

    async def get_entity(self, entity_id: str, tome: str | None = None) -> dict[str, object] | None:
        """Fetch an ent_<id>.json entity record, or None if it doesn't exist yet."""
        try:
            payload = await self.get_memory(entity_key(entity_id), tome=tome)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        content = payload["content"]
        assert isinstance(content, str)
        return json.loads(content)

    async def browse_all(self, tome: str | None = None) -> dict[str, object]:
        """List all stored memory and entity keys in a tome (the default tome when omitted)."""
        params = {"tome": tome} if tome is not None else None
        response = await self._request("GET", "/memory/list", params=params)
        return response.json()

    async def forget(self, key: str, tome: str | None = None) -> dict[str, str]:
        """Delete a stored memory or entity record by key.

        tome must match the tome the record was written with."""
        body: dict[str, object] = {"key": key}
        if tome is not None:
            body["tome"] = tome
        _ = await self._request("DELETE", "/memory/", json_body=body)
        return {"message": "deleted", "key": key}

    async def destroy_tome(self, tome: str, confirm: bool = False) -> dict[str, object]:
        """Destroy every memory, entity record, and embedding stored under a tome.

        Irreversible. The backend refuses to destroy the default tome, and
        refuses any tome id that doesn't start with "temp-" or "test-" unless
        confirm=True - see DestroyTome in backend/resources/tome.go. Those
        refusals surface as httpx.HTTPStatusError (403)."""
        if not tome:
            raise ValueError("refusing to destroy the default tome; pass a non-empty tome id")
        params = {"confirm": "true"} if confirm else None
        response = await self._request("DELETE", f"/tome/{quote(tome, safe='')}", params=params)
        return response.json()
