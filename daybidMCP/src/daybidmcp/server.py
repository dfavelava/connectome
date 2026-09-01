import httpx
import json
import os
import uuid
import yaml

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from mcp.server import MCPServer
from pydantic import BaseModel, Field

DEFAULT_API_BASE_URL = "http://localhost:8080/connectome"
DEFAULT_TIMEOUT_SECONDS = 30.0
USER_AGENT = "connectome/0.1.0"
MEMORY_SCHEMA_VERSION = "connectome/memory/0.1"

_ = load_dotenv(Path(__file__).resolve().parents[2] / ".env")

mcp = MCPServer("connectome")

class Entity(BaseModel):
    id: str
    name: str | None = None

class EntityWithMemories(Entity):
    memory_ids: list[str] | None = None

class Relationship(BaseModel):
    subjectEntityId: str
    predicate: str
    objectEntityId: str | None = None

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
) -> str:
    metadata = MemoryMetadata(
        id=id,
        type="",
        created_at=created_at,
        source=MemorySource(type="", created_at=created_at),
        entities=[entity.id for entity in entities],
        relationships=relationships,
    )
    yaml_data = yaml.dump(metadata.model_dump(mode="json"), sort_keys=False)
    return f"""
    ---
    {yaml_data}
    ---
    {content}
    """

def format_entity(entity: EntityWithMemories) -> str:
    return entity.model_dump_json(indent=2)


def merge_memory_ids(existing: list[str] | None, new_memory_id: str) -> list[str]:
    memory_ids = list(existing or [])
    if new_memory_id not in memory_ids:
        memory_ids.append(new_memory_id)
    return memory_ids

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


@mcp.tool()
async def get_memory(key: str) -> str:
    """Retrieve memory content by key."""
    response = await request("GET", "/memory/", params={"key": key})
    return response.text


@mcp.tool()
async def remember(
    content: str,
    entities: list[Entity] | None = None,
    relationships: list[Relationship] | None = None,
) -> str:
    """Store memory content and return the generated key."""
    # TODO: Check if memory already exists and update if found
    memory_id = f"{generate_id('mem')}.md"
    now = datetime.now(UTC)
    memory_entities = entities or []
    memory_relationships = relationships or []

    memory = format_memory(
        memory_id,
        content,
        memory_entities,
        memory_relationships,
        now.isoformat(),
    )

    entity_keys = [f"ent_{entity.id}.json" for entity in memory_entities]
    existing_entity_contents = await batch_read(entity_keys)

    files: list[tuple[str, tuple[str, BytesIO, str]]] = [
        ("file", (memory_id, BytesIO(memory.encode("utf-8")), "text/plain; charset=utf-8"))
    ]

    for entity, entity_id in zip(memory_entities, entity_keys):
        existing_entity = existing_entity_contents.get(entity_id)
        existing_memory_ids: list[str] | None = None
        if existing_entity:
            existing_memory_ids = EntityWithMemories.model_validate_json(existing_entity).memory_ids

        entity_with_memories = EntityWithMemories(
            **entity.model_dump(),
            memory_ids=merge_memory_ids(existing_memory_ids, memory_id),
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
            "memory": json.loads(memory),
            "entity_keys": entity_keys,
        }
    )


@mcp.tool()
async def browse_all() -> str:
    """List stored memory keys."""
    response = await request("GET", "/memory/list")
    payload = response.json()
    keys = [item["Key"] for item in payload.get("Contents", []) if "Key" in item]
    return json.dumps({"keys": keys})


@mcp.tool()
async def forget(key: str) -> str:
    """Delete a memory by key."""
    _ = await request("DELETE", "/memory/", json_body={"key": key})
    return json.dumps({"message": "deleted", "key": key})


def main() -> None:
    mcp.run(transport="stdio")
