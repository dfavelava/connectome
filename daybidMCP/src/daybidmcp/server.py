import json
import os
from io import BytesIO
from pathlib import Path

import httpx
from mcp.server import MCPServer
from dotenv import load_dotenv

DEFAULT_API_BASE_URL = "http://localhost:8080/connectome"
DEFAULT_TIMEOUT_SECONDS = 30.0
USER_AGENT = "connectome/0.1.0"

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

mcp = MCPServer("connectome")


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


async def request(
    method: str,
    path: str,
    *,
    params: dict[str, str] | None = None,
    files: dict[str, tuple[str, BytesIO, str]] | None = None,
    json_body: dict[str, str] | None = None,
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


@mcp.tool()
async def get_memory(key: str) -> str:
    """Retrieve memory content by key."""
    response = await request("GET", "/memory/", params={"key": key})
    return response.text


@mcp.tool()
async def remember(key: str, content: str) -> str:
    """Store memory content under a key."""
    file_obj = BytesIO(content.encode("utf-8"))
    _ = await request(
        "POST",
        "/memory/",
        files={"file": (key, file_obj, "text/plain; charset=utf-8")},
    )
    return json.dumps({"message": "stored", "key": key})


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
