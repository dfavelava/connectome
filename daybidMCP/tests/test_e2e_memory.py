"""End-to-end test for the memory lifecycle.

Exercises ``remember -> browse_all -> get_memory -> forget`` from the MCP server
against a real instance of the Go backend running with the local filesystem
memory manager pointed at a temporary ``$HOME``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
API_KEY = "e2e-test-token"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_ready(url: str, proc: subprocess.Popen[str], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"backend exited early with code {proc.returncode}:\n{proc.stdout.read() if proc.stdout else ''}"
            )
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError("backend did not become ready in time")


@dataclass
class Backend:
    base_url: str
    connectome_dir: Path


@pytest.fixture
def backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Backend:
    go = shutil.which("go")
    if go is None:
        pytest.skip("go toolchain not available")

    home = tmp_path / "home"
    connectome_dir = home / ".connectome"
    connectome_dir.mkdir(parents=True)

    binary = tmp_path / "connectome-backend"
    build = subprocess.run(
        [go, "build", "-o", str(binary), "."],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        pytest.fail(f"go build failed:\n{build.stderr}")

    port = _free_port()
    env = {
        **os.environ,
        "HOME": str(home),
        "PORT": str(port),
        "MEMORY_MANAGER": "local",
        "apikey": API_KEY,
        "GIN_MODE": "release",
    }
    proc = subprocess.Popen(
        [str(binary)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_until_ready(f"http://127.0.0.1:{port}/api/", proc)
    except Exception:
        proc.kill()
        raise

    base_url = f"http://127.0.0.1:{port}/api/connectome"
    monkeypatch.setenv("DAYBID_API_BASE_URL", base_url)
    monkeypatch.setenv("DAYBID_API_KEY", API_KEY)

    try:
        yield Backend(base_url=base_url, connectome_dir=connectome_dir)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _parse_frontmatter(document: str) -> tuple[dict, str]:
    """Split a stored memory document into (metadata, body).

    Tolerates the leading indentation the current ``format_memory`` emits on the
    fence lines and the first frontmatter line.
    """
    lines = document.splitlines()
    fences = [i for i, line in enumerate(lines) if line.strip() == "---"]
    assert len(fences) >= 2, f"expected two frontmatter fences, got: {document!r}"

    fm_lines = lines[fences[0] + 1 : fences[1]]
    body_lines = lines[fences[1] + 1 :]
    if fm_lines:
        fm_lines[0] = fm_lines[0].lstrip()

    metadata = yaml.safe_load("\n".join(fm_lines))
    return metadata, "\n".join(body_lines).strip()


def test_remember_browse_get_forget_roundtrip(backend: Backend) -> None:
    asyncio.run(_roundtrip(backend))


def test_remember_recall_roundtrip(backend: Backend) -> None:
    asyncio.run(_recall_roundtrip(backend))


async def _roundtrip(backend: Backend) -> None:
    from daybidmcp.server import Entity, browse_all, forget, get_memory, remember

    ada = Entity(id="ada", name="Ada Lovelace")

    # --- remember -----------------------------------------------------------
    first = json.loads(
        await remember(
            content="Ada enjoys analytical engines.",
            entities=[ada],
            relationships=[],
            memory_type="fact",
        )
    )
    memory_key = first["key"]
    assert memory_key.startswith("mem_") and memory_key.endswith(".md")
    assert first["entity_keys"] == ["ent_ada.json"]

    memory_path = backend.connectome_dir / memory_key
    assert memory_path.is_file(), "memory file was not written to the local FS"

    metadata, body = _parse_frontmatter(memory_path.read_text())
    assert metadata["version"] == "connectome/memory/0.1"
    assert metadata["id"] == memory_key
    assert metadata["type"] == "fact"
    assert metadata["entities"] == ["ada"]
    assert metadata["relationships"] == []
    assert metadata["created_at"]
    assert "source" in metadata
    assert body == "Ada enjoys analytical engines."

    entity_path = backend.connectome_dir / "ent_ada.json"
    assert entity_path.is_file(), "entity record was not created"
    entity_record = json.loads(entity_path.read_text())
    assert entity_record["id"] == "ada"
    assert entity_record["name"] == "Ada Lovelace"
    assert entity_record["memory_ids"] == [memory_key]

    # --- remember again: memory_ids merge on the shared entity -------------
    second = json.loads(
        await remember(
            content="Ada wrote the first algorithm.",
            entities=[ada],
            relationships=[],
            memory_type="fact",
        )
    )
    second_key = second["key"]
    assert second_key != memory_key

    entity_record = json.loads(entity_path.read_text())
    assert entity_record["memory_ids"] == [memory_key, second_key]

    # --- browse_all -------------------------------------------------------
    listed = json.loads(await browse_all())
    listed_keys = {item["key"] for item in listed["keys"]}
    assert {memory_key, second_key, "ent_ada.json"} <= listed_keys

    # --- get_memory ------------------------------------------------------
    fetched = json.loads(await get_memory(memory_key))
    assert "Ada enjoys analytical engines." in fetched["content"]

    # --- forget --------------------------------------------------------
    deleted = json.loads(await forget(memory_key))
    assert deleted == {"message": "deleted", "key": memory_key}
    assert not memory_path.exists(), "memory file still present after forget"

    remaining = {item["key"] for item in json.loads(await browse_all())["keys"]}
    assert memory_key not in remaining
    assert second_key in remaining
    assert "ent_ada.json" in remaining


async def _recall_roundtrip(backend: Backend) -> None:
    from daybidmcp.server import Entity, forget, recall, remember

    ada = Entity(id="ada", name="Ada Lovelace")
    grace = Entity(id="grace", name="Grace Hopper")

    tea = json.loads(
        await remember(
            content="David prefers tea over coffee in the afternoon.",
            entities=[],
            relationships=[],
            memory_type="preference",
        )
    )
    ada_fact = json.loads(
        await remember(
            content="Ada Lovelace wrote the first published algorithm.",
            entities=[ada],
            relationships=[],
            memory_type="fact",
        )
    )
    grace_fact = json.loads(
        await remember(
            content="Grace Hopper popularized the term debugging.",
            entities=[grace],
            relationships=[],
            memory_type="fact",
        )
    )
    tea_key, ada_key, grace_key = tea["key"], ada_fact["key"], grace_fact["key"]

    # recall is an @mcp.tool()-decorated function: its parameters default to
    # Field(...) sentinels that only resolve to real values when the MCP
    # protocol layer binds arguments from JSON. Calling it directly, as this
    # test does, means every argument must be passed explicitly - an omitted
    # one stays a raw FieldInfo object and fails to JSON-encode.
    try:
        # --- plain semantic search surfaces the relevant memory first ------
        results = json.loads(
            await recall(query="What does David like to drink?", k=5, memory_type=None, entity=None, since=None, until=None, hydrate=False)
        )["results"]
        keys = [r["key"] for r in results]
        assert keys, "expected at least one recall result"
        assert keys[0] == tea_key, f"expected {tea_key} ranked first, got {keys}"
        assert results[0]["snippet"]
        assert "content" not in results[0]

        # --- type filter narrows to facts only -----------------------------
        fact_keys = {
            r["key"]
            for r in json.loads(
                await recall(query="algorithms and debugging", k=5, memory_type="fact", entity=None, since=None, until=None, hydrate=False)
            )["results"]
        }
        assert tea_key not in fact_keys
        assert ada_key in fact_keys or grace_key in fact_keys

        # --- entity filter narrows to memories mentioning that entity ------
        ada_keys = {
            r["key"]
            for r in json.loads(
                await recall(query="Ada Lovelace", k=5, memory_type=None, entity="ada", since=None, until=None, hydrate=False)
            )["results"]
        }
        assert ada_key in ada_keys
        assert grace_key not in ada_keys

        # --- hydrate returns the full memory body, not just a snippet ------
        hydrated = json.loads(
            await recall(query="What does David like to drink?", k=1, memory_type=None, entity=None, since=None, until=None, hydrate=True)
        )["results"]
        assert hydrated
        assert "David prefers tea over coffee in the afternoon." in hydrated[0]["content"]
    finally:
        for key in (tea_key, ada_key, grace_key):
            await forget(key)
