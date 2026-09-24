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
    monkeypatch.setenv("CONNECTOME_API_BASE_URL", base_url)
    monkeypatch.setenv("CONNECTOME_API_KEY", API_KEY)

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


def test_remember_creates_stub_entities_and_writes_acl(backend: Backend) -> None:
    asyncio.run(_stub_entities_and_acl(backend))


def test_remember_merges_kind_and_meta_onto_entity_record(backend: Backend) -> None:
    asyncio.run(_kind_and_meta_merge(backend))


def test_supersede_relationship_patches_in_place(backend: Backend) -> None:
    asyncio.run(_supersede_relationship(backend))


def test_recall_as_scopes_results_by_acl_and_member_of(backend: Backend) -> None:
    asyncio.run(_recall_as_acl_scope(backend))


def test_facet_recalls_correctly_for_its_own_audience_alongside_root(backend: Backend) -> None:
    asyncio.run(_facet_recall(backend))


def test_tome_scopes_remember_get_memory_forget_and_entity_records(backend: Backend) -> None:
    asyncio.run(_tome_scoping(backend))


def test_recall_tome_scopes_results_to_the_default_tome(backend: Backend) -> None:
    asyncio.run(_recall_tome_scope(backend))


def test_recall_occurred_at_filters_are_distinct_from_created_at(backend: Backend) -> None:
    asyncio.run(_recall_occurred_at_filter(backend))


async def _roundtrip(backend: Backend) -> None:
    from connectomemcp.server import Entity, browse_all, forget, get_memory, remember

    ada = Entity(id="ada", name="Ada Lovelace")

    # --- remember -----------------------------------------------------------
    first = json.loads(
        await remember(
            content="Ada enjoys analytical engines.",
            entities=[ada],
            relationships=[],
            memory_type="fact",
            acl=None,
            derived_from=None,
            tome=None,
            occurred_at=None,
        )
    )
    memory_key = first["key"]
    second_key: str | None = None
    try:
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

        # --- remember again: memory_ids merge on the shared entity ---------
        second = json.loads(
            await remember(
                content="Ada wrote the first algorithm.",
                entities=[ada],
                relationships=[],
                memory_type="fact",
                acl=None,
                derived_from=None,
                tome=None,
                occurred_at=None,
            )
        )
        second_key = second["key"]
        assert second_key != memory_key

        entity_record = json.loads(entity_path.read_text())
        assert entity_record["memory_ids"] == [memory_key, second_key]

        # --- browse_all -----------------------------------------------------
        listed = json.loads(await browse_all(tome=None))
        listed_keys = {item["key"] for item in listed["keys"]}
        assert {memory_key, second_key, "ent_ada.json"} <= listed_keys

        # --- get_memory ------------------------------------------------------
        fetched = json.loads(await get_memory(memory_key, tome=None))
        assert "Ada enjoys analytical engines." in fetched["content"]

        # --- forget -----------------------------------------------------------
        deleted = json.loads(await forget(memory_key, tome=None))
        assert deleted == {"message": "deleted", "key": memory_key}
        assert not memory_path.exists(), "memory file still present after forget"

        remaining = {item["key"] for item in json.loads(await browse_all(tome=None))["keys"]}
        assert memory_key not in remaining
        assert second_key in remaining
        assert "ent_ada.json" in remaining
    finally:
        # memory_key is already forgotten above; second_key and the entity
        # record are only forgotten here so a real (non-ephemeral) Postgres
        # instance backing the embeddings table isn't left with an orphaned
        # row when this test runs against it.
        if second_key is not None:
            await forget(second_key, tome=None)
        await forget("ent_ada.json", tome=None)


async def _kind_and_meta_merge(backend: Backend) -> None:
    from connectomemcp.server import Entity, forget, get_memory, remember

    cave = Entity(id="cave", kind="location", meta={"status": "rumored"})

    first = json.loads(
        await remember(
            content="Adventurers hear rumors of a cave to the north.",
            entities=[cave],
            relationships=[],
            memory_type="note",
            acl=None,
            derived_from=None,
            tome=None,
            occurred_at=None,
        )
    )
    key = first["key"]
    second_key: str | None = None
    try:
        entity_path = backend.connectome_dir / "ent_cave.json"
        entity_record = json.loads(entity_path.read_text())
        assert entity_record["kind"] == "location"
        assert entity_record["meta"] == {"status": "rumored"}

        # --- remember again: kind is replaced, meta shallow-merges ---------
        scouted_cave = Entity(id="cave", kind="dungeon", meta={"status": "scouted"})
        second = json.loads(
            await remember(
                content="Scouts confirm the cave and map its entrance.",
                entities=[scouted_cave],
                relationships=[],
                memory_type="note",
                acl=None,
                derived_from=None,
                tome=None,
                occurred_at=None,
            )
        )
        second_key = second["key"]

        entity_record = json.loads(entity_path.read_text())
        assert entity_record["kind"] == "dungeon"
        assert entity_record["meta"] == {"status": "scouted"}

        # --- get_memory returns the entity record's kind/meta unchanged ----
        fetched = json.loads(await get_memory("ent_cave.json", tome=None))
        fetched_entity = json.loads(fetched["content"])
        assert fetched_entity["kind"] == "dungeon"
        assert fetched_entity["meta"] == {"status": "scouted"}
    finally:
        await forget(key, tome=None)
        if second_key is not None:
            await forget(second_key, tome=None)
        await forget("ent_cave.json", tome=None)


async def _recall_roundtrip(backend: Backend) -> None:
    from connectomemcp.server import Entity, forget, recall, remember

    ada = Entity(id="ada", name="Ada Lovelace")
    grace = Entity(id="grace", name="Grace Hopper")

    tea = json.loads(
        await remember(
            content="David prefers tea over coffee in the afternoon.",
            entities=[],
            relationships=[],
            memory_type="preference",
            acl=None,
            derived_from=None,
            tome=None,
            occurred_at=None,
        )
    )
    ada_fact = json.loads(
        await remember(
            content="Ada Lovelace wrote the first published algorithm.",
            entities=[ada],
            relationships=[],
            memory_type="fact",
            acl=None,
            derived_from=None,
            tome=None,
            occurred_at=None,
        )
    )
    grace_fact = json.loads(
        await remember(
            content="Grace Hopper popularized the term debugging.",
            entities=[grace],
            relationships=[],
            memory_type="fact",
            acl=None,
            derived_from=None,
            tome=None,
            occurred_at=None,
        )
    )
    tea_key, ada_key, grace_key = tea["key"], ada_fact["key"], grace_fact["key"]

    # recall (like remember, above) is an @mcp.tool()-decorated function: its
    # parameters default to Field(...) sentinels that only resolve to real
    # values when the MCP protocol layer binds arguments from JSON. Calling
    # it directly, as this test does, means every argument must be passed
    # explicitly - an omitted one stays a raw FieldInfo object and fails to
    # JSON-encode (or, for remember's acl, fails MemoryMetadata validation).
    try:
        # --- plain semantic search surfaces the relevant memory first ------
        results = json.loads(
            await recall(query="What does David like to drink?", k=5, memory_type=None, entity=None, since=None, until=None, occurred_since=None, occurred_until=None, hydrate=False, as_=None, tome=None)
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
                await recall(query="algorithms and debugging", k=5, memory_type="fact", entity=None, since=None, until=None, occurred_since=None, occurred_until=None, hydrate=False, as_=None, tome=None)
            )["results"]
        }
        assert tea_key not in fact_keys
        assert ada_key in fact_keys or grace_key in fact_keys

        # --- entity filter narrows to memories mentioning that entity ------
        ada_keys = {
            r["key"]
            for r in json.loads(
                await recall(query="Ada Lovelace", k=5, memory_type=None, entity="ada", since=None, until=None, occurred_since=None, occurred_until=None, hydrate=False, as_=None, tome=None)
            )["results"]
        }
        assert ada_key in ada_keys
        assert grace_key not in ada_keys

        # --- hydrate returns the full memory body, not just a snippet ------
        hydrated = json.loads(
            await recall(query="What does David like to drink?", k=1, memory_type=None, entity=None, since=None, until=None, occurred_since=None, occurred_until=None, hydrate=True, as_=None, tome=None)
        )["results"]
        assert hydrated
        assert "David prefers tea over coffee in the afternoon." in hydrated[0]["content"]
    finally:
        for key in (tea_key, ada_key, grace_key):
            await forget(key, tome=None)


async def _recall_tome_scope(backend: Backend) -> None:
    from connectomemcp.server import forget, recall, remember

    # tome=None lands this memory in the default tome (tome_id ""), same as
    # everything written before tomes existed.
    stored = json.loads(
        await remember(
            content="The lighthouse keeper at Ashvale hums an old sea shanty every dawn.",
            entities=[],
            relationships=[],
            memory_type="note",
            acl=None,
            derived_from=None,
            tome=None,
            occurred_at=None,
        )
    )
    memory_key = stored["key"]

    try:
        # --- omitting tome finds it in the default tome, as before tomes -------
        # --- existed -------------------------------------------------------
        default_keys = {
            r["key"]
            for r in json.loads(
                await recall(
                    query="lighthouse keeper humming at dawn",
                    k=5,
                    memory_type=None,
                    entity=None,
                    since=None,
                    until=None,
                    occurred_since=None,
                    occurred_until=None,
                    hydrate=False,
                    as_=None,
                    tome=None,
                )
            )["results"]
        }
        assert memory_key in default_keys

        # --- a different tome sees none of the default tome's memories ---------
        other_tome_keys = {
            r["key"]
            for r in json.loads(
                await recall(
                    query="lighthouse keeper humming at dawn",
                    k=5,
                    memory_type=None,
                    entity=None,
                    since=None,
                    until=None,
                    occurred_since=None,
                    occurred_until=None,
                    hydrate=False,
                    as_=None,
                    tome="west-marches",
                )
            )["results"]
        }
        assert memory_key not in other_tome_keys
    finally:
        await forget(memory_key, tome=None)


async def _stub_entities_and_acl(backend: Backend) -> None:
    from connectomemcp.server import Entity, Relationship, forget, get_memory, remember

    david = Entity(id="david", name="David")

    result = json.loads(
        await remember(
            content="David likes tea, which Grace also enjoys.",
            entities=[david],
            relationships=[
                Relationship(subjectEntityId="david", predicate="likes", objectEntityId="tea"),
                Relationship(subjectEntityId="grace", predicate="likes", objectEntityId="tea"),
            ],
            memory_type="fact",
            acl=["GM"],
            derived_from=None,
            tome=None,
            occurred_at=None,
        )
    )
    memory_key = result["key"]

    try:
        # --- relationship endpoints not in `entities` get bare stub records -
        assert set(result["entity_keys"]) == {"ent_david.json", "ent_tea.json", "ent_grace.json"}

        tea_entity = json.loads(await get_memory("ent_tea.json", tome=None))
        tea_record = json.loads(tea_entity["content"])
        assert tea_record["id"] == "tea"
        assert tea_record["name"] is None
        assert tea_record["memory_ids"] == [memory_key]

        grace_entity = json.loads(await get_memory("ent_grace.json", tome=None))
        grace_record = json.loads(grace_entity["content"])
        assert grace_record["id"] == "grace"
        assert grace_record["name"] is None

        # --- acl round-trips through the stored frontmatter -----------------
        fetched = json.loads(await get_memory(memory_key, tome=None))
        metadata, _ = _parse_frontmatter(fetched["content"])
        assert metadata["acl"] == ["GM"]
    finally:
        await forget(memory_key, tome=None)
        await forget("ent_david.json", tome=None)
        await forget("ent_tea.json", tome=None)
        await forget("ent_grace.json", tome=None)


async def _supersede_relationship(backend: Backend) -> None:
    from connectomemcp.server import (
        Entity,
        Relationship,
        forget,
        get_memory,
        remember,
        supersede_relationship,
    )

    david = Entity(id="david", name="David")

    result = json.loads(
        await remember(
            content="David likes tea, and also likes coffee.",
            entities=[david],
            relationships=[
                Relationship(subjectEntityId="david", predicate="likes", objectEntityId="tea"),
                Relationship(subjectEntityId="david", predicate="likes", objectEntityId="coffee"),
            ],
            memory_type="fact",
            acl=None,
            derived_from=None,
            tome=None,
            occurred_at=None,
        )
    )
    memory_key = result["key"]

    try:
        original = json.loads(await get_memory(memory_key, tome=None))
        original_metadata, original_body = _parse_frontmatter(original["content"])

        # --- superseding one relationship leaves the other, and the content, alone ---
        await supersede_relationship(
            memory_id=memory_key,
            subjectEntityId="david",
            predicate="likes",
            objectEntityId="tea",
            superseded_by="mem_correction.md",
            tome=None,
        )

        patched = json.loads(await get_memory(memory_key, tome=None))
        metadata, body = _parse_frontmatter(patched["content"])

        relationships = {(r["subjectEntityId"], r["objectEntityId"]): r for r in metadata["relationships"]}
        assert relationships[("david", "tea")]["superseded_by"] == "mem_correction.md"
        assert relationships[("david", "coffee")]["superseded_by"] is None
        assert body == original_body
        assert metadata["created_at"] == original_metadata["created_at"]
        assert metadata["id"] == original_metadata["id"]

        # --- clearing a supersession with superseded_by=None un-supersedes it ---
        await supersede_relationship(
            memory_id=memory_key,
            subjectEntityId="david",
            predicate="likes",
            objectEntityId="tea",
            superseded_by=None,
            tome=None,
        )
        cleared = json.loads(await get_memory(memory_key, tome=None))
        cleared_metadata, _ = _parse_frontmatter(cleared["content"])
        cleared_relationships = {(r["subjectEntityId"], r["objectEntityId"]): r for r in cleared_metadata["relationships"]}
        assert cleared_relationships[("david", "tea")]["superseded_by"] is None

        # --- a relationship that doesn't exist on the memory is an error --------
        with pytest.raises(httpx.HTTPStatusError):
            await supersede_relationship(
                memory_id=memory_key,
                subjectEntityId="david",
                predicate="dislikes",
                objectEntityId="tea",
                superseded_by="mem_correction.md",
                tome=None,
            )

        # --- a memory key that doesn't exist is an error -------------------------
        with pytest.raises(httpx.HTTPStatusError):
            await supersede_relationship(
                memory_id="mem_does_not_exist.md",
                subjectEntityId="david",
                predicate="likes",
                objectEntityId="tea",
                superseded_by="mem_correction.md",
                tome=None,
            )
    finally:
        await forget(memory_key, tome=None)
        await forget("ent_david.json", tome=None)
        await forget("ent_tea.json", tome=None)
        await forget("ent_coffee.json", tome=None)


async def _recall_as_acl_scope(backend: Backend) -> None:
    from connectomemcp.server import (
        Entity,
        Relationship,
        forget,
        get_memory,
        recall,
        remember,
    )

    alice = Entity(id="alice", name="Alice")

    membership = json.loads(
        await remember(
            content="Alice joins Party A.",
            entities=[alice],
            relationships=[Relationship(subjectEntityId="alice", predicate="member_of", objectEntityId="Party A")],
            memory_type="fact",
            acl=None,
            derived_from=None,
            tome=None,
            occurred_at=None,
        )
    )
    membership_key = membership["key"]

    open_memory = json.loads(
        await remember(
            content="The ruins north of Ashvale are said to be cursed.",
            entities=[],
            relationships=[],
            memory_type="note",
            acl=None,
            derived_from=None,
            tome=None,
            occurred_at=None,
        )
    )
    party_memory = json.loads(
        await remember(
            content="Party A found a hidden door in the ruins north of Ashvale.",
            entities=[],
            relationships=[],
            memory_type="note",
            acl=["Party A"],
            derived_from=None,
            tome=None,
            occurred_at=None,
        )
    )
    gm_memory = json.loads(
        await remember(
            content="The ruins north of Ashvale actually hide a rival GM plot twist.",
            entities=[],
            relationships=[],
            memory_type="note",
            acl=["GM"],
            derived_from=None,
            tome=None,
            occurred_at=None,
        )
    )
    open_key, party_key, gm_key = open_memory["key"], party_memory["key"], gm_memory["key"]

    try:
        # --- member_of, special-cased from the relationship, merges onto the ---
        # --- entity record without introducing a new primitive -----------------
        alice_entity = json.loads(await get_memory("ent_alice.json", tome=None))
        alice_record = json.loads(alice_entity["content"])
        assert alice_record["member_of"] == ["Party A"]

        # --- as="alice" sees unrestricted and Party A memories, not GM's -------
        scoped = {
            r["key"]
            for r in json.loads(
                await recall(
                    query="what do we know about the ruins north of Ashvale",
                    k=10,
                    memory_type=None,
                    entity=None,
                    since=None,
                    until=None,
                    occurred_since=None,
                    occurred_until=None,
                    hydrate=False,
                    as_="alice",
                    tome=None,
                )
            )["results"]
        }
        assert open_key in scoped
        assert party_key in scoped
        assert gm_key not in scoped

        # --- omitting `as` applies no acl filtering at all ---------------------
        unscoped = {
            r["key"]
            for r in json.loads(
                await recall(
                    query="what do we know about the ruins north of Ashvale",
                    k=10,
                    memory_type=None,
                    entity=None,
                    since=None,
                    until=None,
                    occurred_since=None,
                    occurred_until=None,
                    hydrate=False,
                    as_=None,
                    tome=None,
                )
            )["results"]
        }
        assert {open_key, party_key, gm_key} <= unscoped
    finally:
        for key in (membership_key, open_key, party_key, gm_key):
            await forget(key, tome=None)
        await forget("ent_alice.json", tome=None)


async def _facet_recall(backend: Backend) -> None:
    from connectomemcp.server import forget, get_memory, recall, remember

    root = json.loads(
        await remember(
            content="The party found a strange amulet in the ruins north of Ashvale.",
            entities=[],
            relationships=[],
            memory_type="event",
            acl=None,
            derived_from=None,
            tome=None,
            occurred_at=None,
        )
    )
    root_key = root["key"]

    facet = json.loads(
        await remember(
            content="Grace secretly suspects the amulet is cursed and means to hide it from the rest of the party.",
            entities=[],
            relationships=[],
            memory_type="event",
            acl=["GM"],
            derived_from=root_key,
            tome=None,
            occurred_at=None,
        )
    )
    facet_key = facet["key"]

    try:
        # --- derived_from round-trips through the stored frontmatter -----------
        fetched_facet = json.loads(await get_memory(facet_key, tome=None))
        facet_metadata, _ = _parse_frontmatter(fetched_facet["content"])
        assert facet_metadata["derived_from"] == root_key

        fetched_root = json.loads(await get_memory(root_key, tome=None))
        root_metadata, _ = _parse_frontmatter(fetched_root["content"])
        assert root_metadata["derived_from"] is None

        # --- a facet is chunked, embedded, and ACL-filtered like any other ------
        # --- memory: the GM audience recalls both the root and its facet -------
        gm_keys = {
            r["key"]
            for r in json.loads(
                await recall(
                    query="what happened with the amulet in the ruins",
                    k=10,
                    memory_type=None,
                    entity=None,
                    since=None,
                    until=None,
                    occurred_since=None,
                    occurred_until=None,
                    hydrate=False,
                    as_="GM",
                    tome=None,
                )
            )["results"]
        }
        assert root_key in gm_keys
        assert facet_key in gm_keys

        # --- an audience outside the facet's acl sees the root but not the -----
        # --- facet, which is narrower ---------------------------------------
        party_keys = {
            r["key"]
            for r in json.loads(
                await recall(
                    query="what happened with the amulet in the ruins",
                    k=10,
                    memory_type=None,
                    entity=None,
                    since=None,
                    until=None,
                    occurred_since=None,
                    occurred_until=None,
                    hydrate=False,
                    as_="alice",
                    tome=None,
                )
            )["results"]
        }
        assert root_key in party_keys
        assert facet_key not in party_keys
    finally:
        await forget(root_key, tome=None)
        await forget(facet_key, tome=None)


async def _tome_scoping(backend: Backend) -> None:
    from connectomemcp.server import (
        Entity,
        Relationship,
        browse_all,
        forget,
        get_memory,
        recall,
        remember,
    )

    tome = "west-marches"
    ada = Entity(id="ada", name="Ada Lovelace")

    result = json.loads(
        await remember(
            content="In the West Marches, Ada charts the ruins.",
            entities=[ada],
            relationships=[Relationship(subjectEntityId="ada", predicate="member_of", objectEntityId="cartographers")],
            memory_type="fact",
            acl=None,
            derived_from=None,
            tome=tome,
            occurred_at=None,
        )
    )
    memory_key = result["key"]

    try:
        # --- the memory round-trips when read back under the same tome ------
        fetched = json.loads(await get_memory(memory_key, tome=tome))
        assert "West Marches" in fetched["content"]

        # --- the same key resolves to nothing under the default tome --------
        with pytest.raises(httpx.HTTPStatusError):
            await get_memory(memory_key, tome=None)

        # --- or under a different tome ---------------------------------------
        with pytest.raises(httpx.HTTPStatusError):
            await get_memory(memory_key, tome="other-tome")

        # --- entity records written via remember's member_of merge are also -
        # --- scoped to the tome, not reachable from outside it ---------------
        ada_entity = json.loads(await get_memory("ent_ada.json", tome=tome))
        ada_record = json.loads(ada_entity["content"])
        assert ada_record["member_of"] == ["cartographers"]

        with pytest.raises(httpx.HTTPStatusError):
            await get_memory("ent_ada.json", tome=None)

        # --- browse_all lists the tome's keys bare, the same shape remember -
        # --- returned, so each one reads straight back with the same tome ---
        tome_keys = {item["key"] for item in json.loads(await browse_all(tome=tome))["keys"]}
        assert {memory_key, "ent_ada.json", "ent_cartographers.json"} <= tome_keys
        for key in tome_keys:
            await get_memory(key, tome=tome)

        # --- recall returns the same bare key, which reads back with the tome
        recalled = json.loads(
            await recall(query="Ada charts the ruins", k=5, memory_type=None, entity=None, since=None, until=None, occurred_since=None, occurred_until=None, hydrate=False, as_=None, tome=tome)
        )["results"]
        assert memory_key in {r["key"] for r in recalled}
        assert "West Marches" in json.loads(await get_memory(memory_key, tome=tome))["content"]

        # --- and stays invisible to a browse_all of the default tome --------
        default_keys = {item["key"] for item in json.loads(await browse_all(tome=None))["keys"]}
        assert f"tomes/{tome}/{memory_key}" not in default_keys
        assert memory_key not in default_keys
    finally:
        await forget(memory_key, tome=tome)
        await forget("ent_ada.json", tome=tome)
        await forget("ent_cartographers.json", tome=tome)


async def _recall_occurred_at_filter(backend: Backend) -> None:
    from connectomemcp.server import forget, get_memory, recall, remember

    # old_key's occurred_at is far in the past even though it's written (and
    # thus created_at'd) now, proving occurred_since/occurred_until bound
    # occurred_at rather than created_at.
    old_occurred_at = "1969-07-20T20:17:00+00:00"
    old = json.loads(
        await remember(
            content="Ashvale's founding charter was signed the day the tide ran red.",
            entities=[],
            relationships=[],
            memory_type="event",
            acl=None,
            derived_from=None,
            tome=None,
            occurred_at=old_occurred_at,
        )
    )
    # recent has no occurred_at at all.
    recent = json.loads(
        await remember(
            content="Ashvale's harbor market reopened this morning after repairs.",
            entities=[],
            relationships=[],
            memory_type="event",
            acl=None,
            derived_from=None,
            tome=None,
            occurred_at=None,
        )
    )
    old_key, recent_key = old["key"], recent["key"]

    try:
        # --- get_memory surfaces occurred_at when present -------------------
        fetched = json.loads(await get_memory(old_key, tome=None))
        content = fetched["content"]
        assert old_occurred_at.replace("+00:00", "") in content or "1969-07-20" in content

        # --- occurred_until bounded to the past excludes the undated memory,
        # --- (undated is never treated as a match) and finds the old one ----
        bounded_results = json.loads(
            await recall(
                query="Ashvale",
                k=10,
                memory_type=None,
                entity=None,
                since=None,
                until=None,
                occurred_since=None,
                occurred_until="1970-01-01T00:00:00Z",
                hydrate=False,
                as_=None,
                tome=None,
            )
        )["results"]
        bounded_keys = {r["key"] for r in bounded_results}
        assert old_key in bounded_keys
        assert recent_key not in bounded_keys

        # --- occurred_since bounded to the recent past excludes both: the
        # --- old memory fails the bound, and the undated one has no
        # --- occurred_at to satisfy it -----------------------------------
        recent_bound_results = json.loads(
            await recall(
                query="Ashvale",
                k=10,
                memory_type=None,
                entity=None,
                since=None,
                until=None,
                occurred_since="2000-01-01T00:00:00Z",
                occurred_until=None,
                hydrate=False,
                as_=None,
                tome=None,
            )
        )["results"]
        recent_bound_keys = {r["key"] for r in recent_bound_results}
        assert old_key not in recent_bound_keys
        assert recent_key not in recent_bound_keys
    finally:
        await forget(old_key, tome=None)
        await forget(recent_key, tome=None)
