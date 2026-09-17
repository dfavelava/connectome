"""Acceptance test for phase 1.2F.

Exercises a slice wider than test_e2e_memory.py's unit/integration-style
tests - remember -> recall -> supersede_relationship - against the shared
`acceptance_backend`, isolated in its own `temp_tome`. Nothing here calls
`forget`: the whole tome, and everything seeded under it, is destroyed by the
`temp_tome` fixture when the test ends, whether it passes or fails.
"""

from __future__ import annotations

import asyncio
import json

import yaml


def _parse_frontmatter(document: str) -> dict:
    """Extract the YAML frontmatter from a stored memory document."""
    lines = document.splitlines()
    fences = [i for i, line in enumerate(lines) if line.strip() == "---"]
    assert len(fences) >= 2, f"expected two frontmatter fences, got: {document!r}"

    fm_lines = lines[fences[0] + 1 : fences[1]]
    if fm_lines:
        fm_lines[0] = fm_lines[0].lstrip()
    return yaml.safe_load("\n".join(fm_lines))


def test_remember_recall_supersede_relationship_in_a_temp_tome(temp_tome: str) -> None:
    asyncio.run(_remember_recall_supersede(temp_tome))


async def _remember_recall_supersede(tome: str) -> None:
    from daybidmcp.server import (
        Entity,
        Relationship,
        get_memory,
        recall,
        remember,
        supersede_relationship,
    )

    party = Entity(id="party-a", name="Party A")

    sighting = json.loads(
        await remember(
            content="Party A spots the lighthouse keeper's boat moored at Ashvale's north dock.",
            entities=[party],
            relationships=[
                Relationship(subjectEntityId="party-a", predicate="located_at", objectEntityId="ashvale-north-dock")
            ],
            memory_type="fact",
            acl=None,
            derived_from=None,
            tome=tome,
        )
    )
    sighting_key = sighting["key"]

    # --- recall (seeded via remember) surfaces the memory, scoped to this tome ---
    results = json.loads(
        await recall(
            query="where did Party A spot the lighthouse keeper's boat",
            k=5,
            memory_type=None,
            entity="party-a",
            since=None,
            until=None,
            hydrate=True,
            as_=None,
            tome=tome,
        )
    )["results"]
    assert results, "expected recall to surface the seeded memory"
    assert results[0]["key"] == sighting_key
    assert "Ashvale" in results[0]["content"]

    # --- neither the default tome nor another tome can see into this one ---------
    for other_tome in (None, "west-marches"):
        other_keys = {
            r["key"]
            for r in json.loads(
                await recall(
                    query="lighthouse keeper's boat at Ashvale",
                    k=5,
                    memory_type=None,
                    entity=None,
                    since=None,
                    until=None,
                    hydrate=False,
                    as_=None,
                    tome=other_tome,
                )
            )["results"]
        }
        assert sighting_key not in other_keys

    # --- supersede the located_at claim once the party moves on ------------------
    await supersede_relationship(
        memory_id=sighting_key,
        subjectEntityId="party-a",
        predicate="located_at",
        objectEntityId="ashvale-north-dock",
        superseded_by="mem_party_moved_on.md",
        tome=tome,
    )

    patched = json.loads(await get_memory(sighting_key, tome=tome))
    metadata = _parse_frontmatter(patched["content"])
    relationships = {(r["subjectEntityId"], r["objectEntityId"]): r for r in metadata["relationships"]}
    assert relationships[("party-a", "ashvale-north-dock")]["superseded_by"] == "mem_party_moved_on.md"
