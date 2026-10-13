import asyncio

import pytest

from westmarches_eval import cli


class FakeClient:
    """Records the calls ingest and retrieve make; recall returns a fixed
    ranking of campaign ids, by key."""

    base_url = "http://fake"

    def __init__(self, ranking=None):
        self.calls = []
        self.ranking = ranking or {}
        self.keys = {}

    async def assert_relationship(self, subject, predicate, obj, tome=None):
        self.calls.append(("assert", subject, predicate, obj, tome))
        return {}

    async def remember(self, content, **kwargs):
        key = f"mem_{len(self.keys)}.md"
        self.keys[kwargs["sources"][0]["id"]] = key
        self.calls.append(("remember", key, kwargs))
        return {"key": key}

    async def supersede_relationship(self, key, subject, predicate, obj, superseded_by=None, tome=None):
        self.calls.append(("supersede", key, subject, predicate, obj, superseded_by, tome))
        return {}

    async def recall(self, question, k=5, tome=None, hydrate=False, as_=None):
        self.calls.append(("recall", question, k, tome, hydrate, as_))
        ids = self.ranking.get(question, [])
        return {"results": [{"key": self.keys.get(i, "mem_unknown.md"), "snippet": i} for i in ids]}


def test_ingest_writes_what_the_bot_would(small):
    client = FakeClient()
    keys = asyncio.run(cli.ingest(client, small, "temp-westmarches-t"))
    assert len(keys) == len(small.memories)
    asserts = [c for c in client.calls if c[0] == "assert"]
    assert ("assert", "ann", "member_of", "party-a", "temp-westmarches-t") in asserts
    assert len(asserts) == 3
    remembered = {kwargs["sources"][0]["id"]: kwargs for kind, _, kwargs in (c for c in client.calls if c[0] == "remember")}
    assert remembered["rumor"]["acl"] == []
    assert remembered["event-a"]["acl"] == ["party-a", "gm"]
    assert remembered["event-a"]["derived_from"] == keys["event"]
    assert remembered["new-captain"]["occurred_at"] == "2026-03-08T00:00:00Z"
    assert remembered["rumor"]["relationships"][0]["kind"] == "rumor"
    supersedes = [c for c in client.calls if c[0] == "supersede"]
    assert supersedes == [("supersede", keys["old-captain"], "vane", "captain_of", "watch", keys["new-captain"], "temp-westmarches-t")]


def test_retrieve_asks_as_the_viewer_in_both_modes(small):
    client = FakeClient(ranking={"Who is captain?": ["new-captain", "old-captain"]})
    keys = asyncio.run(cli.ingest(client, small, "t"))
    questions = asyncio.run(cli.retrieve(client, small, "t", keys, 10, 4))
    recalls = [c for c in client.calls if c[0] == "recall" and c[1] == "Who is captain?"]
    assert {(c[4], c[5]) for c in recalls} == {(False, "bea"), (True, "bea")}
    (stale,) = [q for q in questions if q["id"] == "q-stale"]
    assert stale["hits"] == {"hydrate_off": ["new-captain", "old-captain"], "hydrate_on": ["new-captain", "old-captain"]}
    assert stale["contexts"]["hydrate_off"] == ["new-captain", "old-captain"]


def test_retrieve_refuses_keys_it_did_not_ingest(small):
    client = FakeClient(ranking={"Who is captain?": ["not-ingested"]})
    keys = asyncio.run(cli.ingest(client, small, "t"))
    with pytest.raises(RuntimeError, match="not ingested"):
        asyncio.run(cli.retrieve(client, small, "t", keys, 10, 4))


def test_leaked_takes_the_worse_mode():
    run = {"retrieval": {"hydrate_off": {"leakage": {"leaked": 0}}, "hydrate_on": {"leakage": {"leaked": 2}}}}
    assert cli.leaked(run) == 2


def test_tome_names_are_scratch_tomes():
    assert cli.tome_for("20261013T120000Z") == "temp-westmarches-20261013t120000z"


def test_answer_models_go_together():
    with pytest.raises(SystemExit):
        cli.parse_args(["--answer-model", "ollama:a"])
    args = cli.parse_args(["--answer-model", "ollama:a", "--judge-model", "ollama:j", "--ks", "10,5"])
    assert args.ks == [5, 10]
