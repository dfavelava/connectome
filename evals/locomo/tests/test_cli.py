import argparse
import asyncio

import pytest

from locomo_eval import cli
from locomo_eval.dataset import parse_sample
from tests.test_dataset import RAW_SAMPLE


class FakeClient:
    """In-memory stand-in for ConnectomeClient: recall returns the tome's
    memories whose text shares a word with the query, in insertion order."""

    base_url = "http://fake/api/connectome"

    def __init__(self, *args, **kwargs):
        self.tomes: dict[str, dict[str, str]] = {}
        self.destroyed: list[str] = []
        self.remember_calls: list[dict] = []
        self.fail_recall = False

    async def remember(self, content, memory_type, tome, occurred_at):
        self.remember_calls.append({"content": content, "tome": tome, "occurred_at": occurred_at})
        key = f"mem_{len(self.remember_calls)}.md"
        self.tomes.setdefault(tome, {})[key] = content
        return {"key": key}

    async def recall(self, query, k, tome):
        if self.fail_recall:
            raise RuntimeError("boom")
        words = set(query.lower().strip("?").split())
        hits = [key for key, text in self.tomes.get(tome, {}).items() if words & set(text.lower().split())]
        return {"results": [{"key": key} for key in hits[:k]]}

    async def destroy_tome(self, tome, confirm=False):
        self.destroyed.append(tome)
        self.tomes.pop(tome, None)
        return {}


@pytest.fixture
def client():
    return FakeClient()


def args(**overrides):
    defaults = {"ks": [1, 5], "concurrency": 2, "no_occurred_at": False, "keep_tomes": False}
    return argparse.Namespace(**{**defaults, **overrides})


def test_run_ingests_scores_and_destroys_tome(client):
    sample = parse_sample(RAW_SAMPLE)
    results, skipped = asyncio.run(cli.run(client, args(), [sample], "r1"))

    assert [c["content"] for c in client.remember_calls][0] == "[1:56 pm on 8 May, 2023] Caroline: Hey Mel!"
    assert client.remember_calls[0]["tome"] == "temp-locomo-r1-conv-1"
    assert client.remember_calls[0]["occurred_at"] == "2023-05-08T13:56:00+00:00"
    assert client.destroyed == ["temp-locomo-r1-conv-1"]

    assert [r.question for r in results] == ["When?", "Adversarial?"]
    assert all(r.retrieved == () or set(r.retrieved) <= {"D1:1", "D1:2", "D2:1"} for r in results)
    assert skipped == {"no_evidence": 1, "unknown_evidence_only": 0, "unknown_evidence_ids": 0}


def test_run_destroys_tome_on_failure(client):
    sample = parse_sample(RAW_SAMPLE)
    client.fail_recall = True
    with pytest.raises(RuntimeError):
        asyncio.run(cli.run(client, args(), [sample], "r2"))
    assert client.destroyed == ["temp-locomo-r2-conv-1"]


def test_no_occurred_at_and_keep_tomes(client):
    sample = parse_sample(RAW_SAMPLE)
    asyncio.run(cli.run(client, args(no_occurred_at=True, keep_tomes=True), [sample], "r3"))
    assert all(c["occurred_at"] is None for c in client.remember_calls)
    assert client.destroyed == []


def test_ks_parsing():
    assert cli._ks("10,1,5,5") == [1, 5, 10]
    with pytest.raises(argparse.ArgumentTypeError):
        cli._ks("0,5")
    with pytest.raises(argparse.ArgumentTypeError):
        cli._ks("51")
