import asyncio
import copy
import hashlib
import json
import re

import pytest

from locomo_eval import cli, extraction
from locomo_eval.dataset import CATEGORY_NAMES, parse_sample
from locomo_eval.extraction import (
    EXTRACT,
    EXTRACT_OPTIONS,
    EXTRACT_VERSION,
    EXTRACTION_SCHEMA,
    FAILED,
    OK,
    Entity,
    ExtractionCache,
    ExtractionParseError,
    OllamaExtractor,
    extract_sample,
    extraction_prompt,
    load_extractions,
    normalize_entity_id,
    parse_extraction,
    pending_sessions,
    sessions_of,
)
from locomo_eval.llm import Completion, SamplingOptions
from locomo_eval.prompts import PROMPTS_DIR, load_prompt
from tests.test_dataset import RAW_SAMPLE

DATASET_SHA = "d" * 64
MODEL = "ollama:fake-extractor"

RAW = copy.deepcopy(RAW_SAMPLE)
RAW["conversation"]["session_2_date_time"] = "10:00 am on 15 May, 2023"
RAW["conversation"]["session_2"] = [
    {"speaker": "Caroline", "dia_id": "D2:1", "text": "I went to the LGBTQ support group yesterday."},
    {"speaker": "Melanie", "dia_id": "D2:2", "text": "That's great!"},
]
RAW["qa"] = [
    {"question": "When did Caroline go to the LGBTQ support group?", "answer": "14 May 2023", "evidence": ["D2:1"], "category": 2},
    {"question": "What is Melanie's favourite zebra-striped umbrella?", "adversarial_answer": "the blue one", "evidence": ["D1:2"], "category": 5},
]
SAMPLE = parse_sample(RAW)


def reply(memories=None, entities=None) -> str:
    return json.dumps({"entities": entities or [], "memories": memories or []})


def memory(content="Caroline said hello.", sources=("D1:1",), entities=(), relationships=(), memory_type="fact", occurred_at=None) -> dict:
    return {
        "content": content,
        "memory_type": memory_type,
        "occurred_at": occurred_at,
        "entities": list(entities),
        "relationships": list(relationships),
        "source_dia_ids": list(sources),
    }


class FakeLLM:
    """Stands in for LLMClient. Replies from `script` in order when given,
    else with one valid memory citing the transcript's first dialog id and
    an entity for Caroline. Every call reports 1000 input and 100 output tokens."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls: list[dict] = []

    async def complete(self, model, system, user, *, stage="default", format=None, options=None) -> Completion:
        self.calls.append({"model": model, "system": system, "user": user, "stage": stage, "format": format, "options": options})
        await asyncio.sleep(0)
        if self.script:
            return Completion(self.script.pop(0), 1000, 100)
        first = re.search(r"^(D\d+:\d+) ", user, re.MULTILINE).group(1)
        text = reply(
            [memory(f"Caroline spoke in {first}.", sources=[first], entities=["caroline"])],
            [{"id": "caroline", "name": "Caroline", "kind": "person"}],
        )
        return Completion(text, 1000, 100)


def extractor(llm, **kwargs):
    return OllamaExtractor(llm, MODEL, load_prompt(EXTRACT_VERSION), **kwargs)


def run(coro):
    return asyncio.run(coro)


# --- sessions and prompt --------------------------------------------------------


def test_sessions_of_groups_turns_in_order():
    sessions = sessions_of(SAMPLE.sample_id, SAMPLE.turns)
    assert [s.number for s in sessions] == [1, 2]
    assert [t.dia_id for t in sessions[0].turns] == ["D1:1", "D1:2"]
    assert sessions[1].date == "10:00 am on 15 May, 2023"
    assert sessions[1].occurred_at == "2023-05-15T10:00:00+00:00"
    assert sessions[1].dia_ids == {"D2:1", "D2:2"}


def test_extraction_prompt_has_transcript_date_and_entities():
    session = sessions_of(SAMPLE.sample_id, SAMPLE.turns)[1]
    text = extraction_prompt(load_prompt(EXTRACT_VERSION), session, [Entity("caroline", "Caroline", "person")])
    assert "Session date: 10:00 am on 15 May, 2023" in text
    assert "D2:1 Caroline: I went to the LGBTQ support group yesterday." in text
    assert "caroline | Caroline | person" in text
    assert "$" not in text
    empty = extraction_prompt(load_prompt(EXTRACT_VERSION), session, [])
    assert "(none yet)" in empty


def test_extract_prompt_is_committed_and_generic():
    prompt = load_prompt(EXTRACT_VERSION)
    assert prompt.sha256 == hashlib.sha256((PROMPTS_DIR / f"{EXTRACT_VERSION}.txt").read_bytes()).hexdigest()
    lowered = prompt.text.lower()
    for name in CATEGORY_NAMES.values():
        assert name not in lowered
    assert "question" not in lowered


def test_no_question_text_reaches_any_prompt(tmp_path):
    llm = FakeLLM()
    cache = ExtractionCache(tmp_path / "cache.jsonl")
    tally = run(extract_sample(extractor(llm), cache, DATASET_SHA, SAMPLE.sample_id, SAMPLE.turns))
    assert tally.extracted == 2
    prompts = [c["system"] + "\n" + c["user"] for c in llm.calls]
    assert prompts
    for item in SAMPLE.qa:
        for text in (item.question, item.answer, item.adversarial_answer):
            if text:
                assert all(text not in p for p in prompts), text


# --- validation -------------------------------------------------------------


def test_parse_extraction_valid_reply():
    text = reply(
        [
            memory(
                "Caroline went to the LGBTQ support group on 14 May 2023.",
                sources=["D2:1", "D2:1"],
                entities=["Caroline", "support group"],
                relationships=[{"subjectEntityId": "caroline", "predicate": "attended", "objectEntityId": "Support Group", "kind": "fact"}],
                memory_type="event",
                occurred_at="2023-05-14",
            )
        ],
        [{"id": "Support Group", "name": "LGBTQ support group", "kind": "organization"}],
    )
    result = parse_extraction(text, {"D2:1", "D2:2"}, ["caroline"])
    assert result.entities == (Entity("support-group", "LGBTQ support group", "organization"),)
    (m,) = result.memories
    assert m.source_dia_ids == ("D2:1",)
    assert m.entities == ("caroline", "support-group")
    assert m.occurred_at == "2023-05-14T00:00:00+00:00"
    assert m.relationships == ({"subjectEntityId": "caroline", "predicate": "attended", "objectEntityId": "support-group", "kind": "fact"},)
    assert sum(result.dropped.values()) == 0


def test_parse_extraction_drops_invalid_ids_and_counts_them():
    text = reply(
        [
            memory(
                sources=["D2:1", "D9:9", "D1:1", "D:2:2", "nonsense"],
                entities=["caroline", "ghost"],
                relationships=[
                    {"subjectEntityId": "ghost", "predicate": "haunts", "objectEntityId": None, "kind": "rumor"},
                    {"subjectEntityId": "caroline", "predicate": "knows", "objectEntityId": "ghost", "kind": "hypothesis"},
                    {"subjectEntityId": "caroline", "predicate": "is_happy", "objectEntityId": None, "kind": "fact"},
                ],
                occurred_at="last Tuesday",
            )
        ],
        [{"id": "!!!", "name": "?", "kind": "object"}],
    )
    result = parse_extraction(text, {"D2:1", "D2:2"}, ["caroline"])
    (m,) = result.memories
    assert m.source_dia_ids == ("D2:1", "D2:2")
    assert m.entities == ("caroline",)
    assert m.relationships == ({"subjectEntityId": "caroline", "predicate": "is_happy", "objectEntityId": None, "kind": "fact"},)
    assert m.occurred_at is None
    assert result.dropped == {"source_dia_ids": 3, "entity_ids": 1, "entity_refs": 1, "relationships": 2, "occurred_at": 1}


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        json.dumps({"memories": []}),
        json.dumps({"entities": [], "memories": {}}),
        reply([memory(memory_type="opinion")]),
        reply([memory(content="  ")]),
        reply([{**memory(), "source_dia_ids": "D1:1"}]),
        reply([{**memory(), "occurred_at": 20230514}]),
        reply([memory(relationships=[{"subjectEntityId": "a", "predicate": "x", "objectEntityId": None, "kind": "certain"}])]),
        reply(entities=[{"id": "a", "name": "A"}]),
    ],
)
def test_parse_extraction_rejects_malformed_replies(text):
    with pytest.raises(ExtractionParseError):
        parse_extraction(text, {"D1:1"})


def test_normalize_entity_id():
    assert normalize_entity_id("  Support Group! ") == "support-group"
    assert normalize_entity_id("Mel's_Dog") == "mel-s-dog"
    assert normalize_entity_id("--") == ""


# --- extractor retries -------------------------------------------------------


def test_extractor_retries_invalid_reply_with_next_seed():
    session = sessions_of(SAMPLE.sample_id, SAMPLE.turns)[0]
    llm = FakeLLM(["not json", reply([memory()])])
    result = run(extractor(llm).extract(session, []))
    assert result.status == OK
    assert result.attempts == 2
    assert (result.input_tokens, result.output_tokens) == (2000, 200)
    assert [c["options"].seed for c in llm.calls] == [EXTRACT_OPTIONS.seed, EXTRACT_OPTIONS.seed + 1]
    assert all(c["format"] == EXTRACTION_SCHEMA and c["stage"] == EXTRACT for c in llm.calls)


def test_extractor_gives_up_after_bounded_attempts():
    session = sessions_of(SAMPLE.sample_id, SAMPLE.turns)[0]
    llm = FakeLLM(["nope"] * 5)
    result = run(extractor(llm, attempts=2).extract(session, []))
    assert result.status == FAILED
    assert result.extraction is None
    assert result.attempts == 2
    assert len(llm.calls) == 2
    assert "not JSON" in result.error


def test_failed_session_does_not_abort_the_sample(tmp_path):
    llm = FakeLLM(["bad", "bad", "bad"])
    cache = ExtractionCache(tmp_path / "cache.jsonl")
    tally = run(extract_sample(extractor(llm), cache, DATASET_SHA, SAMPLE.sample_id, SAMPLE.turns))
    assert tally.failed == [1]
    assert tally.extracted == 1
    assert len(llm.calls) == 4
    records = [json.loads(line) for line in (tmp_path / "cache.jsonl").read_text().splitlines()]
    assert [(r["session"], r["status"]) for r in records] == [(1, FAILED), (2, OK)]
    assert records[0]["error"]


# --- cache ------------------------------------------------------------------


def test_cache_records_provenance_usage_and_config(tmp_path):
    llm = FakeLLM()
    cache = ExtractionCache(tmp_path / "cache.jsonl")
    run(extract_sample(extractor(llm), cache, DATASET_SHA, SAMPLE.sample_id, SAMPLE.turns, model_digest="sha256:abc"))
    records = [json.loads(line) for line in (tmp_path / "cache.jsonl").read_text().splitlines()]
    assert len(records) == 2
    prompt = load_prompt(EXTRACT_VERSION)
    for number, record in enumerate(records, start=1):
        assert record["dataset_sha256"] == DATASET_SHA
        assert record["sample_id"] == "conv-1"
        assert record["session"] == number
        assert record["extractor_model"] == MODEL
        assert record["model_digest"] == "sha256:abc"
        assert (record["prompt_version"], record["prompt_sha256"]) == (EXTRACT_VERSION, prompt.sha256)
        assert record["status"] == OK
        assert record["memories"][0]["source_dia_ids"] == [f"D{number}:1"]
        assert (record["input_tokens"], record["output_tokens"]) == (1000, 100)
        assert record["seconds"] >= 0
    # Session 2 is told about the entity session 1 recorded.
    assert records[0]["known_entity_ids"] == []
    assert records[1]["known_entity_ids"] == ["caroline"]
    assert "caroline | Caroline | person" in llm.calls[1]["user"]
    loaded = load_extractions(tmp_path / "cache.jsonl", DATASET_SHA, "conv-1", MODEL, prompt)
    assert [r["session"] for r in loaded] == [1, 2]


def test_second_run_is_all_cache_hits(tmp_path):
    path = tmp_path / "cache.jsonl"
    run(extract_sample(extractor(FakeLLM()), ExtractionCache(path), DATASET_SHA, SAMPLE.sample_id, SAMPLE.turns))
    llm = FakeLLM()
    cache = ExtractionCache(path)
    assert pending_sessions(cache, extractor(llm), DATASET_SHA, [(SAMPLE.sample_id, SAMPLE.turns)]) == 0
    tally = run(extract_sample(extractor(llm), cache, DATASET_SHA, SAMPLE.sample_id, SAMPLE.turns))
    assert llm.calls == []
    assert (tally.cached, tally.extracted, tally.memories) == (2, 0, 2)
    assert len(path.read_text().splitlines()) == 2


def test_resume_retries_failed_sessions_only(tmp_path):
    path = tmp_path / "cache.jsonl"
    run(extract_sample(extractor(FakeLLM(["bad"] * 3)), ExtractionCache(path), DATASET_SHA, SAMPLE.sample_id, SAMPLE.turns))
    llm = FakeLLM()
    tally = run(extract_sample(extractor(llm), ExtractionCache(path), DATASET_SHA, SAMPLE.sample_id, SAMPLE.turns))
    assert len(llm.calls) == 1
    assert "D1:1 Caroline" in llm.calls[0]["user"]
    assert (tally.extracted, tally.cached, tally.failed) == (1, 1, [])


@pytest.mark.parametrize(
    "change",
    [
        {"dataset": "e" * 64},
        {"model": "ollama:other"},
        {"options": SamplingOptions(num_ctx=4096, num_predict=4096)},
    ],
)
def test_cache_misses_on_changed_key(tmp_path, change):
    path = tmp_path / "cache.jsonl"
    run(extract_sample(extractor(FakeLLM()), ExtractionCache(path), DATASET_SHA, SAMPLE.sample_id, SAMPLE.turns))
    llm = FakeLLM()
    ex = OllamaExtractor(llm, change.get("model", MODEL), load_prompt(EXTRACT_VERSION), options=change.get("options", EXTRACT_OPTIONS))
    run(extract_sample(ex, ExtractionCache(path), change.get("dataset", DATASET_SHA), SAMPLE.sample_id, SAMPLE.turns))
    assert len(llm.calls) == 2


def test_cache_misses_on_changed_prompt(tmp_path):
    path = tmp_path / "cache.jsonl"
    run(extract_sample(extractor(FakeLLM()), ExtractionCache(path), DATASET_SHA, SAMPLE.sample_id, SAMPLE.turns))
    (tmp_path / "extract_v1.txt").write_text((PROMPTS_DIR / "extract_v1.txt").read_text() + "\nBe brief.")
    llm = FakeLLM()
    ex = OllamaExtractor(llm, MODEL, load_prompt(EXTRACT_VERSION, tmp_path))
    run(extract_sample(ex, ExtractionCache(path), DATASET_SHA, SAMPLE.sample_id, SAMPLE.turns))
    assert len(llm.calls) == 2


def test_cache_ignores_truncated_line(tmp_path):
    path = tmp_path / "cache.jsonl"
    run(extract_sample(extractor(FakeLLM()), ExtractionCache(path), DATASET_SHA, SAMPLE.sample_id, SAMPLE.turns))
    with path.open("a") as f:
        f.write('{"dataset_sha256": "d')
    assert len(ExtractionCache(path).records) == 2


# --- command ------------------------------------------------------------------


def test_extract_command_runs_and_resumes(tmp_path, monkeypatch, capsys):
    data = tmp_path / "locomo10.json"
    data.write_text(json.dumps([RAW, {**copy.deepcopy(RAW), "sample_id": "conv-2"}]))
    cache = tmp_path / "extractions.jsonl"
    clients: list = []

    class FakeClient(FakeLLM):
        def __init__(self, **kwargs):
            super().__init__()
            self.described: list[str] = []
            clients.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def describe_model(self, model):
            self.described.append(model)
            return {"id": model, "digest": "sha256:fake"}

    monkeypatch.setattr(extraction, "LLMClient", FakeClient)
    argv = ["extract", "--data", str(data), "--cache", str(cache), "--samples", "conv-1", "--extractor-model", MODEL]
    cli.main(argv)
    assert len(clients[0].calls) == 2
    assert clients[0].described == [MODEL]
    records = [json.loads(line) for line in cache.read_text().splitlines()]
    assert {r["sample_id"] for r in records} == {"conv-1"}
    assert all(r["model_digest"] == "sha256:fake" and r["dataset_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest() for r in records)

    cli.main(argv)
    assert clients[1].calls == []
    assert clients[1].described == []
    assert "0 sessions to extract" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="unknown sample ids: conv-9"):
        cli.main([*argv[:-4], "--samples", "conv-9"])
