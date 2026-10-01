import asyncio
import copy
import hashlib
import json
import re
from dataclasses import asdict

import pytest

from locomo_eval import cli, extraction
from locomo_eval.dataset import CATEGORY_NAMES, parse_sample
from locomo_eval.extraction import (
    EXTRACT,
    EXTRACT_MAX_NUM_PREDICT,
    EXTRACT_OPTIONS,
    EXTRACT_V2_VERSION,
    EXTRACT_VERSION,
    EXTRACTION_SCHEMA,
    FAILED,
    LIFECYCLE_SCHEMA,
    LIFECYCLE_V2_VERSION,
    LIFECYCLE_VERSION,
    OK,
    Duplicate,
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
    recall_memories,
    sessions_of,
)
from locomo_eval.llm import RETRY_TEMPERATURE, Completion, SamplingOptions
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
    """Stands in for LLMClient. Replies from `script` in order when given
    (strings, or Completions to control token counts and done_reason),
    else with one valid memory citing the transcript's first dialog id and
    an entity for Caroline. Every call reports 1000 input and 100 output tokens."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls: list[dict] = []

    async def complete(self, model, system, user, *, stage="default", format=None, options=None) -> Completion:
        self.calls.append({"model": model, "system": system, "user": user, "stage": stage, "format": format, "options": options})
        await asyncio.sleep(0)
        if self.script:
            item = self.script.pop(0)
            return item if isinstance(item, Completion) else Completion(item, 1000, 100)
        first = re.search(r"^(D\d+:\d+) ", user, re.MULTILINE).group(1)
        make = lifecycle_reply if format == LIFECYCLE_SCHEMA else reply
        text = make(
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


@pytest.mark.parametrize("version", [EXTRACT_VERSION, EXTRACT_V2_VERSION])
def test_extraction_prompt_has_transcript_date_and_entities(version):
    session = sessions_of(SAMPLE.sample_id, SAMPLE.turns)[1]
    text = extraction_prompt(load_prompt(version), session, [Entity("caroline", "Caroline", "person")])
    assert "Session date: 10:00 am on 15 May, 2023" in text
    assert "D2:1 Caroline: I went to the LGBTQ support group yesterday." in text
    assert "caroline | Caroline | person" in text
    assert "$" not in text
    empty = extraction_prompt(load_prompt(version), session, [])
    assert "(none yet)" in empty


@pytest.mark.parametrize("version", [EXTRACT_VERSION, EXTRACT_V2_VERSION])
def test_extract_prompt_is_committed_and_generic(version):
    prompt = load_prompt(version)
    assert not extraction.is_lifecycle(prompt)
    assert prompt.sha256 == hashlib.sha256((PROMPTS_DIR / f"{version}.txt").read_bytes()).hexdigest()
    lowered = prompt.text.lower()
    for name in CATEGORY_NAMES.values():
        assert name not in lowered
    assert "question" not in lowered


@pytest.mark.parametrize("version", [EXTRACT_VERSION, EXTRACT_V2_VERSION, LIFECYCLE_VERSION, LIFECYCLE_V2_VERSION])
def test_no_question_text_reaches_any_prompt(tmp_path, version):
    llm = FakeLLM()
    cache = ExtractionCache(tmp_path / "cache.jsonl")
    ex = OllamaExtractor(llm, MODEL, load_prompt(version))
    tally = run(extract_sample(ex, cache, DATASET_SHA, SAMPLE.sample_id, SAMPLE.turns))
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
    assert result.dropped == {"source_dia_ids": 3, "entity_ids": 1, "entity_refs": 1, "relationships": 2, "occurred_at": 1, "memory_refs": 0}


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
    # The first attempt is the configured options; the retry samples, so its seed matters.
    assert llm.calls[0]["options"] == EXTRACT_OPTIONS
    assert llm.calls[1]["options"].temperature == RETRY_TEMPERATURE
    assert llm.calls[1]["options"].num_predict == EXTRACT_OPTIONS.num_predict
    assert result.options == llm.calls[1]["options"]
    assert result.truncated == 0
    assert all(c["format"] == EXTRACTION_SCHEMA and c["stage"] == EXTRACT for c in llm.calls)


def cut_off(num_predict: int = EXTRACT_OPTIONS.num_predict, input_tokens: int = 2736) -> Completion:
    """A reply stopped at the output cap: unterminated JSON, as Ollama returns it."""
    return Completion('{"entities": [], "memories": [{"content": "Caroline said', input_tokens, num_predict, "length")


def test_extractor_retries_truncated_reply_with_larger_cap():
    session = sessions_of(SAMPLE.sample_id, SAMPLE.turns)[0]
    llm = FakeLLM([cut_off(), reply([memory()])])
    result = run(extractor(llm).extract(session, []))
    assert result.status == OK
    assert (result.attempts, result.truncated) == (2, 1)
    first, retry = (c["options"] for c in llm.calls)
    assert first == EXTRACT_OPTIONS
    assert retry.num_predict == 2 * EXTRACT_OPTIONS.num_predict
    assert retry.temperature == RETRY_TEMPERATURE
    assert retry.seed == EXTRACT_OPTIONS.seed + 1
    # 2736 + 8192 fits in the configured context, so it isn't changed (a change reloads the model).
    assert retry.num_ctx == EXTRACT_OPTIONS.num_ctx


def test_truncated_session_fails_with_clear_error_and_cap_stops_growing():
    session = sessions_of(SAMPLE.sample_id, SAMPLE.turns)[0]
    llm = FakeLLM([cut_off(4096), cut_off(8192), cut_off(8192)])
    result = run(extractor(llm).extract(session, []))
    assert result.status == FAILED
    assert result.truncated == 3
    assert [c["options"].num_predict for c in llm.calls] == [4096, 8192, EXTRACT_MAX_NUM_PREDICT]
    assert result.error == f"reply truncated at the {EXTRACT_MAX_NUM_PREDICT}-token output cap (num_predict)"


def test_truncation_is_detected_from_token_count_without_done_reason():
    session = sessions_of(SAMPLE.sample_id, SAMPLE.turns)[0]
    llm = FakeLLM([Completion("{", 100, EXTRACT_OPTIONS.num_predict), reply([memory()])])
    result = run(extractor(llm).extract(session, []))
    assert (result.status, result.truncated) == (OK, 1)
    assert llm.calls[1]["options"].num_predict == 2 * EXTRACT_OPTIONS.num_predict


def test_truncation_retry_grows_context_when_prompt_and_cap_do_not_fit():
    session = sessions_of(SAMPLE.sample_id, SAMPLE.turns)[0]
    llm = FakeLLM([cut_off(4096, input_tokens=10_000), reply([memory()])])
    run(extractor(llm).extract(session, []))
    assert llm.calls[1]["options"].num_ctx >= 10_000 + 8192


def test_retried_session_is_cached_under_the_configured_options(tmp_path):
    path = tmp_path / "cache.jsonl"
    llm = FakeLLM([cut_off(), reply([memory()]), reply([memory("Melanie shared a photo.", sources=["D2:2"])])])
    run(extract_sample(extractor(llm), ExtractionCache(path), DATASET_SHA, SAMPLE.sample_id, SAMPLE.turns))
    first, second = (json.loads(line) for line in path.read_text().splitlines())
    assert first["options"] == second["options"] == asdict(EXTRACT_OPTIONS)
    assert first["options_hash"] == second["options_hash"]
    assert first["truncated_attempts"] == 1
    assert first["attempt_options"]["num_predict"] == 2 * EXTRACT_OPTIONS.num_predict
    assert second["attempt_options"] == asdict(EXTRACT_OPTIONS)
    # A later run finds both, retried or not.
    rerun = FakeLLM()
    run(extract_sample(extractor(rerun), ExtractionCache(path), DATASET_SHA, SAMPLE.sample_id, SAMPLE.turns))
    assert rerun.calls == []


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


# --- lifecycle variant ----------------------------------------------------------


def lifecycle_reply(memories=None, entities=None, duplicates=None) -> str:
    return json.dumps({"entities": entities or [], "memories": [{"supersedes": [], **m} for m in memories or []], "duplicates": duplicates or []})


def stored(id_, content, occurred_at=None):
    return {"id": id_, "content": content, "occurred_at": occurred_at}


@pytest.mark.parametrize("version", [LIFECYCLE_VERSION, LIFECYCLE_V2_VERSION])
def test_lifecycle_prompt_is_committed_and_generic(version):
    prompt = load_prompt(version)
    assert extraction.is_lifecycle(prompt) and not extraction.is_lifecycle(load_prompt(EXTRACT_VERSION))
    assert prompt.sha256 == hashlib.sha256((PROMPTS_DIR / f"{version}.txt").read_bytes()).hexdigest()
    lowered = prompt.text.lower()
    for name in CATEGORY_NAMES.values():
        assert name not in lowered
    assert "question" not in lowered


@pytest.mark.parametrize("version", [LIFECYCLE_VERSION, LIFECYCLE_V2_VERSION])
def test_lifecycle_prompt_shows_recalled_memories_with_ids(version):
    session = sessions_of(SAMPLE.sample_id, SAMPLE.turns)[1]
    prompt = load_prompt(version)
    text = extraction_prompt(prompt, session, [], [stored("M1.1", "Caroline plans to join a support group.", "2023-05-08T00:00:00+00:00"), stored("M1.2", "Melanie likes sunsets.")])
    assert "M1.1 | 2023-05-08 | Caroline plans to join a support group." in text
    assert "M1.2 | - | Melanie likes sunsets." in text
    assert "$" not in text
    assert "Stored memories that may relate to this session (id | date | content):\n(none yet)" in extraction_prompt(prompt, session, [])


def test_lifecycle_v2_keeps_extract_v2_rules():
    v2 = load_prompt(LIFECYCLE_V2_VERSION).text
    extract_v2 = load_prompt(EXTRACT_V2_VERSION).text
    # The add-only rules carry over word for word, ahead of the lifecycle ones.
    rules = extract_v2[extract_v2.index("1. Be exact."):extract_v2.index("4. One memory")]
    assert rules in v2
    assert extract_v2[extract_v2.index('- "occurred_at"'):extract_v2.index('- "entities": the ids')] in v2


def test_parse_lifecycle_reply_supersedes_and_duplicates():
    text = lifecycle_reply(
        [
            {**memory("Caroline joined the support group on 14 May 2023.", sources=["D2:1"]), "supersedes": ["m1.1", "M9.9"]},
            {**memory("Caroline goes to the support group.", sources=["D2:1"]), "supersedes": ["M1.1"]},
        ],
        duplicates=[
            {"memory_id": "M1.2", "source_dia_ids": ["D2:2", "D1:1"]},
            {"memory_id": "M1.2", "source_dia_ids": ["D2:2"]},
            {"memory_id": "M1.1", "source_dia_ids": ["D2:1"]},
        ],
    )
    result = parse_extraction(text, {"D2:1", "D2:2"}, [], ["M1.1", "M1.2"])
    first, second = result.memories
    assert first.supersedes == ("M1.1",)
    # Each stored memory is superseded once, and one that is superseded isn't also repeated.
    assert second.supersedes == ()
    assert result.duplicates == (Duplicate("M1.2", ("D2:2",)),)
    assert result.dropped["memory_refs"] == 4
    assert result.dropped["source_dia_ids"] == 1


def test_parse_lifecycle_reply_needs_its_fields():
    with pytest.raises(ExtractionParseError, match="duplicates"):
        parse_extraction(reply([memory()]), {"D1:1"}, [], [])
    with pytest.raises(ExtractionParseError, match="supersedes"):
        parse_extraction(json.dumps({"entities": [], "memories": [memory()], "duplicates": []}), {"D1:1"}, [], [])
    # The add-only variant ignores lifecycle fields it didn't ask for.
    assert parse_extraction(lifecycle_reply([memory()]), {"D1:1"}).memories[0].supersedes == ()


def test_recall_memories_ranks_per_turn_and_keeps_stored_order():
    session = sessions_of(SAMPLE.sample_id, SAMPLE.turns)[1]
    memories = [
        stored("M1.1", "Melanie likes sunsets."),
        stored("M1.2", "Caroline wants to find an LGBTQ support group."),
        stored("M1.3", "Caroline went to a support group meeting in April 2023."),
        stored("M1.4", "Melanie paints."),
    ]
    assert [m["id"] for m in recall_memories(memories, session)] == ["M1.2", "M1.3"]
    assert [m["id"] for m in recall_memories(memories, session, per_turn=1)] == ["M1.2"]
    assert [m["id"] for m in recall_memories(memories, session, limit=1)] == ["M1.2"]
    assert recall_memories([], session) == []


def test_lifecycle_extraction_recalls_supersedes_and_caches(tmp_path):
    path = tmp_path / "cache.jsonl"
    llm = FakeLLM(
        [
            lifecycle_reply(
                [memory("Caroline says hey to Mel.", sources=["D1:1"]), memory("Melanie shared a photo of a sunset.", sources=["D1:2"])],
            ),
            lifecycle_reply(
                [{**memory("Caroline is back and says hey to Mel again.", sources=["D2:1"]), "supersedes": ["M1.1"]}],
                duplicates=[{"memory_id": "M1.2", "source_dia_ids": ["D2:2"]}],
            ),
        ]
    )
    ex = OllamaExtractor(llm, MODEL, load_prompt(LIFECYCLE_VERSION))
    raw = copy.deepcopy(RAW)
    raw["conversation"]["session_2"] = [
        {"speaker": "Caroline", "dia_id": "D2:1", "text": "Hey Mel, I'm back."},
        {"speaker": "Melanie", "dia_id": "D2:2", "text": "Another sunset photo!"},
    ]
    turns = parse_sample(raw).turns
    tally = run(extract_sample(ex, ExtractionCache(path), DATASET_SHA, "conv-1", turns))
    assert all(c["format"] == LIFECYCLE_SCHEMA for c in llm.calls)
    assert "Stored memories that may relate to this session (id | date | content):\n(none yet)" in llm.calls[0]["user"]
    assert "M1.1 | - | Caroline says hey to Mel." in llm.calls[1]["user"]
    assert "M1.2 | - | Melanie shared a photo of a sunset." in llm.calls[1]["user"]
    assert (tally.memories, tally.duplicates, tally.superseded) == (3, 1, 1)

    first, second = (json.loads(line) for line in path.read_text().splitlines())
    assert first["prompt_version"] == LIFECYCLE_VERSION
    assert first["recalled_memory_ids"] == []
    assert second["recalled_memory_ids"] == ["M1.1", "M1.2"]
    assert second["memories"][0]["supersedes"] == ["M1.1"]
    assert second["duplicates"] == [{"memory_id": "M1.2", "source_dia_ids": ["D2:2"]}]

    # The add-only variant is cached apart; a rerun of this one is all hits with the same counts.
    assert pending_sessions(ExtractionCache(path), extractor(FakeLLM()), DATASET_SHA, [("conv-1", turns)]) == 2
    rerun = FakeLLM()
    tally = run(extract_sample(OllamaExtractor(rerun, MODEL, load_prompt(LIFECYCLE_VERSION)), ExtractionCache(path), DATASET_SHA, "conv-1", turns))
    assert rerun.calls == []
    assert (tally.cached, tally.duplicates, tally.superseded) == (2, 1, 1)

    loaded = extraction.cached_run(ExtractionCache(path), DATASET_SHA, [("conv-1", turns)], MODEL, load_prompt(LIFECYCLE_VERSION))
    assert [m["id"] for m in loaded.memories("conv-1")] == ["M1.1", "M1.2", "M2.1"]
    assert loaded.config["variant"] == extraction.LIFECYCLE
    assert (loaded.config["totals"]["duplicates"], loaded.config["totals"]["superseded"]) == (1, 1)


def test_superseded_memories_are_not_recalled_again(tmp_path):
    raw = copy.deepcopy(RAW)
    raw["conversation"]["session_3_date_time"] = "9:00 am on 1 June, 2023"
    raw["conversation"]["session_3"] = [{"speaker": "Caroline", "dia_id": "D3:1", "text": "Hey Mel, the support group was great."}]
    turns = parse_sample(raw).turns
    llm = FakeLLM(
        [
            lifecycle_reply([memory("Caroline says hey to Mel.", sources=["D1:1"])]),
            lifecycle_reply([{**memory("Caroline went to the LGBTQ support group.", sources=["D2:1"]), "supersedes": ["M1.1"]}]),
            lifecycle_reply([]),
        ]
    )
    run(extract_sample(OllamaExtractor(llm, MODEL, load_prompt(LIFECYCLE_VERSION)), ExtractionCache(tmp_path / "c.jsonl"), DATASET_SHA, "conv-1", turns))
    assert "M1.1 |" in llm.calls[1]["user"]
    assert "M1.1 |" not in llm.calls[2]["user"]
    assert "M2.1 | - | Caroline went to the LGBTQ support group." in llm.calls[2]["user"]
