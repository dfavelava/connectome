import argparse
import asyncio
import copy
import hashlib
import json

import httpx
import pytest

from locomo_eval import cli
from locomo_eval.dataset import parse_sample
from locomo_eval.extraction import (
    EXTRACT_OPTIONS,
    EXTRACT_VERSION,
    LIFECYCLE_VERSION,
    ExtractionCache,
    cached_run,
    options_hash,
)
from locomo_eval.metrics import Context, summarize
from locomo_eval.prompts import load_prompt
from tests.test_dataset import RAW_SAMPLE


class FakeClient:
    """In-memory stand-in for ConnectomeClient: recall returns the tome's
    memories whose text shares a word with the query, in insertion order."""

    base_url = "http://fake/api/connectome"

    def __init__(self, *args, **kwargs):
        self.tomes: dict[str, dict[str, str]] = {}
        self.destroyed: list[str] = []
        self.remember_calls: list[dict] = []
        self.recall_calls: list[dict] = []
        self.forgotten: list[str] = []
        self.superseded: list[dict] = []
        self.fail_recall = False

    async def remember(self, content, memory_type, tome, occurred_at, entities=None, relationships=None):
        self.remember_calls.append(
            {"content": content, "memory_type": memory_type, "tome": tome, "occurred_at": occurred_at, "entities": entities, "relationships": relationships}
        )
        key = f"mem_{len(self.remember_calls)}.md"
        self.tomes.setdefault(tome, {})[key] = content
        return {"key": key}

    async def recall(self, query, k, tome, hydrate=False, ranking=None):
        if self.fail_recall:
            raise RuntimeError("boom")
        self.recall_calls.append({"query": query, "k": k, "hydrate": hydrate, "ranking": ranking})
        words = set(query.lower().strip("?").split())
        memories = self.tomes.get(tome, {})
        hits = [key for key, text in memories.items() if words & set(text.lower().split())][:k]
        # The backend echoes the effective settings: its defaults plus overrides.
        echoed = {"vector_weight": 0.6, "text_weight": 0.4, "text_query": "plain", **(ranking or {})}
        if hydrate:
            # Hydrated results carry the whole memory document, frontmatter included.
            return {"results": [{"key": key, "content": f"---\ntype: event\n---\n{memories[key]}\n"} for key in hits], "ranking": echoed}
        return {"results": [{"key": key} for key in hits], "ranking": echoed}

    async def forget(self, key, tome=None):
        self.forgotten.append(key)
        del self.tomes[tome][key]
        return {"message": "deleted", "key": key}

    async def supersede_relationship(self, key, subject_entity_id, predicate, object_entity_id=None, superseded_by=None, tome=None):
        assert key in self.tomes[tome]
        self.superseded.append({"key": key, "subject": subject_entity_id, "predicate": predicate, "object": object_entity_id, "superseded_by": superseded_by})
        return {}

    async def get_memory(self, key, tome=None):
        if key not in self.tomes.get(tome, {}):
            request = httpx.Request("GET", f"{self.base_url}/memory/")
            raise httpx.HTTPStatusError("not found", request=request, response=httpx.Response(404, request=request))
        return {"content": self.tomes[tome][key]}

    async def destroy_tome(self, tome, confirm=False):
        self.destroyed.append(tome)
        self.tomes.pop(tome, None)
        return {}


@pytest.fixture
def client():
    return FakeClient()


def args(**overrides):
    defaults = {"ks": [1, 5], "answer_k": 10, "concurrency": 2, "no_occurred_at": False, "keep_tomes": False, "reuse_tomes": None, "ingest": "turns", "superseded": "mark"}
    return argparse.Namespace(**{**defaults, **overrides})


def run(client, a, samples, run_id, extracted=None, echoed=None):
    """ingest, retrieve, then (unless keep_tomes) cleanup, as the default command does."""

    async def stages():
        key_maps = await cli.ingest_samples(client, a, samples, run_id, extracted)
        try:
            results, skipped = await cli.retrieve_samples(client, a, samples, run_id, key_maps, echoed)
        finally:
            if not a.keep_tomes:
                await cli.cleanup(client, [s.sample_id for s in samples], run_id)
        return results, skipped, key_maps

    return asyncio.run(stages())


def test_run_ingests_scores_and_destroys_tome(client):
    sample = parse_sample(RAW_SAMPLE)
    results, skipped, key_maps = run(client, args(), [sample], "r1")

    assert [c["content"] for c in client.remember_calls][0] == "[1:56 pm on 8 May, 2023] Caroline: Hey Mel!"
    assert client.remember_calls[0]["tome"] == "temp-locomo-r1-conv-1"
    assert client.remember_calls[0]["occurred_at"] == "2023-05-08T13:56:00+00:00"
    assert client.destroyed == ["temp-locomo-r1-conv-1"]

    assert [r.question for r in results] == ["When?", "Adversarial?"]
    assert skipped == {"no_evidence": 1, "unknown_evidence_only": 0, "unknown_evidence_ids": 0}


def test_run_maps_tome_scoped_search_keys_back_to_dialog_ids(client):
    raw = copy.deepcopy(RAW_SAMPLE)
    raw["qa"].append({"question": "Back?", "evidence": ["D2:1"], "category": 4})
    results, _, _ = run(client, args(), [parse_sample(raw)], "r4")
    assert results[-1].retrieved == ("D2:1",)


def test_run_fails_loudly_on_unmapped_search_keys(client):
    raw = copy.deepcopy(RAW_SAMPLE)
    raw["qa"].append({"question": "Back?", "evidence": ["D2:1"], "category": 4})

    async def recall(query, k, tome, hydrate=False, ranking=None):
        return {"results": [{"key": "mem_from_somewhere_else.md"}]}

    client.recall = recall
    with pytest.raises(RuntimeError, match="not ingested"):
        run(client, args(), [parse_sample(raw)], "r5")
    assert client.destroyed == ["temp-locomo-r5-conv-1"]


def test_run_destroys_tome_on_failure(client):
    sample = parse_sample(RAW_SAMPLE)
    client.fail_recall = True
    with pytest.raises(RuntimeError):
        run(client, args(), [sample], "r2")
    assert client.destroyed == ["temp-locomo-r2-conv-1"]


def test_no_occurred_at_and_keep_tomes(client):
    sample = parse_sample(RAW_SAMPLE)
    run(client, args(no_occurred_at=True, keep_tomes=True), [sample], "r3")
    assert all(c["occurred_at"] is None for c in client.remember_calls)
    assert client.destroyed == []


def test_ks_parsing():
    assert cli._ks("10,1,5,5") == [1, 5, 10]
    with pytest.raises(argparse.ArgumentTypeError):
        cli._ks("0,5")
    with pytest.raises(argparse.ArgumentTypeError):
        cli._ks("51")


def test_retrieve_queries_kept_tomes_without_ingesting(client):
    sample = parse_sample(RAW_SAMPLE)
    _, _, key_maps = run(client, args(keep_tomes=True), [sample], "r6")
    assert set(key_maps["conv-1"].values()) == {(t.dia_id,) for t in sample.turns}
    ingested = len(client.remember_calls)

    results, _ = asyncio.run(cli.retrieve_samples(client, args(), [sample], "r6", key_maps))

    assert len(client.remember_calls) == ingested
    assert client.destroyed == []
    assert [r.question for r in results] == ["When?", "Adversarial?"]


def test_run_saves_hydrated_contexts_and_gold_answers(client):
    raw = copy.deepcopy(RAW_SAMPLE)
    raw["qa"].append({"question": "Hey back again?", "answer": 2022, "evidence": ["D2:1"], "category": 4})
    results, _, _ = run(client, args(ks=[1], answer_k=3), [parse_sample(raw)], "r7")

    assert all(c["hydrate"] and c["k"] == 3 for c in client.recall_calls)
    when, adversarial, back = results
    assert (when.question_id, when.answer, when.adversarial_answer) == ("conv-1#0", "7 May 2023", None)
    assert (adversarial.question_id, adversarial.answer, adversarial.adversarial_answer) == ("conv-1#1", None, "x")
    assert back.question_id == "conv-1#3"
    assert back.answer == "2022"
    assert back.contexts == (
        Context("D1:1", "[1:56 pm on 8 May, 2023] Caroline: Hey Mel!"),
        Context("D2:1", "[not a date] Caroline: Back again."),
    )
    assert back.retrieved == ("D1:1", "D2:1")


def test_recall_k_covers_answer_k():
    assert cli.recall_k(args(ks=[1, 5, 10], answer_k=10)) == 10
    assert cli.recall_k(args(ks=[1, 5], answer_k=20)) == 20
    assert cli.recall_k(args(ks=[1, 50], answer_k=10)) == 50


def test_memory_body_strips_frontmatter():
    assert cli.memory_body("---\ntype: event\ntags: []\n---\n[date] A: hi\n") == "[date] A: hi"
    assert cli.memory_body("no frontmatter") == "no frontmatter"


# --- Extracted ingestion ------------------------------------------------------

EXTRACTOR = "ollama:fake-extractor"


def extraction_record(dataset_sha256, session, memories, entities=(), status="ok", prompt_version=EXTRACT_VERSION, duplicates=()):
    prompt = load_prompt(prompt_version)
    return {
        "dataset_sha256": dataset_sha256,
        "sample_id": "conv-1",
        "session": session,
        "extractor_model": EXTRACTOR,
        "model_digest": "sha256:fake",
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "options_hash": options_hash(EXTRACT_OPTIONS),
        "options": {},
        "status": status,
        "entities": [{"id": e, "name": e.title(), "kind": "person"} for e in entities],
        "memories": memories if status == "ok" else [],
        "duplicates": list(duplicates),
        "dropped": {"source_dia_ids": 1} if status == "ok" else {},
        "attempts": 1 if status == "ok" else 3,
        "error": None if status == "ok" else "bad reply",
        "input_tokens": 1000,
        "output_tokens": 100,
        "seconds": 2.5,
    }


def extracted_memory(content, sources, entities=(), relationships=(), occurred_at=None, memory_type="fact"):
    return {
        "content": content,
        "memory_type": memory_type,
        "occurred_at": occurred_at,
        "entities": list(entities),
        "relationships": list(relationships),
        "source_dia_ids": list(sources),
    }


# Session 1's memories cite both of its turns; session 2 failed, so D2:1 is
# cited by nothing.
SESSION_1 = [
    extracted_memory(
        "Caroline greeted Melanie and Melanie shared a sunset photo.",
        ["D1:1", "D1:2"],
        entities=["caroline", "melanie"],
        relationships=[{"subjectEntityId": "caroline", "predicate": "friend_of", "objectEntityId": "melanie", "kind": "fact"}],
        occurred_at="2023-05-08T00:00:00+00:00",
        memory_type="event",
    ),
    extracted_memory("Melanie likes sunsets.", ["D1:2"], entities=["melanie"], memory_type="preference"),
    extracted_memory("Caroline is friendly.", [], entities=["caroline"]),
]


def write_cache(path, dataset_sha256, *, session_2_failed=True):
    records = [extraction_record(dataset_sha256, 1, SESSION_1, entities=["caroline", "melanie"])]
    if session_2_failed:
        records.append(extraction_record(dataset_sha256, 2, [], status="failed"))
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def cached(tmp_path, sample):
    write_cache(tmp_path / "extractions.jsonl", "d" * 64)
    return cached_run(ExtractionCache(tmp_path / "extractions.jsonl"), "d" * 64, [(sample.sample_id, sample.turns)], EXTRACTOR, load_prompt(EXTRACT_VERSION))


def test_cached_run_reads_records_and_totals(tmp_path):
    sample = parse_sample(RAW_SAMPLE)
    run = cached(tmp_path, sample)

    assert [m["content"] for m in run.memories("conv-1")] == [m["content"] for m in SESSION_1]
    assert run.entity_ids("conv-1") == {"caroline", "melanie"}
    totals = run.config["totals"]
    assert totals["sessions"] == 2
    assert totals["extracted_sessions"] == 1
    assert totals["failed_sessions"] == {"conv-1": [2]}
    assert totals["unextracted_sessions"] == {}
    assert (totals["memories"], totals["input_tokens"], totals["output_tokens"]) == (3, 1000, 100)
    assert totals["dropped"]["source_dia_ids"] == 1
    assert run.config["extractor_model"] == EXTRACTOR
    assert run.config["model_digests"] == ["sha256:fake"]
    assert run.config["prompt"]["version"] == EXTRACT_VERSION


def test_cached_run_counts_sessions_never_extracted(tmp_path):
    sample = parse_sample(RAW_SAMPLE)
    write_cache(tmp_path / "x.jsonl", "d" * 64, session_2_failed=False)
    run = cached_run(ExtractionCache(tmp_path / "x.jsonl"), "d" * 64, [("conv-1", sample.turns)], EXTRACTOR, load_prompt(EXTRACT_VERSION))
    assert run.config["totals"]["failed_sessions"] == {}
    assert run.config["totals"]["unextracted_sessions"] == {"conv-1": [2]}


def test_cached_run_needs_an_extraction_per_sample(tmp_path):
    sample = parse_sample(RAW_SAMPLE)
    write_cache(tmp_path / "x.jsonl", "d" * 64)
    with pytest.raises(ValueError, match="no cached extraction of conv-1"):
        cached_run(ExtractionCache(tmp_path / "x.jsonl"), "e" * 64, [("conv-1", sample.turns)], EXTRACTOR, load_prompt(EXTRACT_VERSION))
    with pytest.raises(ValueError, match="no cached extraction"):
        cached_run(ExtractionCache(tmp_path / "x.jsonl"), "d" * 64, [("conv-1", sample.turns)], "ollama:other", load_prompt(EXTRACT_VERSION))


def test_extracted_ingestion_writes_memories_and_maps_keys_to_sources(client, tmp_path):
    raw = copy.deepcopy(RAW_SAMPLE)
    raw["qa"] = [
        {"question": "Did Caroline greet Melanie?", "answer": "yes", "evidence": ["D1:1", "D1:2"], "category": 1},
        {"question": "Is Caroline back again?", "answer": "yes", "evidence": ["D2:1", "D1:1"], "category": 4},
    ]
    sample = parse_sample(raw)
    extracted = cached(tmp_path, sample)
    results, _, key_maps = run(client, args(ingest="extracted"), [sample], "x1", extracted=extracted)

    first = client.remember_calls[0]
    assert first["content"] == SESSION_1[0]["content"]
    assert first["memory_type"] == "event"
    assert first["tome"] == "temp-locomo-x1-conv-1"
    assert first["occurred_at"] == "2023-05-08T00:00:00+00:00"
    assert first["entities"] == ["caroline", "melanie"]
    assert first["relationships"] == SESSION_1[0]["relationships"]
    assert len(client.remember_calls) == 3
    assert client.destroyed == ["temp-locomo-x1-conv-1"]
    assert sorted(key_maps["conv-1"].values()) == [(), ("D1:1", "D1:2"), ("D1:2",)]

    greeted, back = results
    # Retrieved memories are expanded to their sources in rank order.
    assert greeted.retrieved_sources == (("D1:1", "D1:2"), ("D1:2",), ())
    assert greeted.retrieved == ("D1:1", "D1:2")
    assert greeted.contexts[0] == Context("D1:1", SESSION_1[0]["content"])
    assert greeted.contexts[2] == Context("", "Caroline is friendly.")
    # D2:1 is still evidence, though no memory cites it.
    assert back.evidence == ("D2:1", "D1:1")
    assert back.covered == ("D1:1",)
    assert greeted.covered == ("D1:1", "D1:2")

    summary = summarize(results, [1])
    assert summary["single-hop"]["coverage"] == 0.5
    assert summary["multi-hop"]["recall@1"] == 1.0


def test_extracted_ingestion_can_leave_occurred_at_unset(client, tmp_path):
    sample = parse_sample(RAW_SAMPLE)
    run(client, args(ingest="extracted", no_occurred_at=True), [sample], "x2", extracted=cached(tmp_path, sample))
    assert all(c["occurred_at"] is None for c in client.remember_calls)


def test_turns_ingestion_covers_all_evidence(client):
    results, _, _ = run(client, args(), [parse_sample(RAW_SAMPLE)], "t1")
    assert all(r.covered == r.evidence for r in results)


def test_key_maps_round_trip_and_load_single_id_maps():
    maps = {"conv-1": {"mem_1.md": ("D1:1", "D1:2"), "mem_2.md": ()}}
    assert cli.load_key_maps(json.loads(json.dumps(cli.dump_key_maps(maps)))) == maps
    # Written by a turns run before extracted ingestion existed.
    assert cli.load_key_maps({"conv-1": {"mem_1.md": "D1:1"}}) == {"conv-1": {"mem_1.md": ("D1:1",)}}


def test_memory_stats():
    maps = {"a": {"k1": ("D1:1", "D1:2"), "k2": ()}, "b": {"k3": ("D1:1",)}}
    stats = cli.memory_stats(maps)
    assert stats["a"] == {"memories": 2, "sources_per_memory": 1.0, "entities": 0}
    assert stats["overall"] == {"memories": 3, "sources_per_memory": 1.0, "entities_per_conversation": 0.0}


def test_main_ingests_extracted_and_compares_with_a_turns_run(tmp_path, monkeypatch, capsys):
    raw = copy.deepcopy(RAW_SAMPLE)
    raw["qa"] = [{"question": "Did Caroline greet Melanie?", "answer": "yes", "evidence": ["D1:1", "D2:1"], "category": 1}]
    data = tmp_path / "locomo.json"
    data.write_text(json.dumps([raw]), encoding="utf-8")
    write_cache(tmp_path / "extractions.jsonl", hashlib.sha256(data.read_bytes()).hexdigest())
    monkeypatch.setattr(cli, "ConnectomeClient", FakeClient)
    common = ["--data", str(data), "--results-dir", str(tmp_path), "--ks", "1,5", "--budgets", "8,64"]

    cli.main([*common, "--run-id", "base"])
    capsys.readouterr()
    cli.main([*common, "--run-id", "ext", "--ingest", "extracted", "--extraction-cache", str(tmp_path / "extractions.jsonl"), "--extractor-model", EXTRACTOR, "--compare", "base"])
    out = capsys.readouterr().out

    result = json.loads((tmp_path / "ext.json").read_text(encoding="utf-8"))
    ingestion = result["config"]["ingestion"]
    assert ingestion["mode"] == "extracted"
    assert ingestion["extraction"]["extractor_model"] == EXTRACTOR
    assert ingestion["extraction"]["totals"]["failed_sessions"] == {"conv-1": [2]}
    assert result["config"]["budgets"] == [8, 64]
    assert result["summary"]["overall"]["coverage"] == 0.5
    assert set(result["summary"]["overall"]) >= {"recall@1", "hit@5", "recall@8t", "underfilled@64t"}
    assert result["memories"]["overall"]["memories"] == 3
    assert result["memories"]["overall"]["entities_per_conversation"] == 2.0
    assert json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))["config"]["ingestion"] == {"mode": "turns", "extraction": None}

    assert "  ext (extracted: add-only)" in out and "  base (turns)" in out
    assert "  diff" in out and "-0.500" in out
    assert "coverage" in out and "recall@8t" in out
    assert "3 memories" in out and "3 memories, 1.00 sources/memory, 0.0 entities" in out


def test_main_exits_without_a_cached_extraction(tmp_path, monkeypatch):
    data = tmp_path / "locomo.json"
    data.write_text(json.dumps([RAW_SAMPLE]), encoding="utf-8")
    monkeypatch.setattr(cli, "ConnectomeClient", FakeClient)
    with pytest.raises(SystemExit, match="no extraction cache"):
        cli.main(["--data", str(data), "--results-dir", str(tmp_path), "--ingest", "extracted", "--extraction-cache", str(tmp_path / "none.jsonl")])
    (tmp_path / "other.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="run `locomo-eval extract`"):
        cli.main(["--data", str(data), "--results-dir", str(tmp_path), "--ingest", "extracted", "--extraction-cache", str(tmp_path / "other.jsonl")])


def write_lifecycle_cache(path, dataset_sha256):
    """Session 2 supersedes session 1's friendship memory and repeats its sunset one."""
    session_2 = [{**extracted_memory("Caroline came back to see Melanie.", ["D2:1"], entities=["caroline"]), "supersedes": ["M1.1"]}]
    records = [
        extraction_record(dataset_sha256, 1, [{**m, "supersedes": []} for m in SESSION_1], entities=["caroline", "melanie"], prompt_version=LIFECYCLE_VERSION),
        extraction_record(dataset_sha256, 2, session_2, prompt_version=LIFECYCLE_VERSION, duplicates=[{"memory_id": "M1.2", "source_dia_ids": ["D2:1"]}]),
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def lifecycle_cached(tmp_path, sample):
    write_lifecycle_cache(tmp_path / "extractions.jsonl", "d" * 64)
    return cached_run(ExtractionCache(tmp_path / "extractions.jsonl"), "d" * 64, [(sample.sample_id, sample.turns)], EXTRACTOR, load_prompt(LIFECYCLE_VERSION))


def test_lifecycle_ingestion_marks_superseded_relationships(client, tmp_path):
    sample = parse_sample(RAW_SAMPLE)
    extracted = lifecycle_cached(tmp_path, sample)
    assert (extracted.config["totals"]["duplicates"], extracted.config["totals"]["superseded"]) == (1, 1)
    _, _, key_maps = run(client, args(ingest="extracted", keep_tomes=True), [sample], "l1", extracted=extracted)
    assert len(client.remember_calls) == 4
    old_key, new_key = "mem_1.md", "mem_4.md"
    assert client.superseded == [{"key": old_key, "subject": "caroline", "predicate": "friend_of", "object": "melanie", "superseded_by": new_key}]
    assert client.forgotten == []
    # A marked memory is still stored and can still be recalled.
    assert old_key in key_maps["conv-1"] and old_key in client.tomes["temp-locomo-l1-conv-1"]


def test_lifecycle_ingestion_can_forget_superseded_memories(client, tmp_path):
    sample = parse_sample(RAW_SAMPLE)
    extracted = lifecycle_cached(tmp_path, sample)
    _, _, key_maps = run(client, args(ingest="extracted", superseded="forget"), [sample], "l2", extracted=extracted)
    assert client.forgotten == ["mem_1.md"]
    assert client.superseded == []
    assert sorted(key_maps["conv-1"].values()) == [(), ("D1:2",), ("D2:1",)]


def test_add_only_ingestion_supersedes_nothing(client, tmp_path):
    sample = parse_sample(RAW_SAMPLE)
    run(client, args(ingest="extracted", superseded="forget"), [sample], "a1", extracted=cached(tmp_path, sample))
    assert client.forgotten == [] and client.superseded == []


def test_main_compares_lifecycle_with_add_only(tmp_path, monkeypatch, capsys):
    raw = copy.deepcopy(RAW_SAMPLE)
    raw["qa"] = [{"question": "Did Caroline come back?", "answer": "yes", "evidence": ["D2:1"], "category": 2}]
    data = tmp_path / "locomo.json"
    data.write_text(json.dumps([raw]), encoding="utf-8")
    sha = hashlib.sha256(data.read_bytes()).hexdigest()
    write_cache(tmp_path / "add.jsonl", sha)
    write_lifecycle_cache(tmp_path / "life.jsonl", sha)
    monkeypatch.setattr(cli, "ConnectomeClient", FakeClient)
    common = ["--data", str(data), "--results-dir", str(tmp_path), "--ks", "1", "--budgets", "0", "--ingest", "extracted", "--extractor-model", EXTRACTOR]

    cli.main([*common, "--run-id", "add", "--extraction-cache", str(tmp_path / "add.jsonl")])
    capsys.readouterr()
    cli.main([*common, "--run-id", "life", "--extraction-cache", str(tmp_path / "life.jsonl"), "--extract-prompt", LIFECYCLE_VERSION, "--superseded", "forget", "--compare", "add"])
    captured = capsys.readouterr()

    config = json.loads((tmp_path / "life.json").read_text(encoding="utf-8"))["config"]
    assert config["ingestion"]["extraction"]["variant"] == "lifecycle"
    assert config["ingestion"]["extraction"]["recall"]["limit"] > 0
    assert config["chunking"]["superseded"] == "forget"
    assert "1 duplicates not written, 1 memories superseded (forget)" in captured.err
    out = captured.out
    assert "  life (extracted: lifecycle, forget)" in out and "  add (extracted: add-only)" in out
    # Session 2 failed in the add-only run, so only the lifecycle run covers D2:1.
    diff = next(line for line in out.splitlines()[out.splitlines().index("temporal") :] if line.startswith("  diff"))
    # diff, n, coverage, recall@1
    assert diff.split()[1:3] == ["-", "+1.000"]


def test_ranking_option_parses_json_and_key_value_and_merges():
    parsed = cli.parse_args(["--ranking", '{"vector_weight": 0.2, "rrf_k": 10}', "--ranking", "text_query=bm25", "--ranking", "rrf_k=30"])
    assert parsed.ranking == {"vector_weight": 0.2, "rrf_k": 30.0, "text_query": "bm25"}
    assert cli.parse_args([]).ranking == {}


def test_ranking_option_rejects_unknown_keys_and_bad_numbers():
    for bad in ("nope=1", "vector_weight=abc", "vector_weight", '{"nope": 1}', "{bad"):
        with pytest.raises(SystemExit):
            cli.parse_args(["--ranking", bad])


def test_run_sends_ranking_and_search_config_is_the_echoed_one(client):
    sample = parse_sample(RAW_SAMPLE)
    echoed: list[dict] = []
    run(client, args(ranking={"text_weight": 0.9}), [sample], "r1", echoed=echoed)

    assert {c["ranking"]["text_weight"] for c in client.recall_calls} == {0.9}
    # The config records what the backend said it ran, defaults included.
    assert cli.search_config(echoed) == {"vector_weight": 0.6, "text_weight": 0.9, "text_query": "plain"}


def test_search_config_is_none_without_echo_and_rejects_drift():
    assert cli.search_config([]) is None
    with pytest.raises(RuntimeError):
        cli.search_config([{"rrf_k": 60}, {"rrf_k": 10}])


# --- Stages: ingest, retrieve, cleanup -----------------------------------------


@pytest.fixture
def stage_env(tmp_path, monkeypatch):
    """A dataset on disk and one FakeClient every main() call shares, as one backend would."""
    raw = copy.deepcopy(RAW_SAMPLE)
    raw["qa"].append({"question": "Did Caroline come back?", "answer": "yes", "evidence": ["D2:1"], "category": 2})
    data = tmp_path / "locomo.json"
    data.write_text(json.dumps([raw]), encoding="utf-8")
    backend = FakeClient()
    monkeypatch.setattr(cli, "ConnectomeClient", lambda *a, **kw: backend)
    return backend, ["--data", str(data), "--results-dir", str(tmp_path)], tmp_path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_ingest_then_retrieve_matches_the_default_command(stage_env, capsys):
    backend, common, tmp_path = stage_env
    cli.main([*common, "--run-id", "all"])
    cli.main(["ingest", *common, "--run-id", "base"])
    assert "temp-locomo-base-conv-1" in backend.tomes
    cli.main(["retrieve", "base", *common])

    staged, default = read(tmp_path / "base.json"), read(tmp_path / "all.json")
    assert staged["summary"] == default["summary"]
    assert staged["memories"] == default["memories"]
    assert staged["skipped"] == default["skipped"]
    assert [q["retrieved"] for q in staged["questions"]] == [q["retrieved"] for q in default["questions"]]
    assert staged["config"]["reused_tomes"] == "base" and default["config"]["reused_tomes"] is None
    assert list(staged["config"]) == list(default["config"])
    # The default command cleaned up after itself; the ingest's tomes stay.
    assert read(tmp_path / "all.ingest.json")["tomes"] == "destroyed"
    assert "temp-locomo-all-conv-1" not in backend.tomes and "temp-locomo-base-conv-1" in backend.tomes
    assert "locomo-eval retrieve base" in capsys.readouterr().out


def test_ingest_manifest_round_trip(stage_env):
    backend, common, tmp_path = stage_env
    cli.main(["ingest", *common, "--run-id", "base", "--no-occurred-at", "--embedding-model", "m"])
    manifest = read(tmp_path / "base.ingest.json")

    assert manifest["dataset"]["sha256"] == hashlib.sha256((tmp_path / "locomo.json").read_bytes()).hexdigest()
    assert manifest["samples"] == ["conv-1"]
    assert manifest["embedding_model"] == "m"
    assert manifest["ingestion"] == {"mode": "turns", "extraction": None}
    assert manifest["chunking"]["occurred_at"] is None
    assert manifest["tomes"] == "kept"
    assert manifest["memories"]["conv-1"]["memories"] == 3
    key_maps = cli.load_key_maps(manifest["key_maps"])
    assert set(key_maps["conv-1"]) == set(backend.tomes["temp-locomo-base-conv-1"])
    assert sorted(key_maps["conv-1"].values()) == [("D1:1",), ("D1:2",), ("D2:1",)]
    assert not (tmp_path / "base.keys.json").exists()


def test_retrieve_runs_tagged_configs_against_one_ingest(stage_env, capsys):
    backend, common, tmp_path = stage_env
    cli.main(["ingest", *common, "--run-id", "base"])
    ingested = len(backend.remember_calls)
    cli.main(["retrieve", "base", *common, "--tag", "plain"])
    cli.main(["retrieve", "base", *common, "--tag", "bm25", "--ranking", "text_query=bm25", "--compare", "base.plain"])

    assert len(backend.remember_calls) == ingested
    plain, bm25 = read(tmp_path / "base.plain.json"), read(tmp_path / "base.bm25.json")
    assert plain["config"]["search"]["text_query"] == "plain"
    assert bm25["config"]["search"]["text_query"] == "bm25"
    assert bm25["config"]["run_id"] == "base.bm25"
    assert not (tmp_path / "base.json").exists()
    out = capsys.readouterr().out
    assert "  base.bm25 (turns)" in out and "  base.plain (turns)" in out and "  diff" in out


def test_retrieve_rejects_bad_tags():
    for bad in ("ingest", "keys", "a.b", "", "-x"):
        with pytest.raises(SystemExit):
            cli.parse_retrieve_args(["base", "--tag", bad])
    assert cli.parse_retrieve_args(["base", "--tag", "bm25_k1-2"]).tag == "bm25_k1-2"


def test_retrieve_fails_without_a_manifest(stage_env):
    _, common, _ = stage_env
    with pytest.raises(SystemExit, match="run `locomo-eval ingest --run-id nope` first"):
        cli.main(["retrieve", "nope", *common])


def test_retrieve_fails_when_tomes_are_missing(stage_env):
    backend, common, _ = stage_env
    cli.main(["ingest", *common, "--run-id", "base"])
    backend.tomes.clear()  # e.g. the backend's volume was reset
    with pytest.raises(SystemExit, match="tomes missing .*temp-locomo-base-conv-1"):
        cli.main(["retrieve", "base", *common])
    assert backend.recall_calls == []


def test_retrieve_fails_after_cleanup(stage_env):
    backend, common, tmp_path = stage_env
    cli.main(["ingest", *common, "--run-id", "base"])
    cli.main(["cleanup", "base", *common])
    assert backend.destroyed == ["temp-locomo-base-conv-1"]
    assert read(tmp_path / "base.ingest.json")["tomes"] == "destroyed"
    with pytest.raises(SystemExit, match="tomes were destroyed"):
        cli.main(["retrieve", "base", *common])
    # With its tomes gone, the run id can be ingested again.
    cli.main(["ingest", *common, "--run-id", "base"])


def test_retrieve_fails_on_another_dataset_or_unknown_samples(stage_env):
    _, common, tmp_path = stage_env
    cli.main(["ingest", *common, "--run-id", "base"])
    with pytest.raises(SystemExit, match="did not ingest: conv-9"):
        cli.main(["retrieve", "base", *common, "--samples", "conv-9"])
    (tmp_path / "locomo.json").write_text(json.dumps([RAW_SAMPLE]), encoding="utf-8")
    with pytest.raises(SystemExit, match="is not the dataset run base ingested"):
        cli.main(["retrieve", "base", *common])


def test_ingest_refuses_a_run_whose_tomes_are_kept(stage_env):
    _, common, _ = stage_env
    cli.main(["ingest", *common, "--run-id", "base"])
    with pytest.raises(SystemExit, match="locomo-eval cleanup base"):
        cli.main(["ingest", *common, "--run-id", "base"])
    with pytest.raises(SystemExit, match="locomo-eval cleanup base"):
        cli.main([*common, "--run-id", "base"])


def test_ingest_failure_destroys_the_tomes_it_wrote(client):
    first = parse_sample(RAW_SAMPLE)
    second = parse_sample({**RAW_SAMPLE, "sample_id": "conv-2"})
    remember = client.remember

    async def flaky(content, memory_type, tome, occurred_at, **kw):
        if tome.endswith("conv-2"):
            raise RuntimeError("backend down")
        return await remember(content, memory_type, tome, occurred_at, **kw)

    client.remember = flaky
    with pytest.raises(RuntimeError):
        asyncio.run(cli.ingest_samples(client, args(), [first, second], "f1"))
    assert client.destroyed == ["temp-locomo-f1-conv-1", "temp-locomo-f1-conv-2"]


def test_default_command_keep_tomes_reuse_tomes_and_cleanup_flag(stage_env, capsys):
    backend, common, tmp_path = stage_env
    cli.main([*common, "--run-id", "base", "--keep-tomes"])
    assert read(tmp_path / "base.ingest.json")["tomes"] == "kept" and backend.destroyed == []
    ingested = len(backend.remember_calls)

    cli.main([*common, "--run-id", "vector", "--reuse-tomes", "base", "--ranking", "text_weight=0"])
    assert len(backend.remember_calls) == ingested
    vector = read(tmp_path / "vector.json")
    assert vector["config"]["reused_tomes"] == "base" and vector["config"]["search"]["text_weight"] == 0
    with pytest.raises(SystemExit, match="ingested turns, not extracted"):
        cli.main([*common, "--reuse-tomes", "base", "--ingest", "extracted"])

    cli.main([*common, "--cleanup", "base"])
    assert backend.destroyed == ["temp-locomo-base-conv-1"]
    assert "--cleanup is deprecated" in capsys.readouterr().err
    assert read(tmp_path / "base.ingest.json")["tomes"] == "destroyed"


def test_reuse_tomes_reads_an_older_keep_tomes_run(stage_env):
    """A --keep-tomes run from before manifests left <run-id>.keys.json and <run-id>.json."""
    backend, common, tmp_path = stage_env
    cli.main([*common, "--run-id", "old", "--keep-tomes"])
    manifest = read(tmp_path / "old.ingest.json")
    (tmp_path / "old.keys.json").write_text(json.dumps(manifest["key_maps"]), encoding="utf-8")
    (tmp_path / "old.ingest.json").unlink()

    cli.main([*common, "--run-id", "again", "--reuse-tomes", "old"])
    assert read(tmp_path / "again.json")["summary"] == read(tmp_path / "old.json")["summary"]
    cli.main(["cleanup", "old", *common])
    assert backend.destroyed == ["temp-locomo-old-conv-1"]


def test_retrieve_takes_lifecycle_options_from_the_manifest(stage_env):
    backend, common, tmp_path = stage_env
    write_lifecycle_cache(tmp_path / "life.jsonl", hashlib.sha256((tmp_path / "locomo.json").read_bytes()).hexdigest())
    extract = ["--ingest", "extracted", "--extraction-cache", str(tmp_path / "life.jsonl"), "--extractor-model", EXTRACTOR, "--extract-prompt", LIFECYCLE_VERSION]
    cli.main(["ingest", *common, "--run-id", "life", *extract, "--superseded", "forget"])
    manifest = read(tmp_path / "life.ingest.json")
    assert manifest["ingestion"]["extraction"]["variant"] == "lifecycle"
    assert manifest["ingestion"]["extraction"]["recall"]["limit"] > 0
    assert manifest["chunking"]["superseded"] == "forget"
    # Written after supersession: the forgotten memory is gone from the key map.
    assert backend.forgotten[0] not in manifest["key_maps"]["conv-1"]
    assert manifest["memories"]["conv-1"]["entities"] == 2

    cli.main(["retrieve", "life", *common])
    result = read(tmp_path / "life.json")
    assert result["config"]["chunking"]["superseded"] == "forget"
    assert result["memories"]["overall"]["entities_per_conversation"] == 2.0
    assert cli.ingestion_label(result["config"]) == "extracted: lifecycle, forget"
    with pytest.raises(SystemExit, match="--superseded forget, not mark"):
        cli.main([*common, "--reuse-tomes", "life", "--superseded", "mark"])
