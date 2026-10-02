"""Run the LoCoMo retrieval-recall eval against a live Connectome backend.

Each conversation is ingested into its own scratch tome
(temp-locomo-<run>-<sample>): one memory per dialog turn with --ingest turns
(the default), or the memories a cached `locomo-eval extract` run chose with
--ingest extracted. Every QA item with evidence is then sent to recall, the
returned memory keys are mapped back to their source dialog ids, and evidence
coverage, recall@k / hit@k and recall at equal token budgets are computed per
category. The tomes are destroyed once every conversation is scored,
including on failure.

The default command runs three stages in one go, which can also run apart:
`locomo-eval ingest` fills the tomes and writes results/<run-id>.ingest.json,
`locomo-eval retrieve <run-id>` scores them (as often as you like, with
--tag), and `locomo-eval cleanup <run-id>` destroys them; see each one's
--help. `locomo-eval answer <run-id> ...` then scores a finished run's answers
offline; see `locomo-eval answer --help`. `locomo-eval extract ...` has a
local LLM choose the memories to store instead; see `locomo-eval extract --help`.
`locomo-eval judge-agreement <run-id> ...` checks the judges that scored a run
against hand labels; see `locomo-eval judge-agreement --help`.
"""

import argparse
import asyncio
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
from connectomeclient import ConnectomeClient
from dotenv import load_dotenv

from locomo_eval import answering, extraction, judge_labels
from locomo_eval.dataset import Sample, Turn, load_dataset, sample_ids
from locomo_eval.extraction import (
    DEFAULT_CACHE_PATH,
    DEFAULT_EXTRACTOR_MODEL,
    EXTRACT_VERSION,
    CachedRun,
    ExtractionCache,
    cached_run,
)
from locomo_eval.metrics import Context, QuestionResult, expand, summarize
from locomo_eval.prompts import load_prompt

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "locomo10.json"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"
DEFAULT_KS = [1, 5, 10]
DEFAULT_ANSWER_K = 10
DEFAULT_BUDGETS = [64, 128, 256]
TURNS = "turns"
EXTRACTED = "extracted"
# What becomes of a memory the lifecycle extractor superseded: its
# relationships are marked superseded_by its successor (recall still returns
# it), or it is forgotten.
MARK = "mark"
FORGET = "forget"
# The backend caps search k at 50 - see maxSearchK in backend/resources/searchResource.go.
MAX_SEARCH_K = 50
MEMORY_TEMPLATE = "[{session_date}] {speaker}: {text}"
SOURCE_TYPE = "locomo-eval"
# Whether an ingest manifest's tomes still exist.
KEPT = "kept"
DESTROYED = "destroyed"
# A --tag names results/<run-id>.<tag>.json, so it can't clash with the other
# files a run writes (the manifest, answer checkpoints).
TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
RESERVED_TAGS = {"ingest", "keys"}


def memory_text(turn: Turn) -> str:
    return MEMORY_TEMPLATE.format(session_date=turn.session_date, speaker=turn.speaker, text=turn.text)


def tome_for(run_id: str, sample_id: str) -> str:
    return f"temp-locomo-{run_id}-{sample_id}".lower()


def recall_k(args: argparse.Namespace) -> int:
    return max([*args.ks, args.answer_k])


def memory_body(content: str) -> str:
    """A hydrated memory document's body, without its YAML frontmatter."""
    if content.startswith("---\n") and (end := content.find("\n---\n", 4)) != -1:
        content = content[end + len("\n---\n") :]
    return content.strip()


# A memory key -> the dialog ids it came from. Keys are random, so this is
# the only way back from a recall hit to the turns it can be scored against.
KeyMap = dict[str, tuple[str, ...]]


async def ingest(client: ConnectomeClient, sample: Sample, tome: str, concurrency: int, use_occurred_at: bool) -> KeyMap:
    """Write one memory per turn and return the key map."""
    semaphore = asyncio.Semaphore(concurrency)

    async def write(turn: Turn) -> tuple[str, tuple[str, ...]]:
        async with semaphore:
            result = await client.remember(
                memory_text(turn),
                memory_type="event",
                tome=tome,
                occurred_at=turn.occurred_at if use_occurred_at else None,
            )
        return result["key"], (turn.dia_id,)

    return dict(await asyncio.gather(*(write(turn) for turn in sample.turns)))


async def ingest_extracted(
    client: ConnectomeClient,
    memories: Sequence[dict],
    tome: str,
    concurrency: int,
    use_occurred_at: bool,
    superseded: str = MARK,
) -> KeyMap:
    """Write each extracted memory as it was extracted - content, type,
    entities, relationships and occurred_at - and return the key map to its
    source_dia_ids. No LLM is involved.

    Memories the lifecycle variant superseded are then marked (each of their
    relationships gets superseded_by the successor's key) or, with FORGET,
    forgotten and left out of the key map."""
    semaphore = asyncio.Semaphore(concurrency)

    async def write(memory: dict) -> tuple[str, tuple[str, ...]]:
        async with semaphore:
            result = await client.remember(
                memory["content"],
                memory_type=memory["memory_type"],
                entities=list(memory["entities"]),
                relationships=[dict(r) for r in memory["relationships"]],
                tome=tome,
                occurred_at=memory["occurred_at"] if use_occurred_at else None,
            )
        return result["key"], tuple(memory["source_dia_ids"])

    written = await asyncio.gather(*(write(memory) for memory in memories))
    key_map = dict(written)
    by_id = {memory["id"]: (memory, key) for memory, (key, _) in zip(memories, written, strict=True) if memory.get("id")}

    async def supersede(old: dict, old_key: str, new_key: str) -> None:
        async with semaphore:
            if superseded == FORGET:
                _ = await client.forget(old_key, tome=tome)
                return
            for rel in old["relationships"]:
                _ = await client.supersede_relationship(
                    old_key, rel["subjectEntityId"], rel["predicate"], rel.get("objectEntityId"), superseded_by=new_key, tome=tome
                )

    pending = {}
    for memory, (new_key, _) in zip(memories, written, strict=True):
        for old_id in memory.get("supersedes") or ():
            # An id no memory has can only come from a session re-extracted after its successor.
            if old_id in by_id and old_id not in pending:
                old, old_key = by_id[old_id]
                pending[old_id] = supersede(old, old_key, new_key)
                if superseded == FORGET:
                    key_map.pop(old_key)
    await asyncio.gather(*pending.values())
    return key_map


async def query(
    client: ConnectomeClient,
    sample: Sample,
    tome: str,
    key_map: KeyMap,
    k: int,
    concurrency: int,
    ranking: dict[str, object] | None = None,
    echoed: list[dict[str, object]] | None = None,
) -> tuple[list[QuestionResult], dict[str, int]]:
    # Evidence is checked against the conversation, not the memories, so
    # turns no memory cites still count against coverage and recall.
    known_dia_ids = {turn.dia_id for turn in sample.turns}
    cited = {d for sources in key_map.values() for d in sources}
    skipped = {"no_evidence": 0, "unknown_evidence_only": 0, "unknown_evidence_ids": 0}
    semaphore = asyncio.Semaphore(concurrency)

    async def ask(question: str) -> tuple[tuple[Context, ...], tuple[tuple[str, ...], ...]]:
        async with semaphore:
            response = await client.recall(question, k=k, tome=tome, hydrate=True, ranking=ranking)
        if echoed is not None and isinstance(response.get("ranking"), dict):
            echoed.append(response["ranking"])
        hits = response.get("results") or []
        assert isinstance(hits, list)
        keys = [hit["key"] for hit in hits]
        if unknown := [key for key in keys if key not in key_map]:
            # Every hit comes from this run's own tome, so an unmapped key means
            # the key shape changed - fail rather than silently score zero.
            raise RuntimeError(f"recall returned keys not ingested into {tome}: {unknown[:3]}")
        sources = tuple(key_map[key] for key in keys)
        contexts = tuple(Context(dia_id=next(iter(s), ""), text=memory_body(hit.get("content") or "")) for hit, s in zip(hits, sources, strict=True))
        return contexts, sources

    scored = []
    for item in sample.qa:
        if not item.evidence:
            skipped["no_evidence"] += 1
            continue
        evidence = tuple(e for e in item.evidence if e in known_dia_ids)
        skipped["unknown_evidence_ids"] += len(item.evidence) - len(evidence)
        if not evidence:
            skipped["unknown_evidence_only"] += 1
            continue
        scored.append((item, evidence))

    retrieved = await asyncio.gather(*(ask(item.question) for item, _ in scored))
    results = [
        QuestionResult(
            sample_id=sample.sample_id,
            question=item.question,
            category=item.category_name,
            evidence=evidence,
            retrieved=expand(sources),
            question_id=f"{sample.sample_id}#{item.qa_index}",
            answer=item.answer,
            adversarial_answer=item.adversarial_answer,
            contexts=contexts,
            retrieved_sources=sources,
            covered=tuple(e for e in evidence if e in cited),
        )
        for (item, evidence), (contexts, sources) in zip(scored, retrieved, strict=True)
    ]
    return results, skipped


async def destroy(client: ConnectomeClient, tome: str) -> None:
    try:
        _ = await client.destroy_tome(tome)
    except httpx.HTTPError as exc:
        print(f"warning: failed to destroy tome {tome}: {exc}", file=sys.stderr)


async def ingest_samples(
    client: ConnectomeClient,
    args: argparse.Namespace,
    samples: list[Sample],
    run_id: str,
    extracted: CachedRun | None = None,
) -> dict[str, KeyMap]:
    """Ingest each sample into its own tome and return each sample's key map.

    With extracted, the cached extraction's memories are ingested instead of
    the turns. If ingestion fails, the tomes written so far are destroyed,
    since no manifest will point at them."""
    key_maps: dict[str, KeyMap] = {}
    written: list[str] = []
    try:
        for sample in samples:
            tome = tome_for(run_id, sample.sample_id)
            written.append(tome)
            started = time.monotonic()
            if extracted is not None:
                memories = extracted.memories(sample.sample_id)
                key_map = await ingest_extracted(client, memories, tome, args.concurrency, not args.no_occurred_at, args.superseded)
            else:
                key_map = await ingest(client, sample, tome, args.concurrency, not args.no_occurred_at)
            key_maps[sample.sample_id] = key_map
            print(f"{sample.sample_id}: {len(key_map)} memories ingested in {time.monotonic() - started:.1f}s", file=sys.stderr)
    except BaseException:
        for tome in written:
            await destroy(client, tome)
        raise
    return key_maps


async def retrieve_samples(
    client: ConnectomeClient,
    args: argparse.Namespace,
    samples: list[Sample],
    run_id: str,
    key_maps: dict[str, KeyMap],
    echoed: list[dict[str, object]] | None = None,
) -> tuple[list[QuestionResult], dict[str, int]]:
    """Query the tomes run_id ingested and return the results and skip
    counts. Nothing is written or destroyed.

    Each search response's echoed ranking settings are appended to echoed.
    """
    k = recall_k(args)
    results: list[QuestionResult] = []
    skipped: dict[str, int] = {}
    for sample in samples:
        started = time.monotonic()
        tome = tome_for(run_id, sample.sample_id)
        sample_results, sample_skipped = await query(client, sample, tome, key_maps[sample.sample_id], k, args.concurrency, getattr(args, "ranking", None), echoed)
        results.extend(sample_results)
        for reason, count in sample_skipped.items():
            skipped[reason] = skipped.get(reason, 0) + count
        print(f"{sample.sample_id}: {len(sample_results)} questions scored in {time.monotonic() - started:.1f}s", file=sys.stderr)
    return results, skipped


async def missing_tomes(client: ConnectomeClient, run_id: str, key_maps: dict[str, KeyMap]) -> list[str]:
    """The tomes of run_id that no longer hold their first ingested memory.
    A search of a missing tome just comes back empty, so without this
    check a destroyed tome would score as zero recall."""
    missing = []
    for sample_id, key_map in key_maps.items():
        if not key_map:
            continue
        tome = tome_for(run_id, sample_id)
        try:
            _ = await client.get_memory(next(iter(key_map)), tome=tome)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            missing.append(tome)
    return missing


def search_config(echoed: list[dict[str, object]]) -> dict[str, object] | None:
    """The ranking settings the backend reported running with, or None when it
    reported none (a backend from before per-request ranking, or no questions
    scored). The settings are the same for every request of a run, since the
    harness sends one --ranking; a difference means the backend changed
    mid-run, which is worth failing over rather than recording one of them."""
    if not echoed:
        return None
    if any(e != echoed[0] for e in echoed):
        raise RuntimeError("the backend reported different ranking settings within one run")
    return echoed[0]


def memory_stats(key_maps: dict[str, KeyMap], entities: dict[str, int] | None = None) -> dict[str, dict[str, float | int]]:
    """Per sample and overall: memories, mean sources per memory, and
    entities (0 for raw turns, which carry none)."""
    stats: dict[str, dict[str, float | int]] = {}
    for sample_id, key_map in key_maps.items():
        stats[sample_id] = {
            "memories": len(key_map),
            "sources_per_memory": _ratio(sum(map(len, key_map.values())), len(key_map)),
            "entities": (entities or {}).get(sample_id, 0),
        }
    memories = sum(int(s["memories"]) for s in stats.values())
    sources = sum(len(v) for key_map in key_maps.values() for v in key_map.values())
    stats["overall"] = {
        "memories": memories,
        "sources_per_memory": _ratio(sources, memories),
        "entities_per_conversation": _ratio(sum(int(s["entities"]) for s in stats.values()), len(key_maps)),
    }
    return stats


def dump_key_maps(key_maps: dict[str, KeyMap]) -> dict[str, dict[str, list[str]]]:
    return {sample_id: {key: list(sources) for key, sources in key_map.items()} for sample_id, key_map in key_maps.items()}


def load_key_maps(raw: dict[str, dict[str, str | list[str]]]) -> dict[str, KeyMap]:
    """Key maps as an ingest manifest (or an older --keep-tomes run's
    keys.json) holds them. Maps from before extracted ingestion hold one
    dialog id per key rather than a list."""
    return {
        sample_id: {key: (sources,) if isinstance(sources, str) else tuple(sources) for key, sources in key_map.items()}
        for sample_id, key_map in raw.items()
    }


async def cleanup(client: ConnectomeClient, sample_ids: Sequence[str], run_id: str) -> None:
    for sample_id in sample_ids:
        tome = tome_for(run_id, sample_id)
        await destroy(client, tome)
        print(f"destroyed {tome}", file=sys.stderr)


def chunking_config(args: argparse.Namespace, extracted: CachedRun | None) -> dict[str, object]:
    if extracted is None:
        return {"unit": "one memory per dialog turn", "template": MEMORY_TEMPLATE, "occurred_at": "session date" if not args.no_occurred_at else None}
    return {
        "unit": "one memory per extracted memory",
        "template": None,
        "occurred_at": "extracted" if not args.no_occurred_at else None,
        "superseded": args.superseded if extracted.config.get("variant") == extraction.LIFECYCLE else None,
    }


def ingest_manifest(
    client: ConnectomeClient,
    args: argparse.Namespace,
    samples: list[Sample],
    run_id: str,
    key_maps: dict[str, KeyMap],
    extracted: CachedRun | None = None,
) -> dict[str, object]:
    """What `retrieve` needs to score an ingest's tomes: the key maps
    (written after supersession, so forgotten memories are gone) and how
    the tomes were filled, which retrieval results copy into their config."""
    entities = {s.sample_id: len(extracted.entity_ids(s.sample_id)) for s in samples} if extracted else None
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "base_url": client.base_url,
        "dataset": {"path": str(args.data), "sha256": _sha256(args.data)},
        "samples": [s.sample_id for s in samples],
        "embedding_model": args.embedding_model,
        "ingestion": {"mode": args.ingest, "extraction": extracted.config if extracted else None},
        "chunking": chunking_config(args, extracted),
        "concurrency": args.concurrency,
        "tomes": KEPT,
        "memories": memory_stats(key_maps, entities),
        "key_maps": dump_key_maps(key_maps),
    }


def manifest_path(results_dir: Path, run_id: str) -> Path:
    return results_dir / f"{run_id}.ingest.json"


def write_manifest(results_dir: Path, manifest: dict[str, object]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(results_dir, str(manifest["run_id"]))
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def read_manifest(results_dir: Path, run_id: str) -> dict | None:
    """run_id's ingest manifest, or one rebuilt from an older --keep-tomes
    run's <run-id>.keys.json and results; None when there is neither."""
    path = manifest_path(results_dir, run_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    keys_path, run_path = results_dir / f"{run_id}.keys.json", results_dir / f"{run_id}.json"
    if not (keys_path.exists() and run_path.exists()):
        return None
    run = json.loads(run_path.read_text(encoding="utf-8"))
    config = run["config"]
    key_maps = load_key_maps(json.loads(keys_path.read_text(encoding="utf-8")))
    entities = {sample_id: int(row.get("entities", 0)) for sample_id, row in (run.get("memories") or {}).items() if sample_id != "overall"}
    return {
        "run_id": run_id,
        "created_at": config.get("started_at"),
        "git_commit": config.get("git_commit"),
        "base_url": config.get("base_url"),
        "dataset": config["dataset"],
        "samples": list(key_maps),
        "embedding_model": config.get("embedding_model"),
        "ingestion": config.get("ingestion") or {"mode": TURNS, "extraction": None},
        "chunking": config.get("chunking"),
        "concurrency": config.get("concurrency"),
        "tomes": KEPT,
        "memories": memory_stats(key_maps, entities),
        "key_maps": dump_key_maps(key_maps),
    }


def run_config(
    client: ConnectomeClient,
    args: argparse.Namespace,
    manifest: dict,
    samples: list[Sample],
    run_id: str,
    started_at: str,
    reused_tomes: str | None = None,
) -> dict[str, object]:
    """A retrieval result's config. How the tomes were filled comes from
    the ingest manifest; reused_tomes names the ingest run when this process
    didn't do the ingesting."""
    return {
        "run_id": run_id,
        "started_at": started_at,
        "git_commit": _git_commit(),
        "base_url": client.base_url,
        "dataset": manifest["dataset"],
        "samples": [s.sample_id for s in samples],
        "ks": args.ks,
        "budgets": args.budgets,
        "token_counter": "words and punctuation marks (metrics.count_tokens)",
        "recall_k": recall_k(args),
        "answer_k": args.answer_k,
        # Filled in from the ranking the backend echoes on its search
        # responses (see search_config), not from this process's environment.
        "search": None,
        "reused_tomes": reused_tomes,
        "embedding_model": manifest["embedding_model"],
        "ingestion": manifest["ingestion"],
        "chunking": manifest["chunking"],
        "concurrency": args.concurrency,
    }


def print_summary(
    summary: dict[str, dict[str, float | int]],
    ks: list[int],
    budgets: Sequence[int] = (),
    label: str = "",
    baseline: dict[str, dict[str, float | int]] | None = None,
    baseline_label: str = "",
) -> None:
    """The metrics per category, with each row of a baseline run under the
    matching row. A metric the baseline didn't record prints as "-"."""
    columns = ["n", "coverage", *(f"recall@{k}" for k in ks), *(f"hit@{k}" for k in ks), *(f"recall@{b}t" for b in budgets)]
    width = max(14, len(label), len(baseline_label)) + 2 if baseline is not None else 14
    print(f"{'category':<{width}}" + "".join(f"{c:>12}" for c in columns))

    def line(name: str, row: dict[str, float | int]) -> None:
        cells = ["-" if row.get(c) is None else f"{row[c]}" if c == "n" else f"{row[c]:.3f}" for c in columns]
        print(f"{name:<{width}}" + "".join(f"{c:>12}" for c in cells))

    def delta(row: dict[str, float | int], base: dict[str, float | int]) -> None:
        cells = ["-" if c == "n" or row.get(c) is None or base.get(c) is None else f"{row[c] - base[c]:+.3f}" for c in columns]
        print(f"{'  diff':<{width}}" + "".join(f"{c:>12}" for c in cells))

    for name, row in summary.items():
        if baseline is None:
            line(name, row)
            continue
        print(name)
        line(f"  {label}", row)
        line(f"  {baseline_label}", baseline.get(name, {}))
        delta(row, baseline.get(name, {}))
    if budgets:
        overall = summary.get("overall", {})
        print("underfilled (retrieval ran out before the budget): " + ", ".join(f"{b}t {overall.get(f'underfilled@{b}t', 0):.3f}" for b in budgets))


def print_memory_stats(stats: dict[str, dict[str, float | int]], label: str = "", baseline: dict[str, dict[str, float | int]] | None = None, baseline_label: str = "") -> None:
    rows = [(label or "this run", stats["overall"])]
    if baseline is not None:
        rows.append((baseline_label, baseline["overall"]))
    for name, row in rows:
        print(f"{name}: {row['memories']} memories, {row['sources_per_memory']:.2f} sources/memory, {row['entities_per_conversation']:.1f} entities/conversation")


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="path to locomo10.json (default: %(default)s)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="where a run's files live (default: %(default)s)")
    parser.add_argument("--samples", type=sample_ids, help="comma-separated sample ids (default: all)")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds (default: %(default)s)")


def _ingest_args(parser: argparse.ArgumentParser) -> None:
    # --ingest and --superseded default to None so that --reuse-tomes can
    # tell whether they were given; _ingest_defaults fills them in.
    parser.add_argument(
        "--ingest",
        choices=[TURNS, EXTRACTED],
        help=f"ingest one memory per turn, or the memories a cached `locomo-eval extract` run chose (default: {TURNS})",
    )
    parser.add_argument("--extraction-cache", type=Path, default=DEFAULT_CACHE_PATH, help="--ingest extracted: the extract stage's cache (default: %(default)s)")
    parser.add_argument("--extractor-model", default=DEFAULT_EXTRACTOR_MODEL, help="--ingest extracted: whose extraction to ingest (default: %(default)s)")
    parser.add_argument("--extract-prompt", default=EXTRACT_VERSION, help="--ingest extracted: the extraction prompt version (default: %(default)s)")
    parser.add_argument(
        "--superseded",
        choices=[MARK, FORGET],
        help=f"--ingest extracted with a lifecycle extraction: mark superseded memories' relationships superseded_by their successor, or forget them (default: {MARK})",
    )
    parser.add_argument("--embedding-model", default="nomic-embed-text", help="recorded in the run config only (default: %(default)s)")
    parser.add_argument("--no-occurred-at", action="store_true", help="don't set occurred_at; the session date stays in the memory text")


def _ingest_defaults(args: argparse.Namespace) -> None:
    args.ingest = args.ingest or TURNS
    args.superseded = args.superseded or MARK


def _retrieve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ks", type=_ks, default=DEFAULT_KS, help="comma-separated k values (default: 1,5,10)")
    parser.add_argument(
        "--budgets",
        type=_budgets,
        default=DEFAULT_BUDGETS,
        help="comma-separated token budgets for equal-budget recall; 0 for none (default: 64,128,256)",
    )
    parser.add_argument(
        "--answer-k",
        type=int,
        default=DEFAULT_ANSWER_K,
        help="retrieved turns an answer stage will use; recall fetches max(ks + [answer-k]) (default: %(default)s)",
    )
    parser.add_argument(
        "--ranking",
        action="append",
        type=_ranking,
        default=[],
        metavar="JSON|KEY=VALUE",
        help=f"search ranking overrides sent with every query, as a JSON object or repeatable key=value; keys: {', '.join(RANKING_KEYS)} (default: the backend's)",
    )
    parser.add_argument("--compare", metavar="RUN_ID", help="print a finished run's metrics (results/RUN_ID.json, e.g. a turns run) under this run's, row by row")


def _check_retrieve_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    args.ranking = {k: v for override in args.ranking for k, v in override.items()}
    if not 1 <= args.answer_k <= MAX_SEARCH_K:
        parser.error(f"--answer-k must be between 1 and {MAX_SEARCH_K}")


def _concurrency_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--concurrency", type=int, default=8, help="max in-flight requests (default: %(default)s)")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="locomo-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common_args(parser)
    parser.add_argument("--run-id", help="tome/result name suffix (default: a UTC timestamp)")
    _ingest_args(parser)
    _retrieve_args(parser)
    _concurrency_arg(parser)
    parser.add_argument("--keep-tomes", action="store_true", help="skip cleanup: leave the tomes for `locomo-eval retrieve` (destroy them with `locomo-eval cleanup`)")
    parser.add_argument("--reuse-tomes", metavar="RUN_ID", help="same as `locomo-eval retrieve RUN_ID`, but writing results/<run-id>.json")
    parser.add_argument("--cleanup", metavar="RUN_ID", help="deprecated: use `locomo-eval cleanup RUN_ID`")
    args = parser.parse_args(argv)
    _check_retrieve_args(parser, args)
    if not args.reuse_tomes:
        _ingest_defaults(args)
    return args


def parse_ingest_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="locomo-eval ingest",
        description="Ingest each conversation into its own tome, temp-locomo-<run-id>-<sample-id>, and leave the tomes in place. "
        "Writes results/<run-id>.ingest.json, which `locomo-eval retrieve <run-id>` scores the tomes from.",
    )
    _common_args(parser)
    parser.add_argument("--run-id", help="tome/manifest name suffix (default: a UTC timestamp)")
    _ingest_args(parser)
    _concurrency_arg(parser)
    args = parser.parse_args(argv)
    _ingest_defaults(args)
    return args


def parse_retrieve_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="locomo-eval retrieve",
        description="Score the tomes `locomo-eval ingest` left behind, without ingesting anything. "
        "Writes results/<run-id>.json, or results/<run-id>.<tag>.json with --tag.",
    )
    parser.add_argument("run_id", help="the ingest run to score (results/<run-id>.ingest.json)")
    parser.add_argument("--tag", type=_tag, help="write results/<run-id>.<tag>.json, so several retrieval configs can share one ingest")
    _common_args(parser)
    _retrieve_args(parser)
    _concurrency_arg(parser)
    args = parser.parse_args(argv)
    _check_retrieve_args(parser, args)
    return args


def parse_cleanup_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="locomo-eval cleanup", description="Destroy the tomes a run left behind.")
    parser.add_argument("run_id", help="the run whose tomes to destroy")
    _common_args(parser)
    return parser.parse_args(argv)


def load_cached_run(args: argparse.Namespace, samples: list[Sample]) -> CachedRun:
    """The cached extraction to ingest; exits when it's missing. Only
    sample ids and turns go in, as in the extract stage."""
    try:
        prompt = load_prompt(args.extract_prompt)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))
    if not args.extraction_cache.exists():
        sys.exit(f"no extraction cache at {args.extraction_cache}; run `locomo-eval extract` first")
    cache = ExtractionCache(args.extraction_cache)
    try:
        extracted = cached_run(cache, _sha256(args.data), [(s.sample_id, s.turns) for s in samples], args.extractor_model, prompt)
    except ValueError as exc:
        sys.exit(f"{exc}; run `locomo-eval extract` with the same --samples, --extractor-model and --extract-prompt first")
    totals = extracted.config["totals"]
    if extracted.config["variant"] == extraction.LIFECYCLE:
        print(f"lifecycle extraction: {totals['duplicates']} duplicates not written, {totals['superseded']} memories superseded ({args.superseded})", file=sys.stderr)
    if totals["failed_sessions"] or totals["unextracted_sessions"]:
        print(
            f"warning: sessions with no successful extraction are left out - failed: {totals['failed_sessions'] or 'none'}, "
            f"never extracted: {totals['unextracted_sessions'] or 'none'}",
            file=sys.stderr,
        )
    return extracted


def load_samples(args: argparse.Namespace, wanted: Sequence[str] | None = None) -> list[Sample]:
    """The dataset's samples, only those in wanted (default: --samples) if given."""
    if not args.data.exists():
        sys.exit(f"dataset not found at {args.data}; see evals/locomo/README.md for the download")
    samples = load_dataset(args.data)
    wanted = args.samples if wanted is None else wanted
    if wanted:
        samples = [s for s in samples if s.sample_id in set(wanted)]
        if missing := set(wanted) - {s.sample_id for s in samples}:
            sys.exit(f"unknown sample ids: {', '.join(sorted(missing))}")
    return samples


def load_baseline(args: argparse.Namespace) -> dict | None:
    if not args.compare:
        return None
    compare_path = args.results_dir / f"{args.compare}.json"
    if not compare_path.exists():
        sys.exit(f"no results at {compare_path} to --compare against")
    return json.loads(compare_path.read_text(encoding="utf-8"))


def refuse_kept_tomes(results_dir: Path, run_id: str) -> None:
    """Ingesting into tomes that still hold an earlier ingest would mix the two."""
    path = manifest_path(results_dir, run_id)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("tomes") == KEPT:
        sys.exit(f"run {run_id} still has tomes from an earlier ingest; destroy them with `locomo-eval cleanup {run_id}` or pick another --run-id")


def ingest_stage(client: ConnectomeClient, args: argparse.Namespace, samples: list[Sample], run_id: str) -> dict:
    """Ingest samples into run_id's tomes and write the manifest."""
    refuse_kept_tomes(args.results_dir, run_id)
    extracted = load_cached_run(args, samples) if args.ingest == EXTRACTED else None
    try:
        key_maps = asyncio.run(ingest_samples(client, args, samples, run_id, extracted))
    except httpx.HTTPError as exc:
        sys.exit(f"request to {client.base_url} failed: {exc!r}")
    manifest = ingest_manifest(client, args, samples, run_id, key_maps, extracted)
    write_manifest(args.results_dir, manifest)
    return manifest


def retrieve_stage(
    client: ConnectomeClient,
    args: argparse.Namespace,
    manifest: dict,
    samples: list[Sample],
    name: str,
    started_at: str,
    reused_tomes: str | None,
    baseline: dict | None,
) -> None:
    """Score the manifest's tomes and write results/<name>.json."""
    key_maps = load_key_maps(manifest["key_maps"])
    config = run_config(client, args, manifest, samples, name, started_at, reused_tomes)
    echoed: list[dict[str, object]] = []
    try:
        results, skipped = asyncio.run(retrieve_samples(client, args, samples, manifest["run_id"], key_maps, echoed))
    except httpx.HTTPError as exc:
        sys.exit(f"request to {config['base_url']} failed: {exc!r}")
    config["search"] = search_config(echoed)
    summary = summarize(results, args.ks, args.budgets)
    entities = {sample_id: int(row["entities"]) for sample_id, row in manifest["memories"].items() if sample_id != "overall"}
    stats = memory_stats({s.sample_id: key_maps[s.sample_id] for s in samples}, entities)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"{name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"config": config, "summary": summary, "memories": stats, "skipped": skipped, "questions": [asdict(r) for r in results]},
            f,
            indent=2,
        )

    label = f"{name} ({ingestion_label(config)})"
    if baseline is None:
        print_summary(summary, args.ks, args.budgets)
        print_memory_stats(stats, label)
    else:
        baseline_label = f"{args.compare} ({ingestion_label(baseline['config'])})"
        print_summary(summary, args.ks, args.budgets, label, rescore(baseline, args.ks, args.budgets), baseline_label)
        print_memory_stats(stats, label, baseline.get("memories"), baseline_label)
    print(f"\nskipped: {skipped}\nwrote {out_path}")


def cleanup_stage(client: ConnectomeClient, args: argparse.Namespace, run_id: str) -> None:
    """Destroy run_id's tomes: those its manifest lists, or with no
    manifest, one per dataset sample. A manifest whose tomes are all gone
    is marked so, and retrieve then refuses it."""
    manifest = read_manifest(args.results_dir, run_id)
    if manifest is None:
        sample_ids = [s.sample_id for s in load_samples(args)]
    else:
        if args.samples and (missing := set(args.samples) - set(manifest["samples"])):
            sys.exit(f"run {run_id} did not ingest: {', '.join(sorted(missing))}")
        sample_ids = [s for s in manifest["samples"] if not args.samples or s in args.samples]
    asyncio.run(cleanup(client, sample_ids, run_id))
    if manifest is not None and manifest_path(args.results_dir, run_id).exists() and set(sample_ids) == set(manifest["samples"]):
        manifest["tomes"] = DESTROYED
        write_manifest(args.results_dir, manifest)


def retrieve_existing(args: argparse.Namespace, ingest_run_id: str, name: str, ingest_mode: str | None = None, superseded: str | None = None) -> None:
    """Score the tomes ingest_run_id left behind into results/<name>.json.
    How they were ingested comes from the manifest; ingest_mode and
    superseded, when given, must agree with it."""
    manifest = read_manifest(args.results_dir, ingest_run_id)
    if manifest is None:
        sys.exit(f"no ingest manifest at {manifest_path(args.results_dir, ingest_run_id)}; run `locomo-eval ingest --run-id {ingest_run_id}` first")
    if manifest.get("tomes") != KEPT:
        sys.exit(f"run {ingest_run_id}'s tomes were destroyed; ingest again with `locomo-eval ingest --run-id {ingest_run_id}`")
    kept_mode = (manifest.get("ingestion") or {}).get("mode", TURNS)
    if ingest_mode and ingest_mode != kept_mode:
        sys.exit(f"run {ingest_run_id} ingested {kept_mode}, not {ingest_mode}; pass --ingest {kept_mode}")
    kept_superseded = (manifest.get("chunking") or {}).get("superseded")
    if superseded and kept_superseded and superseded != kept_superseded:
        sys.exit(f"run {ingest_run_id} ingested with --superseded {kept_superseded}, not {superseded}")
    if args.samples and (missing := [s for s in args.samples if s not in manifest["samples"]]):
        sys.exit(f"run {ingest_run_id} did not ingest: {', '.join(missing)}")
    samples = load_samples(args, args.samples or manifest["samples"])
    if _sha256(args.data) != manifest["dataset"]["sha256"]:
        sys.exit(f"{args.data} is not the dataset run {ingest_run_id} ingested ({manifest['dataset']['path']}); pass --data")
    baseline = load_baseline(args)

    client = ConnectomeClient(source_type=SOURCE_TYPE, timeout=args.timeout)
    key_maps = load_key_maps(manifest["key_maps"])
    try:
        missing_tome_ids = asyncio.run(missing_tomes(client, ingest_run_id, {s.sample_id: key_maps[s.sample_id] for s in samples}))
    except httpx.HTTPError as exc:
        sys.exit(f"request to {client.base_url} failed: {exc!r}")
    if missing_tome_ids:
        sys.exit(f"tomes missing from {client.base_url}: {', '.join(missing_tome_ids)}; ingest again with `locomo-eval ingest --run-id {ingest_run_id}`")
    started_at = datetime.now(UTC).isoformat()
    retrieve_stage(client, args, manifest, samples, name, started_at, ingest_run_id, baseline)


def ingest_main(argv: list[str]) -> None:
    args = parse_ingest_args(argv)
    samples = load_samples(args)
    run_id = args.run_id or _timestamp()
    client = ConnectomeClient(source_type=SOURCE_TYPE, timeout=args.timeout)
    manifest = ingest_stage(client, args, samples, run_id)
    print_memory_stats(manifest["memories"], f"{run_id} ({ingestion_label(manifest)})")
    print(f"\nwrote {manifest_path(args.results_dir, run_id)}")
    print(f"score it with `locomo-eval retrieve {run_id}`; destroy its tomes with `locomo-eval cleanup {run_id}`")


def retrieve_main(argv: list[str]) -> None:
    args = parse_retrieve_args(argv)
    retrieve_existing(args, args.run_id, f"{args.run_id}.{args.tag}" if args.tag else args.run_id)


def cleanup_main(argv: list[str]) -> None:
    args = parse_cleanup_args(argv)
    cleanup_stage(ConnectomeClient(source_type=SOURCE_TYPE, timeout=args.timeout), args, args.run_id)


def run_all(args: argparse.Namespace) -> None:
    """ingest, then retrieve, then (unless --keep-tomes) cleanup."""
    samples = load_samples(args)
    run_id = args.run_id or _timestamp()
    baseline = load_baseline(args)
    client = ConnectomeClient(source_type=SOURCE_TYPE, timeout=args.timeout)
    started_at = datetime.now(UTC).isoformat()
    manifest = ingest_stage(client, args, samples, run_id)
    try:
        retrieve_stage(client, args, manifest, samples, run_id, started_at, None, baseline)
    finally:
        if not args.keep_tomes:
            cleanup_stage(client, args, run_id)


SUBCOMMANDS = {
    "ingest": ingest_main,
    "retrieve": retrieve_main,
    "cleanup": cleanup_main,
    "answer": answering.main,
    "extract": extraction.main,
    "judge-agreement": judge_labels.main,
}


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in SUBCOMMANDS:
        SUBCOMMANDS[argv[0]](argv[1:])
        return
    args = parse_args(argv)
    if args.cleanup:
        print(f"warning: --cleanup is deprecated; use `locomo-eval cleanup {args.cleanup}`", file=sys.stderr)
        cleanup_stage(ConnectomeClient(source_type=SOURCE_TYPE, timeout=args.timeout), args, args.cleanup)
    elif args.reuse_tomes:
        retrieve_existing(args, args.reuse_tomes, args.run_id or _timestamp(), args.ingest, args.superseded)
    else:
        run_all(args)


def ingestion_label(config: dict) -> str:
    """What a run ingested: turns, or extracted by the add-only or lifecycle
    variant (and, for lifecycle, what became of superseded memories)."""
    ingestion = config.get("ingestion") or {}
    mode = ingestion.get("mode", TURNS)
    if mode != EXTRACTED:
        return mode
    variant = (ingestion.get("extraction") or {}).get("variant") or extraction.ADD_ONLY
    if variant == extraction.LIFECYCLE:
        variant = f"{variant}, {(config.get('chunking') or {}).get('superseded') or MARK}"
    return f"{mode}: {variant}"


def rescore(run: dict, ks: list[int], budgets: Sequence[int]) -> dict[str, dict[str, float | int]]:
    """A finished run's summary recomputed from its questions with this
    run's ks and budgets, so the two tables line up. Coverage is only there
    when the run recorded it; a run from before stored contexts can't be
    scored at a budget."""
    results = [
        QuestionResult(
            sample_id=q["sample_id"],
            question=q["question"],
            category=q["category"],
            evidence=tuple(q["evidence"]),
            retrieved=tuple(q["retrieved"]),
            question_id=q.get("question_id", ""),
            contexts=tuple(Context(c["dia_id"], c["text"]) for c in q.get("contexts") or ()),
            retrieved_sources=tuple(tuple(s) for s in q.get("retrieved_sources") or ()),
            covered=None if q.get("covered") is None else tuple(q["covered"]),
        )
        for q in run["questions"]
    ]
    has_contexts = all(r.contexts for r in results if r.retrieved)
    return summarize(results, ks, budgets if has_contexts else [])


def _ks(value: str) -> list[int]:
    ks = sorted({int(k) for k in value.split(",")})
    if not ks or ks[0] < 1 or ks[-1] > MAX_SEARCH_K:
        raise argparse.ArgumentTypeError(f"k values must be between 1 and {MAX_SEARCH_K}")
    return ks


def _budgets(value: str) -> list[int]:
    budgets = sorted({int(b) for b in value.split(",")} - {0})
    if budgets and budgets[0] < 1:
        raise argparse.ArgumentTypeError("budgets must be positive token counts")
    return budgets


def _tag(value: str) -> str:
    if not TAG_PATTERN.fullmatch(value) or value in RESERVED_TAGS:
        raise argparse.ArgumentTypeError(f"a tag is letters, digits, '-' and '_', and not {' or '.join(sorted(RESERVED_TAGS))}")
    return value


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dt%H%M%Sz")


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


RANKING_KEYS = ("vector_weight", "text_weight", "rrf_k", "text_query", "bm25_k1", "bm25_b", "text_max_df")


def _ranking(value: str) -> dict[str, object]:
    """One --ranking argument: a JSON object, or key=value (numbers parsed,
    text_query left as text). The backend validates the values."""
    if value.lstrip().startswith("{"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(f"not valid JSON: {exc}") from exc
    else:
        key, sep, raw = value.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError("expected a JSON object or key=value")
        parsed = {key.strip(): raw.strip() if key.strip() == "text_query" else _number(raw)}
    if unknown := set(parsed) - set(RANKING_KEYS):
        raise argparse.ArgumentTypeError(f"unknown ranking keys {sorted(unknown)}; expected {', '.join(RANKING_KEYS)}")
    return parsed


def _number(raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {raw!r}") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()
