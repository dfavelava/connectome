"""Run the LoCoMo retrieval-recall eval against a live Connectome backend.

Each conversation is ingested into its own scratch tome
(temp-locomo-<run>-<sample>): one memory per dialog turn with --ingest turns
(the default), or the memories a cached `locomo-eval extract` run chose with
--ingest extracted. Every QA item with evidence is then sent to recall, the
returned memory keys are mapped back to their source dialog ids, and evidence
coverage, recall@k / hit@k and recall at equal token budgets are computed per
category. The tome is destroyed as soon as its conversation is scored,
including on failure.

`locomo-eval answer <run-id> ...` then scores a finished run's answers
offline; see `locomo-eval answer --help`. `locomo-eval extract ...` has a
local LLM choose the memories to store instead; see `locomo-eval extract --help`.
"""

import argparse
import asyncio
import hashlib
import json
import os
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

from locomo_eval import answering, extraction
from locomo_eval.dataset import Sample, Turn, load_dataset
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


async def query(client: ConnectomeClient, sample: Sample, tome: str, key_map: KeyMap, k: int, concurrency: int) -> tuple[list[QuestionResult], dict[str, int]]:
    # Evidence is checked against the conversation, not the memories, so
    # turns no memory cites still count against coverage and recall.
    known_dia_ids = {turn.dia_id for turn in sample.turns}
    cited = {d for sources in key_map.values() for d in sources}
    skipped = {"no_evidence": 0, "unknown_evidence_only": 0, "unknown_evidence_ids": 0}
    semaphore = asyncio.Semaphore(concurrency)

    async def ask(question: str) -> tuple[tuple[Context, ...], tuple[tuple[str, ...], ...]]:
        async with semaphore:
            response = await client.recall(question, k=k, tome=tome, hydrate=True)
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


async def run(
    client: ConnectomeClient,
    args: argparse.Namespace,
    samples: list[Sample],
    run_id: str,
    key_maps: dict[str, KeyMap] | None = None,
    extracted: CachedRun | None = None,
) -> tuple[list[QuestionResult], dict[str, int], dict[str, KeyMap]]:
    """Ingest, score, and destroy each sample's tome, returning the results,
    skip counts, and each sample's key map.

    With extracted, the cached extraction's memories are ingested instead of
    the turns. Given key_maps (from an earlier --keep-tomes run named
    run_id), ingestion is skipped and that run's tomes are queried and left
    in place instead.
    """
    k = recall_k(args)
    reuse = key_maps is not None
    key_maps = dict(key_maps or {})
    results: list[QuestionResult] = []
    skipped: dict[str, int] = {}
    for sample in samples:
        tome = tome_for(run_id, sample.sample_id)
        started = time.monotonic()
        try:
            if reuse:
                key_map = key_maps[sample.sample_id]
            elif extracted is not None:
                memories = extracted.memories(sample.sample_id)
                key_map = key_maps[sample.sample_id] = await ingest_extracted(client, memories, tome, args.concurrency, not args.no_occurred_at, args.superseded)
            else:
                key_map = key_maps[sample.sample_id] = await ingest(client, sample, tome, args.concurrency, not args.no_occurred_at)
            ingested = time.monotonic()
            sample_results, sample_skipped = await query(client, sample, tome, key_map, k, args.concurrency)
        finally:
            if not args.keep_tomes and not reuse:
                await destroy(client, tome)
        results.extend(sample_results)
        for reason, count in sample_skipped.items():
            skipped[reason] = skipped.get(reason, 0) + count
        print(
            f"{sample.sample_id}: {len(key_map)} memories {'reused' if reuse else f'ingested in {ingested - started:.1f}s'}, "
            f"{len(sample_results)} questions scored in {time.monotonic() - ingested:.1f}s",
            file=sys.stderr,
        )
    return results, skipped, key_maps


def memory_stats(key_maps: dict[str, KeyMap], extracted: CachedRun | None = None) -> dict[str, dict[str, float | int]]:
    """Per sample and overall: memories, mean sources per memory, and
    entities (0 for raw turns, which carry none)."""
    stats: dict[str, dict[str, float | int]] = {}
    for sample_id, key_map in key_maps.items():
        entities = len(extracted.entity_ids(sample_id)) if extracted else 0
        stats[sample_id] = {"memories": len(key_map), "sources_per_memory": _ratio(sum(map(len, key_map.values())), len(key_map)), "entities": entities}
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
    """Key maps as a --keep-tomes run wrote them. Maps from before extracted
    ingestion hold one dialog id per key rather than a list."""
    return {
        sample_id: {key: (sources,) if isinstance(sources, str) else tuple(sources) for key, sources in key_map.items()}
        for sample_id, key_map in raw.items()
    }


async def cleanup(client: ConnectomeClient, samples: list[Sample], run_id: str) -> None:
    for sample in samples:
        tome = tome_for(run_id, sample.sample_id)
        await destroy(client, tome)
        print(f"destroyed {tome}", file=sys.stderr)


def run_config(client: ConnectomeClient, args: argparse.Namespace, samples: list[Sample], run_id: str, extracted: CachedRun | None = None) -> dict[str, object]:
    if extracted is None:
        chunking = {"unit": "one memory per dialog turn", "template": MEMORY_TEMPLATE, "occurred_at": "session date" if not args.no_occurred_at else None}
    else:
        chunking = {
            "unit": "one memory per extracted memory",
            "template": None,
            "occurred_at": "extracted" if not args.no_occurred_at else None,
            "superseded": args.superseded if extracted.config.get("variant") == extraction.LIFECYCLE else None,
        }
    return {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "base_url": client.base_url,
        "dataset": {"path": str(args.data), "sha256": _sha256(args.data)},
        "samples": [s.sample_id for s in samples],
        "ks": args.ks,
        "budgets": args.budgets,
        "token_counter": "words and punctuation marks (metrics.count_tokens)",
        "recall_k": recall_k(args),
        "answer_k": args.answer_k,
        # The backend reads these from its own environment and does not
        # expose them over HTTP, so they are recorded as reported by the
        # harness's environment - see the README for keeping the two in sync.
        "search": {
            "vector_weight": _float_env("SEARCH_VECTOR_WEIGHT", 0.6),
            "text_weight": _float_env("SEARCH_TEXT_WEIGHT", 0.4),
            "rrf_k": _float_env("SEARCH_RRF_K", 60),
            "text_query": os.environ.get("SEARCH_TEXT_QUERY") or "plain",
            "bm25_k1": _float_env("SEARCH_BM25_K1", 1.2),
            "bm25_b": _float_env("SEARCH_BM25_B", 0.75),
            "text_max_df": _float_env("SEARCH_TEXT_MAX_DF", 0.05),
        },
        "reused_tomes": args.reuse_tomes,
        "embedding_model": args.embedding_model,
        "ingestion": {"mode": args.ingest, "extraction": extracted.config if extracted else None},
        "chunking": chunking,
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


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="locomo-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="path to locomo10.json (default: %(default)s)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="where to write <run-id>.json (default: %(default)s)")
    parser.add_argument("--run-id", help="tome/result name suffix (default: a UTC timestamp)")
    parser.add_argument("--ks", type=_ks, default=DEFAULT_KS, help="comma-separated k values (default: 1,5,10)")
    parser.add_argument(
        "--budgets",
        type=_budgets,
        default=DEFAULT_BUDGETS,
        help="comma-separated token budgets for equal-budget recall; 0 for none (default: 64,128,256)",
    )
    parser.add_argument(
        "--ingest",
        choices=[TURNS, EXTRACTED],
        default=TURNS,
        help="ingest one memory per turn, or the memories a cached `locomo-eval extract` run chose (default: %(default)s)",
    )
    parser.add_argument("--extraction-cache", type=Path, default=DEFAULT_CACHE_PATH, help="--ingest extracted: the extract stage's cache (default: %(default)s)")
    parser.add_argument("--extractor-model", default=DEFAULT_EXTRACTOR_MODEL, help="--ingest extracted: whose extraction to ingest (default: %(default)s)")
    parser.add_argument("--extract-prompt", default=EXTRACT_VERSION, help="--ingest extracted: the extraction prompt version (default: %(default)s)")
    parser.add_argument(
        "--superseded",
        choices=[MARK, FORGET],
        default=MARK,
        help="--ingest extracted with a lifecycle extraction: mark superseded memories' relationships superseded_by their successor, or forget them (default: %(default)s)",
    )
    parser.add_argument("--compare", metavar="RUN_ID", help="print a finished run's metrics (e.g. a turns run) under this run's, row by row")
    parser.add_argument(
        "--answer-k",
        type=int,
        default=DEFAULT_ANSWER_K,
        help="retrieved turns an answer stage will use; recall fetches max(ks + [answer-k]) (default: %(default)s)",
    )
    parser.add_argument("--samples", type=lambda s: s.split(","), help="comma-separated sample ids to run (default: all)")
    parser.add_argument("--concurrency", type=int, default=8, help="max in-flight requests (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds (default: %(default)s)")
    parser.add_argument("--embedding-model", default="nomic-embed-text", help="recorded in the run config only (default: %(default)s)")
    parser.add_argument("--no-occurred-at", action="store_true", help="don't set occurred_at; the session date stays in the memory text")
    parser.add_argument("--keep-tomes", action="store_true", help="don't destroy tomes afterwards (for debugging; clean up with --cleanup)")
    parser.add_argument(
        "--reuse-tomes",
        metavar="RUN_ID",
        help="query the tomes an earlier --keep-tomes run left behind instead of ingesting (for comparing backend search settings on one index)",
    )
    parser.add_argument("--cleanup", metavar="RUN_ID", help="destroy the tomes left behind by RUN_ID and exit")
    args = parser.parse_args(argv)
    if not 1 <= args.answer_k <= MAX_SEARCH_K:
        parser.error(f"--answer-k must be between 1 and {MAX_SEARCH_K}")
    return args


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


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["answer"]:
        answering.main(argv[1:])
        return
    if argv[:1] == ["extract"]:
        extraction.main(argv[1:])
        return
    args = parse_args(argv)
    if not args.data.exists():
        sys.exit(f"dataset not found at {args.data}; see evals/locomo/README.md for the download")

    samples = load_dataset(args.data)
    if args.samples:
        wanted = set(args.samples)
        samples = [s for s in samples if s.sample_id in wanted]
        if missing := wanted - {s.sample_id for s in samples}:
            sys.exit(f"unknown sample ids: {', '.join(sorted(missing))}")

    client = ConnectomeClient(source_type=SOURCE_TYPE, timeout=args.timeout)
    if args.cleanup:
        asyncio.run(cleanup(client, samples, args.cleanup))
        return

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dt%H%M%Sz")
    baseline = None
    if args.compare:
        compare_path = args.results_dir / f"{args.compare}.json"
        if not compare_path.exists():
            sys.exit(f"no results at {compare_path} to --compare against")
        baseline = json.loads(compare_path.read_text(encoding="utf-8"))
    extracted = load_cached_run(args, samples) if args.ingest == EXTRACTED else None
    config = run_config(client, args, samples, run_id, extracted)
    tome_run_id, key_maps = run_id, None
    if args.reuse_tomes:
        tome_run_id = args.reuse_tomes
        key_maps_path = args.results_dir / f"{tome_run_id}.keys.json"
        if not key_maps_path.exists():
            sys.exit(f"no key map at {key_maps_path}; --reuse-tomes needs a run made with --keep-tomes")
        kept_path = args.results_dir / f"{tome_run_id}.json"
        if kept_path.exists():
            kept_mode = (json.loads(kept_path.read_text(encoding="utf-8"))["config"].get("ingestion") or {}).get("mode", TURNS)
            if kept_mode != args.ingest:
                sys.exit(f"run {tome_run_id} ingested {kept_mode}, not {args.ingest}; pass --ingest {kept_mode}")
        key_maps = load_key_maps(json.loads(key_maps_path.read_text(encoding="utf-8")))
        if missing := [s.sample_id for s in samples if s.sample_id not in key_maps]:
            sys.exit(f"run {tome_run_id} did not ingest: {', '.join(missing)}")
    try:
        results, skipped, key_maps = asyncio.run(run(client, args, samples, tome_run_id, key_maps, extracted))
    except httpx.HTTPError as exc:
        sys.exit(f"request to {config['base_url']} failed: {exc!r}")
    summary = summarize(results, args.ks, args.budgets)
    stats = memory_stats({s.sample_id: key_maps[s.sample_id] for s in samples}, extracted)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_tomes and not args.reuse_tomes:
        # Memory keys are random, so a later --reuse-tomes run needs this map
        # to score the kept tomes.
        with (args.results_dir / f"{run_id}.keys.json").open("w", encoding="utf-8") as f:
            json.dump(dump_key_maps(key_maps), f, indent=2)
    out_path = args.results_dir / f"{run_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"config": config, "summary": summary, "memories": stats, "skipped": skipped, "questions": [asdict(r) for r in results]},
            f,
            indent=2,
        )

    label = f"{run_id} ({ingestion_label(config)})"
    if baseline is None:
        print_summary(summary, args.ks, args.budgets)
        print_memory_stats(stats, label)
    else:
        baseline_label = f"{args.compare} ({ingestion_label(baseline['config'])})"
        print_summary(summary, args.ks, args.budgets, label, rescore(baseline, args.ks, args.budgets), baseline_label)
        print_memory_stats(stats, label, baseline.get("memories"), baseline_label)
    print(f"\nskipped: {skipped}\nwrote {out_path}")


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


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()
