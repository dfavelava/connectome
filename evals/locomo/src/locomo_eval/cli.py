"""Run the LoCoMo retrieval-recall eval against a live Connectome backend.

Each conversation is ingested into its own scratch tome
(temp-locomo-<run>-<sample>), one memory per dialog turn. Every QA item with
evidence is then sent to recall, the returned memory keys are mapped back to
dialog ids, and evidence recall@k / hit@k are computed per category. The tome
is destroyed as soon as its conversation is scored, including on failure.
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
from connectomeclient import ConnectomeClient
from dotenv import load_dotenv

from locomo_eval.dataset import Sample, Turn, load_dataset
from locomo_eval.metrics import QuestionResult, summarize

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "locomo10.json"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"
DEFAULT_KS = [1, 5, 10]
# The backend caps search k at 50 - see maxSearchK in backend/resources/searchResource.go.
MAX_SEARCH_K = 50
MEMORY_TEMPLATE = "[{session_date}] {speaker}: {text}"
SOURCE_TYPE = "locomo-eval"


def memory_text(turn: Turn) -> str:
    return MEMORY_TEMPLATE.format(session_date=turn.session_date, speaker=turn.speaker, text=turn.text)


def tome_for(run_id: str, sample_id: str) -> str:
    return f"temp-locomo-{run_id}-{sample_id}".lower()


async def ingest(client: ConnectomeClient, sample: Sample, tome: str, concurrency: int, use_occurred_at: bool) -> dict[str, str]:
    """Write one memory per turn and return the key -> dia_id map (keys are random)."""
    semaphore = asyncio.Semaphore(concurrency)

    async def write(turn: Turn) -> tuple[str, str]:
        async with semaphore:
            result = await client.remember(
                memory_text(turn),
                memory_type="event",
                tome=tome,
                occurred_at=turn.occurred_at if use_occurred_at else None,
            )
        return result["key"], turn.dia_id

    return dict(await asyncio.gather(*(write(turn) for turn in sample.turns)))


async def query(client: ConnectomeClient, sample: Sample, tome: str, key_to_dia: dict[str, str], k: int, concurrency: int) -> tuple[list[QuestionResult], dict[str, int]]:
    known_dia_ids = set(key_to_dia.values())
    skipped = {"no_evidence": 0, "unknown_evidence_only": 0, "unknown_evidence_ids": 0}
    semaphore = asyncio.Semaphore(concurrency)

    # Search returns tome-scoped blob keys (tomes/<tome>/mem_<uuid>.md - see
    # TomeScopedKey in backend/resources/tome.go), while remember returns the
    # bare key, so strip the scope before mapping back to a dialog id. Tracked
    # as issue #16; removeprefix stays a no-op once search returns bare keys.
    scope_prefix = f"tomes/{tome}/"

    async def ask(question: str) -> tuple[str, ...]:
        async with semaphore:
            response = await client.recall(question, k=k, tome=tome)
        hits = response.get("results") or []
        assert isinstance(hits, list)
        keys = [hit["key"].removeprefix(scope_prefix) for hit in hits]
        if unknown := [key for key in keys if key not in key_to_dia]:
            # Every hit comes from this run's own tome, so an unmapped key means
            # the key shape changed - fail rather than silently score zero.
            raise RuntimeError(f"recall returned keys not ingested into {tome}: {unknown[:3]}")
        return tuple(key_to_dia[key] for key in keys)

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
            retrieved=hits,
        )
        for (item, evidence), hits in zip(scored, retrieved, strict=True)
    ]
    return results, skipped


async def destroy(client: ConnectomeClient, tome: str) -> None:
    try:
        _ = await client.destroy_tome(tome)
    except httpx.HTTPError as exc:
        print(f"warning: failed to destroy tome {tome}: {exc}", file=sys.stderr)


async def run(client: ConnectomeClient, args: argparse.Namespace, samples: list[Sample], run_id: str) -> tuple[list[QuestionResult], dict[str, int]]:
    k = max(args.ks)
    results: list[QuestionResult] = []
    skipped: dict[str, int] = {}
    for sample in samples:
        tome = tome_for(run_id, sample.sample_id)
        started = time.monotonic()
        try:
            key_to_dia = await ingest(client, sample, tome, args.concurrency, not args.no_occurred_at)
            ingested = time.monotonic()
            sample_results, sample_skipped = await query(client, sample, tome, key_to_dia, k, args.concurrency)
        finally:
            if not args.keep_tomes:
                await destroy(client, tome)
        results.extend(sample_results)
        for reason, count in sample_skipped.items():
            skipped[reason] = skipped.get(reason, 0) + count
        print(
            f"{sample.sample_id}: {len(sample.turns)} turns ingested in {ingested - started:.1f}s, "
            f"{len(sample_results)} questions scored in {time.monotonic() - ingested:.1f}s",
            file=sys.stderr,
        )
    return results, skipped


async def cleanup(client: ConnectomeClient, samples: list[Sample], run_id: str) -> None:
    for sample in samples:
        tome = tome_for(run_id, sample.sample_id)
        await destroy(client, tome)
        print(f"destroyed {tome}", file=sys.stderr)


def run_config(client: ConnectomeClient, args: argparse.Namespace, samples: list[Sample], run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "base_url": client.base_url,
        "dataset": {"path": str(args.data), "sha256": _sha256(args.data)},
        "samples": [s.sample_id for s in samples],
        "ks": args.ks,
        "recall_k": max(args.ks),
        # The backend reads these from its own environment and does not
        # expose them over HTTP, so they are recorded as reported by the
        # harness's environment - see the README for keeping the two in sync.
        "search": {
            "vector_weight": _float_env("SEARCH_VECTOR_WEIGHT", 0.6),
            "text_weight": _float_env("SEARCH_TEXT_WEIGHT", 0.4),
            "rrf_k": _float_env("SEARCH_RRF_K", 60),
        },
        "embedding_model": args.embedding_model,
        "chunking": {
            "unit": "one memory per dialog turn",
            "template": MEMORY_TEMPLATE,
            "occurred_at": "session date" if not args.no_occurred_at else None,
        },
        "concurrency": args.concurrency,
    }


def print_summary(summary: dict[str, dict[str, float | int]], ks: list[int]) -> None:
    columns = ["n", *(f"recall@{k}" for k in ks), *(f"hit@{k}" for k in ks)]
    print(f"{'category':<14}" + "".join(f"{c:>11}" for c in columns))
    for name, row in summary.items():
        cells = [f"{row[c]:>11}" if c == "n" else f"{row[c]:>11.3f}" for c in columns]
        print(f"{name:<14}" + "".join(cells))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="locomo-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="path to locomo10.json (default: %(default)s)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="where to write <run-id>.json (default: %(default)s)")
    parser.add_argument("--run-id", help="tome/result name suffix (default: a UTC timestamp)")
    parser.add_argument("--ks", type=_ks, default=DEFAULT_KS, help="comma-separated k values (default: 1,5,10)")
    parser.add_argument("--samples", type=lambda s: s.split(","), help="comma-separated sample ids to run (default: all)")
    parser.add_argument("--concurrency", type=int, default=8, help="max in-flight requests (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds (default: %(default)s)")
    parser.add_argument("--embedding-model", default="nomic-embed-text", help="recorded in the run config only (default: %(default)s)")
    parser.add_argument("--no-occurred-at", action="store_true", help="don't set occurred_at; the session date stays in the memory text")
    parser.add_argument("--keep-tomes", action="store_true", help="don't destroy tomes afterwards (for debugging; clean up with --cleanup)")
    parser.add_argument("--cleanup", metavar="RUN_ID", help="destroy the tomes left behind by RUN_ID and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
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
    config = run_config(client, args, samples, run_id)
    try:
        results, skipped = asyncio.run(run(client, args, samples, run_id))
    except httpx.HTTPError as exc:
        sys.exit(f"request to {config['base_url']} failed: {exc!r}")
    summary = summarize(results, args.ks)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"{run_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"config": config, "summary": summary, "skipped": skipped, "questions": [asdict(r) for r in results]},
            f,
            indent=2,
        )

    print_summary(summary, args.ks)
    print(f"\nskipped: {skipped}\nwrote {out_path}")


def _ks(value: str) -> list[int]:
    ks = sorted({int(k) for k in value.split(",")})
    if not ks or ks[0] < 1 or ks[-1] > MAX_SEARCH_K:
        raise argparse.ArgumentTypeError(f"k values must be between 1 and {MAX_SEARCH_K}")
    return ks


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
