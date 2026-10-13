"""Run the West Marches belief eval against a live Connectome backend.

The checked-in campaign (campaign/campaign.yaml) is written into a scratch
tome, temp-westmarches-<run-id>, through the same client calls the bot makes:
each player is asserted member_of their party, and each memory is remembered
with its acl, occurred_at and derived_from, then any memory it supersedes has
its relationships marked superseded_by it. Every question is then sent to
recall as its viewer (`as`), with hydrate off and with it on, and the results
are scored for leakage, staleness, perspective and gold recall. With
--answer-model and --judge-model, truth-status and graph answers are scored
too (see `westmarches-eval answer --help`). The tome is destroyed at the end,
also when the run fails.

The default command runs every stage in one go; they can also run apart:
`westmarches-eval ingest` fills the tome and writes
results/<run-id>.ingest.json, `westmarches-eval retrieve <run-id>` scores it
and writes results/<run-id>.json, `westmarches-eval answer <run-id> ...`
scores answers from that file, and `westmarches-eval cleanup <run-id>`
destroys the tome.

--fail-on-leak exits with status 1 when any result leaked.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx
from connectomeclient import ConnectomeClient
from dotenv import load_dotenv

from westmarches_eval import answering, scores
from westmarches_eval.campaign import (
    DEFAULT_CAMPAIGN_PATH,
    Campaign,
    CampaignError,
    load_campaign,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"
DEFAULT_KS = [5, 10]
DEFAULT_ANSWER_K = 10
# The backend caps search k at 50 - see maxSearchK in backend/resources/searchResource.go.
MAX_SEARCH_K = 50
GENERATED_BY = "process:westmarches-eval"
KEPT = "kept"
DESTROYED = "destroyed"
LEAK_EXIT_CODE = 1


def tome_for(run_id: str) -> str:
    return f"temp-westmarches-{run_id}".lower()


# A campaign memory id -> the key remember returned for it.
KeyMap = dict[str, str]


async def ingest(client: ConnectomeClient, campaign: Campaign, tome: str) -> KeyMap:
    """Write the campaign into tome, in order, and return the key map.

    Memories go one at a time: a facet needs its root's key and a
    supersession its successor's."""
    for party, (_, members) in campaign.parties.items():
        for member in members:
            _ = await client.assert_relationship(member, "member_of", party, tome=tome)
    keys: KeyMap = {}
    for memory in campaign.memories:
        result = await client.remember(
            memory.content,
            memory_type=memory.type,
            entities=memory.entities,
            relationships=[r.to_client() for r in memory.relationships],
            acl=list(memory.acl),
            tome=tome,
            occurred_at=memory.occurred_at,
            derived_from=keys[memory.derived_from] if memory.derived_from else None,
            sources=[{"id": memory.id, "title": f"{campaign.name}, session {memory.session}"}],
        )
        keys[memory.id] = result["key"]
        for old_id in memory.supersedes:
            for rel in campaign.memory(old_id).relationships:
                _ = await client.supersede_relationship(keys[old_id], rel.subject, rel.predicate, rel.object, superseded_by=keys[memory.id], tome=tome)
    return keys


async def retrieve(client: ConnectomeClient, campaign: Campaign, tome: str, keys: KeyMap, k: int, concurrency: int) -> list[dict]:
    """Ask every question as its viewer, hydrate off and on. Each question's
    record holds the hits as campaign memory ids and the contexts an answer
    stage reads."""
    ids = {key: memory_id for memory_id, key in keys.items()}
    semaphore = asyncio.Semaphore(concurrency)

    async def ask(question: str, viewer: str, hydrate: bool) -> tuple[list[str], list[str]]:
        async with semaphore:
            response = await client.recall(question, k=k, tome=tome, hydrate=hydrate, as_=viewer)
        hits = response.get("results") or []
        assert isinstance(hits, list)
        if unknown := [hit["key"] for hit in hits if hit["key"] not in ids]:
            # Every hit comes from this run's own tome, so an unmapped key
            # means the key shape changed - fail rather than score it.
            raise RuntimeError(f"recall returned keys not ingested into {tome}: {unknown[:3]}")
        return [ids[hit["key"]] for hit in hits], [answering.context_text(hit) for hit in hits]

    async def record(q) -> dict:
        (off_hits, off_contexts), (on_hits, on_contexts) = await asyncio.gather(ask(q.question, q.viewer, False), ask(q.question, q.viewer, True))
        return {
            "id": q.id,
            "kind": q.kind,
            "as": q.viewer,
            "question": q.question,
            "answer": q.answer,
            "truth": q.truth,
            "gold": list(q.gold),
            "stale": list(q.stale),
            "probes": list(q.probes),
            "hits": {answering.HYDRATE_OFF: off_hits, answering.HYDRATE_ON: on_hits},
            "contexts": {answering.HYDRATE_OFF: off_contexts, answering.HYDRATE_ON: on_contexts},
        }

    return list(await asyncio.gather(*(record(q) for q in campaign.questions)))


def summarize(campaign: Campaign, questions: list[dict], ks: Sequence[int]) -> dict[str, dict]:
    return {mode: scores.summarize(campaign, {q["id"]: q["hits"][mode] for q in questions}, ks) for mode in answering.MODES}


async def destroy(client: ConnectomeClient, tome: str) -> None:
    try:
        _ = await client.destroy_tome(tome)
    except httpx.HTTPError as exc:
        print(f"warning: failed to destroy tome {tome}: {exc}", file=sys.stderr)


async def tome_missing(client: ConnectomeClient, tome: str, keys: KeyMap) -> bool:
    """Whether tome no longer holds its first memory. A search of a missing
    tome just comes back empty, which would score as perfect - no leaks."""
    try:
        _ = await client.get_memory(next(iter(keys.values())), tome=tome)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise
        return True
    return False


def manifest_path(results_dir: Path, run_id: str) -> Path:
    return results_dir / f"{run_id}.ingest.json"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_manifest(results_dir: Path, run_id: str) -> dict:
    path = manifest_path(results_dir, run_id)
    if not path.exists():
        sys.exit(f"no ingest manifest at {path}; run `westmarches-eval ingest --run-id {run_id}` first")
    return json.loads(path.read_text(encoding="utf-8"))


def ingest_stage(client: ConnectomeClient, args: argparse.Namespace, campaign: Campaign, run_id: str) -> dict:
    path = manifest_path(args.results_dir, run_id)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("tomes") == KEPT:
        sys.exit(f"run {run_id} still has a tome from an earlier ingest; destroy it with `westmarches-eval cleanup {run_id}` or pick another --run-id")
    tome = tome_for(run_id)
    started = time.monotonic()
    try:
        keys = asyncio.run(ingest(client, campaign, tome))
    except BaseException as exc:
        # No manifest will point at the tome, so nothing else would destroy it.
        asyncio.run(destroy(client, tome))
        if isinstance(exc, httpx.HTTPError):
            sys.exit(f"request to {client.base_url} failed: {exc!r}")
        raise
    print(f"{tome}: {len(keys)} memories ingested in {time.monotonic() - started:.1f}s", file=sys.stderr)
    manifest = {
        "run_id": run_id,
        "tome": tome,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "base_url": client.base_url,
        "campaign": {"path": str(args.campaign), "sha256": campaign.sha256},
        "embedding_model": args.embedding_model,
        "tomes": KEPT,
        "key_map": keys,
    }
    write_json(path, manifest)
    return manifest


def retrieve_stage(client: ConnectomeClient, args: argparse.Namespace, campaign: Campaign, manifest: dict, name: str) -> dict:
    if manifest["campaign"]["sha256"] != campaign.sha256:
        sys.exit(f"{args.campaign} is not the campaign run {manifest['run_id']} ingested; pass --campaign or ingest again")
    k = max([*args.ks, args.answer_k])
    started_at = datetime.now(UTC).isoformat()
    try:
        if asyncio.run(tome_missing(client, manifest["tome"], manifest["key_map"])):
            sys.exit(f"tome {manifest['tome']} is missing from {client.base_url}; ingest again with `westmarches-eval ingest --run-id {manifest['run_id']}`")
        questions = asyncio.run(retrieve(client, campaign, manifest["tome"], manifest["key_map"], k, args.concurrency))
    except httpx.HTTPError as exc:
        sys.exit(f"request to {client.base_url} failed: {exc!r}")
    run = {
        "config": {
            "run_id": name,
            "ingest_run_id": manifest["run_id"],
            "started_at": started_at,
            "git_commit": _git_commit(),
            "base_url": client.base_url,
            "campaign": manifest["campaign"],
            "embedding_model": manifest["embedding_model"],
            "ks": args.ks,
            "recall_k": k,
            "answer_k": args.answer_k,
            "ranking": "backend default",
        },
        "retrieval": summarize(campaign, questions, args.ks),
        "questions": questions,
    }
    write_json(args.results_dir / f"{name}.json", run)
    return run


def cleanup_stage(client: ConnectomeClient, args: argparse.Namespace, run_id: str) -> None:
    path = manifest_path(args.results_dir, run_id)
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    tome = manifest["tome"] if manifest else tome_for(run_id)
    asyncio.run(destroy(client, tome))
    print(f"destroyed {tome}", file=sys.stderr)
    if manifest is not None:
        manifest["tomes"] = DESTROYED
        write_json(path, manifest)


def answer_stage(args: argparse.Namespace, name: str) -> dict:
    results_path = args.results_dir / f"{name}.json"
    if not results_path.exists():
        sys.exit(f"no results at {results_path}; run `westmarches-eval retrieve {name}` first")
    try:
        config = answering.answer_config(args.answer_model, args.judge_model, args.answer_k, args.answer_prompt, args.judge_prompt)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))
    return asyncio.run(answering.run_answers(results_path, config, args.llm_concurrency))


def leaked(run: dict) -> int:
    return max(int(summary["leakage"]["leaked"]) for summary in run["retrieval"].values())


def print_report(run: dict, answers: dict | None = None) -> None:
    """Every score side by side, hydrate off and on."""
    config = run["config"]
    retrieval = run["retrieval"]
    off, on = retrieval[answering.HYDRATE_OFF], retrieval[answering.HYDRATE_ON]
    print(f"West Marches eval: {config['run_id']} against {config['base_url']}")
    print(f"{'':<34}{'hydrate off':>14}{'hydrate on':>14}")

    def row(label: str, a: object, b: object) -> None:
        print(f"{label:<34}{_cell(a):>14}{_cell(b):>14}")

    row("leakage (leaked/results)", f"{off['leakage']['leaked']}/{off['leakage']['results']}", f"{on['leakage']['leaked']}/{on['leakage']['results']}")
    for k in config["ks"]:
        a, b = off[f"k={k}"], on[f"k={k}"]
        row(f"staleness@{k}", a["staleness"][f"staleness@{k}"], b["staleness"][f"staleness@{k}"])
        row(f"  stale_first@{k}", a["staleness"][f"stale_first@{k}"], b["staleness"][f"stale_first@{k}"])
        row(f"perspective@{k}", a["perspective"][f"perspective@{k}"], b["perspective"][f"perspective@{k}"])
        row(f"  root_first@{k}", a["perspective"][f"root_first@{k}"], b["perspective"][f"root_first@{k}"])
        row(f"graph recall@{k}", a["graph"][f"recall@{k}"], b["graph"][f"recall@{k}"])
        row(f"  graph all_gold@{k}", a["graph"][f"all_gold@{k}"], b["graph"][f"all_gold@{k}"])
        for kind, values in a["recall"].items():
            row(f"recall@{k} {kind}", values[f"recall@{k}"], b["recall"][kind][f"recall@{k}"])
    if answers is not None:
        cfg = answers["config"]
        modes = answers["modes"]
        print(f"answers: {cfg['answer_model']} ({cfg['answer_prompt']['version']}), judged by {cfg['judge_model']}, answer_k={cfg['answer_k']}")
        for key, label in (("truth_status", "truth status accuracy"), ("truth_content", "  truth content accuracy"), ("graph", "graph answer accuracy")):
            a, b = modes[answering.HYDRATE_OFF]["summary"][key], modes[answering.HYDRATE_ON]["summary"][key]
            row(label, a["accuracy"], b["accuracy"])
            if a["judge_null"] or b["judge_null"]:
                row("  unparseable verdicts", a["judge_null"], b["judge_null"])
    n = off["n"]
    print("questions: " + ", ".join(f"{count} {kind}" for kind, count in n.items()))
    for mode, summary in retrieval.items():
        for leak in summary["leakage"]["leaks"]:
            print(f"LEAK ({mode}): {leak['question']} as {leak['as']} got {leak['memory']} at rank {leak['rank']}")


def _cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN_PATH, help="the campaign file (default: %(default)s)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="where a run's files live (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds (default: %(default)s)")


def _ingest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-model", default="nomic-embed-text", help="recorded in the run config only (default: %(default)s)")


def _retrieve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ks", type=_ks, default=DEFAULT_KS, help="comma-separated k values (default: 5,10)")
    parser.add_argument("--answer-k", type=int, default=DEFAULT_ANSWER_K, help="results an answer is given; recall fetches max(ks + [answer-k]) (default: %(default)s)")
    parser.add_argument("--concurrency", type=int, default=8, help="max in-flight recall requests (default: %(default)s)")


def _answer_args(parser: argparse.ArgumentParser, required: bool) -> None:
    parser.add_argument("--answer-model", required=required, help="answer model, provider:model (e.g. ollama:qwen3:8b)")
    parser.add_argument("--judge-model", required=required, help="judge model, provider:model (e.g. ollama:qwen3:14b)")
    parser.add_argument("--answer-prompt", default=answering.ANSWER_VERSION, help="answer prompt version (default: %(default)s)")
    parser.add_argument("--judge-prompt", default=answering.DEFAULT_JUDGE_VERSION, help="LoCoMo judge prompt for content (default: %(default)s)")
    parser.add_argument("--llm-concurrency", type=int, default=1, help="max in-flight LLM calls (default: %(default)s)")


def _check_answer_k(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 1 <= args.answer_k <= MAX_SEARCH_K:
        parser.error(f"--answer-k must be between 1 and {MAX_SEARCH_K}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="westmarches-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common_args(parser)
    parser.add_argument("--run-id", help="tome/result name suffix (default: a UTC timestamp)")
    _ingest_args(parser)
    _retrieve_args(parser)
    _answer_args(parser, required=False)
    parser.add_argument("--keep-tome", action="store_true", help="skip cleanup; destroy the tome later with `westmarches-eval cleanup`")
    parser.add_argument("--fail-on-leak", action="store_true", help=f"exit with status {LEAK_EXIT_CODE} when any result leaked")
    args = parser.parse_args(argv)
    _check_answer_k(parser, args)
    if bool(args.answer_model) != bool(args.judge_model):
        parser.error("--answer-model and --judge-model go together")
    return args


def parse_ingest_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="westmarches-eval ingest", description="Write the campaign into temp-westmarches-<run-id> and leave it there.")
    _common_args(parser)
    parser.add_argument("--run-id", help="tome/manifest name suffix (default: a UTC timestamp)")
    _ingest_args(parser)
    return parser.parse_args(argv)


def parse_retrieve_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="westmarches-eval retrieve", description="Score the tome `westmarches-eval ingest` left behind. Writes results/<run-id>.json, or results/<run-id>.<tag>.json with --tag.")
    parser.add_argument("run_id", help="the ingest run to score")
    parser.add_argument("--tag", help="write results/<run-id>.<tag>.json")
    _common_args(parser)
    _retrieve_args(parser)
    parser.add_argument("--fail-on-leak", action="store_true", help=f"exit with status {LEAK_EXIT_CODE} when any result leaked")
    args = parser.parse_args(argv)
    _check_answer_k(parser, args)
    return args


def parse_answer_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="westmarches-eval answer", description=answering.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="the retrieval results to answer (results/<name>.json)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="where a run's files live (default: %(default)s)")
    parser.add_argument("--answer-k", type=int, default=DEFAULT_ANSWER_K, help="results an answer is given (default: %(default)s)")
    _answer_args(parser, required=True)
    args = parser.parse_args(argv)
    _check_answer_k(parser, args)
    return args


def parse_cleanup_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="westmarches-eval cleanup", description="Destroy the tome a run left behind.")
    parser.add_argument("run_id", help="the run whose tome to destroy")
    _common_args(parser)
    return parser.parse_args(argv)


def client_for(args: argparse.Namespace) -> ConnectomeClient:
    return ConnectomeClient(generated_by=GENERATED_BY, timeout=args.timeout)


def campaign_for(args: argparse.Namespace) -> Campaign:
    try:
        return load_campaign(args.campaign)
    except (CampaignError, OSError) as exc:
        sys.exit(f"{args.campaign}: {exc}")


def ingest_main(argv: list[str]) -> None:
    args = parse_ingest_args(argv)
    run_id = args.run_id or _timestamp()
    ingest_stage(client_for(args), args, campaign_for(args), run_id)
    print(f"wrote {manifest_path(args.results_dir, run_id)}")
    print(f"score it with `westmarches-eval retrieve {run_id}`; destroy its tome with `westmarches-eval cleanup {run_id}`")


def retrieve_main(argv: list[str]) -> None:
    args = parse_retrieve_args(argv)
    manifest = read_manifest(args.results_dir, args.run_id)
    if manifest.get("tomes") != KEPT:
        sys.exit(f"run {args.run_id}'s tome was destroyed; ingest again with `westmarches-eval ingest --run-id {args.run_id}`")
    name = f"{args.run_id}.{args.tag}" if args.tag else args.run_id
    run = retrieve_stage(client_for(args), args, campaign_for(args), manifest, name)
    print_report(run)
    print(f"wrote {args.results_dir / f'{name}.json'}")
    if args.fail_on_leak and leaked(run):
        sys.exit(LEAK_EXIT_CODE)


def answer_main(argv: list[str]) -> None:
    args = parse_answer_args(argv)
    answers = answer_stage(args, args.name)
    run = json.loads((args.results_dir / f"{args.name}.json").read_text(encoding="utf-8"))
    print_report(run, answers)


def cleanup_main(argv: list[str]) -> None:
    args = parse_cleanup_args(argv)
    cleanup_stage(client_for(args), args, args.run_id)


def run_all(args: argparse.Namespace) -> None:
    """ingest, retrieve, answer (with models), then cleanup unless --keep-tome."""
    campaign = campaign_for(args)
    run_id = args.run_id or _timestamp()
    client = client_for(args)
    manifest = ingest_stage(client, args, campaign, run_id)
    try:
        run = retrieve_stage(client, args, campaign, manifest, run_id)
    finally:
        if not args.keep_tome:
            cleanup_stage(client, args, run_id)
    answers = answer_stage(args, run_id) if args.answer_model else None
    print_report(run, answers)
    print(f"wrote {args.results_dir / f'{run_id}.json'}")
    if args.fail_on_leak and leaked(run):
        sys.exit(LEAK_EXIT_CODE)


SUBCOMMANDS = {
    "ingest": ingest_main,
    "retrieve": retrieve_main,
    "answer": answer_main,
    "cleanup": cleanup_main,
}


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in SUBCOMMANDS:
        SUBCOMMANDS[argv[0]](argv[1:])
        return
    run_all(parse_args(argv))


def _ks(value: str) -> list[int]:
    ks = sorted({int(k) for k in value.split(",")})
    if not ks or ks[0] < 1 or ks[-1] > MAX_SEARCH_K:
        raise argparse.ArgumentTypeError(f"k values must be between 1 and {MAX_SEARCH_K}")
    return ks


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dt%H%M%Sz")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()
