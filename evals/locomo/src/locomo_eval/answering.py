"""The answer stage: score a finished retrieval run offline, with no backend calls.

    uv run locomo-eval answer <run-id> --answer-model ollama:<model> --judge-model ollama:<model>

Reads results/<run-id>.json (written by the retrieval run, which stores each
question's gold answer and retrieved contexts). For each question it answers
from the top `answer_k` contexts, scores the answer by token F1, and asks the
judge for a CORRECT/WRONG verdict. The summary goes back into the same file
under "answers", keyed by a hash of the answer config, so one retrieval run
can be scored by several answer and judge configs over the same contexts.

`--answer-think` lets the answer model think before it answers (the judge
never does), with a larger output cap for the thinking tokens. It is part of
the config, so thinking and non-thinking scores sit side by side.

Every finished stage call is appended to results/<run-id>.<cfg-hash>.jsonl
(fsynced per line) keyed by (question_id, stage, model, prompt version and
sha256, context hash, sampling options). A rerun skips keys already there, so
a killed run loses at most the calls in flight. All of the run's checkpoint
files are read, so a config that only changes the judge reuses the answers of
an earlier config. The answer's context hash covers the question and the
contexts it was shown, and the judge's covers the generated and gold answers,
so a changed context forces a fresh answer and a changed answer a fresh
verdict.
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from locomo_eval.dataset import CATEGORY_NAMES, sample_ids
from locomo_eval.llm import (
    Completion,
    LLMClient,
    LLMError,
    SamplingOptions,
    cost_usd,
    load_pricing,
    price_for,
)
from locomo_eval.metrics import Context, QuestionResult, summarize, summarize_answers
from locomo_eval.prompts import (
    ANSWER_VERSION,
    JUDGE_VERSION,
    NOT_MENTIONED,
    Prompt,
    answer_prompt,
    judge,
    judge_prompt,
    load_prompt,
)
from locomo_eval.scoring import ADVERSARIAL, score_answer

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"
CATEGORY_IDS = {name: number for number, name in CATEGORY_NAMES.items()}
ANSWER = "answer"
JUDGE = "judge"
STAGES = (ANSWER, JUDGE)
_CFG_HASH = re.compile(r"^[0-9a-f]{12}$")
# Output cap for an answer call with thinking on: the thinking counts against
# num_predict, and a cut-off thought leaves an empty answer.
THINK_NUM_PREDICT = 4096


@dataclass(frozen=True)
class AnswerConfig:
    answer_model: str
    judge_model: str
    answer_prompt: Prompt
    judge_prompt: Prompt
    answer_k: int
    options: SamplingOptions = SamplingOptions()
    # Let the answer model think; the judge's options are left as they are.
    answer_think: bool = False

    def to_dict(self) -> dict:
        config = {
            "answer_model": self.answer_model,
            "judge_model": self.judge_model,
            "answer_prompt": self.answer_prompt.config(),
            "judge_prompt": self.judge_prompt.config(),
            "answer_k": self.answer_k,
            "temperature": self.options.temperature,
            "options": asdict(self.options),
        }
        if self.answer_think:
            # Only when on, so configs from before the option keep their hashes.
            config["answer_options"] = asdict(self.stage_options(ANSWER))
        return config

    def hash(self) -> str:
        """12 hex digits identifying everything that decides the scores."""
        return _sha256(json.dumps(self.to_dict(), sort_keys=True))[:12]

    def model(self, stage: str) -> str:
        return self.answer_model if stage == ANSWER else self.judge_model

    def prompt(self, stage: str) -> Prompt:
        return self.answer_prompt if stage == ANSWER else self.judge_prompt

    def stage_options(self, stage: str) -> SamplingOptions:
        if stage == ANSWER and self.answer_think:
            return replace(self.options, think=True, num_predict=max(self.options.num_predict, THINK_NUM_PREDICT))
        return self.options


@dataclass(frozen=True)
class Question:
    """One scored question of a retrieval run, as the answer stage needs it."""

    question_id: str
    sample_id: str
    question: str
    category: str
    answer: str | None
    adversarial_answer: str | None
    contexts: tuple[Context, ...]

    @property
    def category_id(self) -> int:
        return CATEGORY_IDS[self.category]


def load_questions(run: dict) -> list[Question]:
    return [
        Question(
            question_id=q["question_id"],
            sample_id=q["sample_id"],
            question=q["question"],
            category=q["category"],
            answer=q.get("answer"),
            adversarial_answer=q.get("adversarial_answer"),
            contexts=tuple(Context(**c) for c in q.get("contexts") or ()),
        )
        for q in run["questions"]
    ]


def answer_context_hash(question: Question, answer_k: int) -> str:
    """What an answer depends on besides the model and prompt: the question
    and the top answer_k contexts it is shown."""
    contexts = [[c.dia_id, c.text] for c in question.contexts[:answer_k]]
    return _sha256(json.dumps([question.question, contexts]))[:16]


def judge_context_hash(question: Question, generated: str) -> str:
    """What a verdict depends on: the question, gold, trap and generated answers."""
    return _sha256(json.dumps([question.question, question.category, question.answer, question.adversarial_answer, generated]))[:16]


def options_hash(options: SamplingOptions) -> str:
    return _sha256(json.dumps(asdict(options), sort_keys=True))[:12]


def record_key(record: dict) -> tuple:
    return (
        record["question_id"],
        record["stage"],
        record["model"],
        record["prompt_version"],
        record["prompt_sha256"],
        record["context_hash"],
        record["options_hash"],
    )


class Checkpoint:
    """Finished stage calls, appended one JSON line at a time to
    <run-id>.<cfg-hash>.jsonl and read back from every checkpoint of the run."""

    def __init__(self, results_dir: Path, run_id: str, cfg_hash: str):
        self.path = results_dir / f"{run_id}.{cfg_hash}.jsonl"
        self.records: dict[tuple, dict] = {}
        for path in checkpoint_paths(results_dir, run_id):
            for record in _read_jsonl(path):
                self.records[record_key(record)] = record

    def get(self, key: tuple) -> dict | None:
        return self.records.get(key)

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.records[record_key(record)] = record


def checkpoint_paths(results_dir: Path, run_id: str) -> list[Path]:
    prefix = f"{run_id}."
    return sorted(
        p
        for p in results_dir.glob(f"{glob_escape(run_id)}.*.jsonl")
        if p.name.startswith(prefix) and _CFG_HASH.match(p.name[len(prefix) : -len(".jsonl")])
    )


def glob_escape(value: str) -> str:
    return re.sub(r"([*?\[])", r"[\1]", value)


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A line cut short by a kill; its call is simply redone.
                continue


@dataclass
class Tally:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


class _Recording:
    """Forwards complete() to the client and tallies one stage call's usage,
    retries included, so each checkpoint record carries its own cost."""

    def __init__(self, client):
        self.client = client
        self.options = client.options
        self.tally = Tally()

    async def complete(self, *args, **kwargs) -> Completion:
        started = time.monotonic()
        completion = await self.client.complete(*args, **kwargs)
        self.tally.calls += 1
        self.tally.input_tokens += completion.input_tokens
        self.tally.output_tokens += completion.output_tokens
        self.tally.seconds += time.monotonic() - started
        return completion


@dataclass
class Progress:
    total: int
    done: int = 0
    # Questions that needed at least one fresh call, and the time and output
    # tokens they took, for the ETA.
    fresh: int = 0
    fresh_calls: int = 0
    cached_calls: int = 0
    output_tokens: int = 0
    started: float = field(default_factory=time.monotonic)

    def line(self) -> str:
        elapsed = time.monotonic() - self.started
        text = f"[{self.done}/{self.total}] {self.fresh_calls} calls, {self.cached_calls} cached, {elapsed:.0f}s"
        if self.fresh and elapsed > 0:
            text += f", {self.output_tokens / elapsed:.1f} output tok/s"
            text += f", ETA {_duration(elapsed / self.fresh * (self.total - self.done))}"
        return text


@dataclass
class AnswerRun:
    questions: list[dict]
    # The records each question's result came from, by stage.
    records: dict[str, list[dict]]


async def run_answers(
    client,
    config: AnswerConfig,
    questions: list[Question],
    checkpoint: Checkpoint,
    *,
    concurrency: int = 1,
    on_progress: Callable[[Progress], None] | None = None,
) -> AnswerRun:
    """Answer, F1-score and judge each question, reusing checkpointed calls.

    `client` is an LLMClient or anything with its `options` and `complete()`.
    An LLM failure propagates once the calls in flight are cancelled; every
    call already finished is in the checkpoint for the rerun."""
    semaphore = asyncio.Semaphore(concurrency)
    progress = Progress(total=len(questions))

    async def stage(question: Question, name: str, context_hash: str, call) -> tuple[dict, bool]:
        prompt = config.prompt(name)
        record = {
            "question_id": question.question_id,
            "stage": name,
            "model": config.model(name),
            "prompt_version": prompt.version,
            "prompt_sha256": prompt.sha256,
            "context_hash": context_hash,
            "options_hash": options_hash(config.stage_options(name)),
        }
        if (cached := checkpoint.get(record_key(record))) is not None:
            progress.cached_calls += 1
            return cached, False
        recording = _Recording(client)
        output = await call(recording)
        record["output"] = output
        record["usage"] = asdict(recording.tally)
        record["finished_at"] = datetime.now(UTC).isoformat()
        checkpoint.append(record)
        progress.fresh_calls += 1
        progress.output_tokens += recording.tally.output_tokens
        return record, True

    async def one(question: Question) -> tuple[dict, dict, dict]:
        async with semaphore:

            async def call_answer(recording: _Recording) -> dict:
                text = answer_prompt(config.answer_prompt, question.question, question.contexts[: config.answer_k])
                completion = await recording.complete(config.answer_model, "", text, stage=ANSWER, options=config.stage_options(ANSWER))
                return {"answer": completion.text}

            answer_record, fresh_answer = await stage(question, ANSWER, answer_context_hash(question, config.answer_k), call_answer)
            generated = answer_record["output"]["answer"]

            async def call_judge(recording: _Recording) -> dict:
                text = judge_prompt(config.judge_prompt, question.question, question.category_id, generated, question.answer, question.adversarial_answer)
                verdict = await judge(recording, config.judge_model, text)
                return {"label": verdict.judge_label, "reasoning": verdict.judge_reasoning, "attempts": verdict.attempts, "error": verdict.judge_error}

            judge_record, fresh_judge = await stage(question, JUDGE, judge_context_hash(question, generated), call_judge)
            verdict = judge_record["output"]
            result = {
                "id": question.question_id,
                "category": question.category,
                "question": question.question,
                "gold": NOT_MENTIONED if question.category_id == ADVERSARIAL else question.answer,
                "answer": generated,
                "f1": score_answer(generated, question.answer, question.category_id),
                "judge_label": verdict["label"],
                "judge_reasoning": verdict["reasoning"],
            }
            progress.done += 1
            progress.fresh += fresh_answer or fresh_judge
            if on_progress:
                on_progress(progress)
            return result, answer_record, judge_record

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(one(q)) for q in questions]
    finished = [task.result() for task in tasks]
    return AnswerRun(
        questions=[result for result, _, _ in finished],
        records={ANSWER: [a for _, a, _ in finished], JUDGE: [j for _, _, j in finished]},
    )


def usage_totals(run: AnswerRun, config: AnswerConfig, pricing: dict[str, dict[str, float]]) -> dict[str, dict]:
    """Per-stage totals over every call behind the results, checkpointed
    ones from earlier invocations included: what this config's scores cost."""
    totals = {}
    for name in STAGES:
        tally = Tally()
        for record in run.records[name]:
            usage = record["usage"]
            tally.calls += usage["calls"]
            tally.input_tokens += usage["input_tokens"]
            tally.output_tokens += usage["output_tokens"]
            tally.seconds += usage["seconds"]
        totals[name] = {
            **asdict(tally),
            "seconds": round(tally.seconds, 3),
            "cost_usd": cost_usd(pricing, config.model(name), tally.input_tokens, tally.output_tokens),
        }
    return totals


def answers_entry(run: AnswerRun, config: AnswerConfig, pricing: dict[str, dict[str, float]], extra_config: dict | None = None) -> dict:
    return {
        "config": {
            **config.to_dict(),
            "pricing": {model: price_for(pricing, model) for model in dict.fromkeys([config.answer_model, config.judge_model])},
            **(extra_config or {}),
        },
        "summary": summarize_answers(run.questions),
        "usage": usage_totals(run, config, pricing),
        "questions": run.questions,
    }


def select_questions(questions: list[Question], samples: list[str] | None, limit: int | None) -> tuple[list[Question], dict[str, int]]:
    """The questions to score, and how many were skipped and why."""
    skipped = {"no_gold_answer": 0}
    selected = []
    for question in questions:
        if samples and question.sample_id not in samples:
            continue
        if question.category_id != ADVERSARIAL and question.answer is None:
            skipped["no_gold_answer"] += 1
            continue
        selected.append(question)
    return selected[:limit] if limit is not None else selected, skipped


def print_table(retrieval: dict[str, dict], answers: dict[str, dict], ks: list[int]) -> None:
    """Retrieval and answer columns side by side, overall_excl_adversarial first of the overall rows."""
    columns = [*(f"recall@{k}" for k in ks), *(f"hit@{k}" for k in ks), "f1", "judge_acc"]
    print(f"{'category':<26}{'n':>6}" + "".join(f"{c:>11}" for c in columns))
    for name, row in answers.items():
        values = {**retrieval.get(name, {}), **row}
        cells = "".join(f"{'-':>11}" if values.get(c) is None else f"{values[c]:>11.3f}" for c in columns)
        print(f"{name:<26}{row['n']:>6}" + cells)


def retrieval_summary(run: dict, question_ids: set[str], ks: list[int]) -> dict[str, dict]:
    """Retrieval metrics over the scored questions, with an
    overall_excl_adversarial row to match the answer summary."""
    results = [
        QuestionResult(
            q["sample_id"],
            q["question"],
            q["category"],
            tuple(q["evidence"]),
            tuple(q["retrieved"]),
            # k counts memories, and an extracted memory can cite several turns.
            retrieved_sources=tuple(tuple(s) for s in q.get("retrieved_sources") or ()),
        )
        for q in run["questions"]
        if q["question_id"] in question_ids
    ]
    summary = summarize(results, ks)
    excl = summarize([r for r in results if r.category != "adversarial"], ks)["overall"]
    return {**summary, "overall_excl_adversarial": excl}


def print_judge_sample(questions: list[dict], n: int, rng: random.Random) -> None:
    judged = [q for q in questions if q["judge_label"] is not None]
    print(f"\n{min(n, len(judged))} random judged items to spot-check:")
    for q in rng.sample(judged, min(n, len(judged))):
        print(f"\n{q['id']} ({q['category']})\n  question: {q['question']}\n  gold:     {q['gold']}\n  answer:   {q['answer']}")
        print(f"  judge:    {q['judge_label']} - {q['judge_reasoning']}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="locomo-eval answer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="the retrieval run to score (results/<run-id>.json)")
    parser.add_argument("--tag", help="score a tagged retrieval, results/<run-id>.<tag>.json from `locomo-eval retrieve --tag`")
    parser.add_argument("--answer-model", required=True, help="provider:model that answers, e.g. ollama:qwen3:8b")
    parser.add_argument("--judge-model", required=True, help="provider:model that judges; ideally at least as large as the answer model and another family")
    parser.add_argument("--answer-prompt", default=ANSWER_VERSION, help="answer prompt version (default: %(default)s)")
    parser.add_argument("--judge-prompt", default=JUDGE_VERSION, help="judge prompt version (default: %(default)s)")
    parser.add_argument("--answer-think", action="store_true", help="let the answer model think before answering; slower, and a separate config")
    parser.add_argument("--answer-k", type=int, help="contexts shown to the answer model (default: the retrieval run's answer_k)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="where the run's files live (default: %(default)s)")
    parser.add_argument("--samples", type=sample_ids, help="comma-separated sample ids to score (default: all in the run)")
    parser.add_argument("--limit", type=int, help="score only the first N questions, for a quick check")
    parser.add_argument("--concurrency", type=int, default=1, help="LLM calls in flight; local Ollama serves one at a time unless OLLAMA_NUM_PARALLEL is set (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-call timeout in seconds (default: %(default)s)")
    parser.add_argument("--judge-sample", type=int, default=0, metavar="N", help="print N random judged items to spot-check the judge")
    args = parser.parse_args(argv)
    if args.tag:
        # A tagged retrieval is a run of its own, answer checkpoints included.
        args.run_id = f"{args.run_id}.{args.tag}"
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_path = args.results_dir / f"{args.run_id}.json"
    if not run_path.exists():
        sys.exit(f"no retrieval run at {run_path}; run `locomo-eval --run-id {args.run_id}` first")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run_config = run["config"]
    answer_k = args.answer_k or run_config.get("answer_k")
    if answer_k is None:
        sys.exit(f"{run_path} records no answer_k; it predates stored contexts, so rerun the retrieval")
    if not 1 <= answer_k <= run_config.get("recall_k", answer_k):
        sys.exit(f"--answer-k must be between 1 and the run's recall_k ({run_config.get('recall_k')})")

    questions, skipped = select_questions(load_questions(run), args.samples, args.limit)
    if args.samples and (missing := set(args.samples) - set(run_config.get("samples", []))):
        sys.exit(f"run {args.run_id} did not score: {', '.join(sorted(missing))}")
    if questions and not any(q.contexts for q in questions):
        sys.exit(f"{run_path} has no stored contexts; rerun the retrieval")

    try:
        config = AnswerConfig(
            answer_model=args.answer_model,
            judge_model=args.judge_model,
            answer_prompt=load_prompt(args.answer_prompt),
            judge_prompt=load_prompt(args.judge_prompt),
            answer_k=answer_k,
            answer_think=args.answer_think,
        )
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))
    pricing = load_pricing()
    checkpoint = Checkpoint(args.results_dir, args.run_id, config.hash())
    print(f"scoring {len(questions)} questions from {run_path} (config {config.hash()}, checkpoint {checkpoint.path.name})", file=sys.stderr)

    last_print = 0.0

    def on_progress(progress: Progress) -> None:
        nonlocal last_print
        if time.monotonic() - last_print >= 5 or progress.done == progress.total:
            last_print = time.monotonic()
            print(progress.line(), file=sys.stderr)

    async def go() -> tuple[AnswerRun, dict]:
        async with LLMClient(options=config.options, concurrency=args.concurrency, timeout=args.timeout, pricing=pricing) as client:
            try:
                llm = await client.run_config({ANSWER: config.answer_model, JUDGE: config.judge_model})
            except (LLMError, ValueError) as exc:
                sys.exit(str(exc))
            return await run_answers(client, config, questions, checkpoint, concurrency=args.concurrency, on_progress=on_progress), llm

    try:
        answer_run, llm = asyncio.run(go())
    except* LLMError as group:
        sys.exit(f"stopped: {group.exceptions[0]}\nfinished calls are saved in {checkpoint.path}; rerun the same command to resume")

    entry = answers_entry(
        answer_run,
        config,
        pricing,
        {
            "llm": llm,
            "subset": {"samples": args.samples, "limit": args.limit},
            "skipped": skipped,
            "finished_at": datetime.now(UTC).isoformat(),
        },
    )
    # Reread in case another config's scores were written meanwhile.
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run.setdefault("answers", {})[config.hash()] = entry
    tmp_path = run_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    tmp_path.replace(run_path)

    ks = run_config.get("ks", [])
    print_table(retrieval_summary(run, {q.question_id for q in questions}, ks), entry["summary"], ks)
    usage = entry["usage"]
    for name in STAGES:
        cost = usage[name]["cost_usd"]
        print(
            f"{name}: {usage[name]['calls']} calls, {usage[name]['input_tokens']} in / {usage[name]['output_tokens']} out tokens, "
            f"{usage[name]['seconds']:.0f}s, cost {'unknown' if cost is None else f'${cost:.4f}'}"
        )
    if skipped["no_gold_answer"]:
        print(f"skipped: {skipped}")
    if args.judge_sample:
        print_judge_sample(answer_run.questions, args.judge_sample, random.Random())
    print(f"\nwrote answers[{config.hash()}] to {run_path}")


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
