"""The answer stage: truth status and graph answers, scored offline.

    uv run westmarches-eval answer <run-id> --answer-model ollama:<model> --judge-model ollama:<model>

Reads results/<run-id>.json, written by `westmarches-eval retrieve`, which
holds each question's top results as recall returned them with hydrate off
and with it on. For the truth and graph questions it answers from the top
answer_k of each, then asks the judge:

- truth questions: did the answer give the gold claim with its gold truth
  status (truth_v1)? That is the truth-status score. The same answers
  are also judged on content alone (LoCoMo's judge_v3).
- graph questions: is the answer right (judge_v3)?

The LLM client, judge parsing and prompt versioning are LoCoMo's
(locomo_eval.llm and locomo_eval.prompts). The scores go back into the
results file under "answers", keyed by a hash of the answer config.
"""

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from locomo_eval.llm import LLMClient, SamplingOptions
from locomo_eval.prompts import (
    DEFAULT_JUDGE_VERSION,
    NO_TRAP_ANSWER,
    Prompt,
    judge,
    judge_summary,
)
from locomo_eval.prompts import load_prompt as load_locomo_prompt

from westmarches_eval.campaign import GRAPH, TRUTH

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
ANSWER_VERSION = "answer_v1"
TRUTH_JUDGE_VERSION = "truth_v1"
# Hydrate off returns a snippet of each memory's body; hydrate on returns the
# whole document, frontmatter and all. These are the two settings a run scores.
HYDRATE_OFF = "hydrate_off"
HYDRATE_ON = "hydrate_on"
MODES = (HYDRATE_OFF, HYDRATE_ON)
ANSWERED_KINDS = (TRUTH, GRAPH)


def load_prompt(version: str) -> Prompt:
    """A westmarches prompt by version, or LoCoMo's judge_vN."""
    if version.startswith("judge_"):
        return load_locomo_prompt(version)
    return load_locomo_prompt(version, PROMPTS_DIR)


def split_frontmatter(content: str) -> tuple[dict, str]:
    """A hydrated memory document's frontmatter and body."""
    if content.startswith("---\n") and (end := content.find("\n---\n", 4)) != -1:
        try:
            metadata = yaml.safe_load(content[4:end]) or {}
        except yaml.YAMLError:
            metadata = {}
        return (metadata if isinstance(metadata, dict) else {}), content[end + len("\n---\n") :].strip()
    return {}, content.strip()


def context_text(hit: dict) -> str:
    """What the answer model sees of one recall result.

    A hydrated result carries the memory's frontmatter, so its line shows what
    a client can read from it: the date it happened, any rumor or hypothesis
    kind on its relationships, and (superseded) when every relationship has
    been superseded. A snippet carries the body text only."""
    if not hit.get("content"):
        return (hit.get("snippet") or "").strip()
    metadata, body = split_frontmatter(hit["content"])
    tags = []
    if occurred_at := metadata.get("occurred_at"):
        tags.append(f"[{str(occurred_at)[:10]}]")
    relationships = [r for r in metadata.get("relationships") or [] if isinstance(r, dict)]
    kinds = dict.fromkeys(r.get("kind") for r in relationships if r.get("kind") in ("rumor", "hypothesis"))
    tags.extend(f"({kind})" for kind in kinds)
    if relationships and all(r.get("superseded_by") for r in relationships):
        tags.append("(superseded)")
    return " ".join([*tags, body])


@dataclass(frozen=True)
class AnswerConfig:
    answer_model: str
    judge_model: str
    answer_prompt: Prompt
    judge_prompt: Prompt
    truth_judge_prompt: Prompt
    answer_k: int
    options: SamplingOptions = SamplingOptions()

    def to_dict(self) -> dict:
        return {
            "answer_model": self.answer_model,
            "judge_model": self.judge_model,
            "answer_prompt": self.answer_prompt.config(),
            "judge_prompt": self.judge_prompt.config(),
            "truth_judge_prompt": self.truth_judge_prompt.config(),
            "answer_k": self.answer_k,
            "options": asdict(self.options),
        }

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


async def answer_question(client: LLMClient, config: AnswerConfig, question: dict, contexts: list[str]) -> dict:
    """Answer one question from its top contexts and judge it."""
    prompt = config.answer_prompt.render(context="\n".join(contexts[: config.answer_k]) or "(none)", question=question["question"])
    completion = await client.complete(config.answer_model, "", prompt, stage="answer")
    generated = completion.text.strip()
    record = {"question_id": question["id"], "kind": question["kind"], "generated": generated, "truncated": completion.truncated}
    content = await judge(
        client,
        config.judge_model,
        config.judge_prompt.render(question=question["question"], gold_answer=question["answer"], trap_answer=NO_TRAP_ANSWER, generated_answer=generated),
    )
    record["content_label"] = content.judge_label
    record["content_reasoning"] = content.judge_reasoning
    if question["kind"] == TRUTH:
        truth = await judge(
            client,
            config.judge_model,
            config.truth_judge_prompt.render(question=question["question"], gold_answer=question["answer"], truth=question["truth"], generated_answer=generated),
        )
        record["truth_label"] = truth.judge_label
        record["truth_reasoning"] = truth.judge_reasoning
    return record


def summarize_answers(records: list[dict]) -> dict[str, dict]:
    truth = [r for r in records if r["kind"] == TRUTH]
    graph = [r for r in records if r["kind"] == GRAPH]
    return {
        "truth_status": judge_summary(r["truth_label"] for r in truth),
        "truth_content": judge_summary(r["content_label"] for r in truth),
        "graph": judge_summary(r["content_label"] for r in graph),
    }


async def answer_run(client: LLMClient, config: AnswerConfig, run: dict) -> dict:
    """Answer and judge every truth and graph question in both hydrate modes."""
    questions = [q for q in run["questions"] if q["kind"] in ANSWERED_KINDS]
    modes = {}
    for mode in MODES:
        records = await asyncio.gather(*(answer_question(client, config, q, q["contexts"][mode]) for q in questions))
        modes[mode] = {"summary": summarize_answers(list(records)), "questions": list(records)}
    return modes


def answer_config(answer_model: str, judge_model: str, answer_k: int, answer_prompt: str = ANSWER_VERSION, judge_prompt: str = DEFAULT_JUDGE_VERSION) -> AnswerConfig:
    return AnswerConfig(
        answer_model=answer_model,
        judge_model=judge_model,
        answer_prompt=load_prompt(answer_prompt),
        judge_prompt=load_prompt(judge_prompt),
        truth_judge_prompt=load_prompt(TRUTH_JUDGE_VERSION),
        answer_k=answer_k,
    )


async def run_answers(results_path: Path, config: AnswerConfig, llm_concurrency: int) -> dict:
    """Answer a finished retrieval run and write the scores back into it.
    Returns the answers entry it wrote."""
    run = json.loads(results_path.read_text(encoding="utf-8"))
    async with LLMClient(options=config.options, concurrency=llm_concurrency) as client:
        llm = await client.run_config({"answer": config.answer_model, "judge": config.judge_model})
        modes = await answer_run(client, config, run)
        entry = {"config": config.to_dict(), "llm": llm, "modes": modes, "usage": client.usage_summary()}
    run.setdefault("answers", {})[config.hash()] = entry
    results_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    return entry
