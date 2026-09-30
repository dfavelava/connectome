"""Versioned answer and judge prompts, and strict parsing of judge verdicts.

Prompts live next to this module as `<name>_v<N>.txt`. A committed version is
never edited: changing the wording means adding the next version, so a
result's recorded version and sha256 always identify the exact text it used.
Templates use `$field` placeholders (string.Template), which leaves the JSON
braces in the judge's examples alone.

The judge runs on a small local model, so its reply is constrained with
Ollama structured outputs (JUDGE_SCHEMA) and then parsed strictly. A reply
that still doesn't parse is retried once with the next seed at
RETRY_TEMPERATURE - at temperature 0 decoding is greedy and would return the
same reply whatever the seed - and after that recorded as a null label, which
summaries count rather than score.
"""

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import NamedTuple

from locomo_eval.llm import LLMClient, retry_options
from locomo_eval.metrics import Context
from locomo_eval.scoring import ADVERSARIAL

PROMPTS_DIR = Path(__file__).resolve().parent
ANSWER_VERSION = "answer_v1"
# Describes both turn and extracted-memory contexts, spells out relative-date
# arithmetic and allows inference the excerpts support.
ANSWER_V2_VERSION = "answer_v2"
JUDGE_VERSION = "judge_v1"
# Lenient, like the Mem0/LoCoMo judge, on a date more specific than the gold
# period, on lists with extra or missing (but no contradicting) items and on a
# left-out qualifier; as strict as judge_v1 on wrong facts and abstentions.
JUDGE_V2_VERSION = "judge_v2"

# What the answer prompt tells the model to say when the context lacks the
# answer; it is also the gold answer the judge sees for adversarial questions.
NOT_MENTIONED = "Not mentioned in the conversation."
NO_TRAP_ANSWER = "(none)"

CORRECT = "CORRECT"
WRONG = "WRONG"
JUDGE_LABELS = (CORRECT, WRONG)
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "label": {"type": "string", "enum": list(JUDGE_LABELS)},
    },
    "required": ["reasoning", "label"],
    "additionalProperties": False,
}
JUDGE_ATTEMPTS = 2

_VERSION = re.compile(r"^[a-z]+_v\d+$")
_DIA_ID = re.compile(r"^D(\d+):(\d+)$")


class Prompt(NamedTuple):
    text: str
    version: str
    sha256: str

    def render(self, **fields: str) -> str:
        """Fill the `$field` placeholders; a missing field raises KeyError."""
        return Template(self.text).substitute(fields)

    def config(self) -> dict[str, str]:
        """What a run config records to pin this prompt."""
        return {"version": self.version, "sha256": self.sha256}


def load_prompt(version: str, prompts_dir: Path = PROMPTS_DIR) -> Prompt:
    """Load a prompt by version (e.g. "answer_v1") with the sha256 of its file."""
    if not _VERSION.match(version):
        raise ValueError(f"prompt version {version!r} is not of the form <name>_v<N> (e.g. answer_v1)")
    path = prompts_dir / f"{version}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt {version!r} in {prompts_dir}")
    data = path.read_bytes()
    return Prompt(text=data.decode("utf-8"), version=version, sha256=hashlib.sha256(data).hexdigest())


def _chronological_key(context: Context) -> tuple[int, int, str]:
    # Sessions are numbered in date order and turns in speaking order, so the
    # dialog id orders turns by time; unparseable ids go last, as retrieved.
    match = _DIA_ID.match(context.dia_id)
    return (int(match.group(1)), int(match.group(2)), "") if match else (1 << 31, 0, context.dia_id)


def answer_prompt(prompt: Prompt, question: str, contexts: Iterable[Context]) -> str:
    """The answer prompt for a question and its retrieved contexts, sorted by
    date. A turn's text already starts with its session date; an extracted
    memory sorts by the first turn it cites."""
    ordered = sorted(contexts, key=_chronological_key)
    return prompt.render(context="\n".join(c.text for c in ordered), question=question)


def judge_prompt(
    prompt: Prompt,
    question: str,
    category: int,
    generated_answer: str,
    answer: str | None = None,
    adversarial_answer: str | None = None,
) -> str:
    """The judge prompt. For an adversarial question the gold answer is an
    abstention and the dataset's adversarial_answer is shown as the trap."""
    if category == ADVERSARIAL:
        gold, trap = NOT_MENTIONED, adversarial_answer or NO_TRAP_ANSWER
    else:
        if answer is None:
            raise ValueError(f"category {category} question {question!r} has no gold answer")
        gold, trap = answer, NO_TRAP_ANSWER
    return prompt.render(question=question, gold_answer=gold, trap_answer=trap, generated_answer=generated_answer)


class JudgeParseError(ValueError):
    pass


class Verdict(NamedTuple):
    reasoning: str
    label: str


def parse_judge_output(text: str) -> Verdict:
    """Parse a judge reply strictly: one JSON object with exactly a string
    `reasoning` and a `label` of CORRECT or WRONG. Nothing is guessed from
    free text."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"judge reply is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise JudgeParseError(f"judge reply is not a JSON object: {text!r}")
    if set(data) != {"reasoning", "label"}:
        raise JudgeParseError(f"judge reply has keys {sorted(data)}, expected ['label', 'reasoning']")
    if not isinstance(data["reasoning"], str):
        raise JudgeParseError("judge reasoning is not a string")
    if data["label"] not in JUDGE_LABELS:
        raise JudgeParseError(f"judge label {data['label']!r} is not one of {', '.join(JUDGE_LABELS)}")
    return Verdict(reasoning=data["reasoning"], label=data["label"])


@dataclass(frozen=True)
class JudgeResult:
    # None when no attempt returned a parseable verdict.
    judge_label: str | None
    judge_reasoning: str | None
    attempts: int
    # The last parse error, when judge_label is None.
    judge_error: str | None = None


async def judge(client: LLMClient, model: str, prompt_text: str, *, attempts: int = JUDGE_ATTEMPTS) -> JudgeResult:
    """Ask the judge for a verdict, retrying a malformed reply with the next
    seed at RETRY_TEMPERATURE. Transport errors are the client's to retry and
    still raise."""
    error = None
    for attempt in range(attempts):
        options = retry_options(client.options, attempt)
        completion = await client.complete(model, "", prompt_text, stage="judge", format=JUDGE_SCHEMA, options=options)
        try:
            verdict = parse_judge_output(completion.text)
        except JudgeParseError as exc:
            error = str(exc)
            continue
        return JudgeResult(judge_label=verdict.label, judge_reasoning=verdict.reasoning, attempts=attempt + 1)
    return JudgeResult(judge_label=None, judge_reasoning=None, attempts=attempts, judge_error=error)


def judge_summary(labels: Iterable[str | None]) -> dict[str, int | float | None]:
    """Counts of CORRECT, WRONG and unparseable (null) verdicts. Accuracy is
    over the judged questions only; the null count is reported beside it so
    judge failures are never scored as either label."""
    labels = list(labels)
    correct = labels.count(CORRECT)
    wrong = labels.count(WRONG)
    judged = correct + wrong
    return {
        "judged": judged,
        "correct": correct,
        "wrong": wrong,
        "judge_null": len(labels) - judged,
        "accuracy": correct / judged if judged else None,
    }
