"""Evidence recall@k scoring, independent of any backend.

For one QA item with evidence set E and ranked retrieved dialog ids R:

- recall@k = |E ∩ R[:k]| / |E|  (fraction of the evidence found)
- hit@k    = 1 if E ∩ R[:k] is non-empty else 0

Both are averaged over questions, overall and per category.
"""

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Context:
    """One retrieved dialog turn and its memory text, as recall returned it."""

    dia_id: str
    text: str


@dataclass(frozen=True)
class QuestionResult:
    sample_id: str
    question: str
    category: str
    evidence: tuple[str, ...]
    retrieved: tuple[str, ...]
    # "<sample_id>#<qa_index>", stable across runs.
    question_id: str = ""
    answer: str | None = None
    adversarial_answer: str | None = None
    # The retrieved turns' text, in rank order, so answering can run offline.
    contexts: tuple[Context, ...] = ()


def recall_at_k(evidence: tuple[str, ...], retrieved: tuple[str, ...], k: int) -> float:
    if not evidence:
        raise ValueError("recall is undefined for an empty evidence set")
    found = set(retrieved[:k]) & set(evidence)
    return len(found) / len(set(evidence))


def hit_at_k(evidence: tuple[str, ...], retrieved: tuple[str, ...], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & set(evidence) else 0.0


def summarize(results: list[QuestionResult], ks: list[int]) -> dict[str, dict[str, float | int]]:
    """Mean recall@k and hit@k per category, plus an "overall" row.

    Categories are returned in sorted order with "overall" last."""
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for result in results:
        groups[result.category].append(result)
        groups["overall"].append(result)

    summary: dict[str, dict[str, float | int]] = {}
    for name in [*sorted(g for g in groups if g != "overall"), "overall"]:
        group = groups.get(name, [])
        row: dict[str, float | int] = {"n": len(group)}
        for k in ks:
            row[f"recall@{k}"] = _mean([recall_at_k(r.evidence, r.retrieved, k) for r in group])
            row[f"hit@{k}"] = _mean([hit_at_k(r.evidence, r.retrieved, k) for r in group])
        summary[name] = row
    return summary


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


ADVERSARIAL_CATEGORY = "adversarial"


def summarize_answers(questions: list[dict]) -> dict[str, dict[str, float | int | None]]:
    """Mean token F1 and judge accuracy per category, plus "overall" and
    "overall_excl_adversarial" (most published LoCoMo numbers leave out the
    adversarial category).

    Each question is a dict with "category", "f1" and "judge_label"
    (CORRECT, WRONG, or None when the judge's reply never parsed). Judge
    accuracy is over judged questions only, with the null count beside it;
    None when nothing was judged. Categories come sorted, then the two
    overall rows."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for question in questions:
        groups[question["category"]].append(question)
        groups["overall"].append(question)
        if question["category"] != ADVERSARIAL_CATEGORY:
            groups["overall_excl_adversarial"].append(question)

    overall_rows = ("overall_excl_adversarial", "overall")
    summary: dict[str, dict[str, float | int | None]] = {}
    for name in [*sorted(g for g in groups if g not in overall_rows), *overall_rows]:
        group = groups.get(name, [])
        labels = [q["judge_label"] for q in group]
        correct, wrong = labels.count("CORRECT"), labels.count("WRONG")
        summary[name] = {
            "n": len(group),
            "f1": _mean([q["f1"] for q in group]),
            "judge_acc": correct / (correct + wrong) if correct + wrong else None,
            "judge_null": len(group) - correct - wrong,
        }
    return summary
