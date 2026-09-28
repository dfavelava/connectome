"""Evidence recall@k scoring, independent of any backend.

For one QA item with evidence set E, a ranked list of retrieved memories, and
T(k) the dialog turns the top k memories came from (each memory's sources, in
rank order, duplicates removed):

- recall@k = |E ∩ T(k)| / |E|  (fraction of the evidence found)
- hit@k    = 1 if E ∩ T(k) is non-empty else 0

A raw-turn memory's only source is its own turn, so there T(k) is simply the
top k turns. An extracted memory can cite several turns, which is why k
counts memories, not turns.

Two more metrics separate what extraction lost from what retrieval lost, and
compare the ingestion modes fairly:

- coverage     = the fraction of E cited by any memory in the conversation,
  retrieved or not: the most retrieval could possibly find.
- recall@<B>t  = recall over the longest prefix of the ranking whose memory
  text fits in B tokens (see count_tokens). A memory citing many turns
  inflates recall@k, so this compares modes at an equal context budget.
  `underfilled@<B>t` is the fraction of questions whose whole retrieved list
  fit in fewer than B tokens - retrieval ran out before the budget did, so fetch more
  (a larger --ks or --answer-k) for a fair number at that budget.

All are averaged over questions, overall and per category.
"""

import re
from collections import defaultdict
from dataclasses import dataclass

_TOKEN = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class Context:
    """One retrieved memory's text, as recall returned it. dia_id is its
    first source turn, which orders it in time; "" when it cites none."""

    dia_id: str
    text: str


@dataclass(frozen=True)
class QuestionResult:
    sample_id: str
    question: str
    category: str
    evidence: tuple[str, ...]
    # The retrieved memories' source turns in rank order, duplicates removed.
    retrieved: tuple[str, ...]
    # "<sample_id>#<qa_index>", stable across runs.
    question_id: str = ""
    answer: str | None = None
    adversarial_answer: str | None = None
    # The retrieved memories' text, in rank order, so answering can run offline.
    contexts: tuple[Context, ...] = ()
    # Each retrieved memory's source turns, in rank order. Empty means one
    # turn per memory: retrieved itself.
    retrieved_sources: tuple[tuple[str, ...], ...] = ()
    # The evidence turns cited by any memory in the conversation; None when
    # coverage wasn't measured.
    covered: tuple[str, ...] | None = None

    def sources(self) -> tuple[tuple[str, ...], ...]:
        return self.retrieved_sources or tuple((d,) for d in self.retrieved)


def count_tokens(text: str) -> int:
    """An approximate token count - words and punctuation marks - applied
    alike to every mode's memory text, so budgets compare like with like."""
    return len(_TOKEN.findall(text))


def expand(sources: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    """Memories' source turns in rank order, duplicates removed."""
    return tuple(dict.fromkeys(d for memory in sources for d in memory))


def recall_at_k(evidence: tuple[str, ...], retrieved: tuple[str, ...], k: int) -> float:
    if not evidence:
        raise ValueError("recall is undefined for an empty evidence set")
    found = set(retrieved[:k]) & set(evidence)
    return len(found) / len(set(evidence))


def hit_at_k(evidence: tuple[str, ...], retrieved: tuple[str, ...], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & set(evidence) else 0.0


def memory_recall_at_k(result: QuestionResult, k: int) -> float:
    """recall over the source turns of the top k memories."""
    turns = expand(result.sources()[:k])
    return recall_at_k(result.evidence, turns, len(turns))


def memory_hit_at_k(result: QuestionResult, k: int) -> float:
    turns = expand(result.sources()[:k])
    return hit_at_k(result.evidence, turns, len(turns))


def coverage(result: QuestionResult) -> float:
    if result.covered is None:
        raise ValueError(f"{result.question_id or result.question!r} has no coverage recorded")
    return len(set(result.covered) & set(result.evidence)) / len(set(result.evidence))


def within_budget(result: QuestionResult, budget: int) -> tuple[int, bool]:
    """How many top-ranked memories fit in `budget` tokens of text, and
    whether that's all of them with room to spare (the list ran out first)."""
    used = 0
    for count, context in enumerate(result.contexts):
        used += count_tokens(context.text)
        if used > budget:
            return count, False
    return len(result.contexts), used < budget


def budget_recall(result: QuestionResult, budget: int) -> float:
    return memory_recall_at_k(result, within_budget(result, budget)[0])


def summarize(results: list[QuestionResult], ks: list[int], budgets: list[int] | tuple[int, ...] = ()) -> dict[str, dict[str, float | int]]:
    """Mean recall@k and hit@k per category, plus an "overall" row. Adds
    coverage when the results carry it, and recall@<B>t with
    underfilled@<B>t for each token budget B.

    Categories are returned in sorted order with "overall" last."""
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for result in results:
        groups[result.category].append(result)
        groups["overall"].append(result)
    with_coverage = any(r.covered is not None for r in results)

    summary: dict[str, dict[str, float | int]] = {}
    for name in [*sorted(g for g in groups if g != "overall"), "overall"]:
        group = groups.get(name, [])
        row: dict[str, float | int] = {"n": len(group)}
        if with_coverage:
            row["coverage"] = _mean([coverage(r) for r in group])
        for k in ks:
            row[f"recall@{k}"] = _mean([memory_recall_at_k(r, k) for r in group])
            row[f"hit@{k}"] = _mean([memory_hit_at_k(r, k) for r in group])
        for budget in budgets:
            row[f"recall@{budget}t"] = _mean([budget_recall(r, budget) for r in group])
            row[f"underfilled@{budget}t"] = _mean([float(within_budget(r, budget)[1]) for r in group])
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
