"""The retrieval scores: leakage, staleness, perspective and gold recall.

They take each question's ranked hits, as campaign memory ids, and need no
LLM. Truth status and graph answers are scored in answering.py.

- Leakage: the share of all results a question's viewer may not see, over every
  question. Anything above zero is a bug, not a score, and each leak is listed.
- Staleness@k: over the staleness questions, the share of their relevant top-k
  results (gold or stale) that are stale - superseded claims. stale_first@k is
  the share of those questions whose highest-ranked relevant result is stale.
- Perspective@k: the share of perspective questions whose viewer's own facet
  is in the top k and ranks above the root and every other facet of the event.
  root_first@k is the share where the root outranks it.
- recall@k: the mean share of a question's gold memories in its top k, per
  question kind; all_gold@k the share of questions with every gold memory
  there, which is what a graph question needs.
"""

from collections.abc import Mapping, Sequence

from westmarches_eval.campaign import (
    GRAPH,
    PERSPECTIVE,
    QUESTION_KINDS,
    STALENESS,
    Campaign,
    Question,
)

# A question's ranked hits, as campaign memory ids.
Hits = Mapping[str, Sequence[str]]


def leakage(campaign: Campaign, hits: Hits) -> dict[str, object]:
    leaks = [
        {"question": q.id, "as": q.viewer, "memory": memory_id, "rank": rank}
        for q in campaign.questions
        for rank, memory_id in enumerate(hits[q.id], start=1)
        if not campaign.visible(q.viewer, memory_id)
    ]
    results = sum(len(hits[q.id]) for q in campaign.questions)
    return {
        "results": results,
        "leaked": len(leaks),
        "rate": _ratio(len(leaks), results),
        "questions_with_leaks": len({leak["question"] for leak in leaks}),
        "leaks": leaks,
    }


def staleness(questions: Sequence[Question], hits: Hits, k: int) -> dict[str, object]:
    stale = relevant = stale_first = 0
    for q in questions:
        top = hits[q.id][:k]
        ranked = [m for m in top if m in q.gold or m in q.stale]
        relevant += len(ranked)
        stale += sum(1 for m in ranked if m in q.stale)
        stale_first += bool(ranked) and ranked[0] in q.stale
    return {
        f"staleness@{k}": _ratio(stale, relevant) if relevant else None,
        f"stale_first@{k}": _ratio(stale_first, len(questions)) if questions else None,
    }


def perspective(campaign: Campaign, questions: Sequence[Question], hits: Hits, k: int) -> dict[str, object]:
    own_first = root_first = 0
    for q in questions:
        (own,) = q.gold
        family = campaign.facet_family(own)
        ranked = [m for m in hits[q.id][:k] if m in family]
        own_first += bool(ranked) and ranked[0] == own
        root_first += bool(ranked) and ranked[0] == family[0]
    n = len(questions)
    return {
        f"perspective@{k}": _ratio(own_first, n) if n else None,
        f"root_first@{k}": _ratio(root_first, n) if n else None,
    }


def recall(questions: Sequence[Question], hits: Hits, k: int) -> dict[str, object]:
    scored = [q for q in questions if q.gold]
    shares = [len(set(q.gold) & set(hits[q.id][:k])) / len(q.gold) for q in scored]
    n = len(scored)
    return {
        f"recall@{k}": _ratio(sum(shares), n) if n else None,
        f"all_gold@{k}": _ratio(sum(1 for s in shares if s == 1), n) if n else None,
    }


def summarize(campaign: Campaign, hits: Hits, ks: Sequence[int]) -> dict[str, object]:
    """Every retrieval score of one set of hits, at each k."""
    by_kind = {kind: [q for q in campaign.questions if q.kind == kind] for kind in QUESTION_KINDS}
    summary: dict[str, object] = {"leakage": leakage(campaign, hits), "n": {kind: len(qs) for kind, qs in by_kind.items()}}
    for k in ks:
        summary[f"k={k}"] = {
            "staleness": staleness(by_kind[STALENESS], hits, k),
            "perspective": perspective(campaign, by_kind[PERSPECTIVE], hits, k),
            "graph": recall(by_kind[GRAPH], hits, k),
            "recall": {kind: recall(qs, hits, k) for kind, qs in by_kind.items() if any(q.gold for q in qs)},
        }
    return summary


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
