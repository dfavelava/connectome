import pytest

from locomo_eval.metrics import (
    QuestionResult,
    hit_at_k,
    recall_at_k,
    summarize,
    summarize_answers,
)


def test_recall_and_hit_at_k():
    evidence = ("D1:1", "D1:2")
    retrieved = ("D3:1", "D1:2", "D1:1")
    assert recall_at_k(evidence, retrieved, 1) == 0.0
    assert recall_at_k(evidence, retrieved, 2) == 0.5
    assert recall_at_k(evidence, retrieved, 3) == 1.0
    assert hit_at_k(evidence, retrieved, 1) == 0.0
    assert hit_at_k(evidence, retrieved, 2) == 1.0
    assert recall_at_k(evidence, (), 5) == 0.0


def test_recall_rejects_empty_evidence():
    with pytest.raises(ValueError):
        recall_at_k((), ("D1:1",), 1)


def test_summarize_per_category_and_overall():
    results = [
        QuestionResult("s", "q1", "temporal", ("D1:1",), ("D1:1",)),
        QuestionResult("s", "q2", "temporal", ("D1:1",), ("D2:2",)),
        QuestionResult("s", "q3", "multi-hop", ("D1:1", "D2:2"), ("D2:2", "D1:1")),
    ]
    summary = summarize(results, [1, 2])
    assert list(summary) == ["multi-hop", "temporal", "overall"]
    assert summary["temporal"] == {"n": 2, "recall@1": 0.5, "hit@1": 0.5, "recall@2": 0.5, "hit@2": 0.5}
    assert summary["multi-hop"]["recall@1"] == 0.5
    assert summary["multi-hop"]["recall@2"] == 1.0
    assert summary["overall"]["n"] == 3
    assert summary["overall"]["recall@2"] == pytest.approx(2 / 3)


def test_summarize_answers_per_category_and_overall_rows():
    questions = [
        {"category": "temporal", "f1": 1.0, "judge_label": "CORRECT"},
        {"category": "temporal", "f1": 0.5, "judge_label": "WRONG"},
        {"category": "single-hop", "f1": 0.0, "judge_label": None},
        {"category": "adversarial", "f1": 1.0, "judge_label": "CORRECT"},
    ]
    summary = summarize_answers(questions)
    assert list(summary) == ["adversarial", "single-hop", "temporal", "overall_excl_adversarial", "overall"]
    assert summary["temporal"] == {"n": 2, "f1": 0.75, "judge_acc": 0.5, "judge_null": 0}
    # Null verdicts are counted, never scored.
    assert summary["single-hop"] == {"n": 1, "f1": 0.0, "judge_acc": None, "judge_null": 1}
    assert summary["overall_excl_adversarial"] == {"n": 3, "f1": 0.5, "judge_acc": 0.5, "judge_null": 1}
    assert summary["overall"]["n"] == 4
    assert summary["overall"]["judge_acc"] == pytest.approx(2 / 3)


def test_summarize_answers_empty():
    assert summarize_answers([]) == {
        "overall_excl_adversarial": {"n": 0, "f1": 0.0, "judge_acc": None, "judge_null": 0},
        "overall": {"n": 0, "f1": 0.0, "judge_acc": None, "judge_null": 0},
    }


def extracted(evidence, sources, texts=None, covered=None, category="multi-hop"):
    """A result whose retrieved memories cite `sources`, each with text `texts`."""
    from locomo_eval.metrics import Context, expand

    texts = texts or ["memory"] * len(sources)
    return QuestionResult(
        "s",
        "q",
        category,
        tuple(evidence),
        expand(tuple(map(tuple, sources))),
        contexts=tuple(Context(next(iter(s), ""), t) for s, t in zip(sources, texts)),
        retrieved_sources=tuple(map(tuple, sources)),
        covered=covered,
    )


def test_recall_at_k_counts_memories_and_expands_their_sources():
    from locomo_eval.metrics import memory_hit_at_k, memory_recall_at_k

    result = extracted(("D1:1", "D1:2", "D2:1"), [("D1:1", "D1:2"), ("D1:1",), ("D2:1",)])
    assert result.retrieved == ("D1:1", "D1:2", "D2:1")
    # One memory citing two evidence turns finds both at k=1.
    assert memory_recall_at_k(result, 1) == pytest.approx(2 / 3)
    # A repeated source isn't counted twice.
    assert memory_recall_at_k(result, 2) == pytest.approx(2 / 3)
    assert memory_recall_at_k(result, 3) == 1.0
    assert memory_hit_at_k(result, 1) == 1.0
    # A memory citing nothing expands to nothing.
    assert memory_recall_at_k(extracted(("D1:1",), [(), ("D1:1",)]), 1) == 0.0


def test_one_turn_per_memory_matches_plain_recall():
    from locomo_eval.metrics import memory_recall_at_k

    evidence, retrieved = ("D1:1", "D1:2"), ("D3:1", "D1:2", "D1:1")
    turns = QuestionResult("s", "q", "temporal", evidence, retrieved)
    for k in (1, 2, 3):
        assert memory_recall_at_k(turns, k) == recall_at_k(evidence, retrieved, k)


def test_coverage_is_evidence_cited_by_any_memory():
    from locomo_eval.metrics import coverage

    assert coverage(extracted(("D1:1", "D1:2"), [], covered=("D1:2",))) == 0.5
    assert coverage(extracted(("D1:1",), [], covered=())) == 0.0
    with pytest.raises(ValueError):
        coverage(extracted(("D1:1",), []))


def test_count_tokens_counts_words_and_punctuation():
    from locomo_eval.metrics import count_tokens

    assert count_tokens("[8 May, 2023] Caroline: Hey Mel!") == 11
    assert count_tokens("") == 0


def test_budget_keeps_the_ranked_prefix_that_fits():
    from locomo_eval.metrics import budget_recall, within_budget

    # 3, 5 and 2 tokens.
    result = extracted(("D1:1", "D1:2", "D2:1"), [("D1:1",), ("D1:2",), ("D2:1",)], ["a b c", "a b c d e", "a b"])
    assert within_budget(result, 2) == (0, False)
    assert within_budget(result, 3) == (1, False)
    # The second memory would overflow, so the third, which would fit, isn't reached.
    assert within_budget(result, 7) == (1, False)
    assert within_budget(result, 10) == (3, False)
    assert within_budget(result, 11) == (3, True)
    assert budget_recall(result, 3) == pytest.approx(1 / 3)
    assert budget_recall(result, 8) == pytest.approx(2 / 3)
    assert budget_recall(result, 2) == 0.0


def test_budget_charges_a_many_source_memory_for_its_length():
    from locomo_eval.metrics import budget_recall, memory_recall_at_k

    evidence = ("D1:1", "D1:2")
    long_memory = extracted(evidence, [("D1:1", "D1:2")], ["one two three four five six"])
    turns = extracted(evidence, [("D1:1",), ("D1:2",)], ["one two three", "four five six"])
    assert memory_recall_at_k(long_memory, 1) == 1.0 > memory_recall_at_k(turns, 1)
    assert budget_recall(long_memory, 3) == 0.0 < budget_recall(turns, 3)
    assert budget_recall(long_memory, 6) == budget_recall(turns, 6) == 1.0


def test_summarize_adds_coverage_and_budgets():
    results = [
        extracted(("D1:1",), [("D1:1",)], ["a b"], covered=("D1:1",), category="temporal"),
        extracted(("D1:1", "D1:2"), [("D2:1",), ("D1:2",)], ["a b", "c"], covered=("D1:2",)),
    ]
    summary = summarize(results, [1], [2, 10])
    assert summary["temporal"] == {"n": 1, "coverage": 1.0, "recall@1": 1.0, "hit@1": 1.0, "recall@2t": 1.0, "underfilled@2t": 0.0, "recall@10t": 1.0, "underfilled@10t": 1.0}
    assert summary["multi-hop"]["coverage"] == 0.5
    assert summary["multi-hop"]["recall@2t"] == 0.0
    assert summary["multi-hop"]["recall@10t"] == 0.5
    assert summary["overall"]["coverage"] == 0.75
