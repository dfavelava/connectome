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
