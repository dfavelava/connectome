import pytest

from westmarches_eval import scores


def hits_for(small, **overrides):
    hits = {q.id: [] for q in small.questions}
    hits.update(overrides)
    return hits


def test_leakage_counts_every_hidden_result(small):
    hits = hits_for(small, **{"q-leak": ["rumor", "secret", "event-a"], "q-truth": ["rumor", "event-b"], "q-graph": ["secret"]})
    leakage = scores.leakage(small, hits)
    assert leakage["results"] == 6
    assert leakage["leaked"] == 3
    assert leakage["rate"] == pytest.approx(0.5)
    assert leakage["questions_with_leaks"] == 2
    assert {(leak["question"], leak["memory"], leak["rank"]) for leak in leakage["leaks"]} == {
        ("q-leak", "secret", 2),
        ("q-leak", "event-a", 3),
        ("q-truth", "event-b", 2),
    }


def test_no_results_is_no_leakage(small):
    assert scores.leakage(small, hits_for(small))["rate"] == 0.0


def test_staleness_is_the_stale_share_of_relevant_results(small):
    (question,) = [q for q in small.questions if q.kind == "staleness"]
    stale_first = scores.staleness([question], {question.id: ["old-captain", "rumor", "new-captain"]}, 10)
    assert stale_first == {"staleness@10": 0.5, "stale_first@10": 1.0}
    current_first = scores.staleness([question], {question.id: ["new-captain", "old-captain"]}, 10)
    assert current_first == {"staleness@10": 0.5, "stale_first@10": 0.0}
    # The stale memory past k doesn't count.
    assert scores.staleness([question], {question.id: ["new-captain", "old-captain"]}, 1) == {"staleness@1": 0.0, "stale_first@1": 0.0}
    assert scores.staleness([question], {question.id: ["rumor"]}, 10)["staleness@10"] is None


def test_perspective_wants_the_viewers_facet_first(small):
    questions = [q for q in small.questions if q.kind == "perspective"]
    hits = {"q-persp-a": ["event-a", "event"], "q-persp-b": ["event", "event-b"]}
    assert scores.perspective(small, questions, hits, 10) == {"perspective@10": 0.5, "root_first@10": 0.5}
    assert scores.perspective(small, questions, hits, 1) == {"perspective@1": 0.5, "root_first@1": 0.5}
    assert scores.perspective(small, questions, {"q-persp-a": [], "q-persp-b": []}, 10) == {"perspective@10": 0.0, "root_first@10": 0.0}


def test_recall_and_all_gold(small):
    (graph,) = [q for q in small.questions if q.kind == "graph"]
    assert scores.recall([graph], {graph.id: ["new-captain", "secret", "rumor"]}, 2) == {"recall@2": 0.5, "all_gold@2": 0.0}
    assert scores.recall([graph], {graph.id: ["new-captain", "secret", "rumor"]}, 3) == {"recall@3": 1.0, "all_gold@3": 1.0}


def test_summarize_has_every_score_at_every_k(small):
    summary = scores.summarize(small, hits_for(small), [5, 10])
    assert summary["n"] == {"truth": 1, "staleness": 1, "perspective": 2, "graph": 1, "leakage": 1}
    for k in (5, 10):
        assert set(summary[f"k={k}"]) == {"staleness", "perspective", "graph", "recall"}
        # Leakage questions have no gold, so no recall.
        assert "leakage" not in summary[f"k={k}"]["recall"]
