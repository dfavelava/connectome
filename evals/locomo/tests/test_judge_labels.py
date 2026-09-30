import argparse
import json
import random

import pytest

from locomo_eval import cli
from locomo_eval.judge_labels import (
    LabelError,
    agreement,
    draw_sheet,
    hand_label_summary,
    load_labels,
    parse_weights,
)


def scored(n: int, category: str, answer: str, label: str | None, sample: str = "conv-26") -> dict:
    return {
        "id": f"{sample}#{n}",
        "category": category,
        "question": f"Question {n}?",
        "gold": "gold",
        "answer": answer,
        "f1": 0.5,
        "judge_label": label,
        "judge_reasoning": "because",
    }


QUESTIONS = [
    scored(0, "multi-hop", "a, b, c", "WRONG"),
    scored(1, "multi-hop", "a", "WRONG"),
    scored(2, "temporal", "3 July 2023", "WRONG"),
    scored(3, "single-hop", "Paris", "CORRECT"),
    scored(4, "adversarial", "Rome", "CORRECT"),
    scored(5, "adversarial", "Not mentioned in the conversation.", None),
    scored(6, "multi-hop", "x", "WRONG", sample="conv-30"),
]


def label(n: int, category: str, answer: str, value: str, sample: str = "conv-26") -> dict:
    return {"id": f"{sample}#{n}", "category": category, "question": f"Question {n}?", "gold": "gold", "answer": answer, "label": value, "note": ""}


LABELS = [
    label(0, "multi-hop", "a, b, c", "CORRECT"),
    label(1, "multi-hop", "a", "WRONG"),
    label(2, "temporal", "3 July 2023", "CORRECT"),
    label(3, "single-hop", "Paris", "CORRECT"),
    label(4, "adversarial", "Rome", "WRONG"),
    label(5, "adversarial", "Not mentioned in the conversation.", "CORRECT"),
]


def test_parse_weights():
    assert parse_weights("multi-hop=20, temporal=3") == {"multi-hop": 20, "temporal": 3}
    for bad in ("multi-hop", "hops=2", "temporal=x"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_weights(bad)


def test_draw_sheet_is_weighted_seeded_and_blind():
    sheet = draw_sheet(QUESTIONS, {"multi-hop": 2, "adversarial": 5, "open-domain": 3}, random.Random(0), samples=["conv-26"])
    assert [item["id"] for item in sheet] == ["conv-26#0", "conv-26#1", "conv-26#4", "conv-26#5"]
    assert sheet == draw_sheet(QUESTIONS, {"multi-hop": 2, "adversarial": 5, "open-domain": 3}, random.Random(0), samples=["conv-26"])
    for item in sheet:
        assert list(item) == ["id", "category", "question", "gold", "answer", "label", "note"]
        assert item["label"] is None
    # Without a sample filter conv-30's question is in the pool too.
    assert len(draw_sheet(QUESTIONS, {"multi-hop": 3}, random.Random(0))) == 3


def test_agreement_counts_false_correct_and_false_wrong():
    report = agreement(LABELS, QUESTIONS)
    assert (report["matched"], report["unmatched"]) == (6, 0)
    rows = report["rows"]
    assert list(rows) == ["adversarial", "multi-hop", "single-hop", "temporal", "overall"]
    assert rows["multi-hop"] == {"n": 2, "agree": 1, "false_correct": 0, "false_wrong": 1, "judge_null": 0, "agreement": 0.5}
    # The judge accepted the trap answer, and gave no verdict on the abstention.
    assert rows["adversarial"] == {"n": 2, "agree": 0, "false_correct": 1, "false_wrong": 0, "judge_null": 1, "agreement": 0.0}
    assert rows["overall"]["agreement"] == pytest.approx(2 / 5)


def test_agreement_only_counts_the_labelled_answer():
    other_answers = [{**q, "answer": "something else"} if q["id"] == "conv-26#0" else q for q in QUESTIONS]
    report = agreement(LABELS, other_answers)
    assert (report["matched"], report["unmatched"]) == (5, 1)
    assert "conv-26#0" not in json.dumps(report) and report["rows"]["multi-hop"]["n"] == 1


def test_hand_label_summary():
    summary = hand_label_summary(LABELS)
    assert summary["multi-hop"] == {"n": 2, "accuracy": 0.5}
    assert summary["overall"] == {"n": 6, "accuracy": pytest.approx(4 / 6)}


def write_jsonl(path, items):
    path.write_text("".join(json.dumps(item) + "\n" for item in items))
    return path


@pytest.mark.parametrize(
    "item, error",
    [
        ({"id": "q", "answer": "a", "label": None}, "label"),
        ({"id": "q", "answer": "a", "label": "correct"}, "label"),
        ({"id": "q", "label": "CORRECT"}, "answer"),
    ],
)
def test_load_labels_rejects_unlabelled_items(tmp_path, item, error):
    with pytest.raises(LabelError, match=error):
        load_labels(write_jsonl(tmp_path / "labels.jsonl", [item]))


def test_load_labels_rejects_duplicates(tmp_path):
    with pytest.raises(LabelError, match="twice"):
        load_labels(write_jsonl(tmp_path / "labels.jsonl", [LABELS[0], LABELS[0]]))


def write_run(tmp_path, *configs):
    answers = {cfg_hash: {"config": config, "questions": questions} for cfg_hash, config, questions in configs}
    path = tmp_path / "base.json"
    path.write_text(json.dumps({"config": {}, "questions": [], "answers": answers}))
    return path


def judge_config(version):
    return {"answer_model": "ollama:a", "judge_model": "ollama:j", "answer_prompt": {"version": "answer_v1"}, "judge_prompt": {"version": version}}


def test_cli_draws_a_sheet_and_reports_agreement(tmp_path, capsys):
    lenient = [{**q, "judge_label": "CORRECT"} for q in QUESTIONS]
    write_run(tmp_path, ("aaaaaaaaaaaa", judge_config("judge_v1"), QUESTIONS), ("bbbbbbbbbbbb", judge_config("judge_v2"), lenient))
    sheet = tmp_path / "labels" / "sheet.jsonl"

    with pytest.raises(SystemExit, match="--config"):
        cli.main(["judge-agreement", "base", "--results-dir", str(tmp_path), "--draw", "--out", str(sheet)])
    argv = ["judge-agreement", "base", "--results-dir", str(tmp_path), "--draw", "--out", str(sheet), "--config", "aaaaaaaaaaaa", "--weights", "multi-hop=5"]
    cli.main(argv)
    assert [json.loads(line)["id"] for line in sheet.read_text().splitlines()] == ["conv-26#0", "conv-26#1", "conv-30#6"]
    with pytest.raises(SystemExit, match="never overwritten"):
        cli.main(argv)

    labels = write_jsonl(tmp_path / "labels.jsonl", LABELS)
    capsys.readouterr()
    cli.main(["judge-agreement", "base", "--results-dir", str(tmp_path), "--labels", str(labels)])
    out = capsys.readouterr().out
    assert "6 hand labels" in out
    assert "aaaaaaaaaaaa: judge ollama:j judge_v1" in out and "bbbbbbbbbbbb: judge ollama:j judge_v2" in out
    assert "6 labelled answers judged" in out


def test_cli_needs_scored_answers(tmp_path):
    with pytest.raises(SystemExit, match="no run"):
        cli.main(["judge-agreement", "base", "--results-dir", str(tmp_path), "--labels", "x.jsonl"])
    (tmp_path / "base.json").write_text(json.dumps({"questions": []}))
    with pytest.raises(SystemExit, match="no scored answers"):
        cli.main(["judge-agreement", "base", "--results-dir", str(tmp_path), "--labels", "x.jsonl"])
