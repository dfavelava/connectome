"""Check a judge against hand labels, offline, from a scored run's results.

    uv run locomo-eval judge-agreement <run-id> --draw --out labels/<name>.jsonl
    uv run locomo-eval judge-agreement <run-id> --labels labels/<name>.jsonl

A judge prompt can't be validated by its own scores, so a fixed sample of
answers is labelled CORRECT or WRONG by hand and every judge that scored the
same answers is compared with it.

--draw writes a labelling sheet: a seeded random sample of one answer
config's questions, weighted by category (--weights), each with the question,
gold answer, generated answer and an empty "label". The judge's label and
reasoning are left out so labelling stays blind. Fill in each "label" (and
optionally a "note") and commit the file.

--labels reports, for every answer config in results/<run-id>.json, how
often its judge agrees with the hand labels: agreement, false CORRECTs (the
judge accepted an answer labelled WRONG), false WRONGs and null verdicts, per
category and overall. A label only counts against a config that judged the
identical generated answer, so configs that rescore the same cached answers
with another judge prompt are compared on the same items. Nothing calls an
LLM.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from locomo_eval.dataset import CATEGORY_NAMES
from locomo_eval.prompts import CORRECT, JUDGE_LABELS, WRONG

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"
# Weighted towards the categories where judge_v1 and token F1 disagree most.
DEFAULT_WEIGHTS = {"multi-hop": 20, "temporal": 20, "single-hop": 10, "open-domain": 5, "adversarial": 5}
SHEET_FIELDS = ("id", "category", "question", "gold", "answer", "label", "note")
ADVERSARIAL_CATEGORY = CATEGORY_NAMES[5]


class LabelError(ValueError):
    pass


def parse_weights(text: str) -> dict[str, int]:
    """"multi-hop=20,temporal=20" -> {"multi-hop": 20, "temporal": 20}."""
    weights = {}
    for part in text.split(","):
        name, sep, count = part.partition("=")
        name = name.strip()
        if not sep or name not in CATEGORY_NAMES.values() or not count.strip().isdigit():
            raise argparse.ArgumentTypeError(f"{part!r} is not <category>=<count> with a category in {', '.join(CATEGORY_NAMES.values())}")
        weights[name] = int(count)
    return weights


def draw_sheet(questions: list[dict], weights: dict[str, int], rng: random.Random, samples: list[str] | None = None) -> list[dict]:
    """A labelling sheet: up to weights[category] random questions of each
    category, in the run's order, without the judge's verdict."""
    by_category = defaultdict(list)
    for index, question in enumerate(questions):
        # Question ids are <sample-id>#<qa-index>.
        if samples and question["id"].partition("#")[0] not in samples:
            continue
        by_category[question["category"]].append(index)
    chosen = []
    for category, count in weights.items():
        pool = by_category.get(category, [])
        chosen += rng.sample(pool, min(count, len(pool)))
    return [{**{name: questions[i].get(name) for name in SHEET_FIELDS[:5]}, "label": None, "note": ""} for i in sorted(chosen)]


def load_labels(path: Path) -> list[dict]:
    """A labelled sheet; every line needs an id, the generated answer it
    labels and a label of CORRECT or WRONG."""
    labels = []
    seen = set()
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            where = f"{path}:{number}"
            if not isinstance(item.get("id"), str) or not isinstance(item.get("answer"), str):
                raise LabelError(f"{where}: needs a string id and answer")
            if item.get("label") not in JUDGE_LABELS:
                raise LabelError(f"{where}: label {item.get('label')!r} is not one of {', '.join(JUDGE_LABELS)}")
            if item["id"] in seen:
                raise LabelError(f"{where}: {item['id']} is labelled twice")
            seen.add(item["id"])
            labels.append(item)
    return labels


def _empty() -> dict[str, int]:
    return {"n": 0, "agree": 0, "false_correct": 0, "false_wrong": 0, "judge_null": 0}


def agreement(labels: list[dict], questions: list[dict]) -> dict[str, dict]:
    """Per category and overall: how many labelled items this config judged
    (same id and identical answer), how many verdicts match the hand label,
    false CORRECTs, false WRONGs and null verdicts. `agreement` is over
    non-null verdicts; `unmatched` counts labels whose answer this config
    didn't judge."""
    judged = {q["id"]: q for q in questions}
    rows: dict[str, dict] = defaultdict(_empty)
    unmatched = 0
    for item in labels:
        question = judged.get(item["id"])
        if question is None or question["answer"] != item["answer"]:
            unmatched += 1
            continue
        verdict = question["judge_label"]
        for name in (question["category"], "overall"):
            row = rows[name]
            row["n"] += 1
            if verdict is None:
                row["judge_null"] += 1
            elif verdict == item["label"]:
                row["agree"] += 1
            elif verdict == CORRECT:
                row["false_correct"] += 1
            elif verdict == WRONG:
                row["false_wrong"] += 1
    ordered = {name: rows[name] for name in [*sorted(r for r in rows if r != "overall"), "overall"] if name in rows}
    for row in ordered.values():
        verdicts = row["n"] - row["judge_null"]
        row["agreement"] = row["agree"] / verdicts if verdicts else None
    return {"rows": ordered, "matched": len(labels) - unmatched, "unmatched": unmatched}


def hand_label_summary(labels: list[dict]) -> dict[str, dict]:
    """The hand labels' own accuracy per category: what a perfect judge would report on the sample."""
    groups = defaultdict(list)
    for item in labels:
        groups[item.get("category") or "?"].append(item["label"])
        groups["overall"].append(item["label"])
    return {
        name: {"n": len(groups[name]), "accuracy": groups[name].count(CORRECT) / len(groups[name])}
        for name in [*sorted(g for g in groups if g != "overall"), "overall"]
    }


def print_agreement(cfg_hash: str, config: dict, report: dict) -> None:
    judge_prompt = config.get("judge_prompt", {}).get("version", "?")
    answer_prompt = config.get("answer_prompt", {}).get("version", "?")
    print(f"\n{cfg_hash}: judge {config.get('judge_model')} {judge_prompt}, answers {config.get('answer_model')} {answer_prompt}")
    print(f"  {report['matched']} labelled answers judged, {report['unmatched']} labels for other answers")
    if not report["matched"]:
        return
    print(f"  {'category':<14}{'n':>5}{'agree':>8}{'false C':>9}{'false W':>9}{'null':>6}")
    for name, row in report["rows"].items():
        rate = "-" if row["agreement"] is None else f"{row['agreement']:.3f}"
        print(f"  {name:<14}{row['n']:>5}{rate:>8}{row['false_correct']:>9}{row['false_wrong']:>9}{row['judge_null']:>6}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="locomo-eval judge-agreement", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", help="the scored run (results/<run-id>.json)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="where the run's files live (default: %(default)s)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--labels", type=Path, help="a hand-labelled sheet to compare every judge in the run with")
    mode.add_argument("--draw", action="store_true", help="write a labelling sheet to --out")
    parser.add_argument("--out", type=Path, help="with --draw: where to write the sheet")
    parser.add_argument("--config", help="with --draw: the answers config (cfg hash) to draw from; needed when the run has several")
    parser.add_argument("--samples", type=lambda s: s.split(","), help="with --draw: comma-separated sample ids to draw from (default: all scored)")
    default_weights = ",".join(f"{k}={v}" for k, v in DEFAULT_WEIGHTS.items())
    parser.add_argument("--weights", type=parse_weights, default=DEFAULT_WEIGHTS, help=f"with --draw: questions per category (default: {default_weights})")
    parser.add_argument("--seed", type=int, default=0, help="with --draw: random seed (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.draw and args.out is None:
        parser.error("--draw needs --out")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_path = args.results_dir / f"{args.run_id}.json"
    if not run_path.exists():
        sys.exit(f"no run at {run_path}")
    answers = json.loads(run_path.read_text(encoding="utf-8")).get("answers") or {}
    if not answers:
        sys.exit(f"{run_path} has no scored answers; run `locomo-eval answer {args.run_id} ...` first")

    if args.draw:
        if args.config is None and len(answers) > 1:
            sys.exit(f"{run_path} has several answer configs ({', '.join(answers)}); pick one with --config")
        cfg_hash = args.config or next(iter(answers))
        if cfg_hash not in answers:
            sys.exit(f"no answers config {cfg_hash} in {run_path}; it has {', '.join(answers)}")
        sheet = draw_sheet(answers[cfg_hash]["questions"], args.weights, random.Random(args.seed), args.samples)
        if args.out.exists():
            sys.exit(f"{args.out} exists; labelling sheets are never overwritten")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for item in sheet:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        counts = defaultdict(int)
        for item in sheet:
            counts[item["category"]] += 1
        print(f"wrote {len(sheet)} items from answers[{cfg_hash}] to {args.out}: {dict(counts)}")
        return

    try:
        labels = load_labels(args.labels)
    except (LabelError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))
    hand = hand_label_summary(labels)
    print(f"{len(labels)} hand labels in {args.labels}; hand-labelled accuracy: " + ", ".join(f"{k} {v['accuracy']:.3f} ({v['n']})" for k, v in hand.items()))
    for cfg_hash, entry in answers.items():
        print_agreement(cfg_hash, entry["config"], agreement(labels, entry["questions"]))
