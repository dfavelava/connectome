"""Loading and normalizing the LoCoMo dataset (locomo10.json).

The dataset is not vendored - see the README for where to download it. Each
sample is one long two-speaker conversation split into dated sessions, plus QA
items whose `evidence` lists the dialog ids (e.g. "D1:3" = session 1, turn 3)
that support the answer. Those ids are what lets retrieval be scored without
an LLM.
"""

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Category ids as used by LoCoMo's own evaluation code (task_eval/evaluation.py
# in snap-research/locomo). The dataset JSON itself carries only the integers.
CATEGORY_NAMES: dict[int, str] = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

_SESSION_KEY = re.compile(r"^session_(\d+)$")
# Tolerates the handful of malformed evidence strings in locomo10.json, e.g.
# "D8:6; D9:17", "D9:1 D4:4 D4:6" and "D:11:26".
_DIA_ID = re.compile(r"D:?(\d+):(\d+)")
_SESSION_DATE_FORMAT = "%I:%M %p on %d %B, %Y"


@dataclass(frozen=True)
class Turn:
    dia_id: str
    speaker: str
    text: str
    session_date: str
    # RFC3339, or None when session_date doesn't parse.
    occurred_at: str | None


@dataclass(frozen=True)
class QAItem:
    question: str
    category: int
    evidence: tuple[str, ...]

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, f"category-{self.category}")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    turns: tuple[Turn, ...]
    qa: tuple[QAItem, ...]


def parse_session_date(value: str) -> str | None:
    """Convert a LoCoMo session timestamp ("1:56 pm on 8 May, 2023") to RFC3339.

    The dataset carries no timezone, so UTC is assumed."""
    try:
        parsed = datetime.strptime(value.strip(), _SESSION_DATE_FORMAT)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC).isoformat()


def normalize_evidence(raw: list[str]) -> tuple[str, ...]:
    """Extract canonical "D<session>:<turn>" ids from raw evidence strings,
    deduplicated and in first-seen order."""
    ids: dict[str, None] = {}
    for entry in raw:
        for session, turn in _DIA_ID.findall(str(entry)):
            ids[f"D{int(session)}:{int(turn)}"] = None
    return tuple(ids)


def turn_text(turn: dict[str, object]) -> str:
    """A turn's spoken text, with the caption of any image the speaker shared."""
    text = str(turn.get("text", "")).strip()
    caption = turn.get("blip_caption")
    if caption:
        text = f"{text} [shares an image: {caption}]".strip()
    return text


def parse_sample(raw: dict[str, object]) -> Sample:
    conversation = raw["conversation"]
    assert isinstance(conversation, dict)

    session_numbers = sorted(int(m.group(1)) for key in conversation if (m := _SESSION_KEY.match(key)))
    turns: list[Turn] = []
    for number in session_numbers:
        session_date = str(conversation.get(f"session_{number}_date_time", "")).strip()
        occurred_at = parse_session_date(session_date)
        for turn in conversation[f"session_{number}"]:
            turns.append(
                Turn(
                    dia_id=turn["dia_id"],
                    speaker=turn["speaker"],
                    text=turn_text(turn),
                    session_date=session_date,
                    occurred_at=occurred_at,
                )
            )

    qa_items = raw["qa"]
    assert isinstance(qa_items, list)
    qa = tuple(
        QAItem(
            question=str(item["question"]),
            category=int(item["category"]),
            evidence=normalize_evidence(item.get("evidence") or []),
        )
        for item in qa_items
    )
    return Sample(sample_id=str(raw["sample_id"]), turns=tuple(turns), qa=qa)


def load_dataset(path: Path) -> list[Sample]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return [parse_sample(sample) for sample in raw]
