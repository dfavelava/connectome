"""The extract stage: a local LLM decides what to remember from each conversation.

    uv run locomo-eval extract --samples conv-26 --extractor-model ollama:<model>

The retrieval run stores every dialog turn as a memory, so it only tests
search. This stage instead has an extractor read each conversation one
session at a time, in order, and return the memories an agent would store:
self-contained text with relative dates resolved, a memory type and
occurred_at, entities and relationships, and `source_dia_ids` - the turns a
memory came from, used only for scoring.

The extractor never sees the QA items. Its input is built from a sample's
turns alone (`sessions_of` and `extract_sample` take turns, not a Sample), and
the prompt is generic: it says nothing about LoCoMo's question categories.

Each call sees the session transcript with its dialog ids and date, and the
entities recorded in earlier sessions, so entity ids stay the same across
sessions. The reply is constrained with Ollama structured outputs
(EXTRACTION_SCHEMA) and validated; a reply that fails validation is retried
with the next seed at a small temperature, and after `attempts` tries the
session is recorded as failed without aborting the run. A reply cut off at the
output cap is retried with twice the cap, up to EXTRACT_MAX_NUM_PREDICT.
Source ids that aren't in the session, and entity references to entities that
don't exist, are dropped and counted.

The lifecycle variant (`--extract-prompt lifecycle_v1`) also checks what is
already stored before writing, as an agent that recalls first would. For each
session it recalls the conversation's own earlier memories - the ones a tome
built from the cache so far would hold, less those already superseded - with a
local BM25 search, each turn a query (`recall_memories`), and shows them with
ids (`M<session>.<n>`) in the prompt. Besides new memories the reply can list
`duplicates`, stored memories the session only repeats, which are not written
again, and a new memory can name the stored memories it `supersedes`. Only the
recalled ids can be referenced; others are dropped and counted. The recall is
deterministic and local, so extraction still needs no backend and the cache
stays valid; RECALL_LIMIT and RECALL_PER_TURN are part of the variant, so
changing them means a new prompt version. When the records are applied -
ingested, or counted in the extract output - copies are folded into the
memory they copy (`collapse_copies`): a memory that supersedes one with the
same text, or repeats a current memory, isn't written. The records themselves
keep them, so recall during extraction is unchanged.

Results are appended to a JSONL cache (fsynced per line), one record per
session, keyed by dataset sha256, sample, session, extractor model, prompt
version and sha256, and sampling options. A rerun skips sessions already
extracted, so a failed run resumes without redoing work; failed sessions are
tried again.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from locomo_eval.dataset import Turn, load_dataset, normalize_evidence, sample_ids
from locomo_eval.llm import (
    LLMClient,
    LLMError,
    SamplingOptions,
    load_pricing,
    retry_options,
)
from locomo_eval.prompts import Prompt, load_prompt

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "locomo10.json"
DEFAULT_CACHE_PATH = PROJECT_DIR / "results" / "extractions.jsonl"
DEFAULT_EXTRACTOR_MODEL = "ollama:qwen3:8b"
EXTRACT_VERSION = "extract_v1"
# Add-only like extract_v1, with stricter rules on accuracy, dates and entities.
EXTRACT_V2_VERSION = "extract_v2"
# Prompts named lifecycle_v<N> select the lifecycle variant; see is_lifecycle.
LIFECYCLE_VERSION = "lifecycle_v1"
# Lifecycle built on extract_v2: keeps every stored specific when superseding
# or adding, prefers adding to merging and supersedes only what is no longer true.
LIFECYCLE_V2_VERSION = "lifecycle_v2"
# lifecycle_v2 with every worthwhile message accounted for (a memory or a
# duplicate), duplicates only when nothing is new, and events dated with their
# own date in words instead of "As of" the session date.
LIFECYCLE_V3_VERSION = "lifecycle_v3"
# lifecycle_v3 with every relative time resolved to a date in the text, and
# no "As of" the session date beside it.
LIFECYCLE_V4_VERSION = "lifecycle_v4"
ADD_ONLY = "add-only"
LIFECYCLE = "lifecycle"
EXTRACT = "extract"
EXTRACT_ATTEMPTS = 3
OK = "ok"
FAILED = "failed"

# Mirror MemoryType and RelationshipKind in connectomeclient.
MEMORY_TYPES = ("note", "fact", "preference", "event")
RELATIONSHIP_KINDS = ("fact", "hypothesis", "rumor")

# A session transcript plus the known entities, and a few thousand tokens of
# memories back.
EXTRACT_OPTIONS = SamplingOptions(num_ctx=16384, num_predict=4096)
# The output cap a retry may grow to after a reply was cut off at the cap.
EXTRACT_MAX_NUM_PREDICT = 8192
# Context left over the prompt and output when a retry needs a larger num_ctx.
_CTX_MARGIN = 256
# Lifecycle variant: the earlier memories recalled for a session - at most
# RECALL_PER_TURN per turn, RECALL_LIMIT in all.
RECALL_LIMIT = 30
RECALL_PER_TURN = 3

_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "kind": {"type": "string"},
    },
    "required": ["id", "name", "kind"],
    "additionalProperties": False,
}
_RELATIONSHIP_SCHEMA = {
    "type": "object",
    "properties": {
        "subjectEntityId": {"type": "string"},
        "predicate": {"type": "string"},
        "objectEntityId": {"type": ["string", "null"]},
        "kind": {"type": "string", "enum": list(RELATIONSHIP_KINDS)},
    },
    "required": ["subjectEntityId", "predicate", "objectEntityId", "kind"],
    "additionalProperties": False,
}
_MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "memory_type": {"type": "string", "enum": list(MEMORY_TYPES)},
        "occurred_at": {"type": ["string", "null"]},
        "entities": {"type": "array", "items": {"type": "string"}},
        "relationships": {"type": "array", "items": _RELATIONSHIP_SCHEMA},
        "source_dia_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["content", "memory_type", "occurred_at", "entities", "relationships", "source_dia_ids"],
    "additionalProperties": False,
}
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": _ENTITY_SCHEMA},
        "memories": {"type": "array", "items": _MEMORY_SCHEMA},
    },
    "required": ["entities", "memories"],
    "additionalProperties": False,
}
# The lifecycle variant's reply: each memory also names the stored memories it
# supersedes, and stored memories the session only repeats are listed apart.
_DUPLICATE_SCHEMA = {
    "type": "object",
    "properties": {
        "memory_id": {"type": "string"},
        "source_dia_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["memory_id", "source_dia_ids"],
    "additionalProperties": False,
}
LIFECYCLE_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": _ENTITY_SCHEMA},
        "memories": {
            "type": "array",
            "items": {
                **_MEMORY_SCHEMA,
                "properties": {**_MEMORY_SCHEMA["properties"], "supersedes": {"type": "array", "items": {"type": "string"}}},
                "required": [*_MEMORY_SCHEMA["required"], "supersedes"],
            },
        },
        "duplicates": {"type": "array", "items": _DUPLICATE_SCHEMA},
    },
    "required": ["entities", "memories", "duplicates"],
    "additionalProperties": False,
}

_DIA_ID = re.compile(r"^D(\d+):(\d+)$")
_ENTITY_ID_JUNK = re.compile(r"[^a-z0-9]+")
_WORD = re.compile(r"\w+")
DROP_REASONS = ("source_dia_ids", "entity_ids", "entity_refs", "relationships", "occurred_at", "memory_refs")


def is_lifecycle(prompt: Prompt) -> bool:
    """Whether a prompt is for the lifecycle variant, which recalls and can supersede."""
    return prompt.version.startswith(f"{LIFECYCLE}_")


def variant_of(prompt: Prompt) -> str:
    return LIFECYCLE if is_lifecycle(prompt) else ADD_ONLY


# --- Sessions -------------------------------------------------------------


@dataclass(frozen=True)
class Session:
    sample_id: str
    number: int
    date: str
    # RFC3339, or None when the session date doesn't parse.
    occurred_at: str | None
    turns: tuple[Turn, ...]

    @property
    def dia_ids(self) -> frozenset[str]:
        return frozenset(t.dia_id for t in self.turns)


def session_number(dia_id: str) -> int:
    match = _DIA_ID.match(dia_id)
    if not match:
        raise ValueError(f"dialog id {dia_id!r} is not of the form D<session>:<turn>")
    return int(match.group(1))


def sessions_of(sample_id: str, turns: Iterable[Turn]) -> list[Session]:
    """Group a sample's turns into its sessions, in session order. Only turns
    go in: nothing here can carry a sample's QA items."""
    grouped: dict[int, list[Turn]] = {}
    for turn in turns:
        grouped.setdefault(session_number(turn.dia_id), []).append(turn)
    return [
        Session(sample_id=sample_id, number=number, date=group[0].session_date, occurred_at=group[0].occurred_at, turns=tuple(group))
        for number, group in sorted(grouped.items())
    ]


# --- Extractor output -------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    kind: str


@dataclass(frozen=True)
class ExtractedMemory:
    content: str
    memory_type: str
    occurred_at: str | None
    entities: tuple[str, ...]
    relationships: tuple[dict, ...]
    source_dia_ids: tuple[str, ...]
    # Lifecycle variant: ids of the stored memories this one replaces.
    supersedes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Duplicate:
    """Lifecycle variant: a stored memory the session only repeats, so nothing is written."""

    memory_id: str
    source_dia_ids: tuple[str, ...]


@dataclass(frozen=True)
class Extraction:
    """A validated reply for one session: the entities it mentions and its memories."""

    entities: tuple[Entity, ...]
    memories: tuple[ExtractedMemory, ...]
    # Items dropped during validation, by DROP_REASONS.
    dropped: dict[str, int] = field(default_factory=dict)
    # Lifecycle variant only.
    duplicates: tuple[Duplicate, ...] = ()


class ExtractionParseError(ValueError):
    pass


def normalize_entity_id(value: str) -> str:
    """Lowercase, with runs of anything but letters and digits as one hyphen."""
    return _ENTITY_ID_JUNK.sub("-", value.lower()).strip("-")


def normalize_memory_id(value: str) -> str:
    return value.strip().upper()


def memory_id(session: int, index: int) -> str:
    """The id a memory is shown under to the lifecycle variant: M<session>.<n>, from 1."""
    return f"M{session}.{index + 1}"


def normalize_occurred_at(value: str | None) -> str | None:
    """An ISO 8601 date or date-time as RFC3339 (UTC when it has no offset).
    Raises ValueError when it doesn't parse."""
    if value is None or not value.strip():
        return None
    parsed = datetime.fromisoformat(value.strip())
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExtractionParseError(message)


def _string_list(value: object, what: str) -> list[str]:
    _require(isinstance(value, list) and all(isinstance(v, str) for v in value), f"{what} is not a list of strings")
    assert isinstance(value, list)
    return value


def _sources(value: object, session_dia_ids: frozenset[str], dropped: dict[str, int]) -> list[str]:
    sources: list[str] = []
    for raw_id in _string_list(value, "source_dia_ids"):
        canonical = normalize_evidence([raw_id])
        if len(canonical) != 1 or canonical[0] not in session_dia_ids:
            dropped["source_dia_ids"] += 1
        elif canonical[0] not in sources:
            sources.append(canonical[0])
    return sources


def parse_extraction(text: str, session_dia_ids: Iterable[str], known_entity_ids: Iterable[str] = (), existing_memory_ids: Iterable[str] | None = None) -> Extraction:
    """Validate an extractor reply against EXTRACTION_SCHEMA, or against
    LIFECYCLE_SCHEMA when existing_memory_ids (the stored memories the
    extractor was shown) is given.

    A reply with the wrong shape raises ExtractionParseError, which is worth a
    retry. Within a well-formed reply, bad references are dropped and counted
    instead: source ids not in this session, entity ids that normalize to
    nothing, references to entities neither known nor returned, relationships
    whose endpoints don't exist, occurred_at values that aren't ISO 8601, and
    memory ids that weren't shown or are superseded or repeated twice."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionParseError(f"reply is not JSON: {exc}") from exc
    _require(isinstance(data, dict), "reply is not a JSON object")
    _require(isinstance(data.get("entities"), list), "reply has no entities list")
    _require(isinstance(data.get("memories"), list), "reply has no memories list")
    lifecycle = existing_memory_ids is not None
    existing = frozenset(normalize_memory_id(m) for m in existing_memory_ids or ())
    if lifecycle:
        _require(isinstance(data.get("duplicates"), list), "reply has no duplicates list")
    session_dia_ids = frozenset(session_dia_ids)
    dropped = dict.fromkeys(DROP_REASONS, 0)
    superseded: set[str] = set()

    entities: dict[str, Entity] = {}
    for raw in data["entities"]:
        _require(isinstance(raw, dict), "an entity is not an object")
        for key in ("id", "name", "kind"):
            _require(isinstance(raw.get(key), str), f"entity {key} is not a string")
        entity_id = normalize_entity_id(raw["id"])
        if not entity_id:
            dropped["entity_ids"] += 1
            continue
        entities.setdefault(entity_id, Entity(id=entity_id, name=raw["name"].strip(), kind=raw["kind"].strip()))
    valid_ids = set(known_entity_ids) | set(entities)

    memories = []
    for raw in data["memories"]:
        _require(isinstance(raw, dict), "a memory is not an object")
        content = raw.get("content")
        _require(isinstance(content, str) and bool(content.strip()), "memory content is not a non-empty string")
        assert isinstance(content, str)
        _require(raw.get("memory_type") in MEMORY_TYPES, f"memory_type {raw.get('memory_type')!r} is not one of {', '.join(MEMORY_TYPES)}")
        occurred_at = raw.get("occurred_at")
        _require(occurred_at is None or isinstance(occurred_at, str), "occurred_at is not a string or null")
        try:
            occurred_at = normalize_occurred_at(occurred_at)
        except ValueError:
            dropped["occurred_at"] += 1
            occurred_at = None

        entity_refs: list[str] = []
        for ref in _string_list(raw.get("entities"), "memory entities"):
            ref = normalize_entity_id(ref)
            if ref not in valid_ids:
                dropped["entity_refs"] += 1
            elif ref not in entity_refs:
                entity_refs.append(ref)

        relationships = []
        raw_relationships = raw.get("relationships")
        _require(isinstance(raw_relationships, list), "memory relationships is not a list")
        assert isinstance(raw_relationships, list)
        for rel in raw_relationships:
            _require(isinstance(rel, dict), "a relationship is not an object")
            _require(isinstance(rel.get("subjectEntityId"), str), "relationship subjectEntityId is not a string")
            _require(isinstance(rel.get("predicate"), str) and bool(rel["predicate"].strip()), "relationship predicate is not a non-empty string")
            _require(rel.get("objectEntityId") is None or isinstance(rel.get("objectEntityId"), str), "relationship objectEntityId is not a string or null")
            _require(rel.get("kind") in RELATIONSHIP_KINDS, f"relationship kind {rel.get('kind')!r} is not one of {', '.join(RELATIONSHIP_KINDS)}")
            subject = normalize_entity_id(rel["subjectEntityId"])
            obj = normalize_entity_id(rel["objectEntityId"]) if rel.get("objectEntityId") else None
            if subject not in valid_ids or (obj is not None and obj not in valid_ids):
                dropped["relationships"] += 1
                continue
            relationships.append({"subjectEntityId": subject, "predicate": rel["predicate"].strip(), "objectEntityId": obj, "kind": rel["kind"]})

        sources = _sources(raw.get("source_dia_ids"), session_dia_ids, dropped)

        supersedes: list[str] = []
        if lifecycle:
            for ref in _string_list(raw.get("supersedes"), "supersedes"):
                ref = normalize_memory_id(ref)
                if ref not in existing or ref in superseded:
                    dropped["memory_refs"] += 1
                else:
                    superseded.add(ref)
                    supersedes.append(ref)

        memories.append(
            ExtractedMemory(
                content=content.strip(),
                memory_type=raw["memory_type"],
                occurred_at=occurred_at,
                entities=tuple(entity_refs),
                relationships=tuple(relationships),
                source_dia_ids=tuple(sources),
                supersedes=tuple(supersedes),
            )
        )

    duplicates: list[Duplicate] = []
    for raw in data["duplicates"] if lifecycle else []:
        _require(isinstance(raw, dict), "a duplicate is not an object")
        _require(isinstance(raw.get("memory_id"), str), "duplicate memory_id is not a string")
        ref = normalize_memory_id(raw["memory_id"])
        sources = _sources(raw.get("source_dia_ids"), session_dia_ids, dropped)
        # A memory this reply supersedes isn't repeated as it stands.
        if ref not in existing or ref in superseded or any(d.memory_id == ref for d in duplicates):
            dropped["memory_refs"] += 1
            continue
        duplicates.append(Duplicate(memory_id=ref, source_dia_ids=tuple(sources)))
    return Extraction(entities=tuple(entities.values()), memories=tuple(memories), dropped=dropped, duplicates=tuple(duplicates))


# --- Prompt -----------------------------------------------------------------


def transcript_line(turn: Turn) -> str:
    return f"{turn.dia_id} {turn.speaker}: {turn.text}"


def existing_memory_line(memory: dict) -> str:
    return f"{memory['id']} | {(memory.get('occurred_at') or '')[:10] or '-'} | {memory['content']}"


def extraction_prompt(prompt: Prompt, session: Session, known_entities: Sequence[Entity], existing_memories: Sequence[dict] = ()) -> str:
    """The extraction prompt for one session, given the entities recorded so
    far and, for the lifecycle variant, the stored memories recalled for it."""
    known = "\n".join(f"{e.id} | {e.name} | {e.kind}" for e in known_entities) or "(none yet)"
    fields = {
        "session_date": session.date or "unknown",
        "known_entities": known,
        "transcript": "\n".join(transcript_line(t) for t in session.turns),
    }
    if is_lifecycle(prompt):
        fields["existing_memories"] = "\n".join(existing_memory_line(m) for m in existing_memories) or "(none yet)"
    return prompt.render(**fields)


# --- Recall (lifecycle variant) ----------------------------------------------


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def recall_memories(stored: Sequence[dict], session: Session, *, limit: int = RECALL_LIMIT, per_turn: int = RECALL_PER_TURN) -> list[dict]:
    """The stored memories related to a session, as the lifecycle extractor
    sees them: each turn is a BM25 query over the memories' content, its top
    `per_turn` hits with any shared word are kept at their best score, and the
    top `limit` of those are returned in the order they were stored. `stored`
    is the conversation's current memories, oldest first; ties go to the
    older memory, so the result is deterministic."""
    if not stored:
        return []
    k1, b = 1.2, 0.75
    docs = [_words(m["content"]) for m in stored]
    average = sum(map(len, docs)) / len(docs) or 1.0
    df: dict[str, int] = {}
    for doc in docs:
        for word in set(doc):
            df[word] = df.get(word, 0) + 1
    idf = {w: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for w, n in df.items()}
    counts = [Counter(doc) for doc in docs]
    norms = [k1 * (1 - b + b * len(doc) / average) for doc in docs]

    best: dict[int, float] = {}
    for turn in session.turns:
        query = set(_words(turn.text)) & idf.keys()
        if not query:
            continue
        scores = []
        for i, tf in enumerate(counts):
            score = sum(idf[w] * tf[w] * (k1 + 1) / (tf[w] + norms[i]) for w in query if w in tf)
            if score > 0:
                scores.append((-score, i))
        for neg, i in sorted(scores)[:per_turn]:
            best[i] = max(best.get(i, 0.0), -neg)
    chosen = sorted(best, key=lambda i: (-best[i], i))[:limit]
    return [stored[i] for i in sorted(chosen)]


def stored_memories(record: dict) -> list[dict]:
    """A successful session record's memories, each with its id."""
    return [{**m, "id": memory_id(record["session"], i)} for i, m in enumerate(record["memories"])]


def superseded_ids(record: dict) -> set[str]:
    return {ref for m in record["memories"] for ref in m.get("supersedes") or ()}


def after_record(stored: list[dict], record: dict) -> list[dict]:
    """The conversation's current memories once a session record is applied:
    what it supersedes goes, its new memories are added."""
    if record["status"] != OK:
        return stored
    gone = superseded_ids(record)
    return [m for m in stored if m["id"] not in gone] + stored_memories(record)


def normalize_content(text: str) -> str:
    """Memory text compared for copies: case-folded, with runs of whitespace as one space."""
    return " ".join(text.casefold().split())


@dataclass(frozen=True)
class Collapsed:
    """A conversation's memories as they are written, once copies are folded
    into the memory they copy (see collapse_copies)."""

    # The memories to write, each with its id, in session order; `supersedes`
    # holds only ids among them.
    memories: list[dict]
    # Memories not written because their text equals a memory they supersede...
    verbatim_supersedes: int
    # ...or a current memory, stored or earlier in the same reply.
    repeats: int
    # Memories superseded once copies are folded.
    superseded: int


def collapse_copies(records: Iterable[dict]) -> Collapsed:
    """Fold a lifecycle extraction's copies into the memories they copy, as
    its session records (in session order) are applied. The records are left
    as they are, so a cache can be replayed without new LLM calls.

    The extractor sometimes supersedes a memory with the same text, or writes
    again what is already stored. A memory whose normalized text equals one it
    supersedes becomes a duplicate of it, and that memory isn't superseded; one
    equal to a current memory - stored and not superseded, or earlier in the
    same reply - becomes a duplicate of that. Either way it isn't written, and
    anything else it supersedes is superseded by the memory it copies. Later
    references to a copy go to that memory too."""
    kept: dict[str, dict] = {}
    alias: dict[str, str] = {}
    current: dict[str, str] = {}
    gone: set[str] = set()
    verbatim = repeats = 0

    def supersede(old_id: str, new_id: str) -> None:
        gone.add(old_id)
        text = normalize_content(kept[old_id]["content"])
        if current.get(text) == old_id:
            del current[text]
        kept[new_id]["supersedes"].append(old_id)

    for record in records:
        if record["status"] != OK:
            continue
        for memory in stored_memories(record):
            targets: list[str] = []
            for ref in memory.get("supersedes") or ():
                ref = alias.get(ref, ref)
                # An id no memory has can only come from a session re-extracted after its successor.
                if ref in kept and ref not in gone and ref not in targets:
                    targets.append(ref)
            text = normalize_content(memory["content"])
            match = next((t for t in targets if normalize_content(kept[t]["content"]) == text), None)
            if match is not None:
                verbatim += 1
                targets.remove(match)
            elif text in current:
                repeats += 1
                match = current[text]
            if match is not None:
                alias[memory["id"]] = match
                for target in targets:
                    if target != match:
                        supersede(target, match)
                continue
            kept[memory["id"]] = {**memory, "supersedes": []}
            current[text] = memory["id"]
            for target in targets:
                supersede(target, memory["id"])
    return Collapsed(memories=list(kept.values()), verbatim_supersedes=verbatim, repeats=repeats, superseded=len(gone))


# --- Extractors -------------------------------------------------------------


@dataclass(frozen=True)
class SessionExtraction:
    """The outcome of extracting one session, successful or not."""

    status: str
    extraction: Extraction | None
    attempts: int
    input_tokens: int
    output_tokens: int
    seconds: float
    # The last validation error, when status is FAILED.
    error: str | None = None
    # The options of the last attempt, which differ from the extractor's on a retry.
    options: SamplingOptions | None = None
    # Attempts whose reply was cut off at num_predict.
    truncated: int = 0


class Extractor(Protocol):
    """Turns one session into memories. `model`, `prompt` and `options` name
    everything besides the input that decides its output, for the cache key."""

    model: str
    prompt: Prompt
    options: SamplingOptions

    async def extract(self, session: Session, known_entities: Sequence[Entity], existing_memories: Sequence[dict] = ()) -> SessionExtraction: ...


class OllamaExtractor:
    """An Extractor on a local Ollama model, via LLMClient and structured outputs.

    The first attempt uses `options` as given. A reply that fails validation
    is retried with the next seed at RETRY_TEMPERATURE - at temperature 0
    decoding is greedy, so a new seed alone would repeat the reply. A reply cut
    off at num_predict is also retried with twice the cap (up to
    `max_num_predict`), and num_ctx grows if the prompt and cap no longer fit.
    Since only retries change, cached first-attempt results stay valid.
    Transport errors are the client's to retry and still raise."""

    def __init__(self, client: LLMClient, model: str, prompt: Prompt, *, options: SamplingOptions = EXTRACT_OPTIONS, attempts: int = EXTRACT_ATTEMPTS, max_num_predict: int = EXTRACT_MAX_NUM_PREDICT):
        self.client = client
        self.model = model
        self.prompt = prompt
        self.options = options
        self.attempts = attempts
        self.max_num_predict = max_num_predict

    async def extract(self, session: Session, known_entities: Sequence[Entity], existing_memories: Sequence[dict] = ()) -> SessionExtraction:
        text = extraction_prompt(self.prompt, session, known_entities, existing_memories)
        known_ids = [e.id for e in known_entities]
        lifecycle = is_lifecycle(self.prompt)
        schema = LIFECYCLE_SCHEMA if lifecycle else EXTRACTION_SCHEMA
        existing_ids = [m["id"] for m in existing_memories] if lifecycle else None
        input_tokens = output_tokens = truncated = 0
        num_predict, num_ctx = self.options.num_predict, self.options.num_ctx
        error = None
        options = self.options
        started = time.monotonic()
        for attempt in range(self.attempts):
            options = replace(retry_options(self.options, attempt), num_predict=num_predict, num_ctx=num_ctx)
            completion = await self.client.complete(self.model, "", text, stage=EXTRACT, format=schema, options=options)
            input_tokens += completion.input_tokens
            output_tokens += completion.output_tokens
            # Ollama reports done_reason "length"; the count is a fallback for servers that don't.
            if completion.truncated or completion.output_tokens >= options.num_predict:
                truncated += 1
                error = f"reply truncated at the {options.num_predict}-token output cap (num_predict)"
                num_predict = min(num_predict * 2, max(self.max_num_predict, num_predict))
                num_ctx = max(num_ctx, completion.input_tokens + num_predict + _CTX_MARGIN)
                continue
            try:
                extraction = parse_extraction(completion.text, session.dia_ids, known_ids, existing_ids)
            except ExtractionParseError as exc:
                error = str(exc)
                continue
            return SessionExtraction(OK, extraction, attempt + 1, input_tokens, output_tokens, time.monotonic() - started, options=options, truncated=truncated)
        return SessionExtraction(FAILED, None, self.attempts, input_tokens, output_tokens, time.monotonic() - started, error, options=options, truncated=truncated)


# --- Cache ------------------------------------------------------------------


def options_hash(options: SamplingOptions) -> str:
    return _sha256(json.dumps(asdict(options), sort_keys=True))[:12]


def cache_key(dataset_sha256: str, sample_id: str, session: int, extractor: Extractor) -> tuple:
    return (dataset_sha256, sample_id, session, extractor.model, extractor.prompt.version, extractor.prompt.sha256, options_hash(extractor.options))


def record_key(record: dict) -> tuple:
    return (
        record["dataset_sha256"],
        record["sample_id"],
        record["session"],
        record["extractor_model"],
        record["prompt_version"],
        record["prompt_sha256"],
        record["options_hash"],
    )


class ExtractionCache:
    """Extracted sessions, appended one JSON line at a time. Only successful
    records are hits; failed ones are kept in the file as a log, and the
    latest failure of a session never extracted successfully is in `failed`."""

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[tuple, dict] = {}
        self.failed: dict[tuple, dict] = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # A line cut short by a kill mid-write.
                        continue
                    self._add(record)

    def get(self, key: tuple) -> dict | None:
        return self.records.get(key)

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._add(record)

    def _add(self, record: dict) -> None:
        key = record_key(record)
        if record.get("status") == OK:
            self.records[key] = record
            self.failed.pop(key, None)
        elif key not in self.records:
            self.failed[key] = record


def session_record(
    dataset_sha256: str,
    session: Session,
    extractor: Extractor,
    known_entities: Sequence[Entity],
    result: SessionExtraction,
    model_digest: str | None,
    recalled: Sequence[dict] | None = None,
) -> dict:
    extraction = result.extraction
    return {
        "dataset_sha256": dataset_sha256,
        "sample_id": session.sample_id,
        "session": session.number,
        "extractor_model": extractor.model,
        "model_digest": model_digest,
        "prompt_version": extractor.prompt.version,
        "prompt_sha256": extractor.prompt.sha256,
        "options_hash": options_hash(extractor.options),
        "options": asdict(extractor.options),
        "status": result.status,
        "session_date": session.date,
        "session_occurred_at": session.occurred_at,
        "turns": len(session.turns),
        "known_entity_ids": [e.id for e in known_entities],
        "entities": [asdict(e) for e in extraction.entities] if extraction else [],
        "memories": [asdict(m) for m in extraction.memories] if extraction else [],
        # Lifecycle variant: the stored memories it was shown, and the ones the session only repeated.
        "recalled_memory_ids": [m["id"] for m in recalled] if recalled is not None else None,
        "duplicates": [asdict(d) for d in extraction.duplicates] if extraction else [],
        "dropped": extraction.dropped if extraction else {},
        "attempts": result.attempts,
        # The options of the attempt recorded here; they differ from "options" after a retry.
        "attempt_options": asdict(result.options) if result.options else None,
        "truncated_attempts": result.truncated,
        "error": result.error,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "seconds": round(result.seconds, 3),
        "extracted_at": datetime.now(UTC).isoformat(),
    }


def merge_entities(registry: dict[str, Entity], entities: Iterable[Entity | dict]) -> None:
    """Add new entities to the registry. A known id keeps its first name, and
    gains a kind only if it had none."""
    for entity in entities:
        if isinstance(entity, dict):
            entity = Entity(**entity)
        existing = registry.get(entity.id)
        if existing is None:
            registry[entity.id] = entity
        elif not existing.kind and entity.kind:
            registry[entity.id] = replace(existing, kind=entity.kind)


# --- Running ------------------------------------------------------------------


@dataclass
class SampleTally:
    sample_id: str
    sessions: int = 0
    extracted: int = 0
    cached: int = 0
    failed: list[int] = field(default_factory=list)
    memories: int = 0
    # Lifecycle variant: stored memories repeated rather than written again,
    # copies folded into the memory they copy (see collapse_copies), and
    # memories replaced once copies are folded.
    duplicates: int = 0
    verbatim_supersedes: int = 0
    repeats: int = 0
    superseded: int = 0
    dropped: dict[str, int] = field(default_factory=lambda: dict.fromkeys(DROP_REASONS, 0))
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def add(self, record: dict, cached: bool) -> None:
        if record["status"] != OK:
            self.failed.append(record["session"])
        elif cached:
            self.cached += 1
        else:
            self.extracted += 1
        self.memories += len(record["memories"])
        self.duplicates += len(record.get("duplicates") or ())
        for reason, count in record["dropped"].items():
            self.dropped[reason] = self.dropped.get(reason, 0) + count
        if not cached:
            self.input_tokens += record["input_tokens"]
            self.output_tokens += record["output_tokens"]
            self.seconds += record["seconds"]

    def collapsed(self, collapsed: Collapsed) -> None:
        self.verbatim_supersedes = collapsed.verbatim_supersedes
        self.repeats = collapsed.repeats
        self.superseded = collapsed.superseded


async def extract_sample(
    extractor: Extractor,
    cache: ExtractionCache,
    dataset_sha256: str,
    sample_id: str,
    turns: Sequence[Turn],
    *,
    model_digest: str | None = None,
    on_session: Callable[[dict, bool], None] | None = None,
) -> SampleTally:
    """Extract a sample's sessions in order, reusing cached ones. Takes the
    sample's turns, never the Sample, so its questions can't reach a prompt.

    The entities recorded so far - from cached and fresh sessions alike - are
    passed to each session, and for the lifecycle variant the current memories
    recalled for it. A failed session contributes none and the run moves on."""
    tally = SampleTally(sample_id)
    registry: dict[str, Entity] = {}
    lifecycle = is_lifecycle(extractor.prompt)
    stored: list[dict] = []
    records: list[dict] = []
    for session in sessions_of(sample_id, turns):
        tally.sessions += 1
        record = cache.get(cache_key(dataset_sha256, sample_id, session.number, extractor))
        cached = record is not None
        if record is None:
            known = list(registry.values())
            recalled = recall_memories(stored, session) if lifecycle else None
            result = await extractor.extract(session, known, recalled or ())
            record = session_record(dataset_sha256, session, extractor, known, result, model_digest, recalled)
            cache.append(record)
        merge_entities(registry, record["entities"])
        # Recall sees the records as extracted, copies and all, so the cache stays valid.
        stored = after_record(stored, record)
        records.append(record)
        tally.add(record, cached)
        if on_session:
            on_session(record, cached)
    if lifecycle:
        tally.collapsed(collapse_copies(records))
    return tally


def pending_sessions(cache: ExtractionCache, extractor: Extractor, dataset_sha256: str, samples: Iterable[tuple[str, Sequence[Turn]]]) -> int:
    """How many sessions aren't cached yet - the ones a run would call the model for."""
    return sum(
        1
        for sample_id, turns in samples
        for session in sessions_of(sample_id, turns)
        if cache.get(cache_key(dataset_sha256, sample_id, session.number, extractor)) is None
    )


def load_extractions(cache_path: Path, dataset_sha256: str, sample_id: str, extractor_model: str, prompt: Prompt, options: SamplingOptions = EXTRACT_OPTIONS) -> list[dict]:
    """A sample's cached session records for one extractor config, in session order."""
    return _sample_records(ExtractionCache(cache_path).records.values(), dataset_sha256, sample_id, extractor_model, prompt, options)


def _sample_records(records: Iterable[dict], dataset_sha256: str, sample_id: str, extractor_model: str, prompt: Prompt, options: SamplingOptions) -> list[dict]:
    wanted = (dataset_sha256, sample_id, extractor_model, prompt.version, prompt.sha256, options_hash(options))
    matching = [r for r in records if (r["dataset_sha256"], r["sample_id"], r["extractor_model"], r["prompt_version"], r["prompt_sha256"], r["options_hash"]) == wanted]
    return sorted(matching, key=lambda r: r["session"])


@dataclass(frozen=True)
class CachedRun:
    """One extractor config's cached extraction of some samples, as the
    retrieval run ingests it. No LLM calls are involved."""

    # Each sample's successful session records, in session order.
    records: dict[str, list[dict]]
    # What the run config records: the extractor config and totals.
    config: dict

    @property
    def lifecycle(self) -> bool:
        return self.config.get("variant") == LIFECYCLE

    def memories(self, sample_id: str) -> list[dict]:
        """Every memory written, each with its id (see memory_id), in session order.
        Superseded ones are included; their successors name them in `supersedes`.
        For the lifecycle variant, copies are folded (see collapse_copies)."""
        if self.lifecycle:
            return collapse_copies(self.records[sample_id]).memories
        return [m for record in self.records[sample_id] for m in stored_memories(record)]

    def entity_ids(self, sample_id: str) -> set[str]:
        return {e["id"] for record in self.records[sample_id] for e in record["entities"]}


def cached_run(
    cache: ExtractionCache,
    dataset_sha256: str,
    conversations: Iterable[tuple[str, Sequence[Turn]]],
    extractor_model: str,
    prompt: Prompt,
    options: SamplingOptions = EXTRACT_OPTIONS,
) -> CachedRun:
    """The cached records for each conversation, and the totals over them.

    A session with no successful record is left out and counted: as failed
    when its last attempt failed validation, else as not extracted. A
    conversation with no records at all raises ValueError - run the extract
    stage first."""
    records: dict[str, list[dict]] = {}
    failed: dict[str, list[int]] = {}
    unextracted: dict[str, list[int]] = {}
    sessions = 0
    for sample_id, turns in conversations:
        found = _sample_records(cache.records.values(), dataset_sha256, sample_id, extractor_model, prompt, options)
        if not found:
            raise ValueError(f"no cached extraction of {sample_id} by {extractor_model} ({prompt.version}) in {cache.path}")
        records[sample_id] = found
        extracted = {r["session"] for r in found}
        failed_here = {r["session"] for r in _sample_records(cache.failed.values(), dataset_sha256, sample_id, extractor_model, prompt, options)}
        for session in sessions_of(sample_id, turns):
            sessions += 1
            if session.number in extracted:
                continue
            (failed if session.number in failed_here else unextracted).setdefault(sample_id, []).append(session.number)

    every = [r for found in records.values() for r in found]
    lifecycle = is_lifecycle(prompt)
    collapsed = [collapse_copies(found) for found in records.values()] if lifecycle else []
    dropped = dict.fromkeys(DROP_REASONS, 0)
    for record in every:
        for reason, count in record["dropped"].items():
            dropped[reason] = dropped.get(reason, 0) + count
    config = {
        "cache": str(cache.path),
        "extractor_model": extractor_model,
        "model_digests": sorted({r["model_digest"] for r in every if r.get("model_digest")}),
        "prompt": prompt.config(),
        "variant": variant_of(prompt),
        "recall": {"limit": RECALL_LIMIT, "per_turn": RECALL_PER_TURN, "method": "bm25 over the conversation's current memories, one query per turn"}
        if lifecycle
        else None,
        "options": asdict(options),
        "options_hash": options_hash(options),
        "totals": {
            "sessions": sessions,
            "extracted_sessions": len(every),
            "failed_sessions": failed,
            "unextracted_sessions": unextracted,
            # As extracted; for the lifecycle variant, verbatim supersedes and
            # repeats aren't written, and superseded counts once they're folded.
            "memories": sum(len(r["memories"]) for r in every),
            "duplicates": sum(len(r.get("duplicates") or ()) for r in every),
            "verbatim_supersedes": sum(c.verbatim_supersedes for c in collapsed) if lifecycle else None,
            "repeats": sum(c.repeats for c in collapsed) if lifecycle else None,
            "superseded": sum(c.superseded for c in collapsed) if lifecycle else 0,
            "attempts": sum(r["attempts"] for r in every),
            "input_tokens": sum(r["input_tokens"] for r in every),
            "output_tokens": sum(r["output_tokens"] for r in every),
            "seconds": round(sum(r["seconds"] for r in every), 3),
            "dropped": dropped,
        },
    }
    return CachedRun(records=records, config=config)


def print_tallies(tallies: list[SampleTally]) -> None:
    columns = ("sessions", "extracted", "cached", "failed", "memories", "duplicates", "verbatim", "repeats", "superseded", "dropped_ids", "in_tokens", "out_tokens", "seconds")
    print(f"{'sample':<10}" + "".join(f"{c:>12}" for c in columns))
    for t in tallies:
        cells = (t.sessions, t.extracted, t.cached, len(t.failed), t.memories, t.duplicates, t.verbatim_supersedes, t.repeats, t.superseded, t.dropped["source_dia_ids"], t.input_tokens, t.output_tokens, f"{t.seconds:.0f}")
        print(f"{t.sample_id:<10}" + "".join(f"{c:>12}" for c in cells))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="locomo-eval extract", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="path to locomo10.json (default: %(default)s)")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH, help="JSONL cache of extracted sessions (default: %(default)s)")
    parser.add_argument("--samples", type=sample_ids, help="comma-separated sample ids to extract (default: all)")
    parser.add_argument("--extractor-model", default=DEFAULT_EXTRACTOR_MODEL, help="provider:model that extracts (default: %(default)s)")
    parser.add_argument(
        "--extract-prompt",
        default=EXTRACT_VERSION,
        help=f"extraction prompt version; {LIFECYCLE}_v<N> runs the variant that recalls and supersedes (default: %(default)s)",
    )
    parser.add_argument("--attempts", type=int, default=EXTRACT_ATTEMPTS, help="tries per session before it is recorded as failed (default: %(default)s)")
    parser.add_argument("--concurrency", type=int, default=1, help="samples extracted at once; sessions within a sample always run in order (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=900.0, help="per-call timeout in seconds (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.data.exists():
        sys.exit(f"dataset not found at {args.data}; see evals/locomo/README.md for the download")
    dataset_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    samples = load_dataset(args.data)
    if args.samples:
        wanted = set(args.samples)
        samples = [s for s in samples if s.sample_id in wanted]
        if missing := wanted - {s.sample_id for s in samples}:
            sys.exit(f"unknown sample ids: {', '.join(sorted(missing))}")
    try:
        prompt = load_prompt(args.extract_prompt)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))
    # Only turns go further; the QA items stay behind.
    conversations = [(s.sample_id, s.turns) for s in samples]
    cache = ExtractionCache(args.cache)

    def on_session(record: dict, cached: bool) -> None:
        if cached:
            return
        if record["status"] != OK:
            status = f"FAILED after {record['attempts']} attempts: {record['error']}"
        elif record.get("recalled_memory_ids") is not None:
            status = f"{len(record['memories'])} memories, {len(record['duplicates'])} duplicates, {len(superseded_ids(record))} superseded of {len(record['recalled_memory_ids'])} recalled"
        else:
            status = f"{len(record['memories'])} memories"
        print(
            f"{record['sample_id']} session {record['session']}: {status} "
            f"({record['input_tokens']} in / {record['output_tokens']} out tokens, {record['seconds']:.1f}s)",
            file=sys.stderr,
        )

    async def go() -> list[SampleTally]:
        async with LLMClient(options=EXTRACT_OPTIONS, concurrency=args.concurrency, timeout=args.timeout, pricing=load_pricing()) as client:
            extractor = OllamaExtractor(client, args.extractor_model, prompt, attempts=args.attempts)
            pending = pending_sessions(cache, extractor, dataset_sha256, conversations)
            print(f"{pending} sessions to extract with {args.extractor_model} ({prompt.version}, {variant_of(prompt)}); cache {args.cache}", file=sys.stderr)
            digest = None
            if pending:
                # Fails before any call if the model isn't pulled; a fully cached run needs no Ollama.
                try:
                    digest = (await client.describe_model(args.extractor_model))["digest"]
                except (LLMError, ValueError) as exc:
                    sys.exit(str(exc))
            semaphore = asyncio.Semaphore(args.concurrency)

            async def one(sample_id: str, turns: Sequence[Turn]) -> SampleTally:
                async with semaphore:
                    return await extract_sample(extractor, cache, dataset_sha256, sample_id, turns, model_digest=digest, on_session=on_session)

            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(one(sample_id, turns)) for sample_id, turns in conversations]
            return [t.result() for t in tasks]

    try:
        tallies = asyncio.run(go())
    except* LLMError as group:
        sys.exit(f"stopped: {group.exceptions[0]}\nfinished sessions are saved in {args.cache}; rerun the same command to resume")

    print_tallies(tallies)
    if failed := {t.sample_id: t.failed for t in tallies if t.failed}:
        sys.exit(f"\nsessions failed validation: {failed}; rerun to retry them")
    print(f"\nwrote {args.cache}")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
