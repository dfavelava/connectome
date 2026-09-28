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
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from locomo_eval.dataset import Turn, load_dataset, normalize_evidence
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

_DIA_ID = re.compile(r"^D(\d+):(\d+)$")
_ENTITY_ID_JUNK = re.compile(r"[^a-z0-9]+")
DROP_REASONS = ("source_dia_ids", "entity_ids", "entity_refs", "relationships", "occurred_at")


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


@dataclass(frozen=True)
class Extraction:
    """A validated reply for one session: the entities it mentions and its memories."""

    entities: tuple[Entity, ...]
    memories: tuple[ExtractedMemory, ...]
    # Items dropped during validation, by DROP_REASONS.
    dropped: dict[str, int] = field(default_factory=dict)


class ExtractionParseError(ValueError):
    pass


def normalize_entity_id(value: str) -> str:
    """Lowercase, with runs of anything but letters and digits as one hyphen."""
    return _ENTITY_ID_JUNK.sub("-", value.lower()).strip("-")


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


def parse_extraction(text: str, session_dia_ids: Iterable[str], known_entity_ids: Iterable[str] = ()) -> Extraction:
    """Validate an extractor reply against EXTRACTION_SCHEMA.

    A reply with the wrong shape raises ExtractionParseError, which is worth a
    retry. Within a well-formed reply, bad references are dropped and counted
    instead: source ids not in this session, entity ids that normalize to
    nothing, references to entities neither known nor returned, relationships
    whose endpoints don't exist, and occurred_at values that aren't ISO 8601."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionParseError(f"reply is not JSON: {exc}") from exc
    _require(isinstance(data, dict), "reply is not a JSON object")
    _require(isinstance(data.get("entities"), list), "reply has no entities list")
    _require(isinstance(data.get("memories"), list), "reply has no memories list")
    session_dia_ids = frozenset(session_dia_ids)
    dropped = dict.fromkeys(DROP_REASONS, 0)

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

        sources: list[str] = []
        for raw_id in _string_list(raw.get("source_dia_ids"), "source_dia_ids"):
            canonical = normalize_evidence([raw_id])
            if len(canonical) != 1 or canonical[0] not in session_dia_ids:
                dropped["source_dia_ids"] += 1
            elif canonical[0] not in sources:
                sources.append(canonical[0])

        memories.append(
            ExtractedMemory(
                content=content.strip(),
                memory_type=raw["memory_type"],
                occurred_at=occurred_at,
                entities=tuple(entity_refs),
                relationships=tuple(relationships),
                source_dia_ids=tuple(sources),
            )
        )
    return Extraction(entities=tuple(entities.values()), memories=tuple(memories), dropped=dropped)


# --- Prompt -----------------------------------------------------------------


def transcript_line(turn: Turn) -> str:
    return f"{turn.dia_id} {turn.speaker}: {turn.text}"


def extraction_prompt(prompt: Prompt, session: Session, known_entities: Sequence[Entity]) -> str:
    """The extraction prompt for one session, given the entities recorded so far."""
    known = "\n".join(f"{e.id} | {e.name} | {e.kind}" for e in known_entities) or "(none yet)"
    return prompt.render(
        session_date=session.date or "unknown",
        known_entities=known,
        transcript="\n".join(transcript_line(t) for t in session.turns),
    )


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

    async def extract(self, session: Session, known_entities: Sequence[Entity]) -> SessionExtraction: ...


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

    async def extract(self, session: Session, known_entities: Sequence[Entity]) -> SessionExtraction:
        text = extraction_prompt(self.prompt, session, known_entities)
        known_ids = [e.id for e in known_entities]
        input_tokens = output_tokens = truncated = 0
        num_predict, num_ctx = self.options.num_predict, self.options.num_ctx
        error = None
        options = self.options
        started = time.monotonic()
        for attempt in range(self.attempts):
            options = replace(retry_options(self.options, attempt), num_predict=num_predict, num_ctx=num_ctx)
            completion = await self.client.complete(self.model, "", text, stage=EXTRACT, format=EXTRACTION_SCHEMA, options=options)
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
                extraction = parse_extraction(completion.text, session.dia_ids, known_ids)
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
    records are hits; failed ones are kept in the file as a log."""

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[tuple, dict] = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # A line cut short by a kill mid-write.
                        continue
                    if record.get("status") == OK:
                        self.records[record_key(record)] = record

    def get(self, key: tuple) -> dict | None:
        return self.records.get(key)

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if record["status"] == OK:
            self.records[record_key(record)] = record


def session_record(dataset_sha256: str, session: Session, extractor: Extractor, known_entities: Sequence[Entity], result: SessionExtraction, model_digest: str | None) -> dict:
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
        for reason, count in record["dropped"].items():
            self.dropped[reason] = self.dropped.get(reason, 0) + count
        if not cached:
            self.input_tokens += record["input_tokens"]
            self.output_tokens += record["output_tokens"]
            self.seconds += record["seconds"]


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
    passed to each session. A failed session contributes none and the run
    moves on."""
    tally = SampleTally(sample_id)
    registry: dict[str, Entity] = {}
    for session in sessions_of(sample_id, turns):
        tally.sessions += 1
        record = cache.get(cache_key(dataset_sha256, sample_id, session.number, extractor))
        cached = record is not None
        if record is None:
            known = list(registry.values())
            result = await extractor.extract(session, known)
            record = session_record(dataset_sha256, session, extractor, known, result, model_digest)
            cache.append(record)
        merge_entities(registry, record["entities"])
        tally.add(record, cached)
        if on_session:
            on_session(record, cached)
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
    cache = ExtractionCache(cache_path)
    records = [
        r
        for r in cache.records.values()
        if (r["dataset_sha256"], r["sample_id"], r["extractor_model"], r["prompt_version"], r["prompt_sha256"], r["options_hash"])
        == (dataset_sha256, sample_id, extractor_model, prompt.version, prompt.sha256, options_hash(options))
    ]
    return sorted(records, key=lambda r: r["session"])


def print_tallies(tallies: list[SampleTally]) -> None:
    columns = ("sessions", "extracted", "cached", "failed", "memories", "dropped_ids", "in_tokens", "out_tokens", "seconds")
    print(f"{'sample':<10}" + "".join(f"{c:>12}" for c in columns))
    for t in tallies:
        cells = (t.sessions, t.extracted, t.cached, len(t.failed), t.memories, t.dropped["source_dia_ids"], t.input_tokens, t.output_tokens, f"{t.seconds:.0f}")
        print(f"{t.sample_id:<10}" + "".join(f"{c:>12}" for c in cells))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="locomo-eval extract", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="path to locomo10.json (default: %(default)s)")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH, help="JSONL cache of extracted sessions (default: %(default)s)")
    parser.add_argument("--samples", type=lambda s: s.split(","), help="comma-separated sample ids to extract (default: all)")
    parser.add_argument("--extractor-model", default=DEFAULT_EXTRACTOR_MODEL, help="provider:model that extracts (default: %(default)s)")
    parser.add_argument("--extract-prompt", default=EXTRACT_VERSION, help="extraction prompt version (default: %(default)s)")
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
        status = f"{len(record['memories'])} memories" if record["status"] == OK else f"FAILED after {record['attempts']} attempts: {record['error']}"
        print(
            f"{record['sample_id']} session {record['session']}: {status} "
            f"({record['input_tokens']} in / {record['output_tokens']} out tokens, {record['seconds']:.1f}s)",
            file=sys.stderr,
        )

    async def go() -> list[SampleTally]:
        async with LLMClient(options=EXTRACT_OPTIONS, concurrency=args.concurrency, timeout=args.timeout, pricing=load_pricing()) as client:
            extractor = OllamaExtractor(client, args.extractor_model, prompt, attempts=args.attempts)
            pending = pending_sessions(cache, extractor, dataset_sha256, conversations)
            print(f"{pending} sessions to extract with {args.extractor_model} ({prompt.version}); cache {args.cache}", file=sys.stderr)
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
