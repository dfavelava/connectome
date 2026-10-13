"""The checked-in West Marches campaign: its memories, viewers and questions.

load_campaign reads campaign/campaign.yaml and checks it is self-consistent
before anything touches a backend: every id it refers to exists, a memory only
derives from or supersedes one written before it, a question's gold memories
are visible to its viewer and its probes are not. A campaign that fails these
checks would score the harness's own mistakes as Connectome's.

Visibility here is the campaign's own, worked out from the acls it writes and
the parties' memberships, not asked of the backend: it is what the leakage
score holds recall's results to.
"""

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN_PATH = PROJECT_DIR / "campaign" / "campaign.yaml"

MEMORY_TYPES = ("note", "fact", "preference", "event")
RELATIONSHIP_KINDS = ("fact", "hypothesis", "rumor")
PUBLIC = "public"

TRUTH = "truth"
STALENESS = "staleness"
PERSPECTIVE = "perspective"
GRAPH = "graph"
LEAKAGE = "leakage"
QUESTION_KINDS = (TRUTH, STALENESS, PERSPECTIVE, GRAPH, LEAKAGE)


class CampaignError(ValueError):
    pass


@dataclass(frozen=True)
class Relationship:
    subject: str
    predicate: str
    object: str | None
    kind: str = "fact"

    def to_client(self) -> dict[str, str | None]:
        """The relationship as connectomeclient's remember takes it."""
        return {"subjectEntityId": self.subject, "predicate": self.predicate, "objectEntityId": self.object, "kind": self.kind}


@dataclass(frozen=True)
class Memory:
    id: str
    session: str
    type: str
    content: str
    relationships: tuple[Relationship, ...]
    # Empty for a public memory. A restricted acl always includes the GM.
    acl: tuple[str, ...]
    occurred_at: str
    derived_from: str | None = None
    supersedes: tuple[str, ...] = ()

    @property
    def entities(self) -> list[str]:
        """The entity ids the relationships name, in order of first mention."""
        ids = [i for r in self.relationships for i in (r.subject, r.object) if i]
        return list(dict.fromkeys(ids))


@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    viewer: str
    question: str
    answer: str
    gold: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    truth: str | None = None
    probes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Campaign:
    name: str
    gm: str
    # party id -> (name, member ids)
    parties: dict[str, tuple[str, tuple[str, ...]]]
    memories: tuple[Memory, ...]
    questions: tuple[Question, ...]
    sha256: str

    @property
    def players(self) -> tuple[str, ...]:
        return tuple(member for _, members in self.parties.values() for member in members)

    @property
    def viewers(self) -> tuple[str, ...]:
        return (self.gm, *self.players)

    def memory(self, memory_id: str) -> Memory:
        return self._by_id()[memory_id]

    def _by_id(self) -> dict[str, Memory]:
        return {m.id: m for m in self.memories}

    def member_of(self, viewer: str) -> tuple[str, ...]:
        return tuple(party for party, (_, members) in self.parties.items() if viewer in members)

    def scope(self, viewer: str) -> frozenset[str]:
        """The ids whose presence in an acl lets viewer see a memory: their
        own and their parties', one level deep, as the backend resolves `as`."""
        return frozenset((viewer, *self.member_of(viewer)))

    def visible(self, viewer: str, memory_id: str) -> bool:
        acl = self.memory(memory_id).acl
        return not acl or bool(self.scope(viewer) & set(acl))

    def superseded(self) -> frozenset[str]:
        """Memories some later memory superseded."""
        return frozenset(old for m in self.memories for old in m.supersedes)

    def facet_family(self, memory_id: str) -> tuple[str, ...]:
        """A facet's root and every facet of that root, root first."""
        root = self.memory(memory_id).derived_from or memory_id
        return (root, *(m.id for m in self.memories if m.derived_from == root))


def load_campaign(path: Path = DEFAULT_CAMPAIGN_PATH) -> Campaign:
    data = path.read_bytes()
    raw = yaml.safe_load(data)
    gm = _str(raw, "gm")
    parties = {
        party_id: (_str(party, "name"), tuple(_list(party, "members")))
        for party_id, party in (raw.get("parties") or {}).items()
    }
    principals = {gm, *parties, *(m for _, members in parties.values() for m in members)}
    if len(principals) != 1 + len(parties) + sum(len(members) for _, members in parties.values()):
        raise CampaignError("the GM, party ids and player ids must all be distinct")

    memories: list[Memory] = []
    seen: dict[str, Memory] = {}
    for session in raw.get("sessions") or []:
        session_id, date = _str(session, "id"), _str(session, "date")
        for item in session.get("memories") or []:
            memory = _memory(item, session_id, date, gm, principals, seen)
            seen[memory.id] = memory
            memories.append(memory)

    campaign = Campaign(
        name=_str(raw, "name"),
        gm=gm,
        parties=parties,
        memories=tuple(memories),
        questions=(),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    questions = tuple(_question(item, campaign) for item in raw.get("questions") or [])
    if len({q.id for q in questions}) != len(questions):
        raise CampaignError("question ids must be unique")
    return replace(campaign, questions=questions)


def _memory(item: dict, session: str, date: str, gm: str, principals: set[str], seen: dict[str, Memory]) -> Memory:
    memory_id = _str(item, "id")
    if memory_id in seen:
        raise CampaignError(f"duplicate memory id {memory_id!r}")
    where = f"memory {memory_id!r}"
    memory_type = _str(item, "type")
    if memory_type not in MEMORY_TYPES:
        raise CampaignError(f"{where}: type {memory_type!r} is not one of {', '.join(MEMORY_TYPES)}")
    relationships = []
    for rel in item.get("rel") or []:
        if not isinstance(rel, list) or len(rel) not in (3, 4) or not all(isinstance(x, str) for x in rel[:2]):
            raise CampaignError(f"{where}: a relationship is [subject, predicate, object, kind?], not {rel!r}")
        kind = rel[3] if len(rel) == 4 else "fact"
        if kind not in RELATIONSHIP_KINDS:
            raise CampaignError(f"{where}: relationship kind {kind!r} is not one of {', '.join(RELATIONSHIP_KINDS)}")
        relationships.append(Relationship(rel[0], rel[1], rel[2], kind))

    acl_raw = item.get("acl")
    if acl_raw == PUBLIC:
        acl: tuple[str, ...] = ()
    elif isinstance(acl_raw, list) and acl_raw:
        if unknown := set(acl_raw) - principals:
            raise CampaignError(f"{where}: acl names unknown ids {sorted(unknown)}")
        acl = tuple(dict.fromkeys([*acl_raw, gm]))
    else:
        raise CampaignError(f"{where}: acl must be {PUBLIC!r} or a non-empty list of ids")

    derived_from = item.get("derived_from")
    if derived_from is not None and derived_from not in seen:
        raise CampaignError(f"{where}: derived_from {derived_from!r} is not an earlier memory")
    supersedes = tuple(item.get("supersedes") or ())
    for old in supersedes:
        if old not in seen:
            raise CampaignError(f"{where}: supersedes {old!r}, which is not an earlier memory")
        if not seen[old].relationships:
            raise CampaignError(f"{where}: supersedes {old!r}, which has no relationships to mark")

    return Memory(
        id=memory_id,
        session=session,
        type=memory_type,
        content=_str(item, "content"),
        relationships=tuple(relationships),
        acl=acl,
        occurred_at=str(item.get("date") or date),
        derived_from=derived_from,
        supersedes=supersedes,
    )


def _question(item: dict, campaign: Campaign) -> Question:
    question = Question(
        id=_str(item, "id"),
        kind=_str(item, "kind"),
        viewer=_str(item, "as"),
        question=_str(item, "question"),
        answer=_str(item, "answer"),
        gold=tuple(item.get("gold") or ()),
        stale=tuple(item.get("stale") or ()),
        truth=item.get("truth"),
        probes=tuple(item.get("probes") or ()),
    )
    where = f"question {question.id!r}"
    if question.kind not in QUESTION_KINDS:
        raise CampaignError(f"{where}: kind {question.kind!r} is not one of {', '.join(QUESTION_KINDS)}")
    if question.viewer not in campaign.viewers:
        raise CampaignError(f"{where}: as {question.viewer!r} is not the GM or a player")
    known = {m.id for m in campaign.memories}
    if unknown := {*question.gold, *question.stale, *question.probes} - known:
        raise CampaignError(f"{where}: unknown memory ids {sorted(unknown)}")
    if hidden := [g for g in (*question.gold, *question.stale) if not campaign.visible(question.viewer, g)]:
        raise CampaignError(f"{where}: {question.viewer} can't see its gold or stale memories {hidden}")
    if seen := [p for p in question.probes if campaign.visible(question.viewer, p)]:
        raise CampaignError(f"{where}: {question.viewer} can see the memories it probes {seen}")
    superseded = campaign.superseded()
    if current := [s for s in question.stale if s not in superseded]:
        raise CampaignError(f"{where}: stale memories {current} were never superseded")
    if old := [g for g in question.gold if g in superseded]:
        raise CampaignError(f"{where}: gold memories {old} are superseded")

    if question.kind == LEAKAGE:
        if not question.probes or question.gold:
            raise CampaignError(f"{where}: a leakage question has probes and no gold")
    elif not question.gold:
        raise CampaignError(f"{where}: a {question.kind} question needs gold memories")
    if (question.kind == TRUTH) != (question.truth is not None):
        raise CampaignError(f"{where}: truth questions, and only they, have a truth status")
    if question.truth is not None and question.truth not in RELATIONSHIP_KINDS:
        raise CampaignError(f"{where}: truth {question.truth!r} is not one of {', '.join(RELATIONSHIP_KINDS)}")
    if question.kind == STALENESS and not question.stale:
        raise CampaignError(f"{where}: a staleness question needs stale memories")
    if question.kind == PERSPECTIVE:
        if len(question.gold) != 1 or campaign.memory(question.gold[0]).derived_from is None:
            raise CampaignError(f"{where}: a perspective question's gold is the viewer's one facet")
    return question


def _str(item: dict, key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise CampaignError(f"missing or non-string {key!r} in {item!r}")
    return value


def _list(item: dict, key: str) -> list[str]:
    value = item.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise CampaignError(f"missing or non-list {key!r} in {item!r}")
    return value
