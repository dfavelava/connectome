from pathlib import Path

import pytest

from westmarches_eval.campaign import Campaign, load_campaign

# Two parties, a public event with a facet per party, a supersession, a rumor
# and a GM secret: enough to exercise every score.
SMALL = """
name: Test Marches
gm: gm
parties:
  party-a: {name: A, members: [ann, abe]}
  party-b: {name: B, members: [bea]}
sessions:
  - id: s1
    date: "2026-03-01T00:00:00Z"
    memories:
      - {id: old-captain, type: fact, content: Vane is captain., rel: [[vane, captain_of, watch]], acl: public}
      - {id: rumor, type: note, content: A ghost haunts the mill., rel: [[ghost, haunts, mill, rumor]], acl: public}
      - {id: secret, type: note, content: The abbot lives., rel: [[abbot, alive, null]], acl: [gm]}
      - {id: event, type: event, content: The bridge fell., rel: [[bridge, fell, null]], acl: public}
      - {id: event-a, type: event, content: A saw it cut., rel: [[figure, cut, bridge]], acl: [party-a], derived_from: event}
      - {id: event-b, type: event, content: B saw rot., rel: [[bridge, rotted, null, hypothesis]], acl: [party-b], derived_from: event}
  - id: s2
    date: "2026-03-08T00:00:00Z"
    memories:
      - {id: new-captain, type: fact, content: Marr is captain., rel: [[marr, captain_of, watch]], acl: public, supersedes: [old-captain]}
questions:
  - {id: q-truth, kind: truth, as: ann, question: "What haunts the mill?", answer: A ghost - rumor., truth: rumor, gold: [rumor]}
  - {id: q-stale, kind: staleness, as: bea, question: "Who is captain?", answer: Marr., gold: [new-captain], stale: [old-captain]}
  - {id: q-persp-a, kind: perspective, as: abe, question: "Why did the bridge fall?", answer: Cut., gold: [event-a]}
  - {id: q-persp-b, kind: perspective, as: bea, question: "Why did the bridge fall?", answer: Rot., gold: [event-b]}
  - {id: q-graph, kind: graph, as: gm, question: "Who has been captain?", answer: Vane then Marr., gold: [new-captain, rumor]}
  - {id: q-leak, kind: leakage, as: bea, question: "Is the abbot alive?", answer: Not known., probes: [secret, event-a]}
"""


@pytest.fixture
def small_path(tmp_path: Path) -> Path:
    path = tmp_path / "campaign.yaml"
    path.write_text(SMALL, encoding="utf-8")
    return path


@pytest.fixture
def small(small_path: Path) -> Campaign:
    return load_campaign(small_path)
