from pathlib import Path

import pytest

from tests.conftest import SMALL
from westmarches_eval.campaign import QUESTION_KINDS, CampaignError, load_campaign


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "edited.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_checked_in_campaign_is_consistent():
    campaign = load_campaign()
    assert len(campaign.parties) == 3
    assert {q.kind for q in campaign.questions} == set(QUESTION_KINDS)
    # The GM sees every memory; each player misses at least the GM's secrets.
    assert all(campaign.visible(campaign.gm, m.id) for m in campaign.memories)
    for player in campaign.players:
        assert not all(campaign.visible(player, m.id) for m in campaign.memories)


def test_visibility_follows_acl_and_parties(small):
    assert small.visible("bea", "rumor")
    assert small.visible("ann", "event-a") and small.visible("abe", "event-a")
    assert not small.visible("bea", "event-a")
    assert not small.visible("ann", "secret")
    assert small.visible("gm", "secret") and small.visible("gm", "event-b")


def test_restricted_acl_gets_the_gm(small):
    assert small.memory("event-a").acl == ("party-a", "gm")
    assert small.memory("rumor").acl == ()


def test_derived_and_superseded(small):
    assert small.facet_family("event-b") == ("event", "event-a", "event-b")
    assert small.superseded() == {"old-captain"}
    assert small.memory("new-captain").occurred_at == "2026-03-08T00:00:00Z"
    assert small.memory("event-b").relationships[0].kind == "hypothesis"
    assert small.memory("rumor").entities == ["ghost", "mill"]


def test_sha256_tracks_the_file(small_path, tmp_path):
    first = load_campaign(small_path).sha256
    assert load_campaign(write(tmp_path, SMALL + "\n# edited\n")).sha256 != first


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        ("acl: [party-a], derived_from: event}", "acl: [party-a], derived_from: nowhere}", "not an earlier memory"),
        ("supersedes: [old-captain]", "supersedes: [secret-later]", "not an earlier memory"),
        ("acl: [gm]}", "acl: [stranger]}", "unknown ids"),
        ("acl: public, supersedes", "acl: none, supersedes", "acl must be"),
        ("gold: [new-captain], stale: [old-captain]", "gold: [new-captain], stale: [rumor]", "never superseded"),
        ("gold: [new-captain], stale: [old-captain]", "gold: [old-captain], stale: [old-captain]", "are superseded"),
        ("as: ann, question: \"What haunts", "as: nobody, question: \"What haunts", "not the GM or a player"),
        ("probes: [secret, event-a]", "probes: [secret, rumor]", "can see the memories it probes"),
        ("gold: [event-b]}", "gold: [event-a]}", "can't see its gold"),
        ("answer: Cut., gold: [event-a]", "answer: Cut., gold: [event]", "the viewer's one facet"),
        ("truth: rumor, gold", "truth: legend, gold", "truth 'legend'"),
        ("[ghost, haunts, mill, rumor]", "[ghost, haunts, mill, legend]", "relationship kind"),
    ],
)
def test_inconsistent_campaigns_are_refused(tmp_path, old, new, error):
    assert old in SMALL
    with pytest.raises(CampaignError, match=error):
        load_campaign(write(tmp_path, SMALL.replace(old, new, 1)))
