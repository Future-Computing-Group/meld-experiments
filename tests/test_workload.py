"""Campaign workload: assign topic-labelled claims to brains (topic-skewed, with
controlled replication) and derive the eval-r needed-oracle. The needed-set of a claim
is the set of brains that HOME its topic -- an INDEPENDENT ground truth (topic labels,
not the router's own sigma), so eval-r precision/recall is not circular. Deterministic
under seed. The HotpotQA load is datasets-gated elsewhere; this is the pure logic.
"""
from __future__ import annotations


CLAIMS = [("Paris is the capital of France", "geo"),
          ("The Seine flows through Paris", "geo"),
          ("Mitochondria produce ATP", "bio"),
          ("DNA encodes proteins", "bio"),
          ("Beethoven wrote nine symphonies", "music")]


def test_replication_1_homes_each_topic_on_one_brain():
    from smp.sync.workload import build_workload
    w = build_workload(CLAIMS, ["A", "B", "C"], replication=1, seed=1)
    for topic, homes in w.topic_home.items():
        assert len(homes) == 1                       # one home per topic
    # every claim ingested by exactly its topic's home brain
    for text, topic in CLAIMS:
        homes = w.topic_home[topic]
        for b in ["A", "B", "C"]:
            assert (text in w.assignments[b]) == (b in homes)


def test_replication_2_homes_each_topic_on_two_brains():
    from smp.sync.workload import build_workload
    w = build_workload(CLAIMS, ["A", "B", "C"], replication=2, seed=1)
    for homes in w.topic_home.values():
        assert len(homes) == 2
    # a geo claim is ingested by both geo-home brains
    geo_homes = set(w.topic_home["geo"])
    for b in geo_homes:
        assert "Paris is the capital of France" in w.assignments[b]


def test_needed_oracle_is_the_topic_home_set():
    from smp.sync.workload import build_workload
    w = build_workload(CLAIMS, ["A", "B", "C", "D"], replication=2, seed=3)
    for text, topic in CLAIMS:
        assert w.needed[text] == frozenset(w.topic_home[topic])   # independent of any router


def test_deterministic_under_seed():
    from smp.sync.workload import build_workload
    a = build_workload(CLAIMS, ["A", "B", "C"], replication=1, seed=7)
    b = build_workload(CLAIMS, ["A", "B", "C"], replication=1, seed=7)
    assert a.topic_home == b.topic_home and a.assignments == b.assignments
    c = build_workload(CLAIMS, ["A", "B", "C"], replication=1, seed=8)
    # a different seed generally permutes the homing (not asserting strict inequality,
    # but the function must accept distinct seeds and stay self-consistent)
    for text, topic in CLAIMS:
        assert c.needed[text] == frozenset(c.topic_home[topic])
