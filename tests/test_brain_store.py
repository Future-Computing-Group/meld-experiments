"""Task 1.8 — the brain store (a memory-store replica of a world-model graph).

A brain holds Concepts (units), Claims, and Mappings (links). It applies a
Patch's emitted deltas to mutate state; a raw delta applied directly is a no-op
(only a Patch mutates). It can extract a delta for PUBLISH (the version set) and
do a graph-walk read (lex + dense + graph fusion, Def. 2/3 -- REUSE-PENDING).

REUSE: the memory-store persistence + claim/link model. NEW (thin): delta
apply/extract + graph-walk read.
"""
from __future__ import annotations

from meld.brain.store import Brain
from meld.merge.status_crdt import StatusState
from meld.proto import Claim, Endpoint, MemoryLink, MemoryUnit
from meld.protocol.messages import Patch


def _patch_merge(unit):
    return Patch(decision="MERGE", target=Endpoint(node_id=unit.id),
                 emitted=[unit], gates_that_fired=["kappa"], version=1, signature=b"s")


def test_apply_patch_mutates_state():
    b = Brain()
    u = MemoryUnit(id=b"u1", claims=[Claim(claim_key=b"k1", text="Paris is in France")])
    assert not b.changed
    b.apply(_patch_merge(u))
    assert b.changed
    assert b.get_unit(b"u1") is not None


def test_raw_delta_does_not_mutate():
    b = Brain()
    u = MemoryUnit(id=b"u2", claims=[Claim(claim_key=b"k2", text="x")])
    b.receive_raw_delta([u])             # raw delta: no Patch -> no mutation
    assert not b.changed
    assert b.get_unit(b"u2") is None


def test_extract_delta_returns_version_set():
    b = Brain()
    u = MemoryUnit(id=b"u3", claims=[Claim(claim_key=b"k3", text="y")])
    b.apply(_patch_merge(u))
    delta = b.extract_delta()
    assert any(getattr(x, "id", None) == b"u3" for x in delta)


def test_reject_patch_is_recorded_noop():
    b = Brain()
    u = MemoryUnit(id=b"u4", claims=[Claim(claim_key=b"k4", text="z")])
    p = Patch(decision="REJECT", target=Endpoint(node_id=u.id), emitted=[u],
              gates_that_fired=["stale"], version=1, signature=b"s")
    b.apply(p)
    assert b.get_unit(b"u4") is None     # REJECT never adds the unit
    assert len(b.audit_log) == 1         # but it is recorded for audit


def test_relate_emits_mapping_link():
    b = Brain()
    ml = MemoryLink(type="ASSOCIATES", version=1, weight=0.6, signature=b"s")
    p = Patch(decision="RELATE", target=Endpoint(node_id=b"u5"), emitted=[ml],
              gates_that_fired=["sigma"], version=1, signature=b"s")
    b.apply(p)
    assert any(isinstance(x, MemoryLink) for x in b.links)


def test_graph_walk_read_fuses_channels():
    b = Brain()
    u = MemoryUnit(id=b"u6", claims=[Claim(claim_key=b"k6", text="the capital of France is Paris")])
    b.apply(_patch_merge(u))
    hits = b.graph_walk_read("France capital", encoder="hashing", k=5)
    assert len(hits) >= 1
    assert hits[0][0] == b"u6"           # (unit_id, score)


def test_status_state_per_claim():
    b = Brain()
    u = MemoryUnit(id=b"u7", claims=[Claim(claim_key=b"k7", text="t")])
    b.apply(_patch_merge(u))
    assert isinstance(b.claim_status(b"k7"), StatusState)


def test_graph_walk_read_excludes_deprecated_units():
    # recall must respect the status CRDT: a unit whose every claim has been
    # superseded (DEPRECATED) is no longer live and must NOT be returned. A
    # status-blind read that still surfaces a deprecated duplicate is the defect
    # that made distributed_merge identical to naive_union at recall.
    b = Brain()
    live = MemoryUnit(id=b"live", claims=[Claim(claim_key=b"kk",
                      text="the capital of France is Paris")])
    dead = MemoryUnit(id=b"dead", claims=[Claim(claim_key=b"kd",
                      text="Paris is the capital city of France")])
    b.apply(_patch_merge(live))
    b.apply(_patch_merge(dead))
    b.apply(Patch(decision="MERGE", target=Endpoint(node_id=b"dead"),
                  emitted=[("STATUS", b"kd", ("dead", "DEPRECATED"))],
                  gates_that_fired=["sigma"], version=1, signature=b"s"))
    ids = [h[0] for h in b.graph_walk_read("France capital Paris",
                                           encoder="hashing", k=5)]
    assert b"live" in ids
    assert b"dead" not in ids


def test_paraphrase_duplicate_is_deduped_at_recall_after_merge():
    # end-to-end: a canonical claim, then a paraphrase fed through
    # merge_on_receive. The sigma-MERGE deprecates the paraphrase, so recall
    # returns ONE unit (the canonical), not two -- the property that makes
    # distributed_merge differ from naive_union.
    from meld.config import load_thresholds
    from meld.merge.signals import EmbeddingSimilarity
    from meld.protocol.merge_on_receive import merge_on_receive
    b = Brain()
    canon = MemoryUnit(id=b"canon", claims=[Claim(claim_key=b"kc",
                       text="Paris is the capital of France")])
    b.apply(_patch_merge(canon))
    para = MemoryUnit(id=b"para", claims=[Claim(claim_key=b"kp",
                      text="Paris is the capital city of France")])
    for patch in merge_on_receive(para, [canon], cfg=load_thresholds(),
                                  es=EmbeddingSimilarity(encoder="hashing"),
                                  nli="lexical"):
        b.apply(patch)
    ids = [h[0] for h in b.graph_walk_read("capital of France Paris",
                                           encoder="hashing", k=5)]
    assert b"canon" in ids
    assert b"para" not in ids
