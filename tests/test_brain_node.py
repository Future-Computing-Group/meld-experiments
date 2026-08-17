"""Integration kernel: a BrainNode ties the MELD brain + merge-on-receive into a
networked node. ingest() writes a local observation; extract_delta() yields the
claims to publish (new since last publish); receive() merges a peer's delta into the
local brain via merge-on-receive; live_claim_texts() reports the status-live claims
(for the convergence/agreement metric). This is the unit the orchestration deploys
per node; the transport (PatchTransport) carries extract_delta() output between nodes.
"""
from __future__ import annotations


def _cfg():
    from meld.agent.run_hotpot_gate import _calibrated_cfg
    return _calibrated_cfg("hashing")


def _node(nid):
    from meld.sync.node import BrainNode
    return BrainNode(nid, encoder="hashing", nli="lexical", cfg=_cfg())


def test_node_receives_peer_delta_and_claim_becomes_live():
    a, b = _node("A"), _node("B")
    a.ingest("Paris is the capital of France")
    delta = a.extract_delta()
    assert delta                                   # non-empty: A has something to publish
    b.receive(delta)
    assert any("Paris" in t for t in b.live_claim_texts())   # B converged to include it


def test_extract_delta_does_not_republish(a_nid="A"):
    a = _node(a_nid)
    a.ingest("Paris is the capital of France")
    first = a.extract_delta()
    assert first
    assert a.extract_delta() == []                 # already published -> empty delta


def test_two_nodes_converge_after_bidirectional_exchange():
    a, b = _node("A"), _node("B")
    a.ingest("Paris is the capital of France")
    b.ingest("Mount Everest is the tallest mountain")
    da, db = a.extract_delta(), b.extract_delta()
    b.receive(da)
    a.receive(db)
    at, bt = a.live_claim_texts(), b.live_claim_texts()
    for texts in (at, bt):
        assert any("Paris" in t for t in texts)
        assert any("Everest" in t for t in texts)


def test_received_claims_are_not_re_published():
    # a node must not echo a peer's claim back as its own delta (gossip loop guard).
    a, b = _node("A"), _node("B")
    a.ingest("Paris is the capital of France")
    b.receive(a.extract_delta())
    assert b.extract_delta() == []                 # B publishes nothing it only received


def test_reset_clears_brain_and_published_for_cell_isolation():
    # per-cell isolation: a campaign cell must start from a FRESH brain so seeds are
    # independent replicates and each RTT regime is actually re-measured (without reset,
    # publish-once means later cells publish nothing and the regime effect is invisible).
    a = _node("A")
    a.ingest("Paris is the capital of France")
    a.extract_delta()                              # consume the delta (publish-once)
    assert a.live_claim_texts()                    # has state
    a.reset()
    assert a.live_claim_texts() == set()           # brain cleared
    a.ingest("Mount Everest is the tallest mountain")
    assert a.extract_delta()                       # publishes again post-reset (fresh _published)
    assert any("Everest" in t for t in a.live_claim_texts())


def test_apply_status_overrules_a_claim():
    a = _node("A")
    a.ingest("Paris is the capital of France")
    assert a.status_of("Paris is the capital of France") == "ACTIVE"
    a.apply_status("Paris is the capital of France", link_id="L1", effect="OVERRULED")
    assert a.status_of("Paris is the capital of France") == "OVERRULED"


def test_status_crdt_converges_order_independently_across_nodes():
    # the eval-c core: two nodes apply the SAME two overrule links in DIFFERENT orders
    # (as a partition would reorder delivery) -> identical effective status (lattice join).
    a, b = _node("A"), _node("B")
    for n in (a, b):
        n.ingest("the policy is X")
    a.apply_status("the policy is X", link_id="L1", effect="DEPRECATED")
    a.apply_status("the policy is X", link_id="L2", effect="OVERRULED")
    b.apply_status("the policy is X", link_id="L2", effect="OVERRULED")
    b.apply_status("the policy is X", link_id="L1", effect="DEPRECATED")
    assert a.status_of("the policy is X") == b.status_of("the policy is X") == "OVERRULED"


def test_full_state_returns_all_live_claims_for_anti_entropy():
    # full_state() is the anti-entropy payload: the ENTIRE live set, independent of the
    # publish-once delta guard, so a peer that missed claims (e.g. during a partition) can
    # be brought up to date after heal.
    a = _node("A")
    a.ingest("Paris is the capital of France")
    a.ingest("Mount Everest is the tallest mountain")
    a.extract_delta()                              # publish-once consumes the delta...
    fs = a.full_state()                            # ...but full_state still returns everything
    assert len(fs) == 2
    assert any("Paris" in t for t in fs) and any("Everest" in t for t in fs)
