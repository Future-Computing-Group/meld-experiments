"""Integration kernel: a BrainNode is one MELD replica (the five replicated sets plus the
admission path) on a networked node. ingest() publishes a signed local delta and admits it;
outbox() lists own objects to PUBLISH once and receive_object() is the receive gate;
live_claim_texts() reports the live claims (for the convergence/agreement metric); status
changes are authored, signed effects.
"""
from __future__ import annotations




def _node(nid):
    from meld.sync.node import BrainNode
    return BrainNode(nid, encoder="hashing", nli="lexical")


def _ship(src, dst):
    """PUBLISH: dst receives the signed bytes of every id on src's outbox."""
    for oid in src.outbox():
        assert dst.receive_object(*src.objects.raw(oid))


def test_node_receives_peer_delta_and_claim_becomes_live():
    a, b = _node("A"), _node("B")
    a.ingest("Paris is the capital of France")
    ids = a.outbox()
    assert ids                                     # non-empty: A has something to publish
    for oid in ids:
        assert b.receive_object(*a.objects.raw(oid))
    assert any("Paris" in t for t in b.live_claim_texts())   # B converged to include it


def test_outbox_does_not_republish(a_nid="A"):
    a = _node(a_nid)
    a.ingest("Paris is the capital of France")
    assert a.outbox()
    assert a.outbox() == []                        # already published -> empty


def test_two_nodes_converge_after_bidirectional_exchange():
    a, b = _node("A"), _node("B")
    a.ingest("Paris is the capital of France")
    b.ingest("Mount Everest is the tallest mountain")
    _ship(a, b)
    _ship(b, a)
    at, bt = a.live_claim_texts(), b.live_claim_texts()
    for texts in (at, bt):
        assert any("Paris" in t for t in texts)
        assert any("Everest" in t for t in texts)


def test_received_claims_are_not_re_published():
    # a node must not echo a peer's claim back as its own (gossip loop guard).
    a, b = _node("A"), _node("B")
    a.ingest("Paris is the capital of France")
    _ship(a, b)
    assert b.outbox() == []                        # B publishes nothing it only received


def test_reset_clears_brain_and_published_for_cell_isolation():
    # per-cell isolation: a campaign cell must start from a FRESH brain so seeds are
    # independent replicates and each RTT regime is actually re-measured (without reset,
    # publish-once means later cells publish nothing and the regime effect is invisible).
    a = _node("A")
    a.ingest("Paris is the capital of France")
    a.outbox()                                     # consume the outbox (publish-once)
    assert a.live_claim_texts()                    # has state
    a.reset()
    assert a.live_claim_texts() == set()           # brain cleared
    a.ingest("Mount Everest is the tallest mountain")
    assert a.outbox()                              # publishes again post-reset (fresh _published)
    assert any("Everest" in t for t in a.live_claim_texts())


def test_authored_overrule_enters_only_as_a_verified_published_effect():
    # design §7.3: a status change is a signed AuthoredEffect received through the MAC gate
    # (or authored locally through publish_local), never an unsigned local write.
    from meld.agent.memory import DEFAULT_CELL
    from meld.proto import claim_key
    from meld.protocol.objects import AuthoredEffect, attach_mac, canonical_bytes
    text = "Paris is the capital of France"
    a, b = _node("A"), _node("B")
    a.ingest(text)
    assert a.status_of(text) == "ACTIVE"
    ovr = AuthoredEffect(scope="agent", kind="OVERRULE", c1=claim_key(text, DEFAULT_CELL), c2=b"",
                         target_cell=DEFAULT_CELL, author="B", author_cell=DEFAULT_CELL,
                         author_auth=0, created_at=1)
    forged = attach_mac(ovr, b"not-the-group-key")
    assert not a.receive_object(canonical_bytes(forged), forged.mac, "agent")
    assert a.status_of(text) == "ACTIVE" and a.objects.ids("A", "agent") == ()
    oid = b.publish_local(ovr)                          # B authors and signs it
    assert a.receive_object(*b.objects.raw(oid))
    assert a.status_of(text) == "OVERRULED"
    assert not hasattr(a, "_stale_links")               # no caller-supplied stale set remains


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


def test_node_admits_through_the_protocol_step_and_publishes_its_sig():
    # T6b wiring: the default receive hook is protocol/admission.py under the node's epoch;
    # every Sig direction the step evaluated is an own object on the outbox.
    from meld.sync.node import BrainNode
    from test_agent_meld_memory import agent_epoch
    a = BrainNode("A", encoder="hashing", nli="lexical", epoch=agent_epoch())
    b = BrainNode("B", encoder="hashing", nli="lexical", epoch=agent_epoch())
    a.ingest("the valve on line two is open")
    a.ingest("the valve on line two is open now")
    assert [d.decision for d in a.mem.decisions] == ["INSERT", "MERGE"]
    out = a.outbox()
    assert any(a.objects.kind_of(i) == "Sig" for i in out)
    for oid in out:                                      # arrival order is id order, not canonical
        assert b.receive_object(*a.objects.raw(oid))
    assert b.mem.admission.decisions("agent") == a.mem.admission.decisions("agent")
    assert b.live_claim_texts() == a.live_claim_texts() == {"the valve on line two is open"}
    bare = BrainNode("C", encoder="hashing", nli="lexical")         # no epoch: no similarity merge
    bare.ingest("the valve on line two is open")
    bare.ingest("the valve on line two is open now")
    assert [d.decision for d in bare.mem.decisions] == ["INSERT", "RELATE"]
