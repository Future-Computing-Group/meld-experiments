"""A-2 echo-safety probe.

MELD's wire discipline holds an echo-safety invariant: a claim a node RECEIVED over the
wire is never re-published (the gossip-loop guard in BrainNode), so an alignment-tagged
edge published once never echoes/amplifies back around the topology. The probe injects a
tagged claim at an origin, propagates it to peers, and checks no peer (and not the origin)
re-publishes it. Contrast: a claim a node INGESTS locally IS published -- so the guard,
not an inability to publish, is what blocks the echo. Pure simulation (no LLM, no testbed).
"""
from __future__ import annotations




def test_received_alignment_tagged_claim_never_echoes():
    from meld.sync.echo_safety import echo_safety_probe
    r = echo_safety_probe(encoder="hashing", nli="lexical", n_peers=3,
                          tagged_claim="ALIGN-TAG: the canonical policy is X")
    assert r["peer_echoes"] == 0          # no peer re-publishes what it received
    assert r["origin_echo"] == 0          # it never comes back to the origin
    assert r["invariant_holds"] is True


def test_guard_not_inability_blocks_the_echo():
    # control: a node that INGESTS the claim locally DOES publish it -> the no-echo property
    # is the receive-side gossip guard, not a general inability to publish the text.
    from meld.sync.node import BrainNode
    n = BrainNode("X", encoder="hashing", nli="lexical")
    n.ingest("ALIGN-TAG: the canonical policy is X")
    assert any("ALIGN-TAG" in n.objects.get("D", i).text for i in n.outbox())   # ingested -> published
