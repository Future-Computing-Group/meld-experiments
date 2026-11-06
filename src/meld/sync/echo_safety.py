"""A-2 echo-safety probe: the gossip-loop invariant of MELD's wire discipline.

A node PUBLISHes only its own objects, never what it RECEIVED (BrainNode marks received ids
published). So an alignment-tagged claim published once propagates outward but never echoes
back: no peer's outbox carries it, and it never returns to the origin's. The probe builds an
origin + ``n_peers``, propagates one tagged claim, and counts echoes (invariant holds iff zero).
Pure simulation (no LLM, no testbed).
"""
from __future__ import annotations

from typing import Dict

from meld.sync.node import BrainNode


def echo_safety_probe(*, encoder: str, nli: str, n_peers: int,
                      tagged_claim: str) -> Dict[str, object]:
    origin = BrainNode("origin", encoder=encoder, nli=nli)
    peers = [BrainNode(f"peer{i}", encoder=encoder, nli=nli)
             for i in range(n_peers)]

    origin.ingest(tagged_claim)
    out = origin.outbox()                                # origin publishes the tagged claim once
    for p in peers:
        for oid in out:
            p.receive_object(*origin.objects.raw(oid))   # peers receive it over the wire
    tagged = {i for i in out if getattr(origin.objects.get("D", i), "text", None) == tagged_claim}

    # echo-safety: a peer must NOT re-publish what it received; the origin must not see it
    # come back (it already published it -> guarded).
    peer_echoes = sum(1 for p in peers if tagged & set(p.outbox()))
    origin_echo = int(bool(tagged & set(origin.outbox())))
    return {"peer_echoes": peer_echoes, "origin_echo": origin_echo,
            "invariant_holds": bool(tagged) and peer_echoes == 0 and origin_echo == 0}


__all__ = ["echo_safety_probe"]
