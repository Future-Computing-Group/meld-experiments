"""A-2 echo-safety probe: the gossip-loop invariant of SMP's wire discipline.

A node re-publishes only what it locally ingested, never what it RECEIVED (BrainNode marks
received claims published). So an alignment-tagged claim published once propagates outward
but never echoes back: no peer re-publishes it, and it never returns to the origin. The
probe builds an origin + ``n_peers``, propagates one tagged claim, and counts echoes
(invariant holds iff zero). Pure simulation (no LLM, no testbed).
"""
from __future__ import annotations

from typing import Dict

from smp.sync.node import BrainNode


def echo_safety_probe(*, encoder: str, nli: str, cfg, n_peers: int,
                      tagged_claim: str) -> Dict[str, object]:
    origin = BrainNode("origin", encoder=encoder, nli=nli, cfg=cfg)
    peers = [BrainNode(f"peer{i}", encoder=encoder, nli=nli, cfg=cfg)
             for i in range(n_peers)]

    origin.ingest(tagged_claim)
    delta = origin.extract_delta()                       # origin publishes the tagged claim once
    for p in peers:
        p.receive(delta)                                 # peers receive it over the wire

    # echo-safety: a peer must NOT re-publish what it received; the origin must not see it
    # come back (it already published it -> guarded).
    peer_echoes = sum(1 for p in peers if any(tagged_claim == t for t in p.extract_delta()))
    origin_echo = int(any(tagged_claim == t for t in origin.extract_delta()))
    return {"peer_echoes": peer_echoes, "origin_echo": origin_echo,
            "invariant_holds": peer_echoes == 0 and origin_echo == 0}


__all__ = ["echo_safety_probe"]
