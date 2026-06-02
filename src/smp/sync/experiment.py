"""In-process sync experiment driver (§7-s latency / §7-r routing).

Composes the sync primitives into one runnable round so the experiment LOGIC is
verifiable end-to-end before any docker/edge-testbed deployment. Each node publishes its local
delta (BrainNode.extract_delta) to its peers, routed semantically (smp.sync.router) or
by broadcast, over PatchTransport with a skew-free SyncLatencyMeter. Deltas are
snapshotted at round start so within-round receives do not cascade. Returns:

  * latency            -- SyncLatencyMeter.summary() (eval-s)
  * delivered          -- claim_text -> set of peer node_ids actually delivered to,
                          for eval-r routing_metrics against an external needed-oracle
  * presence_agreement -- mean over nodes of |node.live & union| / |union| (full
                          propagation == 1.0); a PRESENCE metric, distinct from the
                          status-CRDT agreement owned by eval.crdt_convergence.
  * node_live          -- per-node final live claim texts.

The transport's send seam is wired to peer BrainNode.receive() in-process, so the 2xx
delivery contract + latency integration are exercised exactly as they will be on the wire.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Set

from smp.sync.latency import SyncLatencyMeter
from smp.sync.node import BrainNode
from smp.sync.router import broadcast_route, semantic_route
from smp.sync.transport import PatchTransport


@dataclass
class SyncRunResult:
    latency: dict
    delivered: Dict[str, Set[str]]
    presence_agreement: float
    node_live: Dict[str, Set[str]]


def _targets(claim: str, candidates: List[str], nodes: Dict[str, BrainNode], *,
             routing_mode: str, encoder: str, sigma_lo: float,
             top_k: Optional[int]) -> Set[str]:
    brains = {p: list(nodes[p].live_claim_texts()) for p in candidates}
    if routing_mode == "semantic":
        return semantic_route(claim, brains, encoder=encoder,
                              sigma_lo=sigma_lo, top_k=top_k)
    if routing_mode == "broadcast":
        return broadcast_route(brains)
    raise ValueError(f"unknown routing_mode: {routing_mode!r}")   # no silent default


def run_sync(nodes: Dict[str, BrainNode], peers: Dict[str, List[str]], *,
             routing_mode: str, encoder: str, sigma_lo: float = 0.0,
             top_k: Optional[int] = None,
             clock: Optional[Callable[[], float]] = None) -> SyncRunResult:
    meter = SyncLatencyMeter(clock=clock or time.monotonic)
    delivered: Dict[str, Set[str]] = {}

    # snapshot every node's delta BEFORE any delivery so within-round receives
    # do not cascade into this round's published set.
    round_deltas = {nid: node.extract_delta() for nid, node in nodes.items()}

    for src, claims in round_deltas.items():
        candidates = peers.get(src, [])
        for claim in claims:
            targets = _targets(claim, candidates, nodes, routing_mode=routing_mode,
                               encoder=encoder, sigma_lo=sigma_lo, top_k=top_k)

            def send(peer_id, _payload, _claim=claim):
                nodes[peer_id].receive([_claim])
                return SimpleNamespace(status_code=200)

            transport = PatchTransport(send, latency_meter=meter)
            patch_id = f"{src}:{claim}"
            d = transport.publish(patch_id, claim.encode("utf-8"), sorted(targets))
            delivered[claim] = {p for p, ok in d.items() if ok}

    live = {nid: node.live_claim_texts() for nid, node in nodes.items()}
    union: Set[str] = set().union(*live.values()) if live else set()
    if union and live:
        agreement = sum(len(s & union) for s in live.values()) / (len(live) * len(union))
    else:
        agreement = 1.0
    return SyncRunResult(latency=meter.summary(), delivered=delivered,
                         presence_agreement=agreement, node_live=live)


__all__ = ["run_sync", "SyncRunResult"]
