"""In-process multi-replica driver over the fold (design §7.1 harnesses, §7.2 H1-H2; Props 1, 3).

Replicas are BrainNodes, each with its own signal provider (``node_factory`` builds them with
``nli_provider(nli, own=True)``; ``settle`` refuses replicas that share a provider or a model
instance).

  * ``sig_exchange(nodes, scope)`` -- every replica receives, through its receive gate, every
                     Sig record of the scope another replica holds.
  * ``settle(nodes, scope)``  -- measures the five components (eval/graph_fingerprint.py) of
                     every replica BEFORE any pass (the window: admission state as delivered;
                     a replica's window lists the derived components, reps, links, decisions,
                     that differ from the post-pass mode), then runs Sig exchanges and passes
                     (fold.readjudicate on each replica's own provider) until a pass misses
                     nothing, at most three, and measures the post-pass divergence, the PASS
                     certificate fingerprints, the pass-time misses per pass, L = the number of
                     passes that missed (Prop 3 says L <= 1; a 3 shows a violation), the source
                     counter ``not_last`` per replica, the per-replica views (hex) and group
                     purity against ``truth`` when the workload has it.
  * ``run_sync(nodes, peers, ...)`` -- the eval-s / eval-r round: each node PUBLISHes the
                     signed bytes on its outbox (deltas routed semantically or by broadcast,
                     every other object to every peer) over PatchTransport with a skew-free
                     SyncLatencyMeter, then ``settle`` per shared scope. Returns the latency
                     summary, the delivered peer set per delta text (scored by
                     router.routing_metrics against a needed-oracle), presence_agreement (mean
                     share of the union of live claim texts each node holds, after the pass),
                     the per-node live texts and the convergence report.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Set

from meld.eval.graph_fingerprint import COMPONENTS, WINDOW, components, divergence
from meld.protocol.fold import nli_provider, readjudicate
from meld.sync.latency import SyncLatencyMeter
from meld.sync.node import BrainNode
from meld.sync.router import broadcast_route, semantic_route
from meld.sync.transport import PatchTransport


def node_factory(*, encoder: str, nli: str, pol: Optional[dict] = None, epoch=None,
                 name: Callable[[int], str] = "r{}".format) -> Callable[[int], BrainNode]:
    """i -> a fresh BrainNode with its own model instance."""
    return lambda i: BrainNode(name(i), encoder=encoder, nli=nli, pol=pol, epoch=epoch,
                               provider=nli_provider(nli, own=True))


def _check_own_models(nodes) -> None:
    provs = [n.mem.admission.provider for n in nodes]
    models = [getattr(p, "model", None) for p in provs]
    held = [id(m) for m in models if m is not None]
    if (len(set(map(id, provs))) < len(provs) or len(set(held)) < len(held)
            or any(m is None and n.mem.nli != "lexical" for n, m in zip(nodes, models))):
        raise ValueError("every replica needs its own provider and model instance")


def sig_exchange(nodes, scope: str) -> int:
    """All-to-all exchange of the scope's Sig records; returns the records newly accepted."""
    pool = {i: n.objects.raw(i) for n in nodes for i in n.objects.ids("Sig", scope)}
    moved = 0
    for n in nodes:
        held = set(n.objects.ids("Sig", scope))
        moved += sum(n.receive_object(*raw) for i, raw in pool.items() if i not in held)
    return moved


def settle(nodes, scope: str, *, truth: Optional[Callable[[bytes, bytes], str]] = None) -> Dict:
    nodes = list(nodes)
    _check_own_models(nodes)
    pre = [components(n.mem.store, scope, n.mem.admission.decisions(scope)) for n in nodes]
    passes, moved = [], []
    for _ in range(3):          # Prop 3: L <= 1; a third pass makes an L >= 2 violation visible
        moved.append(sig_exchange(nodes, scope))
        res = [readjudicate(n.mem.store, scope, m_nli=n.mem.m_nli, provider=n.mem.admission.provider,
                            issuer=n.mem.origin, hw=n.mem.admission.hw, truth=truth) for n in nodes]
        passes.append([r.counters["pass_time_misses"] for r in res])
        if not any(passes[-1]):
            break
    post = [components(n.mem.store, scope, r.decisions) for n, r in zip(nodes, res)]
    modal = {c: Counter(v[c] for v in post).most_common(1)[0][0] for c in COMPONENTS}
    # purity is read from replica 0 (res[0]): valid once the replicas have converged
    purity = None
    if truth is not None:
        purity = {k: res[0].counters[k] for k in
                  ("group_pairs", "impure_pairs", "in_group_contradictions", "undetected_outside_k")}
        purity["purity"] = 1.0 - purity["impure_pairs"] / purity["group_pairs"] if purity["group_pairs"] else 1.0
    return {"pre_pass": divergence(pre), "post_pass": divergence(post),
            "window": [[c for c in WINDOW if v[c] != modal[c]] for v in pre],
            "pre_views": [{c: v[c].hex() for c in COMPONENTS} for v in pre],
            "post_views": [{c: v[c].hex() for c in COMPONENTS} for v in post],
            "not_last": [n.mem.admission.counters["not_last"] for n in nodes],
            "sig_moved": moved, "pass_misses": passes, "L": sum(map(any, passes)),
            "fingerprints": [r.certificate.fingerprint.hex() for r in res],
            "decisions": [tuple(d.decision for d in r.decisions) for r in res],
            "n_prior": [tuple(d.n_prior for d in r.decisions) for r in res],
            "purity": purity}


@dataclass
class SyncRunResult:
    latency: dict
    delivered: Dict[str, Set[str]]
    presence_agreement: float
    node_live: Dict[str, Set[str]]
    convergence: Dict[str, Dict]
    deliveries: List[tuple] = field(default_factory=list)   # (src, delta text, delivered peers)


def _targets(claim: str, candidates: List[str], nodes: Dict[str, BrainNode], *,
             routing_mode: str, encoder: str, sigma_lo: float,
             top_k: Optional[int], margin: Optional[float] = None, gap: bool = False,
             es=None) -> Set[str]:
    brains = {p: list(nodes[p].live_claim_texts()) for p in candidates}
    if routing_mode == "semantic":
        return semantic_route(claim, brains, encoder=encoder,
                              sigma_lo=sigma_lo, top_k=top_k, margin=margin, gap=gap, es=es)
    if routing_mode == "broadcast":
        return broadcast_route(brains)
    raise ValueError(f"unknown routing_mode: {routing_mode!r}")   # no silent default


def run_sync(nodes: Dict[str, BrainNode], peers: Dict[str, List[str]], *,
             routing_mode: str, encoder: str, sigma_lo: float = 0.0,
             top_k: Optional[int] = None,
             clock: Optional[Callable[[], float]] = None,
             margin: Optional[float] = None, gap: bool = False, es=None) -> SyncRunResult:
    """``margin`` / ``gap``: budget-free semantic cuts (router.select_peers). ``es``: a shared
    EmbeddingSimilarity cache for the router (decisions unchanged, encodings reused)."""
    meter = SyncLatencyMeter(clock=clock or time.monotonic)
    delivered: Dict[str, Set[str]] = {}
    deliveries: List[tuple] = []

    # snapshot every outbox BEFORE any delivery so within-round receives do not cascade
    round_ids = {nid: node.outbox() for nid, node in nodes.items()}

    for src, ids in round_ids.items():
        for oid in ids:
            raw = nodes[src].objects.raw(oid)
            delta = nodes[src].objects.get("D", oid)
            targets = (_targets(delta.text, peers.get(src, []), nodes, routing_mode=routing_mode,
                                encoder=encoder, sigma_lo=sigma_lo, top_k=top_k,
                                margin=margin, gap=gap, es=es)
                       if delta is not None else set(peers.get(src, [])))

            def send(peer_id, _payload, _raw=raw):
                return SimpleNamespace(status_code=200 if nodes[peer_id].receive_object(*_raw) else 403)

            d = PatchTransport(send, latency_meter=meter).publish(f"{src}:{oid.hex()}", raw[0],
                                                                  sorted(targets))
            if delta is not None:
                delivered[delta.text] = {p for p, ok in d.items() if ok}
                deliveries.append((src, delta.text, delivered[delta.text]))

    scopes = frozenset.intersection(*(n.objects.scopes for n in nodes.values())) if nodes else ()
    convergence = {s: settle(nodes.values(), s) for s in sorted(scopes)}
    live = {nid: node.live_claim_texts() for nid, node in nodes.items()}
    union: Set[str] = set().union(*live.values()) if live else set()
    agreement = (sum(len(s & union) for s in live.values()) / (len(live) * len(union))
                 if union else 1.0)
    return SyncRunResult(latency=meter.summary(), delivered=delivered,
                         presence_agreement=agreement, node_live=live, convergence=convergence,
                         deliveries=deliveries)


__all__ = ["node_factory", "sig_exchange", "settle", "run_sync", "SyncRunResult"]
