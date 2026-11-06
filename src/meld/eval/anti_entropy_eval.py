"""Lossy, reordered multi-replica convergence over the five replicated sets (design §2.2, §7.2 H1-H2;
Props 1, 3; the freeze-gate lossy harness of §7.4).

``lossy_cell``: N replicas (``make_node(i)``, each with its own provider) receive a workload of
signed objects, each replica in its own seeded order and each delivery dropped independently
with probability ``loss``. With ``effects_last`` every replica first receives the deltas in
canonical order (the latency path, no refold over a partial node set), then the authored effects
in its own order: an A_c effect then changes tested pairs no admission read, and the first pass
misses (L = 1) on every replica that lost no delta.
Optionally, anti-entropy rounds (DIGEST / PULL / PUBLISH,
sync.anti_entropy.objectset_round, over each peer's BrainService) run in a ring until every
replica holds the same five sets; every request and reply byte is metered per message class
(/digest, /pull, /patch). Then ``sync.experiment.settle``: the pre-pass divergence, a
Sig exchange, a pass (a second of each if a pass missed), the post-pass divergence on the five
components and group purity when ``roles`` give ground truth. The divergence is also measured
right after the lossy delivery (``delivered``), and with ``probe_queries`` the number of distinct
top-k reads across replicas per query (``topk_distinct``, 1 when the replicas read alike).

``with_effects`` adds authored effects to the gold workload (every ``every``-th seed: a
DISTINGUISHES between c and p, an OVERRULE of x), so A and the A_c refold are exercised.
``loss_sweep`` runs cells without anti-entropy, with it, and with it in the effects-last arm,
over losses x seeds.

Note: the no-anti-entropy arm still runs ``settle``'s lossless all-to-all Sig exchange, so its
authored metric is unaffected by the missing rounds but its Sig set is not lossy there.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Sequence

from meld.eval.graph_fingerprint import components, divergence
from meld.eval.order_shuffle import topk, truth
from meld.protocol.objects import AuthoredEffect, Delta, scope_partition, signed
from meld.sync.anti_entropy import objectset_round, scope_digest
from meld.sync.experiment import settle
from meld.sync.service import BrainService


def with_effects(deltas, roles, pol: dict, every: int = 4) -> list:
    by = {roles[d.claim_key]: d for d in deltas}
    out = list(deltas)
    for i, sid in enumerate(sorted({s for s, _ in roles.values()})):
        if i % every:
            continue
        c, p, x = by[sid, "c"], by[sid, "p"], by[sid, "x"]
        e = dict(scope=scope_partition(pol)[c.cell], target_cell=c.cell, author="auditor",
                 author_cell=c.cell, author_auth=0, created_at=0)
        out += [AuthoredEffect(kind="DISTINGUISHES", c1=min(c.claim_key, p.claim_key),
                               c2=max(c.claim_key, p.claim_key), **e),
                AuthoredEffect(kind="OVERRULE", c1=x.claim_key, c2=b"", **e)]
    return out


def _roots(node, scope: str):
    return tuple(sorted((k, d.set_hash) for k, d in scope_digest(node.objects, scope).items()))


def reconcile(nodes, scope: str, max_rounds: int = 10):
    """Ring anti-entropy rounds until every replica holds the same sets; (rounds, objects moved,
    request + reply bytes per path). ``rounds == max_rounds`` without agreement shows in the
    authored counter."""
    svcs = [BrainService(n) for n in nodes]
    moved, nbytes = 0, Counter({"/digest": 0, "/pull": 0, "/patch": 0})

    def metered(peer):
        def call(path, body):
            status, reply = peer.handle("POST", path, body)
            nbytes[path] += len(body) + len(reply)
            return status, reply
        return call
    for r in range(1, max_rounds + 1):
        for i, n in enumerate(nodes):
            p, q, _ = objectset_round(n.objects, scope, metered(svcs[(i + 1) % len(nodes)]), n.receive_object)
            moved += p + q
        if len({_roots(n, scope) for n in nodes}) == 1:
            return r, moved, dict(nbytes)
    return max_rounds, moved, dict(nbytes)


def lossy_cell(workload, *, make_node, n: int, loss: float, seed: int, scope: str,
               anti_entropy: bool = True, effects_last: bool = False, roles=None,
               probe_queries: Sequence[str] = ()) -> Dict:
    rng = random.Random(seed)
    nodes = [make_node(i) for i in range(n)]
    batches = ([sorted((o for o in workload if isinstance(o, Delta)), key=lambda d: d.created_at),
                [o for o in workload if not isinstance(o, Delta)]] if effects_last else [list(workload)])
    wire = [[signed(o, nodes[0].key) for o in b] for b in batches]
    dropped = 0
    for node in nodes:
        for j, batch in enumerate(wire):
            for received, mac in (batch if effects_last and j == 0 else rng.sample(batch, len(batch))):
                if rng.random() < loss:
                    dropped += 1
                else:
                    node.receive_object(received, mac, None)
    delivered = divergence([components(x.mem.store, scope, x.mem.admission.decisions(scope)) for x in nodes])
    rounds, moved, nbytes = reconcile(nodes, scope) if anti_entropy else (0, 0, {})
    rep = settle(nodes, scope, truth=truth(roles) if roles else None)
    reads = [topk(x, probe_queries) for x in nodes]
    return {"topk_distinct": {q: len({r[q] for r in reads}) for q in probe_queries},"loss": loss, "n": n, "seed": seed, "anti_entropy": anti_entropy,
            "effects_last": effects_last, "n_objects": len(workload), "dropped": dropped,
            "delivered": delivered, "rounds": rounds, "moved": moved, "ae_bytes": nbytes, **rep}


def loss_sweep(workload, *, make_node, losses: Sequence[float], n: int, seeds: List[int],
               scope: str, roles=None, probe_queries: Sequence[str] = ()) -> Dict:
    """Per loss: the post-pass authored divergence without anti-entropy (mean over seeds), the
    largest post-pass counter with it (0 expected), the pre-pass window, rounds, bytes per
    message class, L; and the effects-last arm's largest post-pass counter and L."""
    rows = []
    for loss in losses:
        wo = [lossy_cell(workload, make_node=make_node, n=n, loss=loss, seed=s, scope=scope,
                         anti_entropy=False, roles=roles) for s in seeds]
        w = [lossy_cell(workload, make_node=make_node, n=n, loss=loss, seed=s, scope=scope,
                        roles=roles, probe_queries=probe_queries) for s in seeds]
        el = [lossy_cell(workload, make_node=make_node, n=n, loss=loss, seed=s, scope=scope,
                         effects_last=True, probe_queries=probe_queries) for s in seeds]
        m = len(seeds)
        rows.append({
            "loss": loss, "n": n, "n_seeds": m,
            "without_ae_authored_mean": sum(c["post_pass"]["authored"] for c in wo) / m,
            "with_ae_post_pass_max": max(max(c["post_pass"].values()) for c in w),
            "with_ae_delivered_mean": {k: sum(c["delivered"][k] for c in w) / m for k in w[0]["delivered"]},
            "with_ae_pre_pass_mean": {k: sum(c["pre_pass"][k] for c in w) / m for k in w[0]["pre_pass"]},
            "rounds_mean": sum(c["rounds"] for c in w) / m,
            "ae_bytes_mean": {k: sum(c["ae_bytes"][k] for c in w) / m for k in w[0]["ae_bytes"]},
            "L_max": max(c["L"] for c in w),
            "effects_last_post_pass_max": max(max(c["post_pass"].values()) for c in el),
            "effects_last_L_max": max(c["L"] for c in el),
            "final_misses_max": max(max(c["pass_misses"][-1]) for c in w),
            # distinct top-k reads across replicas, largest over probe queries and seeds (1 = identical)
            "with_ae_topk_distinct_max": max((v for c in w for v in c["topk_distinct"].values()), default=None),
            "effects_last_topk_distinct_max": max((v for c in el for v in c["topk_distinct"].values()),
                                                  default=None),
            "purity_mean": (sum(c["purity"]["purity"] for c in w) / m) if roles else None,
        })
    return {"n": n, "seeds": seeds, "rows": rows}


__all__ = ["with_effects", "reconcile", "lossy_cell", "loss_sweep"]
