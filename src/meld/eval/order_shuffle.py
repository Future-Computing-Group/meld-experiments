"""Order-shuffle harness over the fold (design §7.1 harnesses, §7.2 H3, §7.4 freeze gate; E2-T1).

``run_order_shuffle`` folds one workload in K seeded arrival orders, one order at a time: a fresh
sink replica (``make_node(i)``) receives the order through the receive gate and admission, then
``sync.experiment.settle([node])`` measures its pre-pass view and runs its pass (a second covers
its own misses), and the replica is dropped before the next order, so one model is live and a
caller may reuse one provider across orders. No Sig crosses orders: each order's fold reads only
the payloads its own replica evaluated, so provider nondeterminism shows as distinct
fingerprints rather than being joined away. Reported across orders: distinct post-pass
fingerprints (target 1 of K), the divergence on the five components before and after the pass
(Sig is reported but not a gate: record batching depends on arrival), the per-order window
(reps, links, decisions differing from the post-pass mode), not_last, L and misses per order,
decision variants, top-k reads per probe query, and with ``roles`` group purity and typed-edge
recovery before and after the pass.

``e2t1`` (06 §e, restated for hybrid H): admission refolds on every non-last arrival, so
"at admission against after the pass" is no contrast; the measurement is the missed-link rate
against the selector's m, one fold of the gold workload per epoch (each epoch its own cal_root,
r3 §4.3), with per-(seed, edge type) hits kept for a cluster bootstrap over seeds.

Workloads:
  * ``running_example`` -- the paper's running example (main text §4.3): the sink holds c1; an
    EU paraphrase, the same sentence scoped US, an intra-EU contradiction c2 and the EU
    authority's overrule of c2 arrive in any order.
  * ``gold_workload`` -- 06 §e: the 26 gold seeds, each canonical (c), paraphrase (p), related (r)
    and contradiction (x), written by three source brains (104 claims), with ground truth;
    ``gold_queries`` are its probe queries (each seed's canonical sentence) for the top-k check.
"""
from __future__ import annotations

import itertools
import random
from collections import Counter
from typing import Callable, Dict, List, Sequence, Tuple

from meld.agent.memory import DEFAULT_CELL, DEFAULT_POL, SMPMemory
from meld.merge.m2_gold import _SEEDS
from meld.protocol.objects import AuthoredEffect, Delta, signed
from meld.eval.graph_fingerprint import COMPONENTS, WINDOW, divergence
from meld.sync.experiment import settle

EU, US = "jurisdiction.eu", "jurisdiction.us"
C1_CANONICAL = "Customer data must be retained for seven years"
C1_PARAPHRASE = "We are required to retain customer data for seven years"
C2_CONTRADICTION = "Customer data must be deleted after two years"
ROLES = "cprx"
_LINK = {"CONFLICT": "CONTRADICTS", "RELATE": "ASSOCIATES"}
EDGES = {"MERGE": ("cp",), "CONFLICT": ("cx", "px"), "RELATE": ("cr", "pr")}


def claims(encoder: str, pol: dict, items: Sequence[Tuple[str, str]], n_sources: int = 3) -> List[Delta]:
    """One delta per (text, cell), written round-robin by ``n_sources`` source brains on a shared
    counter clock (created_at = position, reproducible)."""
    t = itertools.count(1)
    srcs = [SMPMemory(encoder=encoder, nli="lexical", pol=pol, origin=f"src{i}", clock=lambda: next(t))
            for i in range(n_sources)]
    return [srcs[i % n_sources].delta(text, cell) for i, (text, cell) in enumerate(items)]


def running_example(encoder: str):
    """(pol, seed objects, messages)."""
    pol = {**DEFAULT_POL, "scopes": {"jur": [EU, US]},
           "auth_policy": {**DEFAULT_POL["auth_policy"], "calibration_authority": {"jur": "calibration"}}}
    c1, p_eu, p_us, c2 = claims(encoder, pol, [(C1_CANONICAL, EU), (C1_PARAPHRASE, EU),
                                               (C1_PARAPHRASE, US), (C2_CONTRADICTION, EU)])
    overrule = AuthoredEffect(scope="jur", kind="OVERRULE", c1=c2.claim_key, c2=b"", target_cell=EU,
                              author="C", author_cell=EU, author_auth=0, created_at=0)
    return pol, [c1], [p_eu, p_us, c2, overrule]


def gold_workload(encoder: str, n_seeds: int = 26):
    """(deltas, roles: claim key -> (seed id, role in 'cprx')) over the first ``n_seeds`` seeds."""
    rows = [(sid, role, text) for sid, _diff, *texts in _SEEDS[:n_seeds] for role, text in zip(ROLES, texts)]
    deltas = claims(encoder, DEFAULT_POL, [(text, DEFAULT_CELL) for _, _, text in rows])
    return deltas, {d.claim_key: (sid, role) for d, (sid, role, _) in zip(deltas, rows)}


def gold_queries(n_seeds: int = 26) -> List[str]:
    """Probe queries of the gold workload: each seed's canonical sentence, read top-k over the store."""
    return [canon for _sid, _diff, canon, *_ in _SEEDS[:n_seeds]]


def topk(node, queries: Sequence[str]) -> Dict[str, tuple]:
    """query -> the replica's top-k unit ids (graph-walk read over live nodes)."""
    return {q: tuple(u for u, _ in node.mem.store.graph_walk_read(q, encoder=node.mem.encoder))
            for q in queries}


def truth(roles) -> Callable[[bytes, bytes], str]:
    """Ground truth of 06 §e: c-p MERGEABLE, x against c or p CONTRADICTION, else NON_MERGEABLE."""
    def t(x: bytes, y: bytes) -> str:
        (sx, rx), (sy, ry) = roles[x], roles[y]
        pair = {rx, ry}
        if sx != sy or not pair & {"c", "p"}:
            return "NON_MERGEABLE"
        return "MERGEABLE" if pair == {"c", "p"} else "CONTRADICTION" if "x" in pair else "NON_MERGEABLE"
    return t


def edge_recovery(store, roles) -> Dict:
    """06 §e: per type, ground-truth edges recovered (MERGE: one group; CONFLICT / RELATE: a raw
    CONTRADICTS / ASSOCIATES link between the two claims in either direction, and at group level
    one between the representatives of their two distinct groups), hits per (seed, type), the spurious cross-seed
    links, and the spurious merges (group pairs whose truth is not MERGEABLE)."""
    by = {v: k for k, v in roles.items()}
    rep, links, t = store.derived.rep, store.derived.links, truth(roles)
    R = lambda k: rep.get(k, k)                                       # noqa: E731
    raw = {(ty, frozenset((s, d))) for ty, s, d, _ in links}
    grp = {(ty, frozenset((R(s), R(d)))) for ty, s, d, _ in links}
    out, per_seed = {}, {sid: {} for sid in sorted({s for s, _ in by})}
    for typ, pairs in EDGES.items():
        hit = ghit = tot = 0
        for sid in per_seed:
            h = 0
            for a, b in pairs:
                x, y = by[sid, a], by[sid, b]
                tot += 1
                if typ == "MERGE":
                    h += R(x) == R(y)
                    ghit += R(x) == R(y)
                else:
                    h += (_LINK[typ], frozenset((x, y))) in raw
                    ghit += R(x) != R(y) and (_LINK[typ], frozenset((R(x), R(y)))) in grp
            per_seed[sid][typ] = h
            hit += h
        out[typ] = {"recovered": hit, "group_recovered": ghit, "total": tot}
    groups: Dict[bytes, List[bytes]] = {}
    for k in roles:
        groups.setdefault(R(k), []).append(k)
    out["per_seed"] = per_seed
    out["spurious_links"] = sum(roles[s][0] != roles[d][0] for ty, s, d, _ in links if ty in _LINK.values())
    out["spurious_merges"] = sum(t(a, b) != "MERGEABLE" for ms in groups.values()
                                 for a, b in itertools.combinations(ms, 2))
    return out


def _edges_summary(rows: List[Dict]) -> Dict:
    def rate(key):
        return {t: 1 - sum(r[t][key] for r in rows) / sum(r[t]["total"] for r in rows) for t in EDGES}
    return {"miss_rate": rate("recovered"), "group_miss_rate": rate("group_recovered"),
            "total": {t: rows[0][t]["total"] for t in EDGES},
            "spurious_links_mean": sum(r["spurious_links"] for r in rows) / len(rows),
            "spurious_merges_mean": sum(r["spurious_merges"] for r in rows) / len(rows)}


def _receive(node, objs, i) -> None:
    for o in objs:
        if not node.receive_object(*signed(o, node.key), None):
            raise ValueError(f"workload object refused by replica {i}: {type(o).__name__}")


def run_order_shuffle(seed_objs, workload, *, k: int, seed: int, make_node, scope: str,
                      roles=None, probe_queries: Sequence[str] = ()) -> Dict:
    """See the module docstring. ``make_node(i)`` builds the replica for order i."""
    rng = random.Random(seed)
    t = truth(roles) if roles else None
    orders, rows = [], []
    for i in range(k):
        order = list(range(len(workload)))
        rng.shuffle(order)
        node = make_node(i)
        _receive(node, list(seed_objs) + [workload[j] for j in order], i)
        row = {"edges_pre": edge_recovery(node.mem.store, roles) if roles else None}
        row["settle"] = settle([node], scope, truth=t)
        row["edges_post"] = edge_recovery(node.mem.store, roles) if roles else None
        row["topk"] = topk(node, probe_queries)
        orders.append(order)
        rows.append(row)
        del node                                    # one replica (one model) live at a time
    st = [r["settle"] for r in rows]
    pre = [x["pre_views"][0] for x in st]
    post = [x["post_views"][0] for x in st]
    modal = {c: Counter(v[c] for v in post).most_common(1)[0][0] for c in COMPONENTS}
    window = [[c for c in WINDOW if v[c] != modal[c]] for v in pre]
    fps = [x["fingerprints"][0] for x in st]
    out = {"k": k, "seed": seed, "n_items": len(workload), "orders": orders, "fingerprints": fps,
           "distinct_fingerprints": len(set(fps)), "pre_pass": divergence(pre),
           "post_pass": divergence(post), "window": window,
           "orders_with_window": sum(bool(w) for w in window),
           "not_last": [x["not_last"][0] for x in st], "L": [x["L"] for x in st],
           "final_misses": [x["pass_misses"][-1][0] for x in st],
           "decision_variants": [[list(v), n] for v, n in Counter(x["decisions"][0] for x in st).items()],
           "purity": st[0]["purity"],
           "topk_distinct": {q: len({r["topk"][q] for r in rows}) for q in probe_queries}}
    if roles:
        out["edges_pre"] = _edges_summary([r["edges_pre"] for r in rows])
        out["edges_post"] = _edges_summary([r["edges_post"] for r in rows])
    return out


def e2t1(deltas, roles, epochs, *, make_node, scope: str) -> Dict:
    """m -> missed-link rates (raw and group level), per-seed hits, purity and L: one fold of
    ``deltas`` per epoch on ``make_node(epoch)``. Epochs differ in m and each names its own
    cal_root (a selector change redefines the calibration population, r3 §4.3)."""
    if len({e.cal_root for e in epochs}) < len(epochs) or len({e.m for e in epochs}) < len(epochs):
        raise ValueError("each epoch needs its own m and its own cal_root")
    out = {}
    for ep in epochs:
        node = make_node(ep)
        _receive(node, deltas, ep.m)
        st = settle([node], scope, truth=truth(roles))
        e = edge_recovery(node.mem.store, roles)
        out[ep.m] = {"miss_rate": {t: 1 - e[t]["recovered"] / e[t]["total"] for t in EDGES},
                     "group_miss_rate": {t: 1 - e[t]["group_recovered"] / e[t]["total"] for t in EDGES},
                     "per_seed": e["per_seed"], "spurious_links": e["spurious_links"],
                     "spurious_merges": e["spurious_merges"], "purity": st["purity"], "L": st["L"]}
        del node
    return out


__all__ = ["EU", "US", "C1_CANONICAL", "C1_PARAPHRASE", "C2_CONTRADICTION", "claims",
           "running_example", "gold_workload", "gold_queries", "topk", "truth", "edge_recovery", "run_order_shuffle", "e2t1"]
