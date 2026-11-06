"""Wire bytes per message class (sync/wire_bytes.py) on the running example and the gold workload,
and anti-entropy bytes per round (eval/anti_entropy_eval.reconcile metering).

Per workload, one replica receives every object, settles (Sig exchange and pass), and every object
it then holds in the scope is metered as the signed envelope it travels in (DELTA, EFFECT, SIG),
plus its PASS certificate (fold.publish_pass). The gold workload also meters CAL and EPOCH: one
non-mergeable tested pair of the pass per seed becomes a CalRecord, and conformal.build_epoch builds
the epoch over them (sized here, not installed). Anti-entropy: lossy delivery to --n replicas, then
ring DIGEST / PULL / PATCH rounds to agreement, bytes per call path in total and per round; and one
quiescent round between replicas that already agree (DIGEST only). Each section and each lossy row
carries topology.stamps (no epoch is installed: the gold epoch is sized, not held).

Run:  PYTHONPATH=src python scripts/run_wire_bytes.py --encoder minilm --nli mnli-cross-encoder \
          [--n 4] [--losses 0.01 0.05 0.10] [--out outputs/wire-bytes.json]
"""
from __future__ import annotations

import argparse
import json
import os

from meld.agent.memory import DEFAULT_POL
from meld.eval.anti_entropy_eval import lossy_cell, reconcile, with_effects
from meld.eval.order_shuffle import gold_workload, running_example, truth
from meld.eval.topology import stamps
from meld.merge.conformal import bin_of, build_epoch
from meld.protocol.fold import publish_pass, readjudicate
from meld.protocol.objects import CalRecord, m_pol_of, signed
from meld.sync.experiment import node_factory, settle
from meld.sync.objectset import KINDS
from meld.sync.wire_bytes import meter

B0 = 256
ALPHA = 20_000


def _held(node, objs, scope, roles=None):
    """Deliver, settle, pass; (every scope object as received, the pass result)."""
    for o in objs:
        if not node.receive_object(*signed(o, node.key), None):
            raise ValueError(f"workload object refused: {type(o).__name__}")
    settle([node], scope, truth=truth(roles) if roles else None)
    res = readjudicate(node.mem.store, scope, m_nli=node.mem.m_nli, provider=node.mem.admission.provider,
                       issuer=node.mem.origin, hw=node.mem.admission.hw)
    entries = [node.objects.raw(i) for k in KINDS for i in node.objects.ids(k, scope)]
    return entries + [(*publish_pass(res.certificate, node.key), scope)], res


def _cal_epoch(res, roles, pol, scope, key, m_n):
    """One non-mergeable tested pair per seed as a CalRecord, and the epoch over them, signed."""
    t, sel, cal, seen = truth(roles), pol["bootstrap_selector"], [], set()
    for d in res.decisions:
        if d.tested is None or d.s_q is None or t(*d.tested) == "MERGEABLE" or roles[d.b][0] in seen:
            continue
        seen.add(roles[d.b][0])
        c_x, c_y = sorted(d.tested)
        cal.append(CalRecord(
            scope=scope, c_x=c_x, c_y=c_y, bin=bin_of(d.n_prior, B0), s_q=d.s_q, hw="cpu", obs=1,
            label="NON_MERGEABLE", subtype="CONTRADICTION" if t(*d.tested) == "CONTRADICTION" else "OTHER",
            source="SEED", item=str(roles[d.b][0]), author="calibration", created_at=0,
            sigma_lo_q=sel[0], m=sel[2], m_pol=m_pol_of(pol), n_prior=d.n_prior))
    ep = build_epoch(cal, scope=scope, seq=1, alpha=ALPHA, alpha_c=ALPHA, policy="SEED", bins=(B0, "doubling"),
                     selector_params=(sel[0], sel[2], m_pol_of(pol)), m_nli=m_n, sigma_c_q=sel[1],
                     author="calibration", created_at=0)
    return [(*signed(o, key), scope) for o in cal + [ep]]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--losses", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rep = {}

    pol, seed_objs, msgs = running_example(a.encoder)
    node = node_factory(encoder=a.encoder, nli=a.nli, pol=pol)(0)
    entries, _ = _held(node, list(seed_objs) + list(msgs), "jur")
    rep["running_example"] = {"classes": meter(entries), **stamps(a.encoder, a.nli, None, pol)}

    deltas, roles = gold_workload(a.encoder)
    work = with_effects(deltas, roles, DEFAULT_POL)
    make = node_factory(encoder=a.encoder, nli=a.nli)
    node = make(0)
    entries, res = _held(node, work, "agent", roles)
    entries += _cal_epoch(res, roles, DEFAULT_POL, "agent", node.key, node.mem.m_nli)
    rep["gold"] = {"classes": meter(entries), "decisions": len(res.decisions),
                   "merges": sum(d.decision == "MERGE" for d in res.decisions),
                   **stamps(a.encoder, a.nli, None, DEFAULT_POL)}

    lossy = []
    for loss in a.losses:
        c = lossy_cell(work, make_node=make, n=a.n, loss=loss, seed=a.seed, scope="agent", roles=roles)
        lossy.append({"loss": loss, "n": a.n, "dropped": c["dropped"], "rounds": c["rounds"], "moved": c["moved"],
                      "bytes": c["ae_bytes"],
                      "per_round": {k: v / c["rounds"] for k, v in c["ae_bytes"].items()} if c["rounds"] else {},
                      **stamps(a.encoder, a.nli, None, DEFAULT_POL)})
    nodes = [make(i) for i in range(a.n)]
    for x in nodes:
        for o in work:
            x.receive_object(*signed(o, x.key), None)
    converge = reconcile(nodes, "agent")
    quiet = reconcile(nodes, "agent")
    rep["anti_entropy"] = {"lossy": lossy, "lossless_converge": {"rounds": converge[0], "bytes": converge[2]},
                           "quiescent_round": quiet[2], "quiescent_rounds": quiet[0],
                           **stamps(a.encoder, a.nli, None, DEFAULT_POL)}
    print(json.dumps(rep, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    return rep


if __name__ == "__main__":
    main()
