#!/usr/bin/env python3
"""Shared merge-on-receive step for the cross-tier brains (relay and Kafka).

One source of truth for the decision logic: reuses the tested src/smp classifier
and status CRDT. ``process_delta`` is transport-independent --- given a delta dict,
the calibrated thresholds, and the brain's status state, it returns the ack dict
(and mutates ``status`` for overrule/deprecate links). The relay brain and the
Kafka brain both call this, so the two transports cannot diverge in semantics.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from smp.merge.classifier import classify                       # noqa: E402
from smp.merge.status_crdt import StatusState, effective_status  # noqa: E402


def process_delta(d: dict, cfg, status: dict, brain_id: str, t_recv: float):
    """Return the ack dict for one delivered delta; mutate ``status`` in place."""
    op = d.get("op", "publish")
    cid = d.get("claim_id", "")
    if op == "publish":
        decision = classify(
            cfg=cfg, kappa=int(d.get("kappa", 0)), sigma=float(d.get("sigma", 0.0)),
            chi=d.get("chi", "neutral"), ctx_ok=bool(d.get("ctx_ok", True)))
        eff = effective_status(status.get(cid, StatusState()))
        return {"nonce": d.get("nonce"), "brain": brain_id, "op": op, "claim_id": cid,
                "decision": decision, "eff_status": eff, "t_recv": t_recv}
    if op in ("overrule", "deprecate"):
        effect = "OVERRULED" if op == "overrule" else "DEPRECATED"
        link_id = d.get("link_id", "%s-%s" % (op, d.get("nonce")))
        st = status.get(cid, StatusState()).add(link_id, effect)
        status[cid] = st  # grow-only set: retained across reconnects -> heal converges
        return {"nonce": d.get("nonce"), "brain": brain_id, "op": op, "claim_id": cid,
                "decision": None, "eff_status": effective_status(st), "t_recv": t_recv}
    return None
