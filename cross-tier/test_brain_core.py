#!/usr/bin/env python3
"""Unit tests for the shared merge-on-receive step (brain_core.process_delta).

``process_delta`` is the one source of truth both cross-tier brains (relay and
Kafka) call, so the two transports cannot diverge in semantics. The broker-gated
end-to-end tests exercise it through a real transport; these tests pin its
contract DIRECTLY, transport-free and broker-free, asserting the causal effect at
the seam (distinct signal in => distinct decision out: a double that ignored the
signals would pass control-flow but fail these) and the order-independent CRDT
join the convergence claim rests on.

Run:  PYTHONPATH=src:cross-tier python3 -m pytest cross-tier/test_brain_core.py -q
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from meld.config import load_thresholds            # noqa: E402
from brain_core import process_delta               # noqa: E402

CFG = load_thresholds(os.path.join(REPO, "src", "meld", "config", "thresholds-minilm.yaml"))

# The verified running-example signal tuples (driver.BEATS): each is built to land
# in exactly one decision class at the calibrated operating point.
MERGE = {"op": "publish", "claim_id": "c1", "kappa": 0, "sigma": 0.93, "chi": "entails", "ctx_ok": True}
RELATE = {"op": "publish", "claim_id": "c1", "kappa": 0, "sigma": 0.93, "chi": "entails", "ctx_ok": False}
CONFLICT = {"op": "publish", "claim_id": "c2", "kappa": 0, "sigma": 0.86, "chi": "contradicts", "ctx_ok": True}
NO_RELATION = {"op": "publish", "claim_id": "c3", "kappa": 0, "sigma": 0.00, "chi": "neutral", "ctx_ok": True}


def _decide(delta):
    return process_delta(dict(delta, nonce="n"), CFG, {}, "b", 0.0)["decision"]


def test_publish_classifies_each_class_distinctly():
    # The seam carries the signals into the decision: distinct signal in => distinct
    # decision out (a test double that ignored the signals would fail this).
    assert _decide(MERGE) == "MERGE"
    assert _decide(RELATE) == "RELATE"
    assert _decide(CONFLICT) == "CONFLICT"
    assert _decide(NO_RELATION) == "NO_RELATION"


def test_publish_ack_shape_carries_identity_and_does_not_mutate_status():
    status = {}
    ack = process_delta(dict(MERGE, nonce="abc"), CFG, status, "hpc", 12.5)
    assert ack["nonce"] == "abc" and ack["brain"] == "hpc" and ack["claim_id"] == "c1"
    assert ack["op"] == "publish" and ack["eff_status"] == "ACTIVE" and ack["t_recv"] == 12.5
    assert status == {}, "a publish must not mutate the status CRDT"


def test_overrule_sets_status_and_mutates_state_in_place():
    status = {}
    ack = process_delta({"op": "overrule", "claim_id": "c2", "link_id": "L_ovr", "nonce": "o"},
                        CFG, status, "b", 0.0)
    assert ack["decision"] is None and ack["op"] == "overrule"
    assert ack["eff_status"] == "OVERRULED"
    assert "c2" in status and effective(status, "c2") == "OVERRULED"


def test_deprecate_sets_deprecated():
    status = {}
    ack = process_delta({"op": "deprecate", "claim_id": "c2", "link_id": "L_dep", "nonce": "d"},
                        CFG, status, "b", 0.0)
    assert ack["eff_status"] == "DEPRECATED" and effective(status, "c2") == "DEPRECATED"


def test_join_is_order_independent_both_orders_converge_to_overruled():
    # The empirical face of the status-CRDT: a claim deprecated by one peer and
    # overruled by another converges to OVERRULED regardless of delivery order.
    dep = {"op": "deprecate", "claim_id": "c2", "link_id": "L_dep", "nonce": "d"}
    ovr = {"op": "overrule", "claim_id": "c2", "link_id": "L_ovr", "nonce": "o"}

    s1 = {}
    process_delta(dep, CFG, s1, "b", 0.0)
    a1 = process_delta(ovr, CFG, s1, "b", 0.0)

    s2 = {}
    process_delta(ovr, CFG, s2, "b", 0.0)
    a2 = process_delta(dep, CFG, s2, "b", 0.0)

    assert a1["eff_status"] == "OVERRULED" == a2["eff_status"]
    assert effective(s1, "c2") == effective(s2, "c2") == "OVERRULED"


def test_idempotent_redelivery_of_same_link_is_a_noop():
    status = {}
    d = {"op": "overrule", "claim_id": "c2", "link_id": "L_ovr", "nonce": "o"}
    process_delta(d, CFG, status, "b", 0.0)
    process_delta(d, CFG, status, "b", 0.0)  # re-deliver (a partition heal would)
    assert effective(status, "c2") == "OVERRULED"
    assert len(status["c2"].links) == 1, "re-adding the same link id must not grow the set"


def test_unknown_op_returns_none():
    assert process_delta({"op": "subscribe", "claim_id": "c1", "nonce": "n"}, CFG, {}, "b", 0.0) is None


def effective(status, cid):
    from meld.merge.status_crdt import effective_status
    return effective_status(status[cid])


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, "->", repr(e)[:200])
    print("=== %d failure(s) ===" % fails)
    sys.exit(1 if fails else 0)
