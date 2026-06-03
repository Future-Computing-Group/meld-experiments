"""T-1 conflict representation + authority tie-break.

(a) Conflict flagging: every injected contradiction must be flagged as a first-class
    CONTRADICTS (the CONFLICT decision), NOT silently merged. SMP flags all; the
    auto-merge baseline flags none (it silently picks/averages a winner).
(b) Authority tie-break: the authority-gated resolution (canonical brain OVERRULES the
    peer via the status-CRDT) yields the canonical claim. So it helps IFF the canonical
    brain actually holds ground truth -- a normative choice, not a truth-finder (the open
    normative limit). Recency-only tie-break is the order-dependent baseline.
Pure simulation over smp.merge.classify + status_crdt (no LLM, no testbed).
"""
from __future__ import annotations


def _cfg():
    from smp.config import load_thresholds
    return load_thresholds()


def _contradictions(cfg):
    hi = min(cfg.theta_merge + 0.05, 1.0)        # same-thing (above merge threshold or exact key)
    return [
        {"kappa": 1, "sigma": hi, "chi": "contradicts", "ctx_ok": True,
         "retention_below_min": False, "status": "ACTIVE"},
        {"kappa": 0, "sigma": hi, "chi": "contradicts", "ctx_ok": True,
         "retention_below_min": False, "status": "ACTIVE"},
    ]


def test_smp_flags_every_contradiction_automerge_flags_none():
    from smp.merge.conflict_trust import conflict_flag_rate
    cfg = _cfg()
    m = conflict_flag_rate(_contradictions(cfg), cfg)
    assert m["smp_flag_rate"] == 1.0             # every contradiction -> CONTRADICTS, never silent
    assert m["auto_merge_flag_rate"] == 0.0      # auto-merge silently resolves, flags none


def test_authority_tie_break_helps_iff_canonical_is_right():
    from smp.merge.conflict_trust import authority_resolution_accuracy
    canonical_right = [{"canonical": "fact-A", "peer": "fact-B", "truth": "fact-A",
                        "canonical_is_later": False} for _ in range(5)]
    peer_right = [{"canonical": "fact-A", "peer": "fact-B", "truth": "fact-B",
                   "canonical_is_later": False} for _ in range(5)]
    assert authority_resolution_accuracy(canonical_right)["smp_authority_acc"] == 1.0
    # authority is normative, not truth-finding: it picks the canonical brain even when wrong.
    assert authority_resolution_accuracy(peer_right)["smp_authority_acc"] == 0.0


def test_recency_baseline_is_order_dependent_not_truth_tracking():
    from smp.merge.conflict_trust import authority_resolution_accuracy
    # truth is the peer's claim, but the canonical claim arrives later -> recency picks the
    # wrong (canonical) one; authority also wrong here -> both miss (truth needs neither rule).
    conflicts = [{"canonical": "fact-A", "peer": "fact-B", "truth": "fact-B",
                  "canonical_is_later": True} for _ in range(4)]
    m = authority_resolution_accuracy(conflicts)
    assert m["recency_acc"] == 0.0               # later=canonical=wrong
    assert m["smp_authority_acc"] == 0.0
