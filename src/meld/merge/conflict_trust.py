"""T-1 conflict representation + authority tie-break (simulation).

Conflict flagging reuses the decision procedure (meld.merge.classify): a same-thing
contradiction is classified CONFLICT (a first-class CONTRADICTS link), never a silent
MERGE. The authority tie-break reuses the status-CRDT: the canonical brain OVERRULES the
peer (peer's claim -> OVERRULED, canonical stays ACTIVE), so the surviving claim is the
canonical one. It therefore helps IFF the canonical brain actually holds ground truth --
MELD resolves status, not truth. No LLM, no testbed (independent of the multi-host runs).
"""
from __future__ import annotations

from typing import Dict, List

from meld.config import Thresholds
from meld.merge import status_crdt
from meld.merge.classifier import classify


def _inputs(c: dict) -> dict:
    return dict(kappa=c["kappa"], sigma=c["sigma"], chi=c["chi"], ctx_ok=c["ctx_ok"],
                retention_below_min=c["retention_below_min"], status=c["status"])


def conflict_flag_rate(cases: List[dict], cfg: Thresholds) -> Dict[str, float]:
    """Fraction of injected contradictions flagged as CONFLICT (first-class CONTRADICTS).
    MELD flags via the decision procedure; auto-merge silently resolves and flags none."""
    if not cases:
        return {"meld_flag_rate": 0.0, "auto_merge_flag_rate": 0.0}
    flagged = sum(1 for c in cases if classify(cfg=cfg, **_inputs(c)) == "CONFLICT")
    return {"meld_flag_rate": flagged / len(cases), "auto_merge_flag_rate": 0.0}


def _authority_survivor(canonical, peer):
    """Canonical OVERRULES peer via the status-CRDT: the peer's claim takes an OVERRULED
    effect, the canonical's stays ACTIVE -> the surviving (effective-ACTIVE) claim is
    canonical. (effective_status of the empty state is ACTIVE; of the overruled state, not.)"""
    canon_state = status_crdt.StatusState()                       # ACTIVE
    peer_state = status_crdt.StatusState().add("authority-overrule", "OVERRULED")
    canon_active = status_crdt.effective_status(canon_state) == "ACTIVE"
    peer_active = status_crdt.effective_status(peer_state) == "ACTIVE"
    return canonical if (canon_active and not peer_active) else peer


def authority_resolution_accuracy(conflicts: List[dict]) -> Dict[str, float]:
    """Post-resolution accuracy of the authority tie-break vs a recency-only baseline.
    Each conflict: {canonical, peer, truth, canonical_is_later}. Authority -> canonical
    survives; recency -> the later-arriving claim survives."""
    if not conflicts:
        return {"meld_authority_acc": 0.0, "recency_acc": 0.0}
    n = len(conflicts)
    meld = sum(1 for x in conflicts
              if _authority_survivor(x["canonical"], x["peer"]) == x["truth"])
    rec = sum(1 for x in conflicts
              if (x["canonical"] if x["canonical_is_later"] else x["peer"]) == x["truth"])
    return {"meld_authority_acc": meld / n, "recency_acc": rec / n}


__all__ = ["conflict_flag_rate", "authority_resolution_accuracy"]
