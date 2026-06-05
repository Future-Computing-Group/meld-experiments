"""The four-way merge decision procedure (architecture A.2).

The genuinely NEW contribution: an executable classifier over the four signals
(kappa/sigma/epsilon/chi) + the Context gate, emitting exactly one of
MERGE / RELATE / CONFLICT / REJECT. Thresholds (theta_merge, sigma_lo) come from
config (CFG), NEVER hardcoded; they are calibrated in M-2, not derived.

Decision procedure (A.2), with the wire-admission (REJECT) gate taking
precedence over everything (Cor. 2.1):

    REJECT    if  stale/overruled gate fails
    CONFLICT  if  chi == contradicts  AND  (kappa == 1 OR sigma >= sigma_lo)
    MERGE     if  kappa == 1
              OR  (sigma >= theta_merge AND chi != contradicts AND ctx_ok)
    RELATE    if  sigma_lo <= sigma < theta_merge          (related, not identical)
              OR  (sigma >= theta_merge AND not ctx_ok)    (same meaning, diff frame)
              with chi != contradicts
    REJECT    otherwise (no actionable relation: sigma below sigma_lo, no key)
"""
from __future__ import annotations

from smp.config import Thresholds


def classify(
    *,
    cfg: Thresholds,
    kappa: int = 0,
    sigma: float = 0.0,
    chi: str = "neutral",
    ctx_ok: bool = True,
    retention_below_min: bool = False,
    status: str = "ACTIVE",
) -> str:
    """Return the merge decision for an (incoming, local) claim pair.

    ``kappa`` exact claim-key identity (0/1); ``sigma`` dense cosine; ``chi``
    NLI label; ``ctx_ok`` Context-gate result; ``retention_below_min`` / ``status``
    drive the wire-admission REJECT gate.
    """
    # --- Gate precedence: stale or non-ACTIVE never propagates (Cor. 2.1) ---
    if retention_below_min or status != "ACTIVE":
        return "REJECT"

    # --- CONFLICT: a contradiction between claims that are at all RELATED
    # (sigma >= sigma_lo) or share a key. Gated on sigma_lo, the actionable-
    # relation floor, NOT theta_merge: a contradiction need not be near-identical
    # in surface form to be a contradiction. Gating it on theta_merge (the round-1
    # design) routed lexically divergent contradictions (sigma in
    # [sigma_lo, theta_merge)) silently to REJECT -- the conflict-recall blind
    # spot. No silent winner; the contradiction is surfaced. ---
    conflict_actionable = (kappa == 1) or (sigma >= cfg.sigma_lo)
    if chi == "contradicts" and conflict_actionable:
        return "CONFLICT"

    # --- MERGE: exact key, OR high-sigma non-contradicting in a shared frame ---
    if kappa == 1:
        return "MERGE"
    if sigma >= cfg.theta_merge and chi != "contradicts" and ctx_ok:
        return "MERGE"

    # --- RELATE: same meaning in a different frame, or mid-band similarity ---
    if chi != "contradicts":
        if sigma >= cfg.theta_merge and not ctx_ok:
            return "RELATE"
        if cfg.sigma_lo <= sigma < cfg.theta_merge:
            return "RELATE"

    # --- otherwise: no actionable relation ---
    return "REJECT"


__all__ = ["classify"]
