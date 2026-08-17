"""A-1 gate ablation: per-gate contribution to false-merge / stale-promotion rates.

Ablating an admission gate = forcing its input to the permissive value, then re-running
the same decision procedure (meld.merge.classify):
  * Context gate   -> ctx_ok forced True (a high-sigma cross-frame pair now MERGEs).
  * inference gate -> a contradicting chi treated as non-contradicting (CONFLICT -> MERGE).
  * staleness gate -> a stale/non-ACTIVE incoming treated as fresh+ACTIVE (REJECT -> propagate).
The full classifier is the safe baseline; each ablation must worsen its target error. Pure
simulation -- no LLM, no testbed (independent of the multi-host runs).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from meld.config import Thresholds
from meld.merge.classifier import classify

GATES = (None, "context", "inference", "staleness")


def classify_ablated(*, cfg: Thresholds, kappa: int, sigma: float, chi: str, ctx_ok: bool,
                     retention_below_min: bool, status: str,
                     ablate: Optional[str] = None) -> str:
    if ablate == "context":
        ctx_ok = True
    elif ablate == "inference":
        chi = "neutral" if chi == "contradicts" else chi
    elif ablate == "staleness":
        retention_below_min = False
        status = "ACTIVE"
    elif ablate is not None:
        raise ValueError(f"unknown gate to ablate: {ablate!r}")     # no silent default
    return classify(cfg=cfg, kappa=kappa, sigma=sigma, chi=chi, ctx_ok=ctx_ok,
                    retention_below_min=retention_below_min, status=status)


def gate_ablation_metrics(cases: List[dict], cfg: Thresholds) -> Dict[Optional[str], dict]:
    """For each gate (and the full classifier, key ``None``) return the false-merge rate
    (non-MERGE-gold cases predicted MERGE -- the safety error) and the stale-promotion rate
    (stale cases predicted non-REJECT). ``cases`` carry the classify inputs + ``gold`` +
    a ``stale`` flag. Deltas vs ``None`` are each gate's contribution."""
    non_merge = [c for c in cases if c["gold"] != "MERGE"]
    stale = [c for c in cases if c.get("stale")]
    out: Dict[Optional[str], dict] = {}
    for gate in GATES:
        preds = [classify_ablated(
            cfg=cfg, kappa=c["kappa"], sigma=c["sigma"], chi=c["chi"], ctx_ok=c["ctx_ok"],
            retention_below_min=c["retention_below_min"], status=c["status"], ablate=gate)
            for c in cases]
        pred = {id(c): p for c, p in zip(cases, preds)}
        fmr = (sum(1 for c in non_merge if pred[id(c)] == "MERGE") / len(non_merge)
               if non_merge else 0.0)
        spr = (sum(1 for c in stale if pred[id(c)] != "REJECT") / len(stale)
               if stale else 0.0)
        out[gate] = {"false_merge_rate": fmr, "stale_promotion_rate": spr}
    return out


__all__ = ["classify_ablated", "gate_ablation_metrics", "GATES"]
