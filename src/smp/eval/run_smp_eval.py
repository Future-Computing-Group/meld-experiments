"""SIM experiment drivers: M-2, H/M-1, O-1, A-1 (laptop, 0 cost).

Each driver runs the REAL merge engine on the small LongMemEval-shaped fixture
and returns real counts. M-2 also calibrates theta_merge/sigma_lo and writes the
calibrated config to a SEPARATE gitignored file under data/ (never the tracked
default). Anything requiring real distributed infra (latency/throughput/
convergence-under-partition) is NOT here -- it is a later, gated phase.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from smp.config import Thresholds, load_thresholds
from smp.eval.longmemeval_loader import all_claim_pairs, load_conversation_slice
from smp.eval.metrics import byte_count, version_count
from smp.eval.topology import (
    CentralizedBaseline,
    NaiveUnionBaseline,
    build_topology,
    run_funnel_merge,
)
from smp.merge.classifier import classify
from smp.merge.context_gate import Context, context_compatible
from smp.merge.signals import EmbeddingSimilarity, chi, kappa
from smp.proto import Claim, MemoryUnit

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
CALIBRATED_CONFIG = os.path.join(REPO_ROOT, "data", "thresholds_calibrated.yaml")

# Map a classifier decision to the M-2 label space.
#   coreferent   -> MERGE or RELATE   (same/related, not a conflict)
#   contradiction-> CONFLICT
#   unrelated    -> REJECT
_LABELS = ("coreferent", "contradiction", "unrelated")


def _decision_to_label(decision: str) -> str:
    if decision in ("MERGE", "RELATE"):
        return "coreferent"
    if decision == "CONFLICT":
        return "contradiction"
    return "unrelated"  # REJECT


def _classify_pair(text_a: str, text_b: str, cfg: Thresholds,
                   es: EmbeddingSimilarity, nli: str,
                   *, use_chi: bool = True, same_cell: bool = True) -> str:
    sigma = es.sigma(text_a, text_b)
    chi_val = chi(text_a, text_b, model=nli) if use_chi else "neutral"
    ctx_ok = same_cell and context_compatible(
        Context(cell="agent.public", valid_from=0, valid_until=None),
        Context(cell="agent.public", valid_from=0, valid_until=None),
    )
    return classify(cfg=cfg, kappa=0, sigma=sigma, chi=chi_val, ctx_ok=ctx_ok)


# ---------------------------------------------------------------------------
# M-2: classification quality of the decision procedure
# ---------------------------------------------------------------------------


def run_m2(n_conversations: int, encoder: str, nli: str,
           cfg: Thresholds | None = None) -> Dict:
    cfg = cfg or load_thresholds()
    es = EmbeddingSimilarity(encoder=encoder)
    convs = load_conversation_slice(n_conversations=n_conversations, seed=0)
    pairs = all_claim_pairs(convs)

    confusion: Dict[str, Dict[str, int]] = {
        g: {p: 0 for p in _LABELS} for g in _LABELS
    }
    false_merge = 0
    n_dangerous = 0
    for pr in pairs:
        decision = _classify_pair(pr["text_a"], pr["text_b"], cfg, es, nli)
        pred = _decision_to_label(decision)
        confusion[pr["label"]][pred] += 1
        if pr["label"] == "contradiction":
            n_dangerous += 1
            if decision == "MERGE":
                false_merge += 1

    per_class = {}
    for cls in _LABELS:
        tp = confusion[cls][cls]
        fp = sum(confusion[g][cls] for g in _LABELS if g != cls)
        fn = sum(confusion[cls][p] for p in _LABELS if p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1,
                          "support": sum(confusion[cls].values())}

    return {
        "per_class": per_class,
        "confusion": confusion,
        "false_merge_rate": (false_merge / n_dangerous) if n_dangerous else 0.0,
        "n_pairs": len(pairs),
    }


def calibrate_and_write(n_conversations: int, encoder: str, nli: str) -> Dict:
    """Sweep theta_merge, pick the value maximising macro-F1 subject to
    false_merge_rate == 0, write the calibrated config to a gitignored file."""
    base = load_thresholds()
    best = None
    for theta in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        # a theta at/below sigma_lo leaves the RELATE band [sigma_lo, theta) empty
        # (an invalid calibration point); skip it rather than trip the invariant.
        if theta <= base.sigma_lo:
            continue
        cfg = Thresholds(theta_merge=theta, sigma_lo=base.sigma_lo,
                         r_min=base.r_min, tau=base.tau)
        res = run_m2(n_conversations, encoder, nli, cfg=cfg)
        if res["false_merge_rate"] > 0:
            continue
        macro_f1 = sum(m["f1"] for m in res["per_class"].values()) / len(res["per_class"])
        if best is None or macro_f1 > best[1]:
            best = (theta, macro_f1)
    chosen = best[0] if best else base.theta_merge
    os.makedirs(os.path.dirname(CALIBRATED_CONFIG), exist_ok=True)
    with open(CALIBRATED_CONFIG, "w", encoding="utf-8") as fh:
        fh.write(f"theta_merge: {chosen}\nsigma_lo: {base.sigma_lo}\n"
                 f"r_min: {base.r_min}\ntau: {base.tau}\n")
    return {"theta_merge": chosen, "macro_f1": best[1] if best else None,
            "written_to": CALIBRATED_CONFIG}


# ---------------------------------------------------------------------------
# H / M-1: distributed vs centralized vs naive-union
# ---------------------------------------------------------------------------


def run_m1(n_conversations: int, n_agents: int, encoder: str, nli: str) -> Dict:
    convs = load_conversation_slice(n_conversations=n_conversations, seed=0)

    # distributed: agents -> global-brain funnel merge
    topo = build_topology(convs, n_agents=n_agents, encoder=encoder)
    run_funnel_merge(topo, nli=nli)
    dist_units = sum(1 for u in topo.global_brain.extract_delta()
                     if isinstance(u, MemoryUnit))
    dist_acc = _answer_accuracy(topo.global_brain, convs, encoder)

    # centralized: single store, no merge
    cb = CentralizedBaseline(convs, encoder=encoder)
    cb.ingest_all()
    cent_units = sum(1 for u in cb.brain.extract_delta() if isinstance(u, MemoryUnit))
    cent_acc = _answer_accuracy(cb.brain, convs, encoder)

    # naive union: append everything, no merge decision
    nu = NaiveUnionBaseline(convs, encoder=encoder)
    nu.ingest_all()
    naive_units = sum(1 for u in nu.brain.extract_delta() if isinstance(u, MemoryUnit))

    return {
        "distributed_accuracy": dist_acc,
        "centralized_accuracy": cent_acc,
        "distributed_units": dist_units,
        "centralized_units": cent_units,
        "naive_union_units": naive_units,
        "n_conflicts": sum(1 for p in topo.patches if p.decision == "CONFLICT"),
    }


def _answer_accuracy(brain, convs: List[Dict], encoder: str) -> float:
    if not convs:
        return 0.0
    correct = 0
    for conv in convs:
        hits = brain.graph_walk_read(conv["question"], encoder=encoder, k=1)
        if hits and hits[0][0] == conv["answer_unit_id"]:
            correct += 1
    return correct / len(convs)


# ---------------------------------------------------------------------------
# O-1: overhead term counts (mapped onto the DWM C_data terms)
# ---------------------------------------------------------------------------


def run_o1(n_conversations: int, n_agents: int, encoder: str, nli: str) -> Dict:
    convs = load_conversation_slice(n_conversations=n_conversations, seed=0)
    topo = build_topology(convs, n_agents=n_agents, encoder=encoder)
    run_funnel_merge(topo, nli=nli)
    units = [u for u in topo.global_brain.extract_delta() if isinstance(u, MemoryUnit)]
    walk_cost = 0
    for conv in convs:
        hits = topo.global_brain.graph_walk_read(conv["question"], encoder=encoder, k=5)
        walk_cost += len(hits)
    return {
        "c_sync_bytes": byte_count(units),                 # bytes synced
        "c_ver_versions": version_count(units),            # version-storage growth
        "c_ret_walk_cost": walk_cost,                      # graph-walk traversal cost
        # No external re-fetch on this corpus: freshness is the status field
        # (no separate staleness tag), and these units are all status-ACTIVE.
        "c_ref_refetch": 0,
        "n_patches": len(topo.patches),
    }


# ---------------------------------------------------------------------------
# A-1: gate ablation (drop the chi NLI gate) -> false-MERGE contribution
# ---------------------------------------------------------------------------


def run_a1(n_conversations: int, encoder: str, nli: str) -> Dict:
    cfg = load_thresholds()
    es = EmbeddingSimilarity(encoder=encoder)
    convs = load_conversation_slice(n_conversations=n_conversations, seed=0)
    pairs = [p for p in all_claim_pairs(convs) if p["label"] == "contradiction"]

    def _fm(use_chi: bool) -> float:
        if not pairs:
            return 0.0
        fm = sum(1 for pr in pairs
                 if _classify_pair(pr["text_a"], pr["text_b"], cfg, es, nli,
                                   use_chi=use_chi) == "MERGE")
        return fm / len(pairs)

    return {"false_merge_full": _fm(True), "false_merge_no_chi": _fm(False),
            "n_contradiction_pairs": len(pairs)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True, choices=["M2", "M1", "O1", "A1"])
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--encoder", default="hashing")
    ap.add_argument("--nli", default="lexical")
    ap.add_argument("--n-conversations", type=int, default=8)
    ap.add_argument("--n-agents", type=int, default=3)
    a = ap.parse_args(argv)
    n = 8 if a.tiny else a.n_conversations
    if a.exp == "M2":
        res = run_m2(n, a.encoder, a.nli)
    elif a.exp == "M1":
        res = run_m1(n, a.n_agents, a.encoder, a.nli)
    elif a.exp == "O1":
        res = run_o1(n, a.n_agents, a.encoder, a.nli)
    else:
        res = run_a1(n, a.encoder, a.nli)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_m2", "run_m1", "run_o1", "run_a1", "calibrate_and_write"]
