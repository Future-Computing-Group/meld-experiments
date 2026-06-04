"""Run the four merge/sync/agent ablation modules and print reportable metrics.

Calls the existing, unit-tested functions (gate_ablation_metrics, conflict_flag_rate,
authority_resolution_accuracy, echo_safety_probe, token_economics) on the SAME calibrated
config and fixtures the unit tests use, so the printed numbers are consistent with the
asserted invariants. Pure simulation: hashing encoder + lexical NLI, no models, no network.

Also runs the M-2 merge-classification eval over the labeled gold set IF the
sentence-transformers encoder/NLI are available (the gold MERGE-vs-RELATE distinction needs
the dense encoder); otherwise it reports that the dense rung is unavailable and prints the
hashing-rung classification result with an explicit degeneracy caveat -- it does NOT
fabricate an ROC.

Run:  PYTHONPATH=src .venv/bin/python scripts/run_ablation_metrics.py
"""
from __future__ import annotations

import json

from smp.config import load_thresholds
from smp.agent.run_hotpot_gate import _calibrated_cfg


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def a1_gate_ablation():
    from smp.merge.gate_ablation import gate_ablation_metrics, GATES
    cfg = load_thresholds()
    hi = min(cfg.theta_merge + 0.05, 1.0)
    cases = [
        {"gold": "RELATE", "kappa": 0, "sigma": hi, "chi": "neutral",
         "ctx_ok": False, "retention_below_min": False, "status": "ACTIVE", "stale": False},
        {"gold": "CONFLICT", "kappa": 1, "sigma": hi, "chi": "contradicts",
         "ctx_ok": True, "retention_below_min": False, "status": "ACTIVE", "stale": False},
        {"gold": "REJECT", "kappa": 1, "sigma": hi, "chi": "neutral",
         "ctx_ok": True, "retention_below_min": False, "status": "DEPRECATED", "stale": True},
        {"gold": "MERGE", "kappa": 1, "sigma": hi, "chi": "neutral",
         "ctx_ok": True, "retention_below_min": False, "status": "ACTIVE", "stale": False},
    ]
    m = gate_ablation_metrics(cases, cfg)
    label = {None: "full-gate baseline", "context": "drop Context gate",
             "inference": "drop inference gate", "staleness": "drop staleness gate"}
    print("== A-1 gate ablation (calibrated cfg: load_thresholds) ==")
    for g in GATES:
        print(f"  {label[g]:<22} false_merge={_pct(m[g]['false_merge_rate'])}  "
              f"stale_promotion={_pct(m[g]['stale_promotion_rate'])}")
    return m


def t1_conflict_trust():
    from smp.merge.conflict_trust import conflict_flag_rate, authority_resolution_accuracy
    cfg = load_thresholds()
    hi = min(cfg.theta_merge + 0.05, 1.0)
    contradictions = [
        {"kappa": 1, "sigma": hi, "chi": "contradicts", "ctx_ok": True,
         "retention_below_min": False, "status": "ACTIVE"},
        {"kappa": 0, "sigma": hi, "chi": "contradicts", "ctx_ok": True,
         "retention_below_min": False, "status": "ACTIVE"},
    ]
    cfr = conflict_flag_rate(contradictions, cfg)
    canonical_right = [{"canonical": "fact-A", "peer": "fact-B", "truth": "fact-A",
                        "canonical_is_later": False} for _ in range(5)]
    peer_right = [{"canonical": "fact-A", "peer": "fact-B", "truth": "fact-B",
                   "canonical_is_later": False} for _ in range(5)]
    acc_canon = authority_resolution_accuracy(canonical_right)
    acc_peer = authority_resolution_accuracy(peer_right)
    print("== T-1 conflict-trust ==")
    print(f"  conflict-flag rate: SMP={_pct(cfr['smp_flag_rate'])}  "
          f"auto-merge={_pct(cfr['auto_merge_flag_rate'])}")
    print(f"  authority-resolution accuracy (canonical holds truth): "
          f"SMP={_pct(acc_canon['smp_authority_acc'])}  recency={_pct(acc_canon['recency_acc'])}")
    print(f"  authority-resolution accuracy (peer holds truth):      "
          f"SMP={_pct(acc_peer['smp_authority_acc'])}  recency={_pct(acc_peer['recency_acc'])}")
    return {"flag": cfr, "canonical_right": acc_canon, "peer_right": acc_peer}


def a2_echo_safety():
    from smp.sync.echo_safety import echo_safety_probe
    cfg = _calibrated_cfg("hashing")
    r = echo_safety_probe(encoder="hashing", nli="lexical", cfg=cfg, n_peers=3,
                          tagged_claim="ALIGN-TAG: the canonical policy is X")
    print("== A-2 echo-safety probe (n_peers=3) ==")
    print(f"  peer_echoes={r['peer_echoes']}  origin_echo={r['origin_echo']}  "
          f"invariant_holds={r['invariant_holds']}")
    return r


def a3_token_economics():
    from smp.agent.token_economics import token_economics
    from smp.agent.hotpot_loader import load_hotpot_fixture
    cfg = _calibrated_cfg("hashing")
    task = load_hotpot_fixture(1)[0]
    rows = {}
    for n in (1, 3, 5):
        r = token_economics(task, [task.question] * n, encoder="hashing", nli="lexical",
                            cfg=cfg, k=20)
        rows[n] = r
    print("== A-3 token economics (hashing/lexical, k=20) ==")
    base = rows[1]
    print(f"  live units: cleaned={base['cleaned_live_units']}  "
          f"uncleaned={base['uncleaned_live_units']}")
    for n in (1, 3, 5):
        r = rows[n]
        print(f"  {n} repeated quer{'y' if n == 1 else 'ies'}: "
              f"cleaned={int(r['cleaned_tokens'])} tok  uncleaned={int(r['uncleaned_tokens'])} tok  "
              f"reduction={_pct(r['reduction_rate'])}")
    return rows


def m2_classification():
    from smp.merge.m2_eval import M2_GOLD, evaluate_m2
    from smp.merge.signals import EmbeddingSimilarity
    print("== M-2 merge-classification / ROC ==")
    dense_ok = True
    try:
        EmbeddingSimilarity(encoder="minilm").sigma("a", "b")
    except Exception as e:
        dense_ok = False
        print(f"  dense encoder (minilm) UNAVAILABLE locally: {type(e).__name__}: {e}")
    if not dense_ok:
        # Report what IS available: the hashing-rung classification on the gold set,
        # with an explicit caveat. A real ROC over theta_merge requires the dense rung.
        cfg = _calibrated_cfg("hashing")
        m = evaluate_m2(M2_GOLD, encoder="hashing", nli="lexical", cfg=cfg)
        print("  No dense rung -> NO real ROC reported (hashing cannot separate")
        print("  paraphrase/related/distinct; lexical chi cannot adjudicate contradiction).")
        print(f"  hashing-rung gold-set accuracy (caveated, NOT an ROC): "
              f"macro_f1={m['macro_f1']:.3f}  false_merge_rate={_pct(m['false_merge_rate'])}  "
              f"n={m['n']}")
        print(f"  confusion: {json.dumps(m['confusion'])}")
        return {"dense_available": False, "hashing_gold": m}
    # dense rung present: report the OPERATING-POINT classification at the calibrated
    # theta_merge (the manuscript Section 7-m2 numbers), then sweep for an ROC.
    m = evaluate_m2(M2_GOLD, encoder="minilm", nli="mnli-cross-encoder",
                    cfg=_calibrated_cfg("minilm"))
    print(f"  dense rung (minilm + cross-encoder NLI), {m['n']}-pair gold set, "
          f"calibrated theta_merge")
    print(f"  false_merge_rate={m['false_merge_rate']:.3f} "
          f"({m['n_false_merges']} of {sum(1 for g,_,_ in M2_GOLD if g!='MERGE')} non-mergeable)  "
          f"macro_f1={m['macro_f1']:.3f}")
    for c in ("MERGE", "RELATE", "CONFLICT", "REJECT"):
        pc = m["per_class"][c]
        print(f"    {c:<9} P={pc['precision']:.2f} R={pc['recall']:.2f} "
              f"F1={pc['f1']:.2f}  (support={int(pc['support'])})")
    print(f"  confusion (gold->pred): {json.dumps(m['confusion'])}")

    import numpy as np
    from smp.config import Thresholds
    from smp.merge.classifier import classify
    from smp.merge.signals import chi as chi_fn
    base = _calibrated_cfg("minilm")
    es = EmbeddingSimilarity(encoder="minilm")
    sig = [(g, es.sigma(a, b), chi_fn(a, b, model="mnli-cross-encoder"))
           for g, a, b in M2_GOLD]
    n_pos = sum(1 for g, _, _ in M2_GOLD if g == "MERGE")
    n_neg = len(M2_GOLD) - n_pos
    pts = []
    for th in np.linspace(0.0, 1.0, 51):
        cfg = Thresholds(theta_merge=float(th), sigma_lo=base.sigma_lo,
                         r_min=base.r_min, tau=base.tau)
        tp = fp = 0
        for g, s, c in sig:
            dec = classify(cfg=cfg, kappa=0, sigma=s, chi=c, ctx_ok=True)
            is_merge = dec == "MERGE"
            if is_merge and g == "MERGE":
                tp += 1
            elif is_merge and g != "MERGE":
                fp += 1
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        pts.append((float(th), fpr, tpr))
    # AUC via trapezoid over fpr-sorted unique points
    uniq = sorted({(fpr, tpr) for _, fpr, tpr in pts})
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(uniq, uniq[1:]):
        auc += (x1 - x0) * (y0 + y1) / 2.0
    print(f"  dense rung (minilm + cross-encoder), {len(M2_GOLD)}-pair gold set")
    print(f"  ROC over theta_merge (MERGE-vs-rest), AUC={auc:.3f}")
    for th, fpr, tpr in pts:
        if th in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0) or abs(th - base.theta_merge) < 1e-9:
            print(f"    theta_merge={th:.2f}  FPR={fpr:.3f}  TPR={tpr:.3f}")
    return {"dense_available": True, "auc": auc, "points": pts}


def main():
    a1_gate_ablation()
    print()
    t1_conflict_trust()
    print()
    a2_echo_safety()
    print()
    a3_token_economics()
    print()
    m2_classification()


if __name__ == "__main__":
    main()
