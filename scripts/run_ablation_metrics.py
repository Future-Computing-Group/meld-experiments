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
    # All probes use kappa=0 (distinct claim keys): each gate is exercised on the
    # DANGEROUS distinct-key / high-sigma semantic path it actually guards, not the
    # trivial exact-key MERGE shortcut. Each non-MERGE probe is constructed so that
    # exactly one gate's input blocks the merge -> dropping that gate is necessary
    # AND sufficient to flip it. This is a per-gate necessity check, not an
    # empirical rate (the measured false-merge rate is the M-2 result).
    cases = [
        {"gold": "RELATE", "kappa": 0, "sigma": hi, "chi": "neutral",
         "ctx_ok": False, "retention_below_min": False, "status": "ACTIVE", "stale": False},
        {"gold": "CONFLICT", "kappa": 0, "sigma": hi, "chi": "contradicts",
         "ctx_ok": True, "retention_below_min": False, "status": "ACTIVE", "stale": False},
        {"gold": "REJECT", "kappa": 0, "sigma": hi, "chi": "neutral",
         "ctx_ok": True, "retention_below_min": False, "status": "DEPRECATED", "stale": True},
        {"gold": "MERGE", "kappa": 0, "sigma": hi, "chi": "neutral",
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
    # E1 (AE #2): the enlarged, balanced 104-pair V2 gold set replaces the
    # round-1 16-pair set. Reports the operating point with bootstrap CIs, the
    # theta_merge ROC (open-interval sweep -> no m-E-1 crash), the easy/hard
    # CONFLICT split (the round-1 blind spot, honestly), and the key-only /
    # embedding-only baselines (what each component buys).
    from smp.merge.m2_gold import M2_GOLD_V2, build_gold_set
    from smp.merge.m2_eval import (evaluate_m2, score_pairs, decisions_full,
                                   classification_metrics, roc_over_theta,
                                   bootstrap_metric_ci, evaluate_baseline)
    from smp.merge.signals import EmbeddingSimilarity
    print("== M-2 merge-classification / ROC (V2 gold set) ==")
    dense_ok = True
    try:
        EmbeddingSimilarity(encoder="minilm").sigma("a", "b")
    except Exception as e:
        dense_ok = False
        print(f"  dense encoder (minilm) UNAVAILABLE locally: {type(e).__name__}: {e}")
    if not dense_ok:
        cfg = _calibrated_cfg("hashing")
        m = evaluate_m2(M2_GOLD_V2, encoder="hashing", nli="lexical", cfg=cfg)
        print("  No dense rung -> NO real ROC (hashing cannot separate paraphrase/")
        print("  related/distinct; lexical chi cannot adjudicate contradiction).")
        print(f"  hashing-rung accuracy (caveated, NOT an ROC): macro_f1={m['macro_f1']:.3f} "
              f"false_merge_rate={_pct(m['false_merge_rate'])} n={m['n']}")
        return {"dense_available": False, "hashing_gold": m}

    base = _calibrated_cfg("minilm")
    theta, sigma_lo = base.theta_merge, base.sigma_lo
    rows = build_gold_set()
    scored = score_pairs(M2_GOLD_V2, encoder="minilm", nli="mnli-cross-encoder")
    gold = [g for g, _, _ in scored]
    pred = decisions_full(scored, theta=theta, sigma_lo=sigma_lo)
    m = classification_metrics(gold, pred)
    n_neg = sum(1 for g in gold if g != "MERGE")
    f1_lo, f1_hi = bootstrap_metric_ci(gold, pred, metric="macro_f1", n_boot=2000, seed=0)
    fmr_lo, fmr_hi = bootstrap_metric_ci(gold, pred, metric="false_merge_rate", n_boot=2000, seed=0)
    pts, auc = roc_over_theta(scored, sigma_lo=sigma_lo, n_points=51)

    print(f"  dense rung (minilm + cross-encoder NLI), {m['n']}-pair balanced gold set, "
          f"theta_merge={theta:.3f} sigma_lo={sigma_lo:.3f}")
    print(f"  macro_f1={m['macro_f1']:.3f} [{f1_lo:.3f},{f1_hi:.3f}]  "
          f"false_merge_rate={m['false_merge_rate']:.3f} [{fmr_lo:.3f},{fmr_hi:.3f}] "
          f"({m['n_false_merges']}/{n_neg} non-mergeable)")
    for c in ("MERGE", "RELATE", "CONFLICT", "REJECT"):
        pc = m["per_class"][c]
        print(f"    {c:<9} P={pc['precision']:.2f} R={pc['recall']:.2f} "
              f"F1={pc['f1']:.2f}  (support={int(pc['support'])})")
    # easy/hard CONFLICT recall (the round-1 lexically-divergent blind spot)
    diff_recall = {}
    for strat in ("easy", "hard"):
        idx = [i for i, r in enumerate(rows) if r["difficulty"] == strat and r["label"] == "CONFLICT"]
        rec = sum(1 for i in idx if pred[i] == "CONFLICT") / len(idx) if idx else float("nan")
        diff_recall[strat] = rec
        print(f"    CONFLICT recall [{strat:<4}] = {rec:.2f}  (n={len(idx)})")
    print(f"  ROC over theta_merge (MERGE-vs-rest), AUC={auc:.3f}")
    # baselines
    ko = evaluate_baseline(M2_GOLD_V2, mode="key_only")
    eo = evaluate_baseline(M2_GOLD_V2, mode="embedding_only", encoder="minilm", theta=theta)
    print(f"  baseline key-only:       macro_f1={ko['macro_f1']:.3f} "
          f"MERGE_recall={ko['per_class']['MERGE']['recall']:.2f} FMR={ko['false_merge_rate']:.3f}")
    print(f"  baseline embedding-only: macro_f1={eo['macro_f1']:.3f} "
          f"FMR={eo['false_merge_rate']:.3f} ({eo['n_false_merges']} false merges)")

    result = {"dense_available": True, "n": m["n"], "theta_merge": theta,
              "sigma_lo": sigma_lo, "macro_f1": m["macro_f1"],
              "macro_f1_ci": [f1_lo, f1_hi], "false_merge_rate": m["false_merge_rate"],
              "false_merge_rate_ci": [fmr_lo, fmr_hi], "n_false_merges": m["n_false_merges"],
              "n_non_mergeable": n_neg, "per_class": m["per_class"], "auc": auc,
              "conflict_recall_by_difficulty": diff_recall, "roc_points": pts,
              "baseline_key_only": {"macro_f1": ko["macro_f1"],
                                    "merge_recall": ko["per_class"]["MERGE"]["recall"],
                                    "false_merge_rate": ko["false_merge_rate"]},
              "baseline_embedding_only": {"macro_f1": eo["macro_f1"],
                                          "false_merge_rate": eo["false_merge_rate"],
                                          "n_false_merges": eo["n_false_merges"]}}
    _persist_m2(result)
    return result


def _persist_m2(result):
    """Persist the M-2 V2 result + provenance to gitignored data/m2-eval/."""
    import datetime
    import os
    import platform
    import subprocess
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    outdir = os.path.join(ROOT, "data", "m2-eval")
    os.makedirs(outdir, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                      text=True).strip()
    except Exception:
        sha = "unknown"
    prov = {"git_sha": sha, "python": platform.python_version(),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "encoder": "all-MiniLM-L6-v2", "nli": "cross-encoder/nli-deberta-v3 (mnli)"}
    payload = {"provenance": prov, "result": result}
    path = os.path.join(outdir, "m2_v2_dense.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  [persisted] {path}")


def e2_perturbation():
    # E2 (AE #4): exercise the adaptive mechanisms under perturbation.
    import datetime
    import os as _os
    from smp.eval.crdt_convergence import partition_reorder_sweep
    print("== E2(a) partition-duration x reorder-rate (status-CRDT vs LWW) ==")
    cells = partition_reorder_sweep(durations=[2, 4, 8, 16],
                                    reorder_rates=[0.0, 0.25, 0.5, 0.75, 1.0],
                                    n_claims=6, n_orderings=200, seed=0)
    crdt_all_one = all(c["crdt_modal_agreement"] == 1.0 for c in cells)
    lww_min = min(c["lww_modal_agreement"] for c in cells)
    lww_maxd = max(c["lww_n_distinct"] for c in cells)
    print(f"  {len(cells)} cells: CRDT modal-agreement==1.0 everywhere: {crdt_all_one} "
          f"(n_distinct=1); LWW min agreement={lww_min:.3f}, up to {lww_maxd} distinct outcomes")

    robustness = None
    try:
        from smp.merge.signals import EmbeddingSimilarity
        EmbeddingSimilarity(encoder="minilm").sigma("a", "b")
        from smp.merge.m2_gold import M2_GOLD_V2
        from smp.merge.m2_eval import score_pairs
        from smp.eval.robustness import robustness_sweep
        base = _calibrated_cfg("minilm")
        scored = score_pairs(M2_GOLD_V2, encoder="minilm", nli="mnli-cross-encoder")
        robustness = robustness_sweep(scored, theta=base.theta_merge, sigma_lo=base.sigma_lo,
                                      noise_sds=[0.0, 0.05, 0.10, 0.20, 0.30],
                                      flip_rates=[0.0, 0.10, 0.25, 0.50], n_trials=50, seed=0)
        print("== E2(b) decision-procedure robustness (dense rung, V2 gold) ==")
        print("  embedding-noise sd -> macro_f1 / FMR: " + "  ".join(
            f"{sd:.2f}:{m['macro_f1']:.3f}/{m['false_merge_rate']:.3f}"
            for sd, m in robustness["noise"].items()))
        print("  degraded-NLI flip  -> macro_f1 / FMR: " + "  ".join(
            f"{fr:.2f}:{m['macro_f1']:.3f}/{m['false_merge_rate']:.3f}"
            for fr, m in robustness["flip"].items()))
    except Exception as e:
        print(f"  E2(b) dense rung unavailable: {type(e).__name__}: {e}")

    HERE = _os.path.dirname(_os.path.abspath(__file__))
    outdir = _os.path.join(_os.path.dirname(HERE), "data", "e2-sweeps")
    _os.makedirs(outdir, exist_ok=True)
    payload = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "partition_reorder_sweep": cells,
               "crdt_modal_agreement_all_one": crdt_all_one,
               "lww_min_agreement": lww_min, "lww_max_distinct": lww_maxd,
               "robustness": robustness}
    with open(_os.path.join(outdir, "e2_sweeps.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  [persisted] {_os.path.join(outdir, 'e2_sweeps.json')}")
    return payload


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
    print()
    e2_perturbation()


if __name__ == "__main__":
    main()
