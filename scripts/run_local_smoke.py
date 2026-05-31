"""Rung-2 LOCAL smoke: full SIM pipeline to completion + a rendered figure.

Usage:  python3 scripts/run_local_smoke.py
Local, laptop-scale, CPU. NO GPU, NO remote, NO billable compute. Real merge
engine, real local hashing embeddings, small LongMemEval-shaped slice. After it
runs, VISUALLY INSPECT the rendered PNG under figures/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from smp.eval.smoke import run_smoke  # noqa: E402

FIG_DIR = os.path.join(ROOT, "figures")

if __name__ == "__main__":
    res = run_smoke(out_dir=FIG_DIR, n_conversations=12, n_agents=3)
    m1, m2 = res["m1"], res["m2"]
    print("== M-1 (distributed vs centralized vs naive-union) ==")
    print(f"  distributed accuracy : {m1['distributed_accuracy']:.3f}")
    print(f"  centralized accuracy : {m1['centralized_accuracy']:.3f}")
    print(f"  distributed units    : {m1['distributed_units']}")
    print(f"  naive-union units    : {m1['naive_union_units']}")
    print(f"  conflicts represented: {m1['n_conflicts']}")
    print("== M-2 (classification quality) ==")
    print(f"  false-MERGE rate     : {m2['false_merge_rate']:.3f}")
    for cls, m in m2["per_class"].items():
        print(f"  {cls:<13} P={m['precision']:.2f} R={m['recall']:.2f} "
              f"F1={m['f1']:.2f} (n={m['support']})")
    print(f"figure: {res['figure']}")
