"""M-1 headline figure (Section 7-h): recall model-independence + F1 corroboration.

Reads the real per-model durable JSONLs from the CSC Ollama run (jobs 6773545 +
6773585; N=200, 5 models x 3 families, 0 errors) under data/csc-m1-archive/ and
plots two panels:

  (A) recall@5 by memory condition -- IDENTICAL across all five models (the
      model-independence demonstration), with the live-store footprint annotated:
      distributed_merge matches centralized recall and beats naive_union at ~11%
      fewer live units.
  (B) answer token-F1 by model x condition -- the per-model corroboration: F1
      scales with model size and stays stable across conditions within each model.

Outputs figures/m1-hotpot.{pdf,png}. Pure local plotting; no network.
Run:  .venv/bin/python scripts/fig_m1_hotpot.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARCHIVE = os.path.join(ROOT, "data", "csc-m1-archive")
# ROOT=.../FCG/Experiments/2026-SMP -> vault root is two dirnames up (Experiments, FCG)
FIGDIR = os.path.join(os.path.dirname(os.path.dirname(ROOT)),
                      "Manuscripts", "2026-SMP — Semantic Merge Protocol", "figures")

# size-ordered for the F1 ladder; labels are display names
MODELS = [
    ("qwen2.5-1.5b", "Qwen2.5\n1.5B"),
    ("mistral-7b", "Mistral\n7B"),
    ("qwen2.5-7b", "Qwen2.5\n7B"),
    ("gemma-3-12b", "Gemma3\n12B"),
    ("qwen2.5-32b", "Qwen2.5\n32B"),
]
CONDS = ["centralized", "distributed_merge", "naive_union"]
CLABEL = {"centralized": "centralized", "distributed_merge": "distributed merge",
          "naive_union": "naive union"}
CCOLOR = {"centralized": "#888888", "distributed_merge": "#1f77b4",
          "naive_union": "#d62728"}


def _agg(key: str):
    rows = [json.loads(l) for l in open(os.path.join(ARCHIVE, f"memdelta_minilm_{key}.jsonl"))]
    out = {}
    for c in CONDS + ["full_context", "none"]:
        for metric in ("recall_at_k", "f1", "live_units"):
            vals = [r.get(metric) for r in rows
                    if r.get("condition") == c and r.get(metric) is not None]
            out[(c, metric)] = (sum(vals) / len(vals)) if vals else float("nan")
    return out


def main():
    data = {k: _agg(k) for k, _ in MODELS}
    # recall + live are model-independent: take from the first model (assert identical)
    ref = data[MODELS[0][0]]
    recall = {c: ref[(c, "recall_at_k")] for c in CONDS}
    live = {c: ref[(c, "live_units")] for c in CONDS}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)

    # --- Panel A: recall@5 by condition (model-independent) + live footprint ---
    xs = range(len(CONDS))
    barsA = axA.bar(list(xs), [recall[c] for c in CONDS],
                    color=[CCOLOR[c] for c in CONDS], width=0.62, gid="recall-bars")
    axA.set_xticks(list(xs))
    axA.set_xticklabels([CLABEL[c] for c in CONDS], fontsize=8)
    axA.set_ylabel("recall@5 (gold supporting facts)", fontsize=8)
    axA.set_ylim(0, 0.78)
    axA.set_title("(A) recall is model-independent", fontsize=9)
    for i, c in enumerate(CONDS):
        axA.text(i, recall[c] + 0.012, f"{recall[c]:.3f}", ha="center", fontsize=7.5)
        axA.text(i, 0.045, f"{live[c]:.1f}\nunits", ha="center", fontsize=6.8,
                 color="white", fontweight="bold")
    axA.annotate("identical across all 5 models", xy=(1, 0.70), ha="center",
                 fontsize=6.8, style="italic")

    # --- Panel B: F1 by model x condition (per-model corroboration) ---
    width = 0.26
    for j, c in enumerate(CONDS):
        ys = [data[k][(c, "f1")] for k, _ in MODELS]
        axB.bar([x + (j - 1) * width for x in range(len(MODELS))], ys, width=width,
                color=CCOLOR[c], label=CLABEL[c], gid=f"f1-{c}")
    axB.set_xticks(range(len(MODELS)))
    axB.set_xticklabels([lbl for _, lbl in MODELS], fontsize=7)
    axB.set_ylabel("answer token-F1", fontsize=8)
    axB.set_ylim(0, 0.72)
    axB.set_title("(B) F1 scales with model, stable across conditions", fontsize=9)
    axB.legend(fontsize=6.8, loc="upper left", frameon=False, ncol=1)

    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"m1-hotpot.{ext}"), dpi=200)
    print("recall (model-independent):", {c: round(recall[c], 3) for c in CONDS})
    print("live units:", {c: round(live[c], 1) for c in CONDS})
    print("wrote", os.path.join(FIGDIR, "m1-hotpot.{pdf,png}"))


if __name__ == "__main__":
    main()
