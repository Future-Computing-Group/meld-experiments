"""Local smoke: full SIM pipeline to completion + a Tufte figure (G2).

Runs M-1 (distributed vs centralized vs naive-union) and M-2 (classification
quality) on a small LongMemEval-shaped slice with real local hashing embeddings,
then renders a two-panel small-multiples figure:
  (left)  M-2 per-class confusion mini-matrix (numbers are the data-ink)
  (right) M-1 distributed-vs-centralized accuracy (grouped bars, direct labels)

Tufte discipline: grayscale-first, no gridlines, no chart borders, no legend
(direct labels), every element earns its ink. Real outputs only -- no fabricated
numbers.
"""
from __future__ import annotations

import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # headless, no GUI (CLI-only rendering)
import matplotlib.pyplot as plt  # noqa: E402

from meld.config import Thresholds, load_thresholds  # noqa: E402
from meld.eval.run_meld_eval import calibrate_and_write, run_m1, run_m2  # noqa: E402

_LABELS = ("coreferent", "contradiction", "unrelated")
_SHORT = {"coreferent": "coref", "contradiction": "contra", "unrelated": "unrel"}


def run_smoke(out_dir: str, n_conversations: int = 8, n_agents: int = 3,
              encoder: str = "hashing", nli: str = "lexical") -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    # M-2 calibrates theta_merge on the gold pairs (writes to a gitignored
    # config), then reports the calibrated classification quality.
    cal = calibrate_and_write(n_conversations, encoder, nli)
    base = load_thresholds()
    cfg = Thresholds(theta_merge=cal["theta_merge"], sigma_lo=base.sigma_lo,
                     r_min=base.r_min, tau=base.tau)
    m2 = run_m2(n_conversations, encoder, nli, cfg=cfg)
    m2["calibrated_theta_merge"] = cal["theta_merge"]
    m1 = run_m1(n_conversations, n_agents, encoder, nli)
    figure = os.path.join(out_dir, "smoke_results.png")
    _render_figure(m1, m2, figure)
    return {"m1": m1, "m2": m2, "figure": figure}


def _render_figure(m1: Dict, m2: Dict, path: str) -> None:
    fig, (axc, axb) = plt.subplots(1, 2, figsize=(9, 3.6))

    # ---- left: M-2 confusion mini-matrix (numbers = data-ink) ----
    conf = m2["confusion"]
    n = len(_LABELS)
    # light shading proportional to count, but the NUMBER is the primary ink
    maxc = max((conf[g][p] for g in _LABELS for p in _LABELS), default=1) or 1
    for i, g in enumerate(_LABELS):           # rows = true
        for j, p in enumerate(_LABELS):       # cols = predicted
            c = conf[g][p]
            shade = 0.12 + 0.5 * (c / maxc) if c else 0.0
            if shade:
                axc.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1,
                                            color=str(1 - shade), zorder=0))
            axc.text(j + 0.5, n - 1 - i + 0.5, str(c), ha="center", va="center",
                     color=("white" if shade > 0.4 else "black"), zorder=1,
                     fontsize=11)
    axc.set_xlim(0, n)
    axc.set_ylim(0, n)
    axc.set_xticks([k + 0.5 for k in range(n)])
    axc.set_yticks([k + 0.5 for k in range(n)])
    axc.set_xticklabels([_SHORT[p] for p in _LABELS], fontsize=8)
    axc.set_yticklabels([_SHORT[g] for g in reversed(_LABELS)], fontsize=8)
    axc.set_xlabel("predicted", fontsize=9)
    axc.set_ylabel("true", fontsize=9)
    axc.set_title(f"M-2 confusion  (false-MERGE rate {m2['false_merge_rate']:.2f})",
                  fontsize=9)
    axc.tick_params(length=0)
    for s in axc.spines.values():
        s.set_visible(False)

    # ---- right: M-1 accuracy grouped bars (direct labels, no legend/grid) ----
    names = ["distributed", "centralized"]
    vals = [m1["distributed_accuracy"], m1["centralized_accuracy"]]
    xs = range(len(names))
    bars = axb.bar(xs, vals, width=0.55, color=["0.25", "0.6"])
    for x, v in zip(xs, vals):
        axb.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    axb.set_xticks(list(xs))
    axb.set_xticklabels(names, fontsize=9)
    axb.set_ylim(0, 1.08)
    axb.set_ylabel("answer accuracy", fontsize=9)
    axb.set_title(f"M-1 merge vs centralized  "
                  f"(units {m1['distributed_units']} vs naive-union "
                  f"{m1['naive_union_units']})", fontsize=9)
    axb.tick_params(length=0)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)
    axb.set_yticks([0, 0.5, 1.0])

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


__all__ = ["run_smoke"]
