#!/usr/bin/env python3
"""Small-multiples scaling figure: the three SMP scaling measurements side by side.

(a) message-rate  -> sink throughput knee (M1, S-2);
(b) cluster size  -> per-claim sync overhead (S-3): semantic O(1) vs broadcast O(N);
(c) cluster size  -> sink state-load (S-4): merge O(1) vs union O(N).

Tufte small multiples: shared visual language, no top/right spines, no chartjunk grid,
direct line labels (no legend), one accent for the SMP line and grey for the baseline.
Reads the run JSONs in outputs/. Writes figures/scaling.{pdf,png}.

Run:  PYTHONPATH=src .venv/bin/python scripts/fig_scaling.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(REPO, "outputs")
FIGS = os.path.join(REPO, "figures")
os.makedirs(FIGS, exist_ok=True)

ACCENT = "#1f5c8a"   # SMP line (semantic / merge / achieved)
BASE = "#9a9a9a"     # baseline (broadcast / union / ideal)

plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.6,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})


def _load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as fh:
        return json.load(fh)


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    m1 = _load("m1-throughput.json")
    s3 = _load("scale-sweep.json")
    s4 = _load("sink-load.json")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(9.6, 3.0))

    # --- (a) message rate -> throughput knee ---
    curve = m1["curve"]
    rates = sorted(float(r) for r in curve)
    achieved = []  # JSON keys are raw strings; match by value
    for r in rates:
        k = next(k for k in curve if abs(float(k) - r) < 1e-6)
        achieved.append(curve[k]["merged_per_s"])
    knee = m1.get("saturation_knee_merges_per_s", max(achieved))
    axA.plot(rates, rates, color=BASE, lw=0.9, ls=(0, (4, 3)))      # ideal y=x
    axA.plot(rates, achieved, color=ACCENT, lw=1.6, marker="o", ms=3)
    axA.axhline(knee, color=ACCENT, lw=0.6, ls=":")
    axA.text(rates[-1], knee + 2, f"knee {knee:.0f}/s", color=ACCENT, fontsize=7,
             ha="right", va="bottom")
    axA.text(rates[1], rates[1] + 6, "ideal (track offer)", color=BASE, fontsize=7,
             rotation=32, ha="left", va="bottom")
    axA.set_xlabel("offered claim rate (claims/s)")
    axA.set_ylabel("sink throughput (merges/s)")
    axA.set_title("(a) message rate", fontsize=9)
    _despine(axA)

    # --- (b) cluster size -> per-claim sync overhead (log-log) ---
    Ns = [r["n_brains"] for r in s3["rows"]]
    sem = [r["deliv_per_claim_semantic"] for r in s3["rows"]]
    bc = [r["deliv_per_claim_broadcast"] for r in s3["rows"]]
    axB.loglog(Ns, bc, color=BASE, lw=1.6, marker="s", ms=3)
    axB.loglog(Ns, sem, color=ACCENT, lw=1.6, marker="o", ms=3)
    axB.text(Ns[-1], bc[-1], " broadcast\n O(N)", color=BASE, fontsize=7, va="center")
    axB.text(Ns[-1], sem[-1] * 1.25, " semantic O(1)", color=ACCENT, fontsize=7, va="bottom",
             ha="right")
    axB.set_xlabel("cluster size N (brains)")
    axB.set_ylabel("deliveries per claim")
    axB.set_title("(b) sync overhead vs nodes", fontsize=9)
    _despine(axB)

    # --- (c) cluster size -> sink state-load (log-log) ---
    Ns4 = [r["n"] for r in s4["rows"]]
    merge = [r["merge_live"] for r in s4["rows"]]
    union = [r["union_live"] for r in s4["rows"]]
    c = s4["n_corpus"]
    axC.loglog(Ns4, union, color=BASE, lw=1.6, marker="s", ms=3)
    axC.loglog(Ns4, merge, color=ACCENT, lw=1.6, marker="o", ms=3)
    axC.text(Ns4[-1], union[-1], " union\n O(N)", color=BASE, fontsize=7, va="center")
    axC.text(Ns4[-1], merge[-1] * 1.3, f" merge O(1) (={c})", color=ACCENT, fontsize=7,
             va="bottom", ha="right")
    axC.set_xlabel("cluster size N (re-publishing brains)")
    axC.set_ylabel("sink live units")
    axC.set_title("(c) state load vs nodes", fontsize=9)
    _despine(axC)

    fig.tight_layout(w_pad=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"scaling.{ext}"), dpi=200, bbox_inches="tight")
    print("wrote figures/scaling.pdf and .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
