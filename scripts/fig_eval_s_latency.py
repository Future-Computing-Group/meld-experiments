"""S-1 sync-latency figure: p50 publish-to-merge latency vs added one-way RTT.

The figure IS the claim (manuscript Section 7.4): semantic routing's p50 stays
low and fans DOWN as fan-out N grows (each publish reaches ~1 needed peer), while
broadcast rises with N and is non-monotone under WAN (the N-inversion). Six lines:
{broadcast, semantic} x {N=2,3,4}. Error bars are 95% bootstrap CIs.

Data: multi-host testbed, HotpotQA workload, 15 seeds/cell, 360 cells, 0 failures.
Numbers transcribed verbatim from the campaign result tables (do not recompute).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RTT = [0, 10, 25, 50]

# mode -> N -> (p50 [ms], ci_lo, ci_hi) per RTT tier, verbatim from spec.
DATA = {
    "Broadcast": {
        2: [(36.0, 32.8, 42.2), (74.6, 74.3, 75.0), (134.8, 134.5, 135.1), (234.7, 234.5, 235.0)],
        3: [(57.1, 56.7, 57.7), (76.7, 76.4, 77.1), (108.1, 107.4, 108.8), (157.3, 156.7, 158.0)],
        4: [(80.0, 79.5, 80.5), (100.9, 100.3, 101.5), (130.8, 130.2, 131.4), (179.2, 178.6, 179.9)],
    },
    "Semantic": {
        2: [(10.4, 10.3, 10.6), (52.4, 52.2, 52.6), (111.8, 111.7, 111.9), (212.4, 212.2, 212.7)],
        3: [(8.4, 8.2, 8.7), (20.1, 16.3, 23.8), (36.1, 24.2, 47.9), (62.6, 37.3, 87.9)],
        4: [(7.2, 7.1, 7.4), (8.8, 8.6, 9.1), (9.1, 8.8, 9.6), (9.0, 8.8, 9.2)],
    },
}

# colourblind-safe: broadcast = vermillion, semantic = blue (Okabe-Ito).
COL = {"Broadcast": "#D55E00", "Semantic": "#0072B2"}
MARK = {2: "o", 3: "s", 4: "^"}

fig, ax = plt.subplots(figsize=(5.4, 3.9))

for mode, by_n in DATA.items():
    ls = "--" if mode == "Broadcast" else "-"
    for n in (2, 3, 4):
        y = [v[0] for v in by_n[n]]
        lo = [v[0] - v[1] for v in by_n[n]]
        hi = [v[2] - v[0] for v in by_n[n]]
        ax.errorbar(
            RTT, y, yerr=[lo, hi], color=COL[mode], linestyle=ls,
            marker=MARK[n], markersize=5, linewidth=1.4, capsize=2.5,
            label=f"{mode}, N={n}", zorder=3,
        )

ax.set_xlabel("added one-way RTT (ms)", fontsize=9)
ax.set_ylabel("p50 publish-to-merge latency (ms, lower better)", fontsize=9)
ax.set_xticks(RTT)
ax.tick_params(labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.margins(x=0.04)

# legend outside the data, to the right.
ax.legend(fontsize=7.5, frameon=False, loc="center left",
          bbox_to_anchor=(1.01, 0.5), handlelength=2.4, title="routing $\\times$ fan-out",
          title_fontsize=8)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"figures/eval-s-latency.{ext}", dpi=200, bbox_inches="tight")
print("wrote figures/eval-s-latency.png / .pdf")
