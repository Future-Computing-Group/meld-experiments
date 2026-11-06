"""PH-10: E2-P2 calibration burden m* under the card's criterion (meld.eval.wp6e), evaluation only, local.

For each encoder x domain cell of the E2-P2 run (outputs/wp6-results/wp6b/p2/a0.02/<enc>-<domain>/strict/rows.jsonl)
the test streams are rebuilt as run_calibration_burden.py builds them (portability.base_for, same labels, checked
by hash against the run's stamp) and folded once without an epoch. Every tested pair with a label is kept as
(bin, s, blocked, mergeable); selection, s and the theta-independent gates do not depend on theta, so each draw's
thresholds (from its row) are replayed on these pairs (wp6e.replay_counts; group gates not replayed, an upper bound
on merges). Checks: per-bin mergeable / non-mergeable counts equal the run's, and the replayed counts of every draw
are compared with the run's realised counts (max macro-F1 difference reported). The oracle is the per-bin threshold
vector with the best replayed macro-F1 on the test pairs (unconstrained, the card's; and under FM <= alpha as a
sensitivity). m* = smallest m with >= 90 % of 200 draws within 0.02 macro-F1 and +0.01 FM of the oracle. The proxy
m* (FM <= full + 0.01, recall >= full - 0.02 from the run's realised counts) is recomputed beside it.

Run (repo root):  OMP_NUM_THREADS=2 .venv/bin/python -I scripts/run_wp6e_burden_card.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from meld.eval import portability as P  # noqa: E402
from meld.eval import wp6e as W  # noqa: E402
from meld.eval.label_queue import resolve  # noqa: E402
from meld.merge.conformal import bin_of  # noqa: E402
from meld.protocol.fold import readjudicate  # noqa: E402
from meld.protocol.objects import FIXED_ONE  # noqa: E402

PJ = lambda *p: os.path.join(ROOT, *p)  # noqa: E731
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
ENC_ID = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
          "mpnet": "sentence-transformers/all-mpnet-base-v2"}


def tested_pairs(ctx, base, domain):
    """(bin, s_q, blocked, mergeable) of every labelled tested pair of the epoch-free fold, as evaluate scores them."""
    out = []
    for b in base["chunks"]:
        store = P._clone(b["store"], base["scope"])
        res = readjudicate(store, base["scope"], m_nli=P._m_nli(base["encoder"], base["nli"]))
        text = {d.claim_key: d.text for _, d in store.nodes().values()}
        for d in res.decisions:
            if d.tested is None:
                continue
            a, t = d.tested
            _, mg, _ = resolve(b["pair"].get(frozenset(d.tested)), text[a], text[t], b["item"][a], b["item"][t],
                               source=domain, source_label=ctx["lookup"], annotations=ctx["ann"])
            if mg is not None:
                out.append({"bin": bin_of(d.n_prior, ctx["a"].B0), "s_q": d.s_q, "mergeable": mg,
                            "blocked": any(g in ("mask", "pair_mask") for g in d.gates)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    P.common_args(ap)
    next(x for x in ap._actions if x.dest == "out_dir").required = False     # default below
    ap.add_argument("--p2", default=PJ("outputs/wp6-results/wp6b/p2/a0.02"))
    ap.add_argument("--cells", nargs="+", default=[f"{e}-{d}" for e in ENC_ID for d in ("paws", "sick", "vitaminc")])
    ap.set_defaults(key_file=PJ("outputs/calibration-labelled/key"),
                    annotations=PJ("outputs/annotate/combined-annotations-final.jsonl"),
                    labels=PJ("outputs/annotate/labels-final.jsonl"), out_dir=PJ("outputs/wp6e/ph10"))
    a = ap.parse_args(argv)
    ctx = P.context(a)
    git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    run = {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
           "nli_id": P.NLI_IDS["mnli-cross-encoder"], "hw": ctx["hw"], "omp": os.environ.get("OMP_NUM_THREADS"),
           "criterion": "card (planning/06-e2-evaluation.md E2-P2): macro-F1 >= oracle - 0.02 and FM <= oracle FM + 0.01, "
                        ">= 90% of draws; oracle = best replayed per-bin thresholds on the test pairs",
           "macro_f1": "mean F1 of MERGE and no-merge over labelled tested pairs (unresolved/excluded left out)",
           "replay": "MERGE iff s > theta(bin) and no mask/pair_mask gate; group gates not replayed (upper bound)",
           "m_unit": "Cal (non-mergeable) records per bin, as E2-P2; not the card's class-stratified labelled pairs"}
    base_prov = P.provider("mnli-cross-encoder")
    memo = {}
    prov = lambda p, h: memo[p, h] if (p, h) in memo else memo.setdefault((p, h), base_prov(p, h))  # noqa: E731
    os.makedirs(a.out_dir, exist_ok=True)
    out, lines = [], []
    for cell in a.cells:
        t0 = time.monotonic()
        enc, dom = cell.rsplit("-", 1)
        f = os.path.join(a.p2, cell, "strict", "rows.jsonl")
        rows = [json.loads(x) for x in open(f)]
        st = rows[0]
        for k in ("manifest_sha256", "annotations_sha256", "labels_sha256", "B_0", "seed"):
            assert st[k] == ctx["stamp"][k], (cell, k)
        assert st["encoder"] == enc and st["test_domain"] == dom
        base = P.base_for(ctx, dom, encoder=enc, nli="mnli-cross-encoder", prov=prov)
        pairs = tested_pairs(ctx, base, dom)
        full = next(r for r in rows if r["m"] == "all")
        for g, c in full["by_bin"].items():
            mine = [p for p in pairs if p["bin"] == int(g)]
            assert (sum(p["mergeable"] for p in mine), sum(not p["mergeable"] for p in mine)) == \
                (c["mergeable"], c["non_mergeable"]), (cell, g)
        draws, exact_gap, proxy = {}, 0.0, {}
        for r in rows:
            th = {int(g): round(t * FIXED_ONE) for g, t in r["theta"].items() if t is not None}
            tp, fp, fn, tn = W.replay_counts(pairs, th)
            ex = (r["merged_mergeable"], r["false_merges"], r["mergeable"] - r["merged_mergeable"],
                  r["non_mergeable"] - r["false_merges"])
            exact_gap = max(exact_gap, abs(W.macro_f1(tp, fp, fn, tn) - W.macro_f1(*ex)))
            if r["m"] == "all":
                full_rep = {"theta_q": th, "macro_f1": W.macro_f1(tp, fp, fn, tn), "fm": fp / (fp + tn),
                            "recall": tp / (tp + fn), "macro_f1_exact": W.macro_f1(*ex)}
                continue
            draws.setdefault(r["m"], []).append((W.macro_f1(tp, fp, fn, tn), fp / (fp + tn) if fp + tn else None))
            proxy.setdefault(r["m"], []).append((r["fm"], r["recall"]))
        prox = {m: sum(fm <= full["fm"] + 0.01 + 1e-12 and rc >= full["recall"] - 0.02 - 1e-12 for fm, rc in v) / len(v)
                for m, v in sorted(proxy.items())}
        row = {"cell": cell, "encoder": enc, "encoder_id": ENC_ID[enc], "domain": dom, "alpha": st["alpha"],
               "p2_rows_sha256": sha(f), "p2_git_sha": st["git_sha"], "p2_hw": st["hw"], "pairs": len(pairs),
               "mergeable": sum(p["mergeable"] for p in pairs), "full_draw": full_rep,
               "max_abs_macro_f1_replay_minus_realised": exact_gap,
               "proxy_share": prox, "proxy_m_star": next((m for m, s in prox.items() if s >= 0.9 - 1e-12), None)}
        for name, cap in (("oracle", None), ("oracle_fm_le_alpha", st["alpha"])):
            th, (tp, fp, fn, tn), f1 = W.oracle(pairs, max_fm=cap)
            card = W.card_mstar(draws, oracle_f1=f1, oracle_fm=fp / (fp + tn))
            mean = {m: sum(x for x, _ in v) / len(v) for m, v in sorted(draws.items())}
            row[name] = {"theta_q": th, "macro_f1": f1, "fm": fp / (fp + tn), "recall": tp / (tp + fn),
                         "mean_draw_macro_f1": mean, **card}
        out.append(row)
        o, c = row["oracle"], row["oracle_fm_le_alpha"]
        lines.append(f"{cell:<20}{len(pairs):>6}{full_rep['macro_f1']:>8.3f}{o['macro_f1']:>8.3f}{o['fm']:>7.3f}"
                     f"{str(o['m_star']):>7}{c['macro_f1']:>8.3f}{c['fm']:>7.3f}{str(c['m_star']):>7}"
                     f"{str(row['proxy_m_star']):>7}{exact_gap:>8.4f}")
        print(lines[-1], f"{time.monotonic() - t0:.0f}s nli-memo {len(memo)}", flush=True)
        with open(os.path.join(a.out_dir, "card-mstar.json"), "w") as fh:
            json.dump({"run": run, "rows": out}, fh, indent=1)
    head = ("PH-10 E2-P2 burden m* under the card criterion (scripts/run_wp6e_burden_card.py; meld.eval.wp6e)\n"
            f"git {run['git_sha']} dirty {run['git_dirty']}  NLI {run['nli_id']}  hw {run['hw']}  alpha 0.02; "
            "P2 source rows sha256 per cell in card-mstar.json\n"
            "fullF1 = full-draw replayed macro-F1; oF1/oFM/m* = unconstrained oracle; cF1/cFM/c-m* = oracle under FM <= alpha; "
            "proxy = the digest's proxy m*; gap = max |replayed - realised| macro-F1 over draws\n\n"
            f"{'cell':<20}{'pairs':>6}{'fullF1':>8}{'oF1':>8}{'oFM':>7}{'m*':>7}{'cF1':>8}{'cFM':>7}{'c-m*':>7}"
            f"{'proxy':>7}{'gap':>8}\n")
    with open(os.path.join(a.out_dir, "card-mstar.txt"), "w") as fh:
        fh.write(head + "\n".join(lines) + "\n")
    print(head + "\n".join(lines))


if __name__ == "__main__":
    main()
