"""WP6g: multi-target admission counterfactual on the E2-B3 benchmark streams, evaluation only, local.

Per domain (PAWS, SICK, VitaminC) the cal stream and the test streams are rebuilt exactly as E2-B3
(calibration.stream; benchmark_test.chunk_rows with the cal stream's node count; MiniLM int8,
nli-deberta-v3-small, seed 0) and the step is replayed by meld.eval.wp6g.vfold in three modes: single (the
released step; must reproduce benchmark_test.evaluate under the released epoch), first-pass and best-pass over
the eligible candidates of the top-m K(b), m in {3, 5, 10}. Two threshold arms per (alpha, m, mode):
  a  the released per-bin table (the m = 3 single-target SEED epoch) applied unchanged;
  b  the table recalibrated on the cal stream under the same variant: the cal stream folded without an epoch
     (theta INF16, so no merge and every eligible candidate is tested), one non-mergeable tested pair per
     source item of the arrival drawn with the released seed rule, conformal.build_epoch per bin (dev key,
     evaluation only; Cal records never leave this run). Check: single, m = 3 rebuilds the released table.
Labels as E2-B3 / PH-13 (label_queue.resolve: manifest, raw source, annotators; AMBIGUOUS non-mergeable).
Unlabelled tested pairs are never invented: counted, left out of the rates and never Cal records; merges into
them are counted, and ``fm_worst_unlabelled`` is the worst case over their labels (every merged one
non-mergeable, every unmerged one mergeable): (k + k_u) / (n + k_u).

Metrics per row (tested-pair units, B.metrics, Clopper-Pearson 95%): false merge among non-mergeable tested
pairs, per-bin false merge (max over bins with n > 0), merge recall over mergeable tested pairs; over the
labelled mergeable manifest pairs of the test streams: target recall (the later arrival's merge partner is the
earlier text) and group recall (the two end in one group, PH-13's complement of not_merged_with_target); wrong-
target merges (an arrival with a mergeable manifest partner merged with another node); NLI directions evaluated
per delta and tested pairs per delta.

NLI: the released path's pairs come from single calls (the provider the release uses, memoised); the remaining
directions of the m = 10 read set are prefetched in length-sorted batches of the same cross-encoder (the max
fixed-point difference against single calls on a sample is recorded in the run header).

Run (repo root):  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/python -I scripts/run_wp6g_multi_target.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from meld.agent.memory import DEFAULT_POL  # noqa: E402
from meld.config import read_epoch  # noqa: E402
from meld.eval import benchmark_test as B  # noqa: E402
from meld.eval import calibration as C  # noqa: E402
from meld.eval import label_queue as Q  # noqa: E402
from meld.eval import wp6g as G  # noqa: E402
from meld.eval.topology import stamps  # noqa: E402
from meld.merge import signals  # noqa: E402
from meld.merge.classifier import select_tested_pair  # noqa: E402
from meld.merge.conformal import bin_of, build_epoch, threshold  # noqa: E402
from meld.merge.signals import sigma_q_cmp, to_fixed16  # noqa: E402
from meld.proto import CANON_VERSION  # noqa: E402
from meld.protocol import fold as F  # noqa: E402
from meld.protocol.objects import ALPHA_ONE, INF16, CalRecord, canonical_bytes, m_nli, m_pol_of, object_id  # noqa: E402

EPOCH_DIR = {"paws": "a{a}", "sick": "a{a}", "vitaminc": "humanS3-a{a}"}
ENC, NLI = "minilm", "mnli-cross-encoder"
PJ = lambda *p: os.path.join(ROOT, *p)  # noqa: E731
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
MN = m_nli(ENC, NLI, "int8", CANON_VERSION)


class World:
    """One stream: store, item and pair maps, nodes, rank, texts, labels of any pair (memoised)."""

    def __init__(self, rows, d, sel, key, lookup, ann, verdicts):
        self.store, self.pol, self.item, self.pair = C.stream(rows, scope=d, encoder=ENC, selector=sel, key=key, seed=0)
        self.nodes, self.rank = F._nodes(self.store, d)
        self.text = {n.key: n.text for n in self.nodes}
        self.raw = {x.claim_key: x.text for _, x in self.store.nodes().values()}
        self.d, self.lookup, self.ann, self.verdicts, self._lab = d, lookup, ann, verdicts, {}

    def label(self, x, y):
        """(class, mergeable, provenance, verdict, manifest row) of the pair, as benchmark_test.units."""
        k = frozenset((x, y))
        if k not in self._lab:
            r = self.pair.get(k)
            a, b = sorted((x, y))
            cls, mg, prov = Q.resolve(r, self.raw[a], self.raw[b], self.item[a], self.item[b], source=self.d,
                                      source_label=self.lookup, annotations=self.ann)
            v = B.verdict_of(r, self.raw[a], self.raw[b], self.item[a], self.item[b], source=self.d, mergeable=mg,
                             prov=prov, verdicts=self.verdicts)
            self._lab[k] = (cls, mg, prov, v, r)
        return self._lab[k]

    def ctx(self, provider, m, theta, hw):
        _, _, _, sig, ctx = F.setup(self.store, self.d, m_nli=MN, provider=provider, issuer="evaluation", hw=hw)
        return ctx._replace(m=m, theta=theta), sig

    def read_set(self, m, sigma_lo, sigma_c):
        """Every (premise, hypothesis) text pair a variant fold at top-m can read (a superset for m' <= m)."""
        from meld.merge.classifier import FoldState
        st, out = FoldState(), set()
        cons = F._distinguishes(self.store, self.d)
        for b in self.nodes:
            K, _ = select_tested_pair(st, b, sigma_lo, m, self.rank, cons)
            for j, r in enumerate(K):
                n = st.nodes[r]
                out.add((b.text, n.text))
                if n.cell == b.cell and self.rank[b.key] <= self.rank[r] and frozenset((b.key, r)) not in cons:
                    out.add((n.text, b.text))
            st.add(b)
        return out


def units(world, decisions, chunk, B_0, pop="tested"):
    """Tested-pair units (as benchmark_test.units); pop="eligible": every eligible candidate of K instead."""
    us = []
    for d in decisions:
        for r in ([r for r, _ in d.tested] if pop == "tested" else d.eligible):
            cls, mg, prov, v, row = world.label(r, d.b)
            us.append({"stratum": row["stratum"] if row else "tested_" + cls, "class": cls,
                       "bin": bin_of(d.n_prior, B_0), "chunk": chunk,
                       "decision": "MERGE" if r == d.target else d.decision if d.decision != "MERGE" else "NOT_CHOSEN",
                       "mergeable": mg, "provenance": prov, "verdict": v})
    return us


def targets(world, decisions):
    """Counters over the labelled mergeable manifest pairs and the arrivals that have one."""
    pos = {d.b: i for i, d in enumerate(decisions)}
    group = {}
    for d in decisions:
        group[d.b] = group[d.target] if d.target else d.b
    partners, c = defaultdict(set), Counter()
    for fs, row in world.pair.items():
        x, y = tuple(fs)
        if world.label(x, y)[1] is not True:
            continue
        t, a = sorted((x, y), key=pos.get)
        partners[a].add(t)
        da = decisions[pos[a]]
        c["pairs"] += 1
        c["target_merged"] += da.target == t
        c["group_merged"] += group[t] == group[a]
        c["target_tested"] += t in {r for r, _ in da.tested}
        c["target_in_K"] += t in da.K
    for a, ts in partners.items():
        da = decisions[pos[a]]
        c["arrivals_with_target"] += 1
        if da.target and da.target not in ts:
            c["wrong_target"] += 1
            c["wrong_target_" + {True: "mergeable", False: "non_mergeable", None: "unlabelled"}[
                world.label(da.target, a)[1]]] += 1
    return c


def recalibrate(world, decisions, *, d, alpha, B_0, m, hw, sel):
    """Arm b: the released Cal draw and quantile rule over the variant's cal-stream tested pairs."""
    lab = Counter()
    for dd in decisions:
        for r, _ in dd.tested:
            mg = world.label(r, dd.b)[1]
            lab["unlabelled" if mg is None else "mergeable" if mg else "non_mergeable"] += 1
    cand = G.cal_candidates(decisions, world.item, lambda x, y: world.label(x, y)[1])
    rng = random.Random(0)
    recs = []
    for it in sorted(cand):
        (cx, cy), s, dd = rng.choice(cand[it])
        row = world.pair.get(frozenset((cx, cy)))
        prov = world.label(cx, cy)[2]
        recs.append(CalRecord(scope=d, c_x=cx, c_y=cy, bin=bin_of(dd.n_prior, B_0), s_q=s, hw=hw, obs=1,
                              label="NON_MERGEABLE",
                              subtype="CONTRADICTION" if prov == "manifest" and row["label"] == "CONFLICT" else "OTHER",
                              source="SEED", item=it, author=C.AUTHOR, created_at=0, sigma_lo_q=sel[0], m=m,
                              m_pol=m_pol_of(world.pol), n_prior=dd.n_prior))
    ep = build_epoch(recs, scope=d, seq=1, alpha=alpha, alpha_c=alpha, policy="SEED", bins=(B_0, "doubling"),
                     selector_params=(sel[0], m, m_pol_of(world.pol)), m_nli=MN, sigma_c_q=sel[1],
                     author=C.AUTHOR, created_at=0, mondrian_c=False)
    return ep, {"cal_records": len(recs), "cal_tested": dict(lab)}


def rate_row(m):
    return {k: m[k] for k in ("false_merge", "merge_recall")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domain", nargs="+", default=list(EPOCH_DIR), choices=list(EPOCH_DIR))
    ap.add_argument("--alpha", nargs="+", default=["0.02", "0.05", "0.10"])
    ap.add_argument("--ms", nargs="+", type=int, default=[3, 5, 10])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out-dir", default=PJ("outputs/wp6g"))
    a = ap.parse_args(argv)
    key = open(PJ("outputs/calibration-labelled/key"), "rb").read().strip()
    annf, labf = PJ("outputs/annotate/combined-annotations-final.jsonl"), PJ("outputs/annotate/labels-final.jsonl")
    manifest = json.load(open(PJ("data/bench/manifest.json")))
    lookup = Q.source_labels(PJ("data/bench"))
    ann = Q.with_ambiguous(Q.load_annotations(annf), Q.load_ambiguous(labf), exclude=False)
    verdicts = B.load_verdicts(annf)
    base = F.nli_provider(NLI)
    memo = {}
    prov = lambda p, h: memo[p, h] if (p, h) in memo else memo.setdefault((p, h), base(p, h))  # noqa: E731
    hw = platform.machine()
    sel = tuple(DEFAULT_POL["bootstrap_selector"][:2]) + (3,)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "multi-target.json")
    run = {"annotations_sha256": sha(annf), "labels_sha256": sha(labf), "ambiguous": "non-mergeable",
           "source_labels": True, "seed": 0, "hw": hw, "omp": os.environ.get("OMP_NUM_THREADS"),
           "key": "local dev key, evaluation only; arm-b tables are not signed epochs", "batch_check": {}}
    rows, t_all = [], time.monotonic()

    def batch_fill(pairs):
        todo = sorted((p for p in pairs if p not in memo), key=lambda p: len(p[0]) + len(p[1]))
        ce = signals._cross_encoder(NLI)
        ce.probs("a", "b")                                   # load the weights
        names = {str(v).lower(): int(k) for k, v in ce._model.config.id2label.items()}
        t0 = time.monotonic()
        for i in range(0, len(todo), 2048):
            ch = todo[i:i + 2048]
            P = ce._model.predict(ch, apply_softmax=True, batch_size=a.batch, show_progress_bar=False)
            for p, x in zip(ch, P):
                memo[p] = to_fixed16(tuple(float(x[names[k]]) for k in ("contradiction", "entailment", "neutral")))
            print(f"  batch NLI {i + len(ch)}/{len(todo)} {time.monotonic() - t0:.0f}s", flush=True)
        return len(todo)

    for d in a.domain:
        t_d = time.monotonic()
        cal_rows, test_rows = C.cal_rows(manifest, d, lookup, ann), C.split_rows(manifest, d, "test")
        n_cal = len({t for r in cal_rows for t in (r["text_a"], r["text_b"])})
        cal_w = World(cal_rows, d, sel, key, lookup, ann, verdicts)
        test_ws = [World(p, d, sel, key, lookup, ann, verdicts) for p in B.chunk_rows(test_rows, n_cal, 0)]
        rel = {al: read_epoch(PJ("outputs/calibration-labelled", EPOCH_DIR[d].format(a=al), f"{d}-seed-a{al}.epoch.json"), key)
               for al in a.alpha}
        B_0 = next(iter(rel.values())).bins[0]
        # released path first, single calls: the E2-B3 reference
        ref = {}
        for al, ep in rel.items():
            ev = B.evaluate(test_rows, ep, key=key, encoder=ENC, nli=NLI, provider=prov, hw=hw, seed=0,
                            source_label=lookup, annotations=ann, verdicts=verdicts, max_unresolved=None, chunk=n_cal)
            ref[al] = ev["summary"]["marginal"]
        C.calibrate(cal_rows, scope=d, encoder=ENC, nli=NLI, provider=prov, key=key, alpha=rel[a.alpha[0]].alpha,
                    B_0=B_0, selector=sel, hw=rel[a.alpha[0]].cal_hw, seed=0, source_label=lookup, annotations=ann,
                    max_unresolved=None)              # the released cal read set, single calls
        n_single = len(memo)
        need = set()
        for w in [cal_w] + test_ws:
            need |= w.read_set(max(a.ms), sel[0], sel[1])
        sample = [p for p in need if p in memo][:200]
        saved = {p: memo.pop(p) for p in sample}
        n_batch = batch_fill(need)
        diff = [max(abs(x - y) for x, y in zip(saved[p], memo[p])) for p in sample]
        run["batch_check"][d] = {"sample": len(sample), "max_fixed16_diff": max(diff, default=0),
                                 "pairs_differing": sum(x > 0 for x in diff)}
        memo.update(saved)                            # the release's single-call values stay authoritative
        print(f"{d}: single-call {n_single}, batched {n_batch}, batch check {run['batch_check'][d]}", flush=True)

        def fold_all(worlds, m, theta, mode):
            ds, calls, nodes = [], 0, 0
            for w in worlds:
                ctx, sig = w.ctx(prov, m, theta, hw)
                dd = G.vfold(w.nodes, ctx, mode)
                ds.append(dd)
                calls, nodes = calls + sig.calls, nodes + len(dd)
            return ds, calls, nodes

        INF = lambda n: INF16  # noqa: E731
        cal_cache = {}
        for al in a.alpha:
            ep_rel = rel[al]
            th_rel = lambda n, e=ep_rel: threshold(e, bin_of(n, B_0))  # noqa: E731
            for m in a.ms:
                for mode in ("single", "first", "best"):
                    if mode == "single" and m != 3:
                        continue
                    for arm in ("a", "b"):
                        if arm == "b":
                            cmode = "single" if mode == "single" else "first"   # theta INF16: first = best
                            if (cmode, m) not in cal_cache:
                                cal_cache[cmode, m] = fold_all([cal_w], m, INF, cmode)[0][0]
                            ep, cstat = recalibrate(cal_w, cal_cache[cmode, m], d=d, alpha=ep_rel.alpha, B_0=B_0,
                                                    m=m, hw=ep_rel.cal_hw, sel=sel)
                            if mode == "single":
                                assert ep.table == ep_rel.table, (d, al, "single m = 3 does not rebuild the table")
                                continue           # identical to arm a
                            th = lambda n, e=ep: threshold(e, bin_of(n, B_0))  # noqa: E731
                        else:
                            ep, cstat, th = ep_rel, None, th_rel
                        ds, calls, nodes = fold_all(test_ws, m, th, mode)
                        us = [u for i, (w, dd) in enumerate(zip(test_ws, ds)) for u in units(w, dd, i, B_0)]
                        sm = B.summarize(us)
                        fm_el = B.metrics([u for i, (w, dd) in enumerate(zip(test_ws, ds))
                                           for u in units(w, dd, i, B_0, "eligible")])["false_merge"]
                        mk = sm["marginal"]
                        if mode == "single":
                            assert rate_row(mk) == rate_row(ref[al]), (d, al, "single does not reproduce E2-B3")
                        tg = Counter()
                        for w, dd in zip(test_ws, ds):
                            tg += targets(w, dd)
                        unl = [u for u in us if u["mergeable"] is None]
                        k_unl = sum(u["decision"] == "MERGE" for u in unl)
                        fmb = {g: v["false_merge"] for g, v in sm["bin"].items() if v["false_merge"]["n"]}
                        gmax = max(fmb, key=lambda g: fmb[g]["rate"]) if fmb else None
                        fm = mk["false_merge"]
                        row = {**stamps(ENC, NLI, ep, ws_pol(test_ws)), "domain": d, "alpha": float(al), "m": m,
                               "mode": mode, "arm": arm, "theta_q": {g: t for g, _, _, t, _ in ep.table},
                               "released_epoch_id": object_id(canonical_bytes(ep_rel)).hex(), "cal": cstat,
                               "false_merge": fm, "fm_per_bin": fmb, "fm_eligible_population": fm_el,
                               "fm_bin_max": {"bin": gmax, **fmb[gmax]} if gmax is not None else None,
                               "fm_worst_unlabelled": B._rate(fm["k"] + k_unl, fm["n"] + k_unl),
                               "unlabelled_tested": len(unl), "merges_into_unlabelled": k_unl,
                               "merge_recall_tested": mk["merge_recall"], "decisions": mk["decisions"],
                               "tested_pairs": len(us), "nodes": nodes,
                               "target_recall": B._rate(tg["target_merged"], tg["pairs"]),
                               "group_recall": B._rate(tg["group_merged"], tg["pairs"]),
                               "target_tested_share": tg["target_tested"] / tg["pairs"] if tg["pairs"] else None,
                               "targets": dict(tg), "wrong_target_merges": tg["wrong_target"],
                               "nli_calls": calls, "nli_calls_per_delta": calls / nodes,
                               "tested_per_delta": len(us) / nodes,
                               "e2b3_reference": rate_row(ref[al]) if mode == "single" else None}
                        rows.append(row)
                        print(f"{d:<9}{al:>6}{m:>4} {mode:<7}{arm} FM {fm['rate']:.4f} ({fm['k']}/{fm['n']}) "
                              f"binmax {row['fm_bin_max']['rate'] if gmax is not None else float('nan'):.4f} "
                              f"recT {mk['merge_recall']['rate']:.4f} tgt {row['target_recall']['rate']:.4f} "
                              f"grp {row['group_recall']['rate']:.4f} worst {row['fm_worst_unlabelled']['rate']:.4f} wrong {tg['wrong_target']} unlM {k_unl}/{len(unl)} "
                              f"nli/d {calls / nodes:.2f}", flush=True)
                        with open(out, "w") as fh:
                            json.dump({"run": run, "rows": rows}, fh, indent=1, default=str)
        print(f"{d} done {time.monotonic() - t_d:.0f}s", flush=True)
    print(f"all done {time.monotonic() - t_all:.0f}s")


def ws_pol(ws):
    return ws[0].pol


if __name__ == "__main__":
    main()
