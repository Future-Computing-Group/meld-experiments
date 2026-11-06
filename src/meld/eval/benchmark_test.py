"""E2-B3: the benchmark's test split admitted under a domain's SEED epoch, evaluation only.

The test stream is built exactly as the calibration stream (``calibration.stream``: the split's
distinct texts shuffled by the seed, one cell, the scope's policy and bootstrap selector), the
epoch is applied to that store as a received signed record, and the scope is folded from the log
start (``fold.readjudicate``); the fold must report that epoch as active. Every decision with a
tested pair (a_e, b) is one evaluation unit. Its mergeability comes from the label store
(``label_queue.resolve``: manifest, else raw source, else annotators), its verdict from the
manifest label of a manifest pair, else the annotators' verdict (``load_verdicts``), else MERGE for
a source-labelled mergeable pair; a verdict that disagrees with the mergeability is dropped.
Tested pairs nobody labels are counted per class and refuse the run beyond ``max_unresolved``.

Test streams of the calibration stream's length (``chunk_rows``). The store-size bin of an arrival
is part of what the calibration stream calibrates: a test stream longer than the calibration stream
reaches bins the calibration stream never reached, which the epoch disables (no row: threshold
INF16), so every arrival there abstains by construction. ``evaluate(chunk=n)`` therefore partitions
the test split into independent streams of at most n distinct texts (n = the calibration stream's
node count): whole source items, shuffled by the seed and packed in that order, the last partial
chunk kept. Each chunk is built as the calibration stream and folded as its own store; the
per-pair outcomes are pooled for the metrics and broken down per chunk and per bin.
``coverage`` reports, for every evaluation, the tested decisions per bin against the Cal records
per bin and the share of each falling in bins the epoch disables.

Metrics (``summarize``), each with a two-sided 95% Clopper-Pearson interval, marginal over the
stream (the guarantee's population), per stratum (the manifest stratum, else ``tested_<class>``)
and per calibration bin of the arriving node:
  false_merge          MERGE among non-mergeable tested pairs
  merge_recall         MERGE among mergeable tested pairs
  contradiction_miss   not CONFLICT among tested pairs with verdict CONFLICT (to_merge, to_relate)
  relate_precision     verdict RELATE among RELATE decisions whose verdict is known

Calibration-draw spread (``spread``): the cal stream is folded once per draw seed and
``calibration.fold_calibrate`` redraws its one-tested-pair-per-item Cal set with that seed, so each
draw is an epoch the calibration authority could have built. Each draw's epoch and its
training-conditional variant are applied to the same test store (seq increasing, so the fold reads
the newest) and the realised test metrics are recorded per draw.

Training-conditional variant (``tc_index``, ``tc_epoch``). Vovk, "Conditional validity of inductive
conformal predictors", ACML 2012 (PMLR 25; arXiv:1209.2673), Proposition 2b: an inductive conformal
predictor with n calibration scores at significance level eps errs on a test example iff at most
j = floor(eps (n+1) - 1) calibration conformity scores are <= the test score, and it is
(E, delta)-valid, i.e. P(error probability > E) <= delta over the calibration draw, provided
delta >= bin_{n,E}(j), the Binomial(n, E) distribution function at j (iff, for a continuous score).
Here a non-mergeable pair errs iff s > theta, so the conformity score is -s and "at most j
calibration scores satisfy -s_i <= -s" is "s > the (n-j)-th smallest s_i". For a target false-merge
rate alpha and confidence delta the variant takes the largest j in {-1, ..., n-1} with
Binomial(n, alpha).cdf(j) <= delta and sets theta_g = the (n-j)-th smallest Cal score of the bin
(INF16, the bin disabled, when j = -1). The marginal threshold is j = floor(alpha (n+1) - 1), the
ceil((n+1)(1-alpha))-th smallest (``conformal.quantile_threshold``), so the variant is never lower.
Per bin, as the epoch's Mondrian bins. Ties in the uint16 score only make the "provided" direction
conservative. The variant table is an evaluation device applied at test time, not a verifiable
epoch: its id differs from the signed one and every row says ``variant``.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import random
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence

from meld.eval import calibration as C
from meld.eval.label_queue import _key, queue_id, resolve
from meld.merge.conformal import bin_of, threshold
from meld.proto import CANON_VERSION
from meld.protocol.fold import readjudicate
from meld.protocol.objects import INF16, canonical_bytes, m_nli, m_pol_of, object_id, signed

METRICS = ("false_merge", "merge_recall", "contradiction_miss", "relate_precision")
VERDICTS = ("MERGE", "RELATE", "CONFLICT", "NO_RELATION")


def cp(k: int, n: int, conf: float = 0.95):
    """Two-sided Clopper-Pearson interval for k successes in n trials ((None, None) for n = 0)."""
    from scipy.stats import beta
    if n == 0:
        return None, None
    a = (1 - conf) / 2
    return (0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k)))


def tc_index(n: int, alpha: float, delta: float) -> Optional[int]:
    """The 1-based order statistic of the training-conditional threshold (module doc), None when
    no index satisfies Vovk's Proposition 2b at (alpha, delta) for n scores (bin disabled)."""
    from scipy.stats import binom
    j = max((j for j in range(-1, n) if binom.cdf(j, n, alpha) <= delta), default=-1)
    return None if j < 0 else n - j


def tc_epoch(epoch, cal: Sequence, delta: float, seq: Optional[int] = None):
    """``epoch`` with each bin's theta replaced by the training-conditional threshold over its
    Cal scores (``cal``, the records the epoch was built from); evaluation only."""
    alpha = epoch.alpha / 1_000_000
    table = []
    for g, n, n_c, _, th_c in epoch.table:
        s = sorted(k.s_q for k in cal if k.bin == g)
        if len(s) != n:
            raise ValueError(f"bin {g}: {len(s)} Cal scores for an epoch row of {n}")
        k = tc_index(n, alpha, delta)
        table.append((g, n, n_c, INF16 if k is None else s[k - 1], th_c))
    return dataclasses.replace(epoch, table=tuple(table), seq=epoch.seq if seq is None else seq)


def chunk_rows(rows: Sequence[dict], size: int, seed: int = 0) -> List[List[dict]]:
    """``rows`` partitioned into streams of at most ``size`` distinct texts (module doc): whole source
    items in seeded order, a chunk closed when the next item would overflow it, the last partial
    chunk kept (an item of more than ``size`` texts is a chunk of its own)."""
    by: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by[r["source_item_id"]].append(r)
    ids = sorted(by)
    random.Random(seed).shuffle(ids)
    out, cur, n = [], [], 0
    for i in ids:
        k = len({t for r in by[i] for t in (r["text_a"], r["text_b"])})   # items are text-disjoint
        if cur and n + k > size:
            out.append(cur)
            cur, n = [], 0
        cur, n = cur + by[i], n + k
    return out + [cur] if cur else out


def coverage(epoch, test_bins) -> Optional[dict]:
    """Tested test decisions per bin (``test_bins``: bin -> count) against the epoch's Cal records
    per bin, and the share of each in bins the epoch disables (threshold INF16)."""
    if epoch is None:
        return None
    cal = {g: n for g, n, *_ in epoch.table}
    off = sorted(g for g in set(cal) | set(test_bins) if threshold(epoch, g) == INF16)
    share = lambda c: sum(c.get(g, 0) for g in off) / sum(c.values()) if sum(c.values()) else None  # noqa: E731
    return {"test_per_bin": {str(g): n for g, n in sorted(test_bins.items())},
            "cal_per_bin": {str(g): n for g, n in sorted(cal.items())}, "disabled_bins": off,
            "test_disabled_share": share(test_bins), "cal_disabled_share": share(cal)}


def load_verdicts(path: str) -> Dict[object, str]:
    """The annotators' four-way verdicts of a label-store JSONL: pair_id and text key -> verdict
    (verdicts outside MERGE/RELATE/CONFLICT/NO_RELATION, e.g. a mergeability-only NON_MERGE, skipped)."""
    out: Dict[object, str] = {}
    with open(path, encoding="utf-8") as fh:
        for r in map(json.loads, filter(str.strip, fh)):
            if r.get("verdict") not in VERDICTS:
                continue
            for k in ([r["pair_id"]] if r.get("pair_id") else []) + \
                    ([_key(r["text_a"], r["text_b"])] if r.get("text_a") and r.get("text_b") else []):
                out[k] = r["verdict"]
    return out


def verdict_of(r, ta, tb, ia, ib, *, source, mergeable, prov, verdicts) -> Optional[str]:
    v = r["label"] if prov == "manifest" and r and r.get("label") in VERDICTS else None
    for k in (r and r["pair_id"], queue_id(source, ia, ib, ta, tb), _key(ta, tb)):
        if v is None and k in (verdicts or {}):
            v = verdicts[k]
    if v is None and mergeable:
        v = "MERGE"
    return v if mergeable is not None and (v == "MERGE") == mergeable else None


def admit(store, epoch, key: bytes) -> None:
    """Apply ``epoch`` to ``store`` as a received signed record; refuse what the store refuses."""
    if store.apply(*signed(epoch, key), epoch.scope) is None:
        raise ValueError("the store refused the epoch record")


def fold_test(store, *, scope: str, epoch, encoder: str, nli: str, provider: Callable, hw: str):
    """Fold the scope; refuse unless ``epoch`` is the one the fold read."""
    res = readjudicate(store, scope, m_nli=m_nli(encoder, nli, "int8", CANON_VERSION), provider=provider,
                       issuer="evaluation", hw=hw)
    if res.epoch is None or canonical_bytes(res.epoch) != canonical_bytes(epoch):
        raise ValueError("the fold did not read the given epoch (scope, M_nli, M_pol or authority differ)")
    return res


def units(decisions, store, item, pair, *, source: str, B_0: int, source_label=None, annotations=None,
          verdicts=None, chunk: int = 0) -> List[dict]:
    """One row per decision with a tested pair: stratum, bin, decision, mergeability, verdict."""
    text = {d.claim_key: d.text for _, d in store.nodes().values()}
    out = []
    for d in decisions:
        if d.tested is None:
            continue
        a, b = d.tested
        r = pair.get(frozenset(d.tested))
        cls, mg, prov = resolve(r, text[a], text[b], item[a], item[b], source=source,
                                source_label=source_label, annotations=annotations)
        out.append({"stratum": r["stratum"] if r else "tested_" + cls, "class": cls, "bin": bin_of(d.n_prior, B_0),
                    "chunk": chunk,
                    "decision": d.decision, "mergeable": mg, "provenance": prov,
                    "verdict": verdict_of(r, text[a], text[b], item[a], item[b], source=source, mergeable=mg,
                                          prov=prov, verdicts=verdicts)})
    return out


def _rate(k, n):
    lo, hi = cp(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def metrics(us: Sequence[dict]) -> dict:
    nm = [u for u in us if u["mergeable"] is False]
    mg = [u for u in us if u["mergeable"] is True]
    con = [u for u in us if u["verdict"] == "CONFLICT"]
    rel = [u for u in us if u["decision"] == "RELATE" and u["verdict"] is not None]
    miss = [u for u in con if u["decision"] != "CONFLICT"]
    return {"false_merge": _rate(sum(u["decision"] == "MERGE" for u in nm), len(nm)),
            "merge_recall": _rate(sum(u["decision"] == "MERGE" for u in mg), len(mg)),
            "contradiction_miss": {**_rate(len(miss), len(con)),
                                   "to_merge": sum(u["decision"] == "MERGE" for u in miss),
                                   "to_relate": sum(u["decision"] == "RELATE" for u in miss)},
            "relate_precision": _rate(sum(u["verdict"] == "RELATE" for u in rel), len(rel)),
            "decisions": dict(Counter(u["decision"] for u in us)), "units": len(us),
            "unresolved": sum(u["mergeable"] is None and u["provenance"] is None for u in us)}


def summarize(us: Sequence[dict]) -> dict:
    """Marginal (pooled over chunks), per-stratum, per-bin and per-chunk metrics (module doc)."""
    by = lambda f: {str(k): metrics([u for u in us if u[f] == k]) for k in sorted({u[f] for u in us})}  # noqa: E731
    return {"marginal": metrics(us), "stratum": by("stratum"), "bin": by("bin"), "chunk": by("chunk")}


def label_counts(us: Sequence[dict]) -> dict:
    c = defaultdict(Counter)
    for u in us:
        c[u["class"]]["excluded" if u["mergeable"] is None and u["provenance"] else "unresolved"
                      if u["mergeable"] is None else "mergeable" if u["mergeable"] else "non_mergeable"] += 1
    return {k: dict(v) for k, v in sorted(c.items())}


def evaluate(test_rows: Sequence[dict], epoch, *, key: bytes, encoder: str, nli: str, provider: Callable,
             hw: str, seed: int = 0, source_label=None, annotations=None, verdicts=None,
             max_unresolved: Optional[int] = 0, chunk: Optional[int] = None) -> dict:
    """Fold the test split of ``epoch.scope`` under ``epoch`` (module doc), as one stream or, with
    ``chunk`` (the calibration stream's node count), as independent streams of that length.
    Returns each chunk's store, item and pair maps (for ``spread``), the pooled units, label counts,
    summary and bin coverage."""
    if not test_rows or any(r["split"] != "test" for r in test_rows):
        raise ValueError("evaluate reads the test split only")
    scope = epoch.scope
    sel = (epoch.sigma_lo_q, epoch.sigma_c_q, epoch.m)
    chunks, us, calls, decisions = [], [], 0, []
    for i, part in enumerate(chunk_rows(test_rows, chunk, seed) if chunk else [list(test_rows)]):
        store, pol, item, pair = C.stream(part, scope=scope, encoder=encoder, selector=sel, key=key, seed=seed)
        if m_pol_of(pol) != epoch.m_pol:
            raise ValueError("the epoch's M_pol is not the test stream's (scope or selector differ)")
        admit(store, epoch, key)
        res = fold_test(store, scope=scope, epoch=epoch, encoder=encoder, nli=nli, provider=provider, hw=hw)
        us += units(res.decisions, store, item, pair, source=scope, B_0=epoch.bins[0], source_label=source_label,
                    annotations=annotations, verdicts=verdicts, chunk=i)
        calls += res.counters["nli_calls"]
        decisions += res.decisions
        chunks.append({"store": store, "item": item, "pair": pair, "decisions": res.decisions,
                       "items": len(set(item.values())),
                       "texts": len(item), "max_bin": max(bin_of(d.n_prior, epoch.bins[0]) for d in res.decisions)})
    labels = label_counts(us)
    n_un = sum(n.get("unresolved", 0) for n in labels.values())
    if max_unresolved is not None and n_un > max_unresolved:
        raise ValueError(f"scope {scope!r} test: {n_un} tested pairs unresolved ({labels}) exceed the "
                         f"tolerance {max_unresolved}: annotate them before evaluating")
    return {"chunks": chunks, "chunk_size": chunk, "pol": pol, "units": us, "labels": labels, "unresolved": n_un,
            "summary": summarize(us), "coverage": coverage(epoch, Counter(u["bin"] for u in us)),
            "nli_calls": calls, "nodes": len(decisions), "untested": sum(d.tested is None for d in decisions)}


def group_purity(ev: dict, *, source: str, source_label=None, annotations=None) -> dict:
    """E2-B5 over ``evaluate``'s chunks: every pair of members of a consolidation group (final
    ``store.derived.rep``), labelled as tested pairs are (``label_queue.resolve``: manifest, source,
    annotators; a pair nobody labels, typically two source items, counts as unlabelled by class);
    purity_labelled = mergeable / labelled; hidden nodes = members other than the representative
    (vis DEPRECATED), and of those the ones whose pair with the representative was never a tested
    MERGE pair (hidden behind a claim they were not compared with)."""
    out = Counter()
    unl = Counter()
    for c in ev["chunks"]:
        store, item, pair = c["store"], c["item"], c["pair"]
        text = {d.claim_key: d.text for _, d in store.nodes().values()}
        rep = store.derived.rep
        groups: Dict[bytes, List[bytes]] = defaultdict(list)
        for k in item:
            groups[rep.get(k, k)].append(k)
        merged = {frozenset(d.tested) for d in c["decisions"] if d.decision == "MERGE"}
        out["nodes"] += len(item)
        for r, ms in groups.items():
            out["groups_multi"] += len(ms) > 1
            for i, x in enumerate(ms):
                if x != r:
                    out["hidden_nodes"] += 1
                    out["hidden_untested_vs_rep"] += frozenset((x, r)) not in merged
                for y in ms[i + 1:]:
                    cls, mg, _ = resolve(pair.get(frozenset((x, y))), text[x], text[y], item[x], item[y],
                                         source=source, source_label=source_label, annotations=annotations)
                    out["group_pairs"] += 1
                    out["mergeable" if mg else "non_mergeable" if mg is False else "unlabelled"] += 1
                    if mg is None:
                        unl[cls] += 1
    lab = out["mergeable"] + out["non_mergeable"]
    return {**{k: out[k] for k in ("nodes", "groups_multi", "group_pairs", "mergeable", "non_mergeable",
                                     "unlabelled", "hidden_nodes", "hidden_untested_vs_rep")},
            "unlabelled_by_class": dict(unl),
            "purity_labelled": out["mergeable"] / lab if lab else None,
            "impure_labelled": _rate(out["non_mergeable"], lab),
            "hidden_share": out["hidden_nodes"] / out["nodes"] if out["nodes"] else None,
            "hidden_untested_share": out["hidden_untested_vs_rep"] / out["nodes"] if out["nodes"] else None}


def spread(cal_rows: Sequence[dict], ev: dict, *, epoch, draws: Sequence[int], delta: float, key: bytes,
           encoder: str, nli: str, provider: Callable, hw: str, seed: int = 0, source_label=None,
           annotations=None, verdicts=None) -> List[dict]:
    """One row per draw seed: the redrawn epoch's id and table, its training-conditional variant's
    table, and the realised test summary and bin coverage of each on ``ev``'s test stores
    (``evaluate``), pooled over its chunks."""
    scope = epoch.scope
    sel = (epoch.sigma_lo_q, epoch.sigma_c_q, epoch.m)
    cstore, pol, citem, cpair = C.stream(cal_rows, scope=scope, encoder=encoder, selector=sel, key=key, seed=seed)
    alpha, B_0 = epoch.alpha, epoch.bins[0]
    seq, out, memo = max(epoch.seq, 0), [], {}
    nli_of = lambda p, h: memo[p, h] if (p, h) in memo else memo.setdefault((p, h), provider(p, h))  # noqa: E731
    for d in draws:
        # a fresh copy of the unfolded cal store per draw, so the pass publishes the Sig records
        # verify_epoch recomputes from; the NLI is memoised, so only the first draw evaluates it
        cres = C.fold_calibrate([(copy.deepcopy(cstore), citem, cpair, {})], scope=scope, pol=pol, source=scope, encoder=encoder,
                                nli=nli, provider=nli_of, alpha=alpha, B_0=B_0, hw=epoch.cal_hw, seed=d,
                                seq=epoch.seq, created_at=epoch.created_at,
                                source_label=source_label, annotations=annotations, max_unresolved=None)
        e_m = cres["epoch"]
        row = {"draw": d, "epoch_id": object_id(canonical_bytes(e_m)).hex(), "cal": len(cres["cal"]),
               "is_given_epoch": canonical_bytes(e_m) == canonical_bytes(epoch)}
        for name, e in (("marginal", e_m), ("training_conditional", tc_epoch(e_m, cres["cal"], delta))):
            seq += 1
            e = dataclasses.replace(e, seq=seq)
            us = []
            for i, c in enumerate(ev["chunks"]):
                admit(c["store"], e, key)
                res = fold_test(c["store"], scope=scope, epoch=e, encoder=encoder, nli=nli, provider=provider, hw=hw)
                us += units(res.decisions, c["store"], c["item"], c["pair"], source=scope, B_0=B_0,
                            source_label=source_label, annotations=annotations, verdicts=verdicts, chunk=i)
            row[name] = {"table": [list(t) for t in e.table], "summary": summarize(us),
                         "coverage": coverage(e, Counter(u["bin"] for u in us))}
        out.append(row)
    return out


def spread_summary(rows: Sequence[dict], alpha: float) -> dict:
    """Per variant and per bin (and marginal): quantiles of the realised test false-merge rate over
    the draws, the share of draws above ``alpha``, and the mean merge recall."""
    import numpy as np
    out = {}
    for name in ("marginal", "training_conditional"):
        cells = defaultdict(lambda: ([], []))
        for r in rows:
            s = r[name]["summary"]
            for g, m in [("marginal", s["marginal"]), *s["bin"].items()]:
                if m["false_merge"]["n"]:
                    cells[g][0].append(m["false_merge"]["rate"])
                if m["merge_recall"]["n"]:
                    cells[g][1].append(m["merge_recall"]["rate"])
        out[name] = {g: {"draws": len(fm), "fm_quantiles": dict(zip(("min", "q05", "q25", "median", "q75", "q95", "max"),
                         map(float, np.quantile(fm, [0, .05, .25, .5, .75, .95, 1])))) if fm else None,
                         "fm_mean": float(np.mean(fm)) if fm else None,
                         "share_fm_above_alpha": float(np.mean([x > alpha for x in fm])) if fm else None,
                         "recall_mean": float(np.mean(rc)) if rc else None}
                     for g, (fm, rc) in sorted(cells.items())}
    return out


__all__ = ["METRICS", "cp", "tc_index", "tc_epoch", "chunk_rows", "coverage", "load_verdicts", "verdict_of", "admit", "fold_test", "units",
           "metrics", "summarize", "label_counts", "evaluate", "spread", "spread_summary"]
