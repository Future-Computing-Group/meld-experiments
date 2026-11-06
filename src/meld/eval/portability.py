"""Test-split evaluation of calibrated epochs, for the portability grid and transfer matrix (E2-P1),
the calibration-burden curve (E2-P2) and recalibration under shift (E2-B4). Evaluation only: the
protocol is used as frozen (calibration.stream, fold.readjudicate, conformal.build_epoch).

A test stream is built exactly as the calibration stream (``calibration.stream`` over the test
split) and folded once without an epoch, so every signal the fold reads is in its Sig set. Its
length is the calibration stream's too: with ``chunk`` (the domain's calibration stream node
count, ``cal_size``) the test split is partitioned into independent streams of that many texts
(``benchmark_test.chunk_rows``: whole source items, seeded, last partial chunk kept), each its own
store, so no test arrival reaches a store-size bin the calibration stream never reached.
``evaluate`` folds a copy of each chunk's store holding one epoch (no new signal is read) and scores
the tested pairs, pooled over the chunks: each takes its mergeability from the label store
(``label_queue.resolve``); a merged non-mergeable pair is a false merge. ``fm`` = false merges /
non-mergeable tested pairs (the guarantee's population, marginal over the stream), ``recall`` =
merged / mergeable tested pairs; unresolved pairs (no label) are counted and left out of both.
Counts are also given per chunk and per bin, with the bin coverage (``benchmark_test.coverage``).

``transfer`` re-addresses an epoch to another stream's scope and M_pol, so domain X's per-bin
thresholds are applied to domain Y's stream. ``subsample`` and ``reepoch`` build an epoch over a
subset of a Cal draw (verified by recomputation). Degraded signals at test time only: ``corrupt``
wraps an NLI provider (a seeded fraction of directions get their label slots permuted), and
``test_stream(noise=sigma)`` adds isotropic Gaussian noise of norm sigma relative to the unit
embedding to every claim's q (the induced cosine change is measured and returned).

``min_provider`` is the in-domain ablation signal: entailment min(p_e(NLI), mean of the paraphrase
probability over both orders), contradiction gate as the NLI decides it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import random
from collections import Counter, defaultdict
from typing import Callable, Optional

import numpy as np

from meld.brain.store import Store
from meld.eval import calibration as C
from meld.eval.benchmark_test import chunk_rows, coverage
from meld.merge.conformal import bin_of, build_epoch, verify_epoch
from meld.merge.signals import quantize_int8, to_fixed16
from meld.proto import CANON_VERSION
from meld.protocol.fold import readjudicate
from meld.protocol.objects import FIXED_ONE, INF16, canonical_bytes, m_nli, m_pol_of, signed


def _m_nli(encoder, nli):
    return m_nli(encoder, nli, "int8", CANON_VERSION)


def _clone(store: Store, scope: str, extra=(), kinds=("D", "Sig")) -> Store:
    s = Store(store.pol, store.key, encoder=store.encoder)
    for kind in kinds:
        for o in store.objects.iter(kind, scope):
            s.apply(*signed(o, store.key))
    for o in extra:
        if s.apply(*signed(o, store.key)) is None:
            raise ValueError("object refused by the evaluation store")
    return s


def _noisy(q: bytes, sigma: float, rng: np.random.Generator) -> np.ndarray:
    v = np.frombuffer(q, np.int8).astype(np.float64)
    v /= np.linalg.norm(v) or 1.0
    return v + sigma * rng.standard_normal(v.size) / np.sqrt(v.size)


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) or 1.0))


def test_stream(rows, *, chunk: Optional[int] = None, seed: int = 0, noise: float = 0.0, noise_seed: int = 0,
                **kw) -> dict:
    """The test split as one stream (``_stream``) or, with ``chunk``, as independent streams of at
    most ``chunk`` texts each (module doc). A chunked base holds its streams under ``chunks`` with
    the item and pair maps merged (chunks are text-disjoint)."""
    parts = chunk_rows(rows, chunk, seed) if chunk else [rows]
    bs = [_stream(p, seed=seed, noise=noise, noise_seed=noise_seed, **kw) for p in parts]
    if not chunk:
        return bs[0]
    info = None
    if noise:
        info = {"sigma": noise, "seed": noise_seed, "per_chunk": [b["noise"] for b in bs],
                **{k: float(np.mean([b["noise"][k] for b in bs])) for k in ("cos_shift_mean", "cos_shift_sd")}}
    return {**{k: bs[0][k] for k in ("pol", "scope", "encoder", "nli", "hw")}, "chunks": bs, "chunk_size": chunk,
            "item": {k: v for b in bs for k, v in b["item"].items()},
            "pair": {k: v for b in bs for k, v in b["pair"].items()}, "noise": info,
            "nodes": sum(b["nodes"] for b in bs), "nli_calls": sum(b["nli_calls"] for b in bs)}


def _stream(rows, *, scope: str, encoder: str, nli: str, selector, key: bytes, provider: Callable,
            hw: str, seed: int = 0, noise: float = 0.0, noise_seed: int = 0) -> dict:
    """One test stream (``calibration.stream``), optionally with embedding noise, folded once
    without an epoch so its Sig set holds every signal the fold reads."""
    store, pol, item, pair = C.stream(rows, scope=scope, encoder=encoder, selector=selector, key=key, seed=seed)
    info = None
    if noise:
        nodes = [d for _, d in store.nodes().values()]
        noisy, clean = {}, {}
        for d in nodes:
            rng = np.random.default_rng([noise_seed, int.from_bytes(d.claim_key[:8], "big")])
            clean[d.claim_key] = np.frombuffer(d.q, np.int8).astype(np.float64)
            noisy[d.claim_key] = quantize_int8(_noisy(d.q, noise, rng))
        store = Store(pol, key, encoder=encoder)
        for d in nodes:
            store.apply(*signed(dataclasses.replace(d, q=noisy[d.claim_key].tobytes()), key))
        r = random.Random(noise_seed)
        ks = list(clean)
        sample = [r.sample(ks, 2) for _ in range(min(2000, len(ks) * 4))] if len(ks) > 1 else []
        dc = [_cos(noisy[a].astype(float), noisy[b].astype(float)) - _cos(clean[a], clean[b]) for a, b in sample]
        info = {"sigma": noise, "seed": noise_seed, "cos_shift_mean": float(np.mean(dc)) if dc else 0.0,
                "cos_shift_sd": float(np.std(dc)) if dc else 0.0}
    res = readjudicate(store, scope, m_nli=_m_nli(encoder, nli), provider=provider, issuer=C.AUTHOR, hw=hw)
    return {"store": store, "pol": pol, "item": item, "pair": pair, "scope": scope, "encoder": encoder,
            "nli": nli, "hw": hw, "noise": info, "nodes": len(res.decisions),
            "nli_calls": res.counters["nli_calls"]}


def replay(base: dict) -> Callable:
    """A provider answering from ``base``'s Sig set (the signals its stream already read), so a
    refold on a signal-free copy (``_clone(kinds=("D",))``, e.g. for ``fold_calibrate``, which
    verifies an epoch against the Sig records its own pass produced) costs no model call."""
    got = {}
    for b in base.get("chunks", [base]):
        store = b["store"]
        text = {d.claim_key: d.text for _, d in store.nodes().values()}
        for r in store.objects.iter("Sig", b["scope"]):
            for (p, h), pay in (((r.c_x, r.c_y), r.pay_xy), ((r.c_y, r.c_x), r.pay_yx)):
                if pay is not None:
                    got[text[p], text[h]] = tuple(pay)
    return lambda p, h: got[p, h]


_COUNTS = ("mergeable", "non_mergeable", "merged_mergeable", "false_merges", "unresolved", "merged_unresolved",
           "excluded")


def _rates(c: Counter) -> dict:
    out = {k: c.get(k, 0) for k in _COUNTS}
    return {**out, "fm": out["false_merges"] / out["non_mergeable"] if out["non_mergeable"] else None,
            "recall": out["merged_mergeable"] / out["mergeable"] if out["mergeable"] else None}


def evaluate(base: dict, epoch, *, source: str, source_label=None, annotations=None,
             items: Optional[set] = None) -> dict:
    """Fold each stream of ``base`` holding ``epoch`` (None: no epoch) and score its tested pairs
    whose arriving node belongs to ``items`` (default all), pooled over the streams, with per-chunk
    and per-bin counts and the bin coverage; see the module doc."""
    import time
    from meld.eval.label_queue import resolve
    t0, scope = time.monotonic(), base["scope"]
    by_class, by_stratum, by_bin, thetas = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), {}
    bins, by_chunk, calls, misses = Counter(), [], 0, 0
    for b in base.get("chunks", [base]):
        store = _clone(b["store"], scope, [epoch] if epoch is not None else [])
        res = readjudicate(store, scope, m_nli=_m_nli(base["encoder"], base["nli"]))
        if epoch is not None and (res.epoch is None or canonical_bytes(res.epoch) != canonical_bytes(epoch)):
            raise ValueError("the evaluation store did not adopt the epoch (scope, M_nli or M_pol differ)")
        calls, misses = calls + res.counters["nli_calls"], misses + res.counters["pass_time_misses"]
        text = {d.claim_key: d.text for _, d in store.nodes().values()}
        item, pair, chunk = b["item"], b["pair"], Counter()
        for d in res.decisions:
            if d.tested is None or (items is not None and item[d.tested[1]] not in items):
                continue
            a, t = d.tested
            r = pair.get(frozenset(d.tested))
            cls, mergeable, prov = resolve(r, text[a], text[t], item[a], item[t], source=source,
                                           source_label=source_label, annotations=annotations)
            g = bin_of(d.n_prior, epoch.bins[0]) if epoch is not None else None
            if epoch is not None:
                thetas[g] = d.theta_q
                bins[g] += 1
            merged = d.decision == "MERGE"
            k = ("excluded" if prov else "unresolved") if mergeable is None else \
                "mergeable" if mergeable else "non_mergeable"
            for c in (by_class[cls], by_stratum[r["stratum"] if r else cls], chunk,
                      *([by_bin[g]] if epoch is not None else [])):
                c[k] += 1
                if merged:
                    c[{"mergeable": "merged_mergeable", "non_mergeable": "false_merges"}.get(k, f"merged_{k}")] += 1
        by_chunk.append({**_rates(chunk), "nodes": len(res.decisions)})
    out = _rates(sum(by_class.values(), Counter()))
    out.update(by_class={c: dict(n) for c, n in sorted(by_class.items())},
               by_stratum={c: dict(n) for c, n in sorted(by_stratum.items())},
               by_bin={str(g): _rates(n) for g, n in sorted(by_bin.items())}, by_chunk=by_chunk,
               chunk_size=base.get("chunk_size"), coverage=coverage(epoch, bins),
               thetas_used={g: t for g, t in sorted(thetas.items())},
               epoch_scope=epoch.scope if epoch is not None else None,
               nli_calls=calls, misses=misses, eval_seconds=round(time.monotonic() - t0, 2))
    return out


def transfer(epoch, pol: dict, scope: str):
    """``epoch``'s operating point re-addressed to ``scope`` under ``pol`` (evaluation only)."""
    return dataclasses.replace(epoch, scope=scope, m_pol=m_pol_of(pol))


def subsample(cal, m: int, seed: int = 0):
    """At most ``m`` Cal records per bin, seeded draw; order of ``cal`` kept."""
    rng, keep = random.Random(seed), set()
    by_bin = defaultdict(list)
    for i, k in enumerate(cal):
        by_bin[k.bin].append(i)
    for g in sorted(by_bin):
        keep.update(rng.sample(by_bin[g], min(m, len(by_bin[g]))))
    return [k for i, k in enumerate(cal) if i in keep]


def reepoch(epoch, cal, sig, seq: Optional[int] = None):
    """The epoch over ``cal`` with every other parameter of ``epoch``; verified by recomputation."""
    ep = build_epoch(cal, scope=epoch.scope, seq=epoch.seq if seq is None else seq, alpha=epoch.alpha,
                     alpha_c=epoch.alpha_c, policy=epoch.policy, bins=epoch.bins,
                     selector_params=(epoch.sigma_lo_q, epoch.m, epoch.m_pol), m_nli=epoch.m_nli,
                     sigma_c_q=epoch.sigma_c_q, author=epoch.author, created_at=epoch.created_at,
                     mondrian_c=epoch.mondrian_c)
    if not verify_epoch(ep, cal, sig):
        raise ValueError("sub-epoch does not verify by recomputation")
    return ep


def corrupt(provider: Callable, rate: float, seed: int = 0) -> Callable:
    """A seeded ``rate`` of directions get their (c, e, n) slots cyclically shifted by 1 or 2."""
    def f(p, h):
        t = tuple(provider(p, h))
        z = hashlib.sha256(f"{seed}\x00{p}\x00{h}".encode()).digest()
        if int.from_bytes(z[:8], "big") / 2 ** 64 >= rate:
            return t
        s = 1 + z[8] % 2
        return t[s:] + t[:s]
    return f


def min_provider(nli: Callable, para: Callable) -> Callable:
    """(p_c, min(p_e, mean para), p_n') with p_n' raised to p_c when the NLI does not flag a
    contradiction, so the contradiction gate is the NLI's. Probabilities in, uint16 triple out."""
    def f(p, h):
        pc, pe, pn = nli(p, h)
        e = min(pe, (para(p, h) + para(h, p)) / 2)
        return to_fixed16((pc, e, pn if pc > max(pe, pn) else max(pn, pc)))
    return f


# ---- providers by name (models load lazily)

NLI_IDS = {"mnli-cross-encoder": "cross-encoder/nli-deberta-v3-small",
           "deberta-base-nli": "cross-encoder/nli-deberta-v3-base",
           "roberta-large-mnli": "FacebookAI/roberta-large-mnli"}
PARA_ID = "semindan/paws_x_xlm_r_only_en"
ABLATION = "min-deberta-base-pawsx"


def _para(model_id: str) -> Callable:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model_id)
    lab = {int(k): str(v).lower() for k, v in ce.config.id2label.items()}
    pos = next(k for k, v in lab.items() if v in ("equivalent", "label_1", "paraphrase", "duplicate"))
    from functools import lru_cache
    return lru_cache(maxsize=None)(lambda p, h: float(ce.predict([(p, h)], apply_softmax=True)[0][pos]))


def provider(name: str) -> Callable:
    """uint16 provider for an NLI alias of ``NLI_IDS``, ``ABLATION``, or 'lexical'."""
    from meld.merge.signals import _CrossEncoderNLI
    from meld.protocol.fold import nli_provider
    if name == "lexical":
        return nli_provider("lexical")
    if name == ABLATION:
        return min_provider(_CrossEncoderNLI(NLI_IDS["deberta-base-nli"]).probs, _para(PARA_ID))
    ce = _CrossEncoderNLI(NLI_IDS[name])
    return lambda p, h: to_fixed16(ce.probs(p, h))


# ---- official PAWS validation pairs for the ablation arm's PAWS calibration

def selection_sample(df):
    """Row indices of the 300 + 300 PAWS validation pairs the signal-model selection used."""
    rng, idx = np.random.RandomState(0), []
    for lab in (1, 0):
        ii = np.flatnonzero(df.label.values == lab)
        idx += list(rng.choice(ii, 300, replace=False))
    return set(int(i) for i in idx)


def pawsval_rows(path: str, n_per_class: int = 1000, seed: int = 0):
    """Manifest-shaped cal rows from official PAWS validation minus the selection sample, ``n_per_class``
    per label (seeded); source items are the connected components of shared texts."""
    import pandas as pd
    from meld.bench.loaders import detok_paws
    df = pd.read_parquet(path)
    used, rng = selection_sample(df), random.Random(seed)
    pick = []
    for lab in (1, 0):
        pool = [i for i in range(len(df)) if df.label.iloc[i] == lab and i not in used]
        pick += rng.sample(pool, min(n_per_class, len(pool)))
    texts = [(detok_paws(df.sentence1.iloc[i]), detok_paws(df.sentence2.iloc[i])) for i in sorted(pick)]
    root = {}

    def find(t):
        while root.setdefault(t, t) != t:
            root[t] = root[root[t]]
            t = root[t]
        return t
    for a, b in texts:
        root[find(a)] = find(b)
    rows = []
    for i, (a, b) in zip(sorted(pick), texts):
        y = bool(df.label.iloc[i] == 1)
        rows.append({"pair_id": f"pawsval:{int(df.id.iloc[i])}", "text_a": a, "text_b": b, "source": "paws",
                     "source_item_id": "pawsval:" + hashlib.sha256(find(a).encode()).hexdigest()[:12],
                     "stratum": "S1_paws_para" if y else "S2_paws_nonpara", "label": "MERGE" if y else "NO_MERGE",
                     "label_provenance": "source", "mergeable": y, "subtype": None, "split": "cal"})
    return rows


def theta_table(epoch) -> dict:
    return {} if epoch is None else {g: (None if th == INF16 else th / FIXED_ONE) for g, _, _, th, _ in epoch.table}


# ---- shared runner plumbing (arguments, label store, stamp, rows)

def common_args(ap) -> None:
    import os
    from meld.agent.memory import DEFAULT_POL
    from meld.eval.benchmark_provenance import REPO_ROOT
    ap.add_argument("--manifest", default=os.path.join(REPO_ROOT, "data", "bench", "manifest.json"))
    ap.add_argument("--alpha", choices=("0.02", "0.05", "0.10"), default="0.02")
    ap.add_argument("--B0", type=int, default=256)
    ap.add_argument("--selector", nargs=3, type=int, default=DEFAULT_POL["bootstrap_selector"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "bench"))
    ap.add_argument("--no-source-labels", action="store_true")
    ap.add_argument("--annotations", help="label store JSONL (label_queue.load_annotations)")
    ap.add_argument("--labels", help="labels.jsonl with the AMBIGUOUS items (non-mergeable)")
    ap.add_argument("--pawsval", default=os.path.join(REPO_ROOT, "outputs", "tournament", "pawsval",
                                                      "validation.parquet"))
    ap.add_argument("--pawsval-n", type=int, default=1000, help="PAWS validation cal pairs per class")
    ap.add_argument("--max-unresolved", type=int, default=0,
                    help="unlabelled tested pairs a cell may hold (cal or test) before it is refused")
    ap.add_argument("--max-items", type=int, help="smoke: keep this many source items per split")
    ap.add_argument("--key-file")
    ap.add_argument("--out-dir", required=True)


def context(a) -> dict:
    """Label store, key, parameters and the stamp every row carries."""
    import json
    import os
    import platform
    import subprocess
    from meld.bench.manifest import manifest_hash
    from meld.eval import label_queue as Q
    from meld.eval.benchmark_provenance import REPO_ROOT
    from meld.merge.conformal import n_min
    from meld.protocol.objects import ALPHA_ONE
    key = open(a.key_file, "rb").read().strip() if a.key_file else os.environ.get("SMP_GROUP_KEY", "").encode()
    if not key:
        raise SystemExit("no signing key: pass --key-file or set SMP_GROUP_KEY")
    manifest = json.load(open(a.manifest, encoding="utf-8"))
    ann = Q.load_annotations(a.annotations) if a.annotations else None
    if a.labels:
        ann = Q.with_ambiguous(ann or {}, Q.load_ambiguous(a.labels))
    lookup = None if a.no_source_labels else Q.source_labels(a.data)
    git = lambda *x: subprocess.run(["git", *x], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest() if p else None  # noqa: E731
    alpha = round(float(a.alpha) * ALPHA_ONE)
    stamp = {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
             "manifest_sha256": manifest_hash(manifest), "annotations_sha256": sha(a.annotations),
             "labels_sha256": sha(a.labels), "source_labels": not a.no_source_labels, "alpha": float(a.alpha),
             "n_min": n_min(alpha), "B_0": a.B0, "selector": list(a.selector), "seed": a.seed,
             "max_unresolved": a.max_unresolved, "max_items": a.max_items, "hw": platform.machine() or "unknown"}
    return {"key": key, "manifest": manifest, "ann": ann, "lookup": lookup, "alpha": alpha, "a": a,
            "stamp": stamp, "hw": stamp["hw"], "pawsval": {}}


def _limit(rows, n, seed):
    if not n:
        return rows
    items = sorted({r["source_item_id"] for r in rows})
    keep = set(random.Random(seed).sample(items, min(n, len(items))))
    return [r for r in rows if r["source_item_id"] in keep]


def rows_for(ctx, domain: str, split: str, nli: str = ""):
    """The domain's rows of ``split`` (cal: refused while unlabelled); the ablation arm's PAWS cal
    is official PAWS validation (``pawsval_rows``)."""
    a = ctx["a"]
    if split == "cal" and domain == "paws" and nli == ABLATION:
        rows = ctx["pawsval"].get("rows") or ctx["pawsval"].setdefault(
            "rows", pawsval_rows(a.pawsval, a.pawsval_n, a.seed))
    elif split == "cal":
        rows = C.cal_rows(ctx["manifest"], domain, ctx["lookup"], ctx["ann"])
    else:
        rows = C.split_rows(ctx["manifest"], domain, split)
    return _limit(rows, a.max_items, a.seed)


def calibrate(ctx, domain: str, *, encoder: str, nli: str, prov: Callable) -> dict:
    """The domain's SEED epoch (development tolerance: the caller refuses by unresolved count); the
    ablation arm's PAWS pairs also take labels from PAWS validation itself."""
    a, lookup = ctx["a"], ctx["lookup"]
    rows = rows_for(ctx, domain, "cal", nli)
    if domain == "paws" and nli == ABLATION:
        import pandas as pd
        from meld.bench.loaders import detok_paws
        from meld.eval.label_queue import _key
        df = pd.read_parquet(a.pawsval)
        used = selection_sample(df)
        val = {_key(detok_paws(df.sentence1.iloc[i]), detok_paws(df.sentence2.iloc[i])): bool(df.label.iloc[i] == 1)
               for i in range(len(df)) if i not in used}
        lookup = lambda x, y, base=lookup: val.get(_key(x, y), base(x, y) if base else None)  # noqa: E731
    return C.calibrate(rows, scope=domain, encoder=encoder, nli=nli, provider=prov, key=ctx["key"],
                       alpha=ctx["alpha"], B_0=a.B0, selector=a.selector, hw=ctx["hw"], seed=a.seed,
                       source_label=lookup, annotations=ctx["ann"], max_unresolved=None)


def cal_size(ctx, domain: str, nli: str = "") -> int:
    """Node count of the domain's calibration stream for this NLI arm (labels not required)."""
    a = ctx["a"]
    rows = rows_for(ctx, domain, "cal", nli) if domain == "paws" and nli == ABLATION else \
        _limit(C.split_rows(ctx["manifest"], domain, "cal"), a.max_items, a.seed)
    return len({t for r in rows for t in (r["text_a"], r["text_b"])})


def base_for(ctx, domain: str, *, encoder: str, nli: str, prov: Callable, scope: Optional[str] = None,
             **kw) -> dict:
    """The domain's test split as streams of its calibration stream's length (module doc)."""
    a = ctx["a"]
    return test_stream(rows_for(ctx, domain, "test"), scope=scope or domain, encoder=encoder, nli=nli,
                       selector=a.selector, key=ctx["key"], provider=prov, hw=ctx["hw"], seed=a.seed,
                       chunk=cal_size(ctx, domain, nli), **kw)


def score(ctx, base, epoch, domain, items=None) -> dict:
    return evaluate(base, epoch, source=domain, source_label=ctx["lookup"], annotations=ctx["ann"], items=items)


def writer(out_dir: str):
    """Append-and-flush JSONL rows (a long grid keeps its partial results); returns (emit, rows)."""
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    path, rows = os.path.join(out_dir, "rows.jsonl"), []
    open(path, "w").close()

    def emit(row):
        rows.append(row)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        print(json.dumps({k: row.get(k) for k in ("kind", "encoder", "nli", "cal_domain", "test_domain", "arm",
                                                  "phase", "m", "rep", "status", "fm", "recall", "unresolved")}),
              flush=True)
    return emit, rows


__all__ = ["chunk_rows", "cal_size", "test_stream", "replay", "evaluate", "transfer", "subsample", "reepoch", "corrupt", "min_provider",
           "provider", "NLI_IDS", "PARA_ID", "ABLATION", "selection_sample", "pawsval_rows", "theta_table",
           "common_args", "context", "rows_for", "calibrate", "base_for", "score", "writer"]
