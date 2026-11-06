"""Index scaling (eval/index_scaling.py): exact top-m against FAISS flat and HNSW as the store
grows, on a fixed snapshot (no write path), plus the admission flag's agreement with exact
admission on the gold seeds and synthetic families.

Corpus: ``synthetic`` (seeded clustered int8, --dim), ``lme`` (distinct LongMemEval_s user turns),
or a text file with one sentence per line (e.g. pooled HotpotQA context sentences). Text corpora
are embedded once under --encoder, in a separate process, into one packed int8 matrix (--emb-cache,
a single .npy file).
Queries: half lightly paraphrased stored sentences (synthetic: perturbed stored rows), half
held-out sentences; rows and queries are spread over four cells (the eligible track).

Every scan cell and admission row carries topology.stamps (a synthetic scan embeds nothing and runs
no NLI: encoder, NLI and M_nli are None there).

Run:  PYTHONPATH=src python scripts/run_index_scaling.py --corpus synthetic --sizes 10000 100000 \
          --encoder minilm --nli mnli-cross-encoder [--queries 1000] [--m 3] [--ef-search 128] \
          [--admission] [--out outputs/index-scaling/synthetic.json]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time

import numpy as np

from meld.agent.memory import DEFAULT_POL
from meld.eval.index_scaling import admission_agreement, clustered_int8, scaling_cell, synthetic_texts
from meld.eval.topology import stamps
from meld.merge.m2_gold import _SEEDS
from meld.protocol.objects import FIXED_ONE

LME = os.path.join("data", "bench", "lme-cleaned", "longmemeval_s_cleaned.json")
CELLS = 4


def _texts(corpus: str):
    if corpus == "lme":
        with open(LME, encoding="utf-8") as fh:
            data = json.load(fh)
        return sorted({t["content"] for q in data for s in q["haystack_sessions"] for t in s
                       if t["role"] == "user" and t["content"].strip()})
    with open(corpus, encoding="utf-8") as fh:
        return sorted({ln.strip() for ln in fh if ln.strip()})


def corpus_and_queries(a):
    """(store rows of the largest size, queries); rows in a seeded order, the first n are cell n."""
    n_max, half = max(a.sizes), a.queries // 2
    rng = np.random.default_rng(a.seed)
    if a.corpus == "synthetic":
        X = clustered_int8(n_max, a.dim, seed=a.seed)
        held = X[rng.integers(0, min(a.sizes), half)].astype(np.int16)
        noise = rng.integers(-6, 7, held.shape)
        Q = np.concatenate([np.clip(held + noise, -127, 127).astype(np.int8),
                            clustered_int8(half, a.dim, seed=a.seed + 1)])
        return X, Q
    from meld.agent.mem_conditions import light_paraphrase
    texts = _texts(a.corpus)
    random.Random(a.seed).shuffle(texts)
    if len(texts) < n_max + half:
        raise SystemExit(f"corpus has {len(texts)} texts, needs {n_max + half}")
    para = [light_paraphrase(texts[i]) for i in rng.integers(0, min(a.sizes), half).tolist()]
    pool = texts[:n_max + half] + para
    if not a.emb_cache:
        raise SystemExit("a text corpus needs --emb-cache (one packed .npy file)")
    if not os.path.exists(a.emb_cache):
        # the encoder runs in its own process: torch and FAISS must not share one (candidate_index)
        os.makedirs(os.path.dirname(a.emb_cache) or ".", exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump(pool, fh)
        t = time.perf_counter()
        subprocess.run([sys.executable, "-c", "import json, sys, numpy as np\n"
                        "from meld.eval.index_scaling import embed_texts\n"
                        "np.save(sys.argv[2], embed_texts(json.load(open(sys.argv[1], encoding='utf-8')), sys.argv[3]))",
                        fh.name, a.emb_cache, a.encoder], check=True)
        os.unlink(fh.name)
        print(f"embedded {len(pool)} texts in {time.perf_counter() - t:.0f} s")
    E = np.load(a.emb_cache)
    if len(E) != len(pool):
        raise SystemExit("--emb-cache does not match this corpus, sizes and seed")
    return E[:n_max], np.concatenate([E[n_max + half:], E[n_max:n_max + half]])


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="synthetic | lme | path to a one-sentence-per-line file")
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--m", type=int, default=DEFAULT_POL["bootstrap_selector"][2])
    ap.add_argument("--sigma-lo-q", type=int, default=DEFAULT_POL["bootstrap_selector"][0])
    ap.add_argument("--overfetch", type=int, default=4)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", type=int, default=128)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--nli", required=True, help="the admission flag's NLI (--admission)")
    ap.add_argument("--emb-cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--admission", action="store_true",
                    help="also measure the admission flag against exact admission")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    X, Q = corpus_and_queries(a)
    rng = np.random.default_rng(a.seed + 2)
    cells, qcells = rng.integers(0, CELLS, len(X)), rng.integers(0, CELLS, len(Q))
    enc = None if a.corpus == "synthetic" else a.encoder
    rep = {"corpus": a.corpus, "seed": a.seed, "machine": platform.machine(), "cells": []}
    for n in sorted(a.sizes):
        r = scaling_cell(X[:n], Q, cells[:n], qcells, m=a.m, sigma_lo=a.sigma_lo_q, overfetch=a.overfetch,
                         M=a.M, ef_construction=a.ef_construction, ef_search=a.ef_search)
        rep["cells"].append({**r, **stamps(enc, None, None, pol=None)})
        print(json.dumps({k: r[k] for k in ("n", "exact_topm", "flat", "hnsw", "peak_rss_bytes")}))
    if a.admission:
        work = [t for _, _, *ts in _SEEDS for t in ts] + synthetic_texts(2000, seed=a.seed)
        rep["admission"] = [admission_agreement(work, index={"kind": "hnsw", "M": M, "ef_search": ef, "k": k},
                                                encoder=a.encoder, nli=a.nli, theta_q=round(0.7 * FIXED_ONE))
                            for M, ef, k in ((32, 128, None), (32, 16, None), (32, 1, 3), (2, 1, 3))]
        for r in rep["admission"]:
            print(json.dumps(r))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    return rep


if __name__ == "__main__":
    main()
