"""Index scaling instrument (off-protocol): FAISS candidate indices against the protocol's exact
top-m (signals.topm_exact) on a fixed store snapshot, as the store grows.

``scaling_cell(X, queries, cells, qcells, m=..., sigma_lo=...)`` on an int8 store matrix X (rows in
canonical order) builds an exact ``IndexFlatIP`` and an ``IndexHNSWFlat`` (merge/candidate_index.py)
and, per query, ranks exactly by ``topm_exact`` (the protocol's selection, over precomputed float32
rows and squared norms as the fold keeps them), by each index's raw top-m, and by each index's
top-``overfetch * m`` re-ranked exactly (what the admission flag does). Reported: build seconds,
serialized index bytes, query latency p50/p95 per method, recall@m of the raw index top-m against
exact K, and the agreement of the re-ranked K with exact K (set and order), of its first element
(a*) and of its first same-cell element (the eligible track, a_e without rank or DISTINGUISHES),
plus the process's peak RSS.

``clustered_int8`` is the synthetic corpus (a seeded Gaussian mixture, quantized per row like a
claim embedding); ``embed_texts`` quantizes real sentences under a sentence encoder into one
int8 matrix (one packed file, no per-text cache). ``synthetic_texts`` and ``admission_agreement``
measure the admission flag end to end on in-order arrival: exact admission against admission with
the index, the steps whose K differs (pre-filter misses), the decisions that differ, and whether
the exact pass over the indexed replica's sets gives the exact admission decisions.
"""
from __future__ import annotations

import random
import resource
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from meld.merge.candidate_index import CandidateIndex
from meld.merge.signals import quantize_int8, topm_exact


def clustered_int8(n: int, dim: int, *, seed: int, centers: int = 256, spread: float = 0.35) -> np.ndarray:
    """n int8 rows from a seeded mixture of ``centers`` Gaussian clusters (the centers are fixed by
    dim alone, so corpora and query sets of one dim share them)."""
    C = np.random.default_rng(dim).standard_normal((centers, dim))
    rng = np.random.default_rng(seed)
    X = C[rng.integers(0, centers, n)] + spread * rng.standard_normal((n, dim))
    m = np.abs(X).max(axis=1, keepdims=True)
    return np.rint(127.0 * X / m).astype(np.int8)


def embed_texts(texts: Sequence[str], encoder: str, batch: int = 256) -> np.ndarray:
    """int8 rows of canon(text) under a sentence-transformers rung, quantized as a Delta's q."""
    from sentence_transformers import SentenceTransformer
    from meld.merge.signals import _SentenceTransformerEncoder
    from meld.proto import canon
    model = SentenceTransformer(_SentenceTransformerEncoder._MODEL_IDS[encoder], device="cpu")
    E = model.encode([canon(t) for t in texts], batch_size=batch, normalize_embeddings=True,
                     show_progress_bar=False)
    return np.stack([quantize_int8(e) for e in np.asarray(E, dtype=np.float64)])


def recall_at_m(got: Sequence[Sequence[int]], exact: Sequence[Sequence[int]]) -> float:
    """Pooled |got ∩ exact| / |exact| over queries (exact sets may be short under the floor)."""
    tot = sum(len(e) for e in exact)
    return sum(len(set(g) & set(e)) for g, e in zip(got, exact)) / tot if tot else 1.0


def _pct(xs: List[float]) -> Dict[str, float]:
    return {"p50_ms": 1e3 * float(np.percentile(xs, 50)), "p95_ms": 1e3 * float(np.percentile(xs, 95))}


def _peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else 1024 * r)


def scaling_cell(X: np.ndarray, queries: np.ndarray, cells: np.ndarray, qcells: np.ndarray, *,
                 m: int, sigma_lo: int, overfetch: int = 4, M: int = 32, ef_construction: int = 200,
                 ef_search: int = 128) -> Dict:
    n, dim = X.shape
    Xf, norms = X.astype(np.float32), (X.astype(np.int64) ** 2).sum(axis=1)
    out: Dict = {"n": n, "dim": dim, "queries": len(queries), "m": m, "sigma_lo_q": sigma_lo,
                 "overfetch": overfetch, "hnsw": {"M": M, "ef_construction": ef_construction,
                                                  "ef_search": ef_search}}
    lat, exact = [], []
    for q in queries:
        t = time.perf_counter()
        exact.append(topm_exact(q, X, m, sigma_lo, Qf=Xf, norms=norms))
        lat.append(time.perf_counter() - t)
    out["exact_topm"] = _pct(lat)
    out["vectors_bytes"] = int(Xf.nbytes)

    def first_cell(K, c):
        return next((r for r in K if cells[r] == c), None)
    for kind in ("flat", "hnsw"):
        t = time.perf_counter()
        idx = CandidateIndex(dim, kind=kind, M=M, ef_construction=ef_construction, ef_search=ef_search)
        idx.add(X)
        row = {"build_s": time.perf_counter() - t, "index_bytes": idx.nbytes()}
        raw, rer, lat_raw, lat_rer = [], [], [], []
        for q in queries:
            t = time.perf_counter()
            raw.append(idx.search(q, m))
            lat_raw.append(time.perf_counter() - t)
            t = time.perf_counter()
            r = np.array(sorted(idx.search(q, overfetch * m)), dtype=np.int64)
            rer.append(r[topm_exact(q, X[r], m, sigma_lo, Qf=Xf[r], norms=norms[r])].tolist())
            lat_rer.append(time.perf_counter() - t)
        has = [i for i, e in enumerate(exact) if e]
        elig = [i for i in has if first_cell(exact[i], qcells[i]) is not None]
        row.update(
            query_raw=_pct(lat_raw), query_reranked=_pct(lat_rer),
            recall_at_m_raw=recall_at_m(raw, exact),
            recall_at_m_reranked=recall_at_m(rer, exact),
            k_agreement=sum(a == e for a, e in zip(rer, exact)) / len(exact),
            primary_agreement=(sum(rer[i][:1] == exact[i][:1] for i in has) / len(has)) if has else 1.0,
            eligible_agreement=(sum(first_cell(rer[i], qcells[i]) == first_cell(exact[i], qcells[i])
                                    for i in elig) / len(elig)) if elig else 1.0,
            queries_with_candidates=len(has), queries_with_eligible=len(elig))
        out[kind] = row
        del idx
    out["peak_rss_bytes"] = _peak_rss_bytes()
    return out


# --------------------------------------------------------------------------- admission flag

_VOCAB = [f"w{i:03d}" for i in range(400)]


def synthetic_texts(n: int, *, seed: int, family: int = 5, words: int = 8, swaps: int = 2) -> List[str]:
    """n distinct texts in families of ``family``: a base of ``words`` vocabulary words, each
    member with ``swaps`` positions replaced, so members are near neighbours under a bag-of-words
    encoder and families are mostly apart."""
    rng = random.Random(seed)
    out: List[str] = []
    seen = set()
    while len(out) < n:
        base = rng.sample(_VOCAB, words)
        for _ in range(family):
            t = list(base)
            for i in rng.sample(range(words), swaps):
                t[i] = rng.choice(_VOCAB)
            s = " ".join(t)
            if s not in seen and len(out) < n:
                seen.add(s)
                out.append(s)
    return out


def admission_agreement(texts: Sequence[str], *, index: dict, encoder: str, nli: str,
                        pol: Optional[dict] = None,
                        theta_q: Optional[int] = None) -> Dict:
    """Admit ``texts`` (one cell, canonical-order arrival) through exact admission and through
    admission with the ``index`` flag; per delta, compare K and the decision. With ``theta_q``
    both replicas first hold one epoch (one bin, that threshold, the policy's bootstrap selector,
    signed by the scope's calibration authority), so similarity MERGE is live. The result carries
    topology.stamps of that run (encoder, NLI, policy, the epoch if any)."""
    from meld.agent.memory import DEFAULT_CELL, DEFAULT_POL
    from meld.brain.store import Store
    from meld.eval.order_shuffle import claims
    from meld.proto import CANON_VERSION
    from meld.protocol.admission import Admission
    from meld.protocol.fold import nli_provider, readjudicate
    from meld.protocol.objects import INF16, EpochRecord, m_nli, m_pol_of, scope_partition, signed
    pol = pol or DEFAULT_POL
    key = b"index-agreement-key"
    mn = m_nli(encoder, nli, "int8", CANON_VERSION)
    deltas = claims(encoder, pol, [(t, DEFAULT_CELL) for t in texts])
    epoch = None
    if theta_q is not None:
        scope = scope_partition(pol)[DEFAULT_CELL]
        lo, c, m = pol["bootstrap_selector"]
        epoch = EpochRecord(scope=scope, seq=1, m_nli=mn, m_pol=m_pol_of(pol), policy="SEED",
                            alpha=20_000, alpha_c=20_000, mondrian_c=False, cal_root=bytes(16),
                            cal_hw="cpu", bins=(1 << 30, "doubling"), table=((0, 100, 0, theta_q, INF16),),
                            sigma_lo_q=lo, sigma_c_q=c, m=m,
                            author=pol["auth_policy"]["calibration_authority"][scope], created_at=0)

    def run(flag):
        s = Store(pol, key, encoder=encoder)
        if epoch is not None and s.apply(*signed(epoch, key)) is None:
            raise ValueError("epoch refused")
        adm = Admission(s, m_nli=mn, provider=nli_provider(nli),
                        issuer="r", hw="cpu", index=flag)
        t = time.perf_counter()
        decs = [adm.admit(s.apply(*signed(d, key))) for d in deltas]
        return decs, adm.counters, time.perf_counter() - t, s
    exact, _, t_exact, _ = run(None)
    approx, counters, t_index, s = run(index)
    passed = readjudicate(s, scope_partition(pol)[DEFAULT_CELL], m_nli=mn, provider=nli_provider(nli),
                          issuer="r", hw="cpu").decisions
    kmiss = [i for i, (a, b) in enumerate(zip(exact, approx)) if a.K != b.K]
    dmiss = [i for i, (a, b) in enumerate(zip(exact, approx)) if a.decision != b.decision]
    pre = counters["prefiltered"]
    from meld.eval.topology import stamps
    return {**stamps(encoder, nli, epoch, pol), "n": len(deltas), "index": dict(index), "prefiltered": pre, "k_miss": len(kmiss),
            "k_miss_rate": len(kmiss) / pre if pre else 0.0, "decision_mismatch": len(dmiss),
            "first_k_miss": kmiss[0] if kmiss else None,
            "first_decision_mismatch": dmiss[0] if dmiss else None,
            "decisions": dict(Counter(d.decision for d in exact)),
            "pass_equals_exact": list(passed) == exact,     # the exact pass repairs every miss
            "seconds_exact": t_exact, "seconds_index": t_index}


__all__ = ["clustered_int8", "embed_texts", "recall_at_m", "scaling_cell", "synthetic_texts",
           "admission_agreement"]
