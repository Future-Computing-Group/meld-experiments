"""The candidate index (merge/candidate_index.py): FAISS flat and HNSW over L2-normalised int8
embeddings, an off-protocol scalability instrument and an optional admission-path pre-filter.
The fold and calibration never read it; the admission path with the flag re-ranks the pre-filtered
rows exactly (signals.topm_exact), so its decision equals the exact path's whenever the exact top-m
lies inside the pre-filter."""
from __future__ import annotations

import functools
import os
import subprocess
import sys

import numpy as np
import pytest

from meld.merge.candidate_index import CandidateIndex
from meld.merge.classifier import FoldState, Node, select_tested_pair
from meld.merge.signals import topm_exact
from meld.eval.index_scaling import (
    admission_agreement, clustered_int8, recall_at_m, scaling_cell, synthetic_texts,
)
from meld.merge.m2_gold import _SEEDS
from meld.protocol.objects import FIXED_ONE

FLOOR = -FIXED_ONE          # no floor: every row is a candidate
_CHILD = "MELD_FAISS_CHILD"


def isolated(fn):
    """Run the test body in a child pytest process: FAISS and torch bundle separate OpenMP
    runtimes on macOS and abort when both load in one process, so the suite's process never
    loads FAISS. The child runs the same test function and its assertions; any failure fails here."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if os.environ.get(_CHILD):
            return fn(*args, **kwargs)
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            f"{__file__}::{fn.__name__}"], env={**os.environ, _CHILD: "1"},
                           capture_output=True, text=True)
        assert r.returncode == 0 and "1 passed" in r.stdout, r.stdout[-4000:] + r.stderr[-2000:]
    return wrapper


def _exact(Q, qs, m):
    return [topm_exact(q, Q, m, FLOOR) for q in qs]


def test_clustered_corpus_is_seeded_int8():
    a, b = clustered_int8(500, 32, seed=3), clustered_int8(500, 32, seed=3)
    assert a.dtype == np.int8 and a.shape == (500, 32) and np.array_equal(a, b)
    assert not np.array_equal(a, clustered_int8(500, 32, seed=4))


@isolated
def test_flat_index_agrees_with_topm_exact():
    Q = clustered_int8(3000, 48, seed=0)
    qs = clustered_int8(100, 48, seed=1)
    idx = CandidateIndex(48, kind="flat")
    idx.add(Q)
    got = [idx.search(q, 10) for q in qs]
    # float32 inner products of normalised vectors: only exact near-ties may swap at the cut
    assert recall_at_m(got, _exact(Q, qs, 10)) >= 0.999


@isolated
def test_hnsw_recall_at_m_against_topm_exact_on_seeded_corpus():
    Q = clustered_int8(5000, 64, seed=0)
    qs = clustered_int8(200, 64, seed=7)
    exact = _exact(Q, qs, 10)
    good = CandidateIndex(64, kind="hnsw", ef_search=128)
    good.add(Q)
    r_good = recall_at_m([good.search(q, 10) for q in qs], exact)
    assert r_good >= 0.99                                      # documented threshold
    poor = CandidateIndex(64, kind="hnsw", M=4, ef_search=1)
    poor.add(Q)
    assert recall_at_m([poor.search(q, 10) for q in qs], exact) < r_good


@isolated
def test_incremental_add_with_explicit_ids_returns_those_ids():
    Q = clustered_int8(400, 16, seed=2)
    idx = CandidateIndex(16, kind="flat")
    idx.add(Q[:150])
    idx.add(Q[150:], ids=np.arange(1150, 1400))
    assert len(idx) == 400
    # the nearest row to a stored vector is itself (ids carry over)
    assert idx.search(Q[10], 1) == [10]
    assert idx.search(Q[200], 1) == [1200]
    # asking for more than held returns what is held, never a -1 placeholder
    small = CandidateIndex(16, kind="hnsw")
    small.add(Q[:5])
    assert sorted(small.search(Q[0], 50)) == [0, 1, 2, 3, 4]


@isolated
def test_zero_vector_is_stored_and_searchable():
    Q = np.zeros((3, 8), np.int8)
    Q[1, 0] = 5
    idx = CandidateIndex(8, kind="flat")
    idx.add(Q)
    assert sorted(idx.search(np.zeros(8, np.int8), 3)) == [0, 1, 2]
    with pytest.raises(TypeError):
        idx.add(Q.astype(np.float32))                          # int8 rows only
    with pytest.raises(ValueError):
        CandidateIndex(8, kind="ivf")


def _state(Q):
    st = FoldState()
    for i, q in enumerate(Q):
        st.add(Node(bytes([i % 256, i // 256]) * 8, "c", (i, b""), q, "", "ACTIVE", 1))
    return st


def test_select_tested_pair_on_rows_equals_full_when_exact_topm_is_inside():
    Q = clustered_int8(600, 24, seed=5)
    st = _state(Q[:-1])
    b = Node(b"\xff" * 16, "c", (10**6, b""), Q[-1], "", "ACTIVE", 1)
    rank = {k: 0 for k in st.keys + [b.key]}
    full = select_tested_pair(st, b, FLOOR, 4, rank, set())
    exact_rows = topm_exact(b.q, st.matrices()[0], 4, FLOOR)
    rows = sorted(set(exact_rows) | {1, 2, 3, 500})
    assert select_tested_pair(st, b, FLOOR, 4, rank, set(), rows=rows) == full
    # a pre-filter that misses one exact member changes K: the miss is visible, not hidden
    short = [r for r in rows if r != exact_rows[0]]
    assert select_tested_pair(st, b, FLOOR, 4, rank, set(), rows=short)[0] != full[0]


def test_select_tested_pair_on_rows_keeps_canonical_order_ties():
    q = np.array([3, 4, 0, 0], np.int8)
    Q = np.stack([q, np.array([0, 0, 5, 0], np.int8), q, q])  # rows 0, 2, 3 tie exactly
    st = _state(Q)
    b = Node(b"\xfe" * 16, "c", (10**6, b""), q, "", "ACTIVE", 1)
    rank = {k: 0 for k in st.keys + [b.key]}
    K, _ = select_tested_pair(st, b, FLOOR, 2, rank, set(), rows=[3, 2, 0])
    assert K == [st.keys[0], st.keys[2]]


def test_synthetic_texts_share_words_within_a_family():
    t = synthetic_texts(40, seed=0)
    assert len(t) == len(set(t)) == 40 and t == synthetic_texts(40, seed=0)


def _workload():
    """The gold seeds (paraphrases, related claims, lexical contradictions) and synthetic families."""
    return [t for _, _, *ts in _SEEDS for t in ts] + synthetic_texts(600, seed=1)


@isolated
def test_admission_with_index_decides_as_exact_admission_at_generous_ef():
    rep = admission_agreement(_workload(), index={"kind": "hnsw", "ef_search": 256}, encoder="hashing", nli="lexical",
                              theta_q=round(0.7 * FIXED_ONE))
    assert rep["n"] == 704 and rep["prefiltered"] == 703       # every latency step used the index
    assert set(rep["decisions"]) == {"INSERT", "MERGE", "RELATE", "CONFLICT"}
    assert rep["k_miss"] == 0 and rep["decision_mismatch"] == 0 and rep["pass_equals_exact"]


@isolated
def test_admission_with_index_counts_prefilter_misses_and_the_pass_repairs_them():
    rep = admission_agreement(_workload(), index={"kind": "hnsw", "M": 2, "ef_search": 1, "k": 3},
                              encoder="hashing", nli="lexical",
                              theta_q=round(0.7 * FIXED_ONE))
    assert rep["k_miss"] > 0 and rep["decision_mismatch"] > 0  # reported, not hidden
    assert rep["k_miss_rate"] == rep["k_miss"] / rep["prefiltered"]
    # a decision can only diverge at or after the first pre-filter miss
    assert rep["first_decision_mismatch"] >= rep["first_k_miss"]
    assert rep["pass_equals_exact"]


@isolated
def test_scaling_cell_reports_exact_flat_and_hnsw():
    X = clustered_int8(2000, 32, seed=0)
    qs = np.concatenate([X[:20] + np.int8(1), clustered_int8(20, 32, seed=9)]).astype(np.int8)
    cells = np.arange(2000) % 4
    rep = scaling_cell(X, qs, cells, np.arange(40) % 4, m=8, sigma_lo=round(0.5 * FIXED_ONE))
    assert rep["n"] == 2000 and rep["queries"] == 40 and rep["peak_rss_bytes"] > 0
    flat, hnsw = rep["flat"], rep["hnsw"]
    assert flat["k_agreement"] == flat["primary_agreement"] == flat["eligible_agreement"] == 1.0
    assert hnsw["recall_at_m_reranked"] >= hnsw["recall_at_m_raw"] - 1e-12
    assert hnsw["queries_with_eligible"] > 0 and hnsw["index_bytes"] > flat["index_bytes"] > 2000 * 32 * 4
    for k in ("exact_topm",):
        assert 0 < rep[k]["p50_ms"] <= rep[k]["p95_ms"]
    for row in (flat, hnsw):
        assert row["build_s"] > 0 and 0 < row["query_raw"]["p50_ms"] <= row["query_raw"]["p95_ms"]


@isolated
def test_run_index_scaling_script_on_a_small_synthetic_corpus(tmp_path):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import run_index_scaling
    out = tmp_path / "idx.json"
    rep = run_index_scaling.main(["--corpus", "synthetic", "--encoder", "hashing", "--nli", "lexical",
                                  "--sizes", "1000", "400", "--queries", "40",
                                  "--dim", "32", "--out", str(out)])
    assert [c["n"] for c in rep["cells"]] == [400, 1000] and out.exists()
    assert all(c["queries"] == 40 and c["flat"]["k_agreement"] == 1.0 for c in rep["cells"])
    # half the queries are perturbed stored rows: they have candidates above the floor
    assert all(c["hnsw"]["queries_with_candidates"] >= 20 for c in rep["cells"])


@isolated
def test_index_scaling_rows_are_stamped():
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_index_scaling.py"
    spec = importlib.util.spec_from_file_location("run_index_scaling_stamps", p)
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    STAMP = {"git_sha", "m_nli", "m_pol", "epoch_id", "seq", "nli_id", "encoder"}
    rep = run.main(["--corpus", "synthetic", "--sizes", "200", "--queries", "10",
                    "--dim", "16", "--encoder", "hashing", "--nli", "lexical"])
    for c in rep["cells"]:              # the synthetic scan embeds nothing and runs no NLI
        assert STAMP <= set(c) and c["encoder"] is None and c["nli_id"] is None and c["m_nli"] is None
    from meld.eval.index_scaling import admission_agreement
    from meld.merge.m2_gold import _SEEDS
    from meld.protocol.objects import FIXED_ONE
    r = admission_agreement([t for _, _, *ts in _SEEDS[:3] for t in ts], index={"kind": "hnsw"},
                            encoder="hashing", nli="lexical", theta_q=round(0.7 * FIXED_ONE))
    assert STAMP <= set(r) and r["seq"] == 1 and r["epoch_id"] and r["nli_id"] == "lexical"
