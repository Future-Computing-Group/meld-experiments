"""S8: smoke tests for the claim-count scaling experiment (candidate_scaling.py).

Hashing encoder + lexical NLI throughout (offline, fast, deterministic) --
a plumbing/shape check, not a re-run of the real minilm/cross-encoder sweep
(scripts/run_candidate_scaling.py)."""
from __future__ import annotations

from smp.config import load_thresholds
from smp.eval.candidate_scaling import (build_incoming_stream, build_local_store,
                                        run_candidate_scaling)


def _run(sizes=(4, 8, 16), k=5, seed=0):
    cfg = load_thresholds()
    return run_candidate_scaling(sizes, k_per_size=k, seed=seed,
                                 encoder="hashing", nli="lexical", cfg=cfg)


def test_shape_and_monotonicity():
    report = _run()
    assert report["store_sizes"] == [4, 8, 16]
    assert len(report["per_size"]) == 3

    sizes_seen = [row["n_local"] for row in report["per_size"]]
    assert sizes_seen == [4, 8, 16]

    for row in report["per_size"]:
        cc = row["candidate_cost"]
        assert cc["n_calls"] == 5
        # every incoming text is distinct from every held text by
        # construction (no kappa==1 early return) -> the scan always covers
        # every ACTIVE local claim.
        assert cc["candidate_set_size_median"] == row["n_local"]
        assert cc["candidate_set_size_p95"] == row["n_local"]
        assert cc["scan_latency_ms_median"] >= 0.0
        assert row["n_matched"] + row["n_unmatched"] == 5

    # candidate-set size scanned grows monotonically with store size.
    medians = [row["candidate_cost"]["candidate_set_size_median"] for row in report["per_size"]]
    assert medians == sorted(medians)
    assert medians[0] < medians[-1]

    fit = report["fit"]
    for key in ("intercept_ms", "slope_ms_per_claim", "r2", "ms_per_1k_claims"):
        assert key in fit


def test_deterministic_fingerprint():
    r1 = _run(sizes=(4, 8), k=6, seed=3)
    r2 = _run(sizes=(4, 8), k=6, seed=3)
    # decisions and candidate-set sizes are content, not timing -- must be
    # byte-for-byte reproducible under a fixed seed.
    for row1, row2 in zip(r1["per_size"], r2["per_size"]):
        assert row1["decision_counts"] == row2["decision_counts"]
        assert row1["candidate_cost"]["candidate_set_size_median"] == \
            row2["candidate_cost"]["candidate_set_size_median"]
        assert row1["n_matched"] == row2["n_matched"]
        assert row1["n_unmatched"] == row2["n_unmatched"]


def test_incoming_stream_is_a_seeded_mix_of_paraphrase_and_novel():
    stream_a = build_incoming_stream(16, 20, seed=1)
    stream_b = build_incoming_stream(16, 20, seed=1)
    assert [u.claims[0].text for u in stream_a] == [u.claims[0].text for u in stream_b]
    held_texts = {u.claims[0].text for u in build_local_store(16)}
    # a mix: at least one incoming text is NOT a held text (novel) and the
    # stream isn't a trivial constant.
    incoming_texts = [u.claims[0].text for u in stream_a]
    assert len(set(incoming_texts)) > 1
    assert not held_texts & set(incoming_texts)  # paraphrases reword, never repeat verbatim
