"""G0 — benchmark identity gate.

Asserts the gitignored, verified-at-source provenance matches the tracked
EXPECTED identity. Fails (gate not satisfied) until the provenance file has
been written from a real source verification. This guards against a silent
LongMemEval -> LongMemBench drift, or a GAIA level-set drift, before any
benchmark data is wired into a loader.
"""
from __future__ import annotations

import pytest

from meld.eval.benchmark_provenance import (
    identity_hash,
    load_expected,
    load_provenance,
)


def test_longmemeval_identity_matches_verified_source():
    expected = load_expected()["longmemeval"]
    got = load_provenance("longmemeval")
    assert got["name"] == expected["name"]
    assert got["arxiv_id"] == expected["arxiv_id"]
    assert identity_hash(got) == expected["identity_sha256"]


def test_longmemeval_is_not_longmembench():
    got = load_provenance("longmemeval")
    # The distinct-from note is load-bearing: confirm the name is exactly
    # LongMemEval and the distractor name is recorded as NOT this benchmark.
    assert got["name"] == "LongMemEval"
    assert got["distinct_from"] == "LongMemBench"


def test_gaia_levels_present():
    got = load_provenance("gaia")
    assert set(got["levels"]) >= {"L1", "L2", "L3"}


def test_gaia_identity_matches_verified_source():
    expected = load_expected()["gaia"]
    got = load_provenance("gaia")
    assert got["name"] == expected["name"]
    assert got["arxiv_id"] == expected["arxiv_id"]
    assert identity_hash(got) == expected["identity_sha256"]


def test_provenance_missing_raises_before_g0(monkeypatch):
    # If the provenance file is absent, the loader must refuse (gate closed),
    # not silently return a default.
    import meld.eval.benchmark_provenance as bp

    monkeypatch.setattr(bp, "PROVENANCE_PATH", "/nonexistent/benchmark_provenance.json")
    with pytest.raises(FileNotFoundError):
        bp.load_provenance("longmemeval")
