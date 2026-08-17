"""Benchmark identity gate (G0) helpers.

Loads the verified-at-source provenance for the evaluation benchmarks and
exposes a deterministic identity hash. The provenance JSON lives under the
gitignored ``data/`` tree (it is operational input, re-derivable by re-fetching
the source); the EXPECTED identity (name + identity hash + levels) is tracked in
``benchmark_identity_expected.json`` next to this module and guarded by
``tests/test_benchmark_identity.py``. No benchmark data is wired before the
identity test is green.

The identity hash is the sha256 over the canonicalised identity fields
(name, arxiv_id, authors, metric, levels). It pins WHICH benchmark a wired
loader is allowed to claim it loaded, catching a silent LongMemEval ->
LongMemBench (or GAIA-level) drift.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
PROVENANCE_PATH = os.path.join(REPO_ROOT, "data", "benchmark_provenance.json")
EXPECTED_PATH = os.path.join(HERE, "benchmark_identity_expected.json")

# Identity fields, in canonical order, that define WHICH benchmark this is.
_IDENTITY_FIELDS = ("name", "arxiv_id", "authors", "metric", "levels")


def identity_hash(entry: Dict[str, Any]) -> str:
    """Deterministic sha256 over the canonical identity fields of a provenance
    entry. Order-independent for ``authors``/``levels`` (sorted)."""
    canon: Dict[str, Any] = {}
    for f in _IDENTITY_FIELDS:
        v = entry.get(f)
        if isinstance(v, list):
            v = sorted(v)
        canon[f] = v
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_provenance(benchmark: str) -> Dict[str, Any]:
    """Load the gitignored, verified-at-source provenance for ``benchmark``.

    Raises ``FileNotFoundError`` (gate not satisfied) if the provenance file
    has not been written from a source verification yet.
    """
    if not os.path.exists(PROVENANCE_PATH):
        raise FileNotFoundError(
            f"benchmark provenance not written yet at {PROVENANCE_PATH}; "
            "verify the benchmark at source before wiring any data (G0)."
        )
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if benchmark not in data:
        raise KeyError(f"no provenance entry for benchmark {benchmark!r}")
    return data[benchmark]


def load_expected() -> Dict[str, Any]:
    with open(EXPECTED_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)
