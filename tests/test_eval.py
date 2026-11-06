"""Task 1.9 — eval loader + topology + metrics.

- longmemeval_loader: a conversation slice yields well-formed Concept/Claim/
  Evidence deltas. (For the local smoke this is a small synthetic LongMemEval-
  SHAPED fixture: it validates the PIPELINE, not a benchmark result. The
  benchmark identity gate G0 governs any real LongMemEval data wiring.)
- metrics: accuracy / recall@k / MRR / byte-count / version-count counters
  correct on a tiny fixture.
"""
from __future__ import annotations

from meld.eval.longmemeval_loader import load_conversation_slice
from meld.eval.metrics import byte_count, mrr, recall_at_k, version_count
from meld.proto import MemoryUnit


def test_loader_yields_wellformed_deltas():
    convs = load_conversation_slice(n_conversations=3, seed=0)
    assert len(convs) == 3
    for c in convs:
        units = c["units"]
        assert units and all(isinstance(u, MemoryUnit) for u in units)
        assert all(u.claims for u in units)


def test_loader_is_deterministic():
    a = load_conversation_slice(n_conversations=4, seed=1)
    b = load_conversation_slice(n_conversations=4, seed=1)
    assert [u.id for c in a for u in c["units"]] == [u.id for c in b for u in c["units"]]


def test_loader_has_known_coreferent_and_contradiction_pairs():
    convs = load_conversation_slice(n_conversations=6, seed=0)
    labels = [p for c in convs for p in c.get("gold_pairs", [])]
    kinds = {lab["kind"] for lab in labels}
    assert "coreferent" in kinds
    assert "contradiction" in kinds


def test_metrics_on_tiny_fixture():
    # recall@k / MRR over a ranked id list against a gold id
    ranked = [b"a", b"b", b"c"]
    assert recall_at_k(ranked, gold=b"b", k=2) == 1.0
    assert recall_at_k(ranked, gold=b"c", k=2) == 0.0
    assert mrr(ranked, gold=b"a") == 1.0
    assert mrr(ranked, gold=b"c") == 1 / 3


def test_byte_and_version_counts():
    units = [MemoryUnit(retention=0.0, id=b"u1"), MemoryUnit(retention=0.0, id=b"u2")]
    assert byte_count(units) > 0
    assert version_count(units) == 2
