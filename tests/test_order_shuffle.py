"""S3: smoke tests for the order-shuffle driver (order_shuffle.py) and the
candidate-cost instrumentation it wires in. Hashing encoder + lexical NLI
throughout (offline, fast, deterministic) -- this is a plumbing/shape test,
not a re-run of the real minilm/cross-encoder workloads (scripts/run_order_shuffle.py)."""
from __future__ import annotations

from meld.agent.memory import _unit_from_text
from meld.config import load_thresholds
from meld.eval.candidate_cost import CandidateCostRecorder, instrument_best_candidate
from meld.eval.order_shuffle import run_order_shuffle
from meld.merge.signals import EmbeddingSimilarity
from meld.protocol.merge_on_receive import merge_on_receive


def _tiny_workload():
    return [_unit_from_text("Paris is the capital of France"),
           _unit_from_text("The capital city of France is Paris"),
           _unit_from_text("Mount Everest is the tallest mountain on Earth")]


def test_driver_k3_smoke_shape():
    cfg = load_thresholds()
    report = run_order_shuffle(_tiny_workload(), k_orderings=3, seed=0,
                               encoder="hashing", nli="lexical", cfg=cfg,
                               probe_queries=["Paris capital France"])
    assert report["k_orderings"] == 3
    assert report["n_workload_items"] == 3
    assert len(report["orderings"]) == 3
    for c in ("live_claims", "status_map", "links", "representatives", "joint"):
        assert report["distinct_fingerprints"][c] >= 1
    assert "Paris capital France" in report["retrieval_agreement"]
    assert report["retrieval_agreement"]["Paris capital France"]["agree"] in (True, False)
    assert report["candidate_cost"]["n_calls"] >= 1
    assert 0.0 <= report["novel_rate"]["median"] <= 1.0


def test_driver_reproducible_for_fixed_seed():
    cfg = load_thresholds()
    kwargs = dict(k_orderings=3, seed=7, encoder="hashing", nli="lexical", cfg=cfg)
    r1 = run_order_shuffle(_tiny_workload(), **kwargs)
    r2 = run_order_shuffle(_tiny_workload(), **kwargs)
    assert [o["joint_digest"] for o in r1["orderings"]] == [o["joint_digest"] for o in r2["orderings"]]
    assert [o["order"] for o in r1["orderings"]] == [o["order"] for o in r2["orderings"]]


def test_status_event_item_delivers_without_going_through_classifier():
    # a bare (claim_key, link_id, effect) workload item is a status-CRDT
    # delivery, not a claim -- it must not touch merge_on_receive at all.
    cfg = load_thresholds()
    workload = [_unit_from_text("Water boils at 100 degrees Celsius"),
               (b"some-claim-key", "overrule.1", "OVERRULED")]
    report = run_order_shuffle(workload, k_orderings=3, seed=1, encoder="hashing",
                               nli="lexical", cfg=cfg)
    assert report["k_orderings"] == 3
    # the status event is excluded from decision_counts (it carries no
    # classifier decision -- see gates_that_fired=["status-link"] filtering).
    for o in report["orderings"]:
        assert sum(o["decision_counts"].values()) == 1   # only the one real claim delta


def test_instrumentation_does_not_change_decisions():
    """Wrapping ``_best_candidate`` must be behaviour-preserving: same inputs,
    same decisions, with sizes/latencies recorded on the side."""
    cfg = load_thresholds()
    es = EmbeddingSimilarity(encoder="hashing")
    local = [_unit_from_text("Paris is the capital of France")]
    incoming = _unit_from_text("The capital city of France is Paris")

    baseline = merge_on_receive(incoming, local, cfg=cfg, es=es, nli="lexical")

    rec = CandidateCostRecorder()
    with instrument_best_candidate(rec):
        instrumented = merge_on_receive(incoming, local, cfg=cfg, es=es, nli="lexical")

    assert [p.decision for p in baseline] == [p.decision for p in instrumented]
    assert [p.emitted for p in baseline] == [p.emitted for p in instrumented]
    assert rec.candidate_set_sizes == [1]
    assert len(rec.scan_latencies_ms) == 1
    assert rec.scan_latencies_ms[0] >= 0.0
