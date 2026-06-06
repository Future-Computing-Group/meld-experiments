#!/usr/bin/env python3
"""M3a: seeded multi-trial cross-tier partition-heal over REAL Apache Kafka (TDD).

The single-trial existence proof (test_kafka_xtier) shows the protocol runs over the
standard fabric; M3a upgrades it to a QUANTITATIVE distributed result --- a
reconvergence-success fraction and a p50/p95 heal-latency distribution over N seeded
trials, the form a reviewer asked for instead of two point values.

Two layers, two test kinds:
  * aggregate_trials() is a pure function over per-trial records -> unit test, no broker.
  * run_partition_heal() drives real kafka_brain subprocesses, partitioning the edge brain
    with SIGSTOP/SIGCONT (its consumer freezes, then re-consumes the missed offset on heal,
    the delivery-driven partition model the main paper states) -> broker-gated integration.

Run:  PYTHONPATH=src:cross-tier python3 -m pytest cross-tier/test_kafka_partition_heal.py -q
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---- unit: pure aggregation (no broker) ----------------------------------------

def test_aggregate_success_fraction_and_percentiles():
    from kafka_partition_heal import aggregate_trials
    trials = [
        {"converged": True,  "heal_ms": 100.0, "diverged_first": True, "seed": 1},
        {"converged": True,  "heal_ms": 200.0, "diverged_first": True, "seed": 2},
        {"converged": True,  "heal_ms": 300.0, "diverged_first": True, "seed": 3},
        {"converged": False, "heal_ms": 0.0,   "diverged_first": True, "seed": 4},
    ]
    agg = aggregate_trials(trials)
    assert agg["n_trials"] == 4
    assert agg["n_converged"] == 3
    assert abs(agg["reconvergence_success_fraction"] - 0.75) < 1e-9
    # percentiles are over the CONVERGED trials' heal latencies only ([100,200,300])
    assert agg["heal_ms_p50"] == 200.0
    assert agg["heal_ms_p95"] == 300.0
    assert abs(agg["heal_ms_mean"] - 200.0) < 1e-9
    assert agg["all_diverged_first"] is True


def test_aggregate_empty_is_well_defined():
    from kafka_partition_heal import aggregate_trials
    agg = aggregate_trials([])
    assert agg["n_trials"] == 0
    assert agg["reconvergence_success_fraction"] == 0.0
    assert agg["heal_ms_p50"] == 0.0 and agg["heal_ms_p95"] == 0.0


def test_aggregate_flags_a_trial_that_never_diverged():
    # if a trial did not diverge first, the partition was not actually exercised;
    # all_diverged_first must drop to False so the run is not silently trusted.
    from kafka_partition_heal import aggregate_trials
    agg = aggregate_trials([
        {"converged": True, "heal_ms": 50.0, "diverged_first": True,  "seed": 1},
        {"converged": True, "heal_ms": 60.0, "diverged_first": False, "seed": 2},
    ])
    assert agg["all_diverged_first"] is False


# ---- integration: real Kafka, bounded (broker-gated) ---------------------------

def _skip_if_no_broker():
    import pytest
    pytest.importorskip("kafka")
    import kafka_partition_heal as m
    if not m.broker_up():
        pytest.skip("no Kafka broker at %s" % m.BOOTSTRAP)


def test_partition_heal_two_trials_over_real_kafka():
    _skip_if_no_broker()
    import kafka_partition_heal as m
    agg = m.run_partition_heal(n_trials=2, seed0=1)
    # every trial must have diverged under partition and reconverged after heal
    assert agg["n_trials"] == 2
    assert agg["all_diverged_first"] is True
    assert agg["reconvergence_success_fraction"] == 1.0
    assert agg["heal_ms_p50"] > 0.0


if __name__ == "__main__":
    test_aggregate_success_fraction_and_percentiles()
    test_aggregate_empty_is_well_defined()
    test_aggregate_flags_a_trial_that_never_diverged()
    print("PASS unit")
