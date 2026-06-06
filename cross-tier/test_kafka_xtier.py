#!/usr/bin/env python3
"""System test: the SMP merge protocol end-to-end over REAL Apache Kafka.

Drives the whole Kafka kit on localhost --- three kafka_brain subprocesses (the
standard-fabric brains) + the driver, over a real broker --- and asserts the same
contract the cross-tier deployment claims: every brain acks every probe and the
running-example decisions (MERGE/RELATE/CONFLICT) + the OVERRULED CRDT join
reproduce identically across all three. This is the in-suite analogue of the
multi-tier Kafka deployment; it skips cleanly when no broker is reachable so the
suite still runs without Kafka.

Run:  PYTHONPATH=src:cross-tier python3 -m pytest cross-tier/test_kafka_xtier.py -q
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _skip_if_no_broker():
    import pytest
    # kafka-python may be absent (e.g. laptop without the Kafka kit): skip, not fail.
    # Importing kafka_xtier pulls in kafka_transport -> `from kafka import ...`, so the
    # module check must precede the import or it raises ModuleNotFoundError at collection.
    pytest.importorskip("kafka")
    import kafka_xtier
    if not kafka_xtier.broker_up():
        pytest.skip("no Kafka broker at %s" % kafka_xtier.BOOTSTRAP)


def test_kafka_end_to_end_three_brains_converge():
    _skip_if_no_broker()
    import kafka_xtier
    rc = kafka_xtier._local_end_to_end()
    assert rc == 0, "Kafka end-to-end did not PASS (every brain acks 10 probes + " \
                    "MERGE/RELATE/CONFLICT/OVERRULED on all three); rc=%r" % rc


if __name__ == "__main__":
    try:
        test_kafka_end_to_end_three_brains_converge()
        print("PASS test_kafka_end_to_end_three_brains_converge")
    except Exception as e:  # noqa: BLE001
        print("FAIL ->", repr(e)[:200]); sys.exit(1)
