#!/usr/bin/env python3
"""Smoke test for the Kafka transport against a local broker (localhost:9092).

Skips cleanly if no broker is reachable, so the suite still runs without Kafka.
Proves the publish -> consume round-trip the cross-tier brain/driver rely on.

Run:  PYTHONPATH=. python3 -m pytest cross-tier/test_kafka_transport.py -q
"""
import os, sys, time, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BOOTSTRAP = os.environ.get("SMP_KAFKA_BOOTSTRAP", "localhost:9092")


def _broker_up():
    try:
        from kafka import KafkaProducer
        KafkaProducer(bootstrap_servers=BOOTSTRAP, request_timeout_ms=2000,
                      api_version_auto_timeout_ms=2000).close(timeout=2)
        return True
    except Exception:
        return False


def _skip_if_no_broker():
    if not _broker_up():
        import pytest
        pytest.skip("no Kafka broker at %s" % BOOTSTRAP)


def test_publish_consume_roundtrip():
    _skip_if_no_broker()
    from kafka_transport import KafkaTransport
    t = KafkaTransport(BOOTSTRAP)
    topic = "meld-test-%s" % uuid.uuid4().hex[:8]
    try:
        t.publish(topic, "hello-continuum")
        c = t.consumer(topic, from_beginning=True, timeout_ms=8000)
        got = [m.value for m in c]
        c.close()
        assert "hello-continuum" in got, got
    finally:
        t.close()


def test_broadcast_two_groups_both_receive():
    """Two consumers in different groups (the broadcast pattern brains use) both
    see the message — the analogue of the relay's deliver-to-all-subscribers."""
    _skip_if_no_broker()
    from kafka_transport import KafkaTransport
    t = KafkaTransport(BOOTSTRAP)
    topic = "meld-bcast-%s" % uuid.uuid4().hex[:8]
    try:
        t.publish(topic, "m1")
        c1 = t.consumer(topic, group="g-a", from_beginning=True, timeout_ms=8000)
        c2 = t.consumer(topic, group="g-b", from_beginning=True, timeout_ms=8000)
        got1 = [m.value for m in c1]; c1.close()
        got2 = [m.value for m in c2]; c2.close()
        assert "m1" in got1 and "m1" in got2, (got1, got2)
    finally:
        t.close()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:  # noqa: BLE001
                import traceback
                if "Skipped" in type(e).__name__ or "skip" in str(e).lower():
                    print("SKIP", name, str(e)[:60]); continue
                fails += 1; print("FAIL", name, "->", repr(e)[:200])
    print("=== %d failure(s) ===" % fails)
    sys.exit(1 if fails else 0)
