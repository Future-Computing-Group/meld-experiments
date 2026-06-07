"""Wire signing on the receive seam: verify-before-admission (TDD, M5).

The manuscript states a delta is signed by the publisher and the receiving brain
VERIFIES the signature before reading any merge signal; a forged or tampered
delta is dropped before admission. HMAC over a shared group key is the
experiment stand-in (benign-fault, keyed-group integrity/attribution -- NOT a
PKI / Byzantine model). These tests pin:
  (a) a validly-signed delta is admitted (the claim merges);
  (b) a tampered or wrong-key delta is rejected before admission (no merge).

Run:  PYTHONPATH=src python3 -m pytest tests/test_wire_signing.py -q
"""
from __future__ import annotations

from smp.sync.node import BrainNode
from smp.sync.wire_delta import deserialize_verified_delta, group_key

KEY = b"smp-test-group-key"


def _cfg():
    from smp.agent.run_hotpot_gate import _calibrated_cfg
    return _calibrated_cfg("hashing")


def _node(nid):
    return BrainNode(nid, encoder="hashing", nli="lexical", cfg=_cfg())


# --------------------------------------------------------------------------- #
# group key
# --------------------------------------------------------------------------- #
def test_group_key_reads_env_and_falls_back_to_documented_test_key():
    assert group_key({"SMP_GROUP_KEY": "abc"}) == b"abc"
    # no env var -> a documented default test key (bytes), never empty
    default = group_key({})
    assert isinstance(default, bytes) and default


# --------------------------------------------------------------------------- #
# (a) validly-signed delta is admitted and merges
# --------------------------------------------------------------------------- #
def test_validly_signed_delta_is_admitted_and_merges():
    a, b = _node("A"), _node("B")
    a.ingest("Paris is the capital of France")
    payload = a.publish_wire_payloads(KEY)[0]
    b.receive_wire(payload, KEY)
    assert any("Paris" in t for t in b.live_claim_texts())


def test_serialize_then_verify_roundtrips_node_and_unit():
    a = _node("A")
    a.ingest("Mount Everest is the tallest mountain")
    payload = a.publish_wire_payloads(KEY)[0]
    node_id, unit = deserialize_verified_delta(payload, KEY)
    assert node_id == "A"
    assert any("Everest" in c.text for c in unit.claims)


# --------------------------------------------------------------------------- #
# (b) tampered / wrong-key delta is rejected BEFORE admission (no merge)
# --------------------------------------------------------------------------- #
def test_tampered_payload_is_rejected_before_admission():
    a, b = _node("A"), _node("B")
    a.ingest("Paris is the capital of France")
    payload = a.publish_wire_payloads(KEY)[0]
    tampered = payload.replace(b"Paris", b"Berlin")
    # verify fails -> None, nothing merged
    assert deserialize_verified_delta(tampered, KEY) is None
    b.receive_wire(tampered, KEY)
    assert b.live_claim_texts() == set()


def test_wrong_key_delta_is_rejected_before_admission():
    a, b = _node("A"), _node("B")
    a.ingest("Paris is the capital of France")
    payload = a.publish_wire_payloads(KEY)[0]
    assert deserialize_verified_delta(payload, b"some-other-group-key") is None
    b.receive_wire(payload, b"some-other-group-key")
    assert b.live_claim_texts() == set()


def test_unsigned_payload_is_rejected():
    import json

    bogus = json.dumps({"node_id": "A", "unit": {}}).encode("utf-8")  # no sig
    assert deserialize_verified_delta(bogus, KEY) is None
