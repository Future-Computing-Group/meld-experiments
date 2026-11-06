"""Wire signing on the receive seam: verify-before-admission (TDD, M5).

The manuscript states a delta is signed by the publisher and the receiving brain
VERIFIES the signature before reading any merge signal; a forged or tampered
delta is dropped before admission. HMAC over a shared group key is the
experiment stand-in (benign-fault, keyed-group integrity/attribution -- NOT a
PKI / Byzantine model). These tests pin:
  (a) a validly-signed delta is admitted into the replicated set;
  (b) a tampered or wrong-key delta is rejected before admission (no merge).

Run:  PYTHONPATH=src python3 -m pytest tests/test_wire_signing.py -q
"""
from __future__ import annotations

from meld.sync.node import BrainNode
from meld.sync.wire_delta import group_key

KEY = b"meld-test-group-key"




def _node(nid, key=KEY):
    return BrainNode(nid, encoder="hashing", nli="lexical", key=key)


def _published(a):
    """A's one signed Delta as it travels: (canonical bytes, mac, scope)."""
    (oid,) = a.outbox()
    return a.objects.raw(oid)


# --------------------------------------------------------------------------- #
# group key
# --------------------------------------------------------------------------- #
def test_group_key_reads_env_and_falls_back_to_documented_test_key():
    assert group_key({"SMP_GROUP_KEY": "abc"}) == b"abc"
    # no env var -> a documented default test key (bytes), never empty
    default = group_key({})
    assert isinstance(default, bytes) and default


# --------------------------------------------------------------------------- #
# (a) validly-signed delta is admitted into the replicated set
# --------------------------------------------------------------------------- #
def test_validly_signed_delta_is_admitted():
    a, b = _node("A"), _node("B")
    a.ingest("Paris is the capital of France")
    assert b.receive_object(*_published(a)) is True
    assert any("Paris" in d.text for d in b.objects.iter("D", "agent"))


def test_signed_bytes_carry_origin_and_text():
    a = _node("A")
    a.ingest("Mount Everest is the tallest mountain")
    received, mac, scope = _published(a)
    b = _node("B")
    b.receive_object(received, mac, scope)
    (d,) = list(b.objects.iter("D", "agent"))
    assert d.origin == "A" and "Everest" in d.text and d.mac == mac


# --------------------------------------------------------------------------- #
# (b) tampered / wrong-key delta is rejected BEFORE admission (not inserted)
# --------------------------------------------------------------------------- #
def test_tampered_payload_is_rejected_before_admission():
    a, b = _node("A"), _node("B")
    a.ingest("Paris is the capital of France")
    received, mac, scope = _published(a)
    assert b.receive_object(received.replace(b"Paris", b"Berlin"), mac, scope) is False
    assert b.objects.ids("D", "agent") == ()


def test_wrong_key_delta_is_rejected_before_admission():
    a, b = _node("A"), _node("B", key=b"some-other-group-key")
    a.ingest("Paris is the capital of France")
    assert b.receive_object(*_published(a)) is False
    assert b.objects.ids("D", "agent") == ()


def test_unsigned_payload_is_rejected():
    a, b = _node("A"), _node("B")
    a.ingest("Paris is the capital of France")
    received, _mac, scope = _published(a)
    assert b.receive_object(received, "", scope) is False
