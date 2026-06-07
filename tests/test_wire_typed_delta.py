"""Typed graph-delta on the wire (CHANGE 4, TDD).

PUBLISH carries a SERIALIZED TYPED DELTA (a MemoryUnit with its Claims + status
+ relevance), not bare claim text. The proto types round-trip losslessly through
JSON (to_dict/from_dict), and a PUBLISHed typed delta, once reconstructed and
merged on the receiver, yields the SAME brain state as the in-process write path.

Run:  PYTHONPATH=src python3 -m pytest tests/test_wire_typed_delta.py -q
"""
from __future__ import annotations

from smp.proto import Claim, Endpoint, MemoryLink, MemoryUnit


# --------------------------------------------------------------------------- #
# (1) lossless JSON round-trip of the typed structures
# --------------------------------------------------------------------------- #
def test_endpoint_roundtrips():
    ep = Endpoint(node_id=b"\x01\x02", claim_key=b"\xaa\xbb")
    assert Endpoint.from_dict(ep.to_dict()) == ep


def test_claim_roundtrips_including_bytes_and_grounding():
    c = Claim(claim_key=b"\xde\xad", text="Paris is the capital of France",
              confidence=0.9, claim_status="ACTIVE",
              grounding=("doc:1", "doc:2"))
    back = Claim.from_dict(c.to_dict())
    assert back == c


def test_memorylink_roundtrips_through_json():
    ml = MemoryLink(type="SUPERSEDES", weight=0.8, version=3,
                    signature=b"\x01", provenance="CONSOLIDATION",
                    src=Endpoint(node_id=b"n", claim_key=b"k"),
                    dst=Endpoint(claim_key=b"k2"), entailment_score=0.7)
    back = MemoryLink.from_dict(ml.to_dict())
    assert back == ml


def test_memoryunit_with_claims_roundtrips_to_equal_typed_structure():
    u = MemoryUnit(
        id=b"\x12\x34", relevance_magnitude=1.0, status="ACTIVE",
        confidence=0.9,
        claims=[Claim(claim_key=b"\x01", text="alpha", confidence=0.9),
                Claim(claim_key=b"\x02", text="beta", claim_status="DEPRECATED")],
    )
    back = MemoryUnit.from_dict(u.to_dict())
    assert back == u                      # full typed structure preserved
    assert [c.text for c in back.claims] == ["alpha", "beta"]


def test_serialized_delta_is_json_not_bare_text():
    import json

    from smp.sync.wire_delta import serialize_signed_delta
    u = MemoryUnit(id=b"\x01", claims=[Claim(claim_key=b"\x02", text="hello world")])
    payload = serialize_signed_delta("A", u, b"k")
    env = json.loads(payload.decode("utf-8"))     # parses as a JSON envelope ...
    assert env["node_id"] == "A"
    assert env["unit"]["claims"][0]["text"] == "hello world"   # typed unit, not raw text
    assert "sig" in env


# --------------------------------------------------------------------------- #
# (2) end-to-end: wire-reconstructed delta merges identically to in-process
# --------------------------------------------------------------------------- #
def _cfg():
    from smp.agent.run_hotpot_gate import _calibrated_cfg
    return _calibrated_cfg("hashing")


def _node(nid):
    from smp.sync.node import BrainNode
    return BrainNode(nid, encoder="hashing", nli="lexical", cfg=_cfg())


def test_wire_typed_delta_merges_identically_to_in_process_path():
    from smp.sync.wire_delta import group_key

    claims = ["Paris is the capital of France",
              "Mount Everest is the tallest mountain",
              "Photosynthesis converts sunlight into chemical energy"]
    key = group_key({})

    # in-process path: write the text directly
    inproc = _node("inproc")
    for t in claims:
        inproc.mem.write(t)

    # wire path: publisher serializes typed deltas, receiver verifies + reconstructs + merges
    pub, recv = _node("pub"), _node("recv")
    for t in claims:
        pub.ingest(t)
    for payload in pub.publish_wire_payloads(key):
        assert recv.receive_wire(payload, key) is True

    # identical observable brain state (the convergence observable)
    assert recv.live_claim_texts() == inproc.live_claim_texts()
