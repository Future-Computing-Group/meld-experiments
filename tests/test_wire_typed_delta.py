"""Typed graph-delta on the wire (CHANGE 4, TDD).

PUBLISH carries a SERIALIZED TYPED DELTA (a MemoryUnit with its Claims + status
+ relevance), not bare claim text. The proto types round-trip losslessly through
JSON (to_dict/from_dict), and a PUBLISHed typed delta, once reconstructed and
merged on the receiver, yields the SAME brain state as the in-process write path.

Run:  PYTHONPATH=src python3 -m pytest tests/test_wire_typed_delta.py -q
"""
from __future__ import annotations

from meld.proto import Claim, Endpoint, MemoryLink, MemoryUnit


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
        id=b"\x12\x34", retention=1.0, status="ACTIVE",
        confidence=0.9,
        claims=[Claim(claim_key=b"\x01", text="alpha", confidence=0.9),
                Claim(claim_key=b"\x02", text="beta", claim_status="DEPRECATED")],
    )
    back = MemoryUnit.from_dict(u.to_dict())
    assert back == u                      # full typed structure preserved
    assert [c.text for c in back.claims] == ["alpha", "beta"]


def test_serialized_delta_is_json_not_bare_text():
    # the envelope carries the Delta's canonical bytes: typed, tagged JSON, not raw text
    import json

    from meld.protocol.objects import attach_mac, canonical_bytes
    from meld.sync.wire_delta import pack
    from test_protocol_objects import _delta
    d = attach_mac(_delta(text="hello world"), b"k")
    env = json.loads(pack([(canonical_bytes(d), d.mac, "eu")]))[0]
    body = json.loads(env["o"])
    assert body["t"] == "Delta" and body["text"] == "hello world" and body["cell"] == "eu.a"
    assert env["m"] == d.mac and env["s"] == "eu"


# --------------------------------------------------------------------------- #
# (2) end-to-end: the wire delivers the publisher's typed objects byte-identically
# --------------------------------------------------------------------------- #


def _node(nid):
    from meld.sync.node import BrainNode
    return BrainNode(nid, encoder="hashing", nli="lexical")


def test_wire_delivers_the_publishers_typed_deltas_byte_identically():
    # Rewritten (WP2 T2): the old test compared live claims after a wire merge; a received
    # delta now reaches the merge only through the admission step (T6), so the wire's own
    # contract is checked here: the receiver holds exactly the publisher's signed Deltas.
    claims = ["Paris is the capital of France",
              "Mount Everest is the tallest mountain",
              "Photosynthesis converts sunlight into chemical energy"]
    pub, recv = _node("pub"), _node("recv")
    for t in claims:
        pub.ingest(t)
    for oid in pub.outbox():
        assert recv.receive_object(*pub.objects.raw(oid)) is True
    ids = pub.objects.ids("D", "agent")
    assert recv.objects.ids("D", "agent") == ids
    assert [recv.objects.raw(i) for i in ids] == [pub.objects.raw(i) for i in ids]
    assert {d.text for d in recv.objects.iter("D", "agent")} == set(claims)
