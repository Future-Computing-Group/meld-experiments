"""Phase 0 — vendored links-first-class proto shape.

The SMP-vendored proto promotes MemoryLink to a first-class object (typed,
versioned, signed, weighted, with an entailment score), adds an addressable
Claim with status + validity, and an Endpoint that may target a claim within a
unit. Freshness is a derived property of the ``status`` field, so there is no
standalone staleness type.

This exercises the Python facade over the vendored proto (the wire types the
merge engine reasons over), not protobuf wire compatibility with firmament --
SMP vendors its own copy per the scoping decision and does not modify firmament.
"""
from __future__ import annotations

import pytest


def test_memorylink_is_first_class_versioned_signed():
    from smp.proto import MemoryLink

    ml = MemoryLink(
        type="SUPERSEDES", version=1, signature=b"x",
        weight=0.9, entailment_score=0.8,
    )
    assert ml.version == 1
    assert ml.signature == b"x"
    assert ml.type == "SUPERSEDES"
    assert ml.weight == pytest.approx(0.9)
    assert ml.entailment_score == pytest.approx(0.8)


def test_memorylink_roundtrips_through_wire_bytes():
    from smp.proto import MemoryLink

    ml = MemoryLink(type="CONTRADICTS", version=3, weight=0.5, signature=b"sig")
    back = MemoryLink.from_bytes(ml.to_bytes())
    assert back.type == "CONTRADICTS"
    assert back.version == 3
    assert back.signature == b"sig"


def test_claim_carries_status_and_validity():
    from smp.proto import Claim

    c = Claim(claim_key=b"k", text="t", confidence=0.9, claim_status="ACTIVE")
    assert c.claim_status == "ACTIVE"
    assert c.claim_key == b"k"
    assert c.text == "t"


def test_endpoint_may_target_a_claim_within_a_unit():
    from smp.proto import Endpoint

    ep = Endpoint(node_id=b"n", claim_key=b"k")
    assert ep.node_id == b"n"
    assert ep.claim_key == b"k"


def test_freshness_is_derived_from_status_not_a_standalone_type():
    # There is no StalenessTag: freshness is read off the status field
    # (is_fresh == status == "ACTIVE"). Assert the removed symbol is gone.
    import smp.proto as proto

    assert not hasattr(proto, "StalenessTag")
    assert "StalenessTag" not in proto.__all__


def test_memoryunit_carries_addressable_claims():
    # The first-class path is MemoryLink + addressable Claims; the node-centric
    # Citation alias has been removed.
    import smp.proto as proto
    from smp.proto import Claim, MemoryUnit

    u = MemoryUnit(id=b"u", claims=[Claim(claim_key=b"k", text="t")])
    assert u.id == b"u"
    assert u.claims[0].claim_key == b"k"
    assert not hasattr(proto, "Citation")
    assert not hasattr(u, "citations")


def test_memoryunit_cell_round_trips_through_dict():
    # The per-claim Context cell (frame) is carried on the unit and survives the
    # wire-delta to_dict/from_dict round trip; default preserves current behavior.
    from smp.proto import Claim, MemoryUnit

    u = MemoryUnit(id=b"u", cell="geo",
                   claims=[Claim(claim_key=b"k", text="t")])
    assert u.cell == "geo"
    d = u.to_dict()
    assert d["cell"] == "geo"
    u2 = MemoryUnit.from_dict(d)
    assert u2.cell == "geo"
    # default cell when absent (back-compat with older serialized units)
    assert MemoryUnit.from_dict({"id": ""}).cell == "agent.public"
    assert MemoryUnit().cell == "agent.public"


def test_linktype_taxonomy_complete():
    from smp.proto import LINK_TYPES

    # the six families from the restructure note must all be representable
    for t in ("ASSOCIATES", "CITES", "GROUNDS", "ENTITY_LINK", "PRECEDENT_OF",
              "OVERRULES", "DISTINGUISHES", "CONTRADICTS", "DERIVED_FROM",
              "CONSOLIDATED_FROM", "TEMPORAL_SUCCESSION", "SUPERSEDES",
              "PARTIALLY_OVERRULES"):
        assert t in LINK_TYPES
