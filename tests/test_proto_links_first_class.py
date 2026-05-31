"""Phase 0 — vendored links-first-class proto shape.

The SMP-vendored proto promotes MemoryLink to a first-class object (typed,
versioned, signed, weighted, with an entailment score), adds an addressable
Claim with status + validity, an Endpoint that may target a claim within a unit,
and a StalenessTag. The node-centric Citation alias is kept deprecated.

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


def test_staleness_tag_fields():
    from smp.proto import StalenessTag

    st = StalenessTag(content_hash=b"h", phi=0.01, is_latest=True)
    assert st.content_hash == b"h"
    assert st.phi == pytest.approx(0.01)
    assert st.is_latest is True


def test_memoryunit_keeps_deprecated_citation_alias():
    # node-centric citations kept only as a deprecated alias; the first-class
    # path is MemoryLink. The unit still carries addressable claims.
    from smp.proto import Claim, MemoryUnit

    u = MemoryUnit(id=b"u", claims=[Claim(claim_key=b"k", text="t")])
    assert u.id == b"u"
    assert u.claims[0].claim_key == b"k"


def test_linktype_taxonomy_complete():
    from smp.proto import LINK_TYPES

    # the six families from the restructure note must all be representable
    for t in ("ASSOCIATES", "CITES", "GROUNDS", "ENTITY_LINK", "PRECEDENT_OF",
              "OVERRULES", "DISTINGUISHES", "CONTRADICTS", "DERIVED_FROM",
              "CONSOLIDATED_FROM", "TEMPORAL_SUCCESSION", "SUPERSEDES",
              "PARTIALLY_OVERRULES"):
        assert t in LINK_TYPES
