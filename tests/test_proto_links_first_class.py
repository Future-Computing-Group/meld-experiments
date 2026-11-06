"""Phase 0 — vendored links-first-class proto shape.

The MELD-vendored proto promotes MemoryLink to a first-class object (typed,
versioned, signed, weighted, with an entailment score), adds an addressable
Claim with status + validity, and an Endpoint that may target a claim within a
unit. Freshness is a derived property of the ``status`` field, so there is no
standalone staleness type.

This exercises the Python facade over the vendored proto (the wire types the
merge engine reasons over), not protobuf wire compatibility with firmament --
MELD vendors its own copy per the scoping decision and does not modify firmament.
"""
from __future__ import annotations

import pytest


def test_memorylink_is_first_class_versioned_signed():
    from meld.proto import MemoryLink

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
    from meld.proto import MemoryLink

    ml = MemoryLink(type="CONTRADICTS", version=3, weight=0.5, signature=b"sig")
    back = MemoryLink.from_bytes(ml.to_bytes())
    assert back.type == "CONTRADICTS"
    assert back.version == 3
    assert back.signature == b"sig"


def test_claim_carries_status_and_version():
    """r3 retired validity intervals: a claim carries a version that is part of its key, and its
    status is read from vis (auth join cons), not from the carried claim_status field. A new
    version under an authorized SUPERSEDE reactivates through the representative."""
    from meld.brain.store import Derived, Store
    from meld.proto import Claim, claim_key
    from meld.protocol.objects import (FIXED_ONE, AuthoredEffect, Delta, attach_mac,
                                       canonical_bytes)

    # version is a field, survives the wire round trip, and is part of the key
    c = Claim(claim_key=claim_key("t", "eu.a", 2), text="t", claim_status="ACTIVE", version=2)
    assert c.version == 2 and Claim.from_dict(c.to_dict()).version == 2
    assert claim_key("t", "eu.a", 0) != claim_key("t", "eu.a", 1)

    key = b"t8-key"
    pol = {"rank_map": {"PRECEDENT": 0, "OBSERVATION": 1}, "scopes": {"eu": ["eu.a"]},
           "auth_policy": {"OVERRULE": 0, "REVOKE": 0, "SUPERSEDE": 1, "DISTINGUISHES": 1},
           "step_version": 0, "r_min": 1}

    def node(version):
        k = claim_key("t", "eu.a", version)
        return Delta(unit_id=k, claim_key=k, text="t", cell="eu.a", version=version, auth=1,
                     status="ACTIVE", retention=FIXED_ONE, created_at=10 + version, origin="A",
                     encoder="hashing", q=b"\x00" * 256)

    def eff(kind, c1, c2=b"", author_auth=0):
        return AuthoredEffect(scope="eu", kind=kind, c1=c1, c2=c2, target_cell="eu.a", author="C",
                              author_cell="eu.a", author_auth=author_auth, created_at=0)

    v0, v1 = node(0), node(1)
    k0, k1 = v0.claim_key, v1.claim_key
    assert k0 != k1
    s = Store(pol, key, encoder="hashing")
    for o in (v0, v1, eff("OVERRULE", k0), eff("SUPERSEDE", k0, k1, author_auth=1)):
        o = attach_mac(o, key)
        assert s.apply(canonical_bytes(o), o.mac) is not None
    # status comes from vis: every carried claim_status is ACTIVE, yet v0 reads OVERRULED
    assert v0.status == v1.status == "ACTIVE"
    assert s.vis(k0) == "OVERRULED"
    # the superseding version is the group's representative and reads ACTIVE
    assert s.rep([k0, k1]) == k1
    s.derived = Derived(rep={k0: k1}, links=[])
    assert s.vis(k1) == "ACTIVE" and s.vis(k0) == "OVERRULED"
    assert s.live_nodes() == [k1]


def test_endpoint_may_target_a_claim_within_a_unit():
    from meld.proto import Endpoint

    ep = Endpoint(node_id=b"n", claim_key=b"k")
    assert ep.node_id == b"n"
    assert ep.claim_key == b"k"


def test_freshness_is_derived_from_status_not_a_standalone_type():
    # There is no StalenessTag: freshness is read off the status field
    # (is_fresh == status == "ACTIVE"). Assert the removed symbol is gone.
    import meld.proto as proto

    assert not hasattr(proto, "StalenessTag")
    assert "StalenessTag" not in proto.__all__


def test_memoryunit_carries_addressable_claims():
    # The first-class path is MemoryLink + addressable Claims; the node-centric
    # Citation alias has been removed.
    import meld.proto as proto
    from meld.proto import Claim, MemoryUnit

    u = MemoryUnit(retention=0.0, id=b"u", claims=[Claim(claim_key=b"k", text="t")])
    assert u.id == b"u"
    assert u.claims[0].claim_key == b"k"
    assert not hasattr(proto, "Citation")
    assert not hasattr(u, "citations")


def test_memoryunit_cell_round_trips_through_dict():
    # The per-claim Context cell (frame) is carried on the unit and survives the
    # wire-delta to_dict/from_dict round trip; default preserves current behavior.
    from meld.proto import Claim, MemoryUnit

    u = MemoryUnit(retention=0.0, id=b"u", cell="geo",
                   claims=[Claim(claim_key=b"k", text="t")])
    assert u.cell == "geo"
    d = u.to_dict()
    assert d["cell"] == "geo"
    u2 = MemoryUnit.from_dict(d)
    assert u2.cell == "geo"
    # default cell when absent (retention is required, so it is always carried)
    assert MemoryUnit.from_dict({"id": "", "retention": 0.0}).cell == "agent.public"
    assert MemoryUnit(retention=0.0).cell == "agent.public"


def test_linktype_taxonomy_complete():
    from meld.proto import LINK_TYPES

    # the six families from the restructure note must all be representable
    for t in ("ASSOCIATES", "CITES", "GROUNDS", "ENTITY_LINK", "PRECEDENT_OF",
              "OVERRULES", "DISTINGUISHES", "CONTRADICTS", "DERIVED_FROM",
              "CONSOLIDATED_FROM", "TEMPORAL_SUCCESSION", "SUPERSEDES",
              "PARTIALLY_OVERRULES"):
        assert t in LINK_TYPES
