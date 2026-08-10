"""S3: tests for graph_fingerprint.py -- a deterministic,
order-invariant-BY-CONSTRUCTION serialization of a Brain's semantically
relevant graph state (live claim set, per-claim status map, typed link
multiset, representative choice per SUPERSEDES-consolidated group)."""
from __future__ import annotations

from smp.brain.store import Brain
from smp.eval.graph_fingerprint import diff_fingerprints, fingerprint
from smp.protocol.messages import Patch
from smp.proto import Claim, Endpoint, MemoryLink, MemoryUnit


def _unit(uid: bytes, key: bytes, text: str) -> MemoryUnit:
    return MemoryUnit(id=uid, status="ACTIVE",
                      claims=[Claim(claim_key=key, text=text, claim_status="ACTIVE")])


def _insert(brain: Brain, unit: MemoryUnit) -> None:
    ck = unit.claims[0].claim_key
    patch = Patch(decision="INSERT", target=Endpoint(node_id=unit.id, claim_key=ck),
                 emitted=[unit, ("STATUS", ck, (unit.id.hex(), "ACTIVE"))],
                 gates_that_fired=["novel"], version=1, signature=b"s")
    brain.apply(patch)


def test_same_state_yields_same_fingerprint():
    a, b = Brain(encoder="hashing"), Brain(encoder="hashing")
    for brain in (a, b):
        _insert(brain, _unit(b"u1", b"k1", "alpha"))
        _insert(brain, _unit(b"u2", b"k2", "beta"))
    fa, fb = fingerprint(a), fingerprint(b)
    assert fa.joint_digest == fb.joint_digest
    assert fa.component_digests == fb.component_digests


def test_permuted_insertion_order_same_content_yields_same_fingerprint():
    a, b = Brain(encoder="hashing"), Brain(encoder="hashing")
    u1 = _unit(b"u1", b"k1", "alpha")
    u2 = _unit(b"u2", b"k2", "beta")
    u3 = _unit(b"u3", b"k3", "gamma")
    for u in (u1, u2, u3):
        _insert(a, u)
    for u in (u3, u1, u2):     # same content, different insertion order
        _insert(b, u)
    fa, fb = fingerprint(a), fingerprint(b)
    assert fa.joint_digest == fb.joint_digest
    assert fa.readable == fb.readable


def test_one_changed_link_yields_different_fingerprint_and_names_it():
    a, b = Brain(encoder="hashing"), Brain(encoder="hashing")
    for brain in (a, b):
        _insert(brain, _unit(b"u1", b"k1", "alpha"))
        _insert(brain, _unit(b"u2", b"k2", "beta"))
        _insert(brain, _unit(b"u3", b"k3", "gamma"))
    # identical live-claim set and status map on both sides; only the RELATE
    # target differs (k2 on a, k3 on b) -- a pure link-structure divergence.
    link_to_k2 = MemoryLink(type="ASSOCIATES", weight=0.9, version=1, signature=b"s",
                            provenance="INFERRED",
                            src=Endpoint(node_id=b"u1", claim_key=b"k1"),
                            dst=Endpoint(claim_key=b"k2"))
    link_to_k3 = MemoryLink(type="ASSOCIATES", weight=0.9, version=1, signature=b"s",
                            provenance="INFERRED",
                            src=Endpoint(node_id=b"u1", claim_key=b"k1"),
                            dst=Endpoint(claim_key=b"k3"))
    a.apply(Patch(decision="RELATE", target=Endpoint(node_id=b"u1", claim_key=b"k1"),
                 emitted=[link_to_k2], gates_that_fired=["sigma"], version=1, signature=b"s"))
    b.apply(Patch(decision="RELATE", target=Endpoint(node_id=b"u1", claim_key=b"k1"),
                 emitted=[link_to_k3], gates_that_fired=["sigma"], version=1, signature=b"s"))
    fa, fb = fingerprint(a), fingerprint(b)

    assert fa.joint_digest != fb.joint_digest
    assert fa.component_digests["links"] != fb.component_digests["links"]
    # the other three components are untouched by this change
    assert fa.component_digests["live_claims"] == fb.component_digests["live_claims"]
    assert fa.component_digests["status_map"] == fb.component_digests["status_map"]
    assert fa.component_digests["representatives"] == fb.component_digests["representatives"]

    d = diff_fingerprints(fa, fb)
    assert set(d.keys()) == {"links"}          # names exactly the changed component
    only_a = {tuple(x) for x in d["links"]["only_in_a"]}
    only_b = {tuple(x) for x in d["links"]["only_in_b"]}
    k2_hex, k3_hex = b"k2".hex(), b"k3".hex()
    assert ("ASSOCIATES", f"claim:{b'k1'.hex()}", f"claim:{k2_hex}") in only_a
    assert ("ASSOCIATES", f"claim:{b'k1'.hex()}", f"claim:{k3_hex}") in only_b


def test_representatives_named_when_canonical_choice_differs():
    # Two runs consolidate the SAME three claims (k1, k2, k3) via SUPERSEDES,
    # but a different member ends up the surfaced canonical (k1 vs k2) -- the
    # "representative choice" component must catch this even though the group
    # membership (and hence live-claim SET) is identical either way.
    a, b = Brain(encoder="hashing"), Brain(encoder="hashing")
    for brain in (a, b):
        for u in (_unit(b"u1", b"k1", "x"), _unit(b"u2", b"k2", "y"), _unit(b"u3", b"k3", "z")):
            _insert(brain, u)

    def _supersede(brain, src_ck, dst_ck):
        link = MemoryLink(type="SUPERSEDES", weight=0.95, version=1, signature=b"s",
                          provenance="CONSOLIDATION",
                          src=Endpoint(node_id=b"x", claim_key=src_ck),
                          dst=Endpoint(claim_key=dst_ck))
        brain.apply(Patch(decision="MERGE", target=Endpoint(claim_key=src_ck),
                          emitted=[("STATUS", src_ck, ("l", "DEPRECATED")), link],
                          gates_that_fired=["sigma"], version=1, signature=b"s"))

    # a: k2, k3 both supersede onto k1 (k1 stays canonical)
    _supersede(a, b"k2", b"k1")
    _supersede(a, b"k3", b"k1")
    # b: k1, k3 both supersede onto k2 (k2 stays canonical) -- same group, different rep
    _supersede(b, b"k1", b"k2")
    _supersede(b, b"k3", b"k2")

    fa, fb = fingerprint(a), fingerprint(b)
    assert fa.component_digests["representatives"] != fb.component_digests["representatives"]
    # live-claim SET differs too here (k1 vs k2 survives) -- but the point is
    # representatives names it explicitly and separately.
    d = diff_fingerprints(fa, fb)
    assert "representatives" in d
