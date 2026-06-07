"""Anti-entropy digest for the status-link CRDT (reviewer gap: SEC is conditional
on eventually-complete delivery, but lossy semantic routing silently drops some
status links and never reconciles them).

The digest is a compact, order-independent summary of a brain's grow-only status-
link set; a round exchanges digests, computes the per-claim symmetric difference of
link ids, and pulls the missing links. Because the set is a CRDT (idempotent union)
re-applying is safe and the round drives the two brains to the SAME link set ->
the SAME effective per-claim status, regardless of routing recall.
"""
from __future__ import annotations


# --- CHANGE 1 RED->GREEN -----------------------------------------------------

def test_digest_is_order_independent_same_set_same_digest():
    # Two link books built by APPLYING the same links in different orders must
    # produce byte-identical digests (the digest is a pure fn of the converged set).
    from smp.sync.anti_entropy import LinkBook, claim_digest
    links = [("c1", "c1.a", "DEPRECATED"), ("c1", "c1.b", "OVERRULED"),
             ("c2", "c2.x", "REVOKED"), ("c2", "c2.y", "DEPRECATED")]
    a = LinkBook()
    for ck, lid, eff in links:
        a.add(ck, lid, eff)
    b = LinkBook()
    for ck, lid, eff in reversed(links):
        b.add(ck, lid, eff)
    # also re-deliver one link to exercise idempotence
    b.add("c1", "c1.a", "DEPRECATED")
    assert claim_digest(a) == claim_digest(b)


def test_digest_lists_sorted_link_ids_and_a_set_hash_per_claim():
    from smp.sync.anti_entropy import LinkBook, claim_digest
    book = LinkBook()
    book.add("c1", "c1.b", "OVERRULED")
    book.add("c1", "c1.a", "DEPRECATED")
    dig = claim_digest(book)
    assert set(dig.keys()) == {"c1"}
    cd = dig["c1"]
    assert cd.link_ids == ("c1.a", "c1.b")          # sorted, deterministic
    assert isinstance(cd.set_hash, str) and len(cd.set_hash) == 16


def test_symmetric_difference_identifies_missing_ids_per_claim():
    from smp.sync.anti_entropy import LinkBook, claim_digest, symmetric_difference
    a = LinkBook()
    a.add("c1", "c1.a", "DEPRECATED")
    a.add("c1", "c1.b", "OVERRULED")          # A has b, B does not
    a.add("c2", "c2.x", "REVOKED")
    b = LinkBook()
    b.add("c1", "c1.a", "DEPRECATED")
    b.add("c3", "c3.z", "DEPRECATED")          # B has a whole claim A lacks
    da, db = claim_digest(a), claim_digest(b)
    a_missing, b_missing = symmetric_difference(da, db)
    # a_missing = ids A lacks (present in B, absent in A)
    assert a_missing == {"c3": {"c3.z"}}
    # b_missing = ids B lacks (present in A, absent in B)
    assert b_missing == {"c1": {"c1.b"}, "c2": {"c2.x"}}


def test_anti_entropy_round_makes_missing_brain_match_complete_peer():
    # B is missing some links (as after lossy routing). One round with the complete
    # peer A drives B's per-claim effective status to A's.
    from smp.sync.anti_entropy import LinkBook, anti_entropy_round
    complete = LinkBook()
    complete.add("c1", "c1.dep", "DEPRECATED")
    complete.add("c1", "c1.ovr", "OVERRULED")
    complete.add("c2", "c2.dep", "DEPRECATED")
    complete.add("c2", "c2.rev", "REVOKED")
    missing = LinkBook()
    missing.add("c1", "c1.dep", "DEPRECATED")     # lost c1.ovr and ALL of c2
    cost = anti_entropy_round(missing, complete)
    assert missing.effective_statuses() == complete.effective_statuses()
    assert missing.effective_statuses() == {"c1": "OVERRULED", "c2": "REVOKED"}
    assert cost.links_pulled_into_a == 3          # c1.ovr, c2.dep, c2.rev
    assert cost.bytes_exchanged > 0


def test_anti_entropy_round_is_idempotent_and_symmetric():
    # Re-applying links is safe (CRDT union); a second round transfers nothing and
    # the round reconciles BOTH directions (each side pulls what it lacks).
    from smp.sync.anti_entropy import LinkBook, anti_entropy_round
    a = LinkBook()
    a.add("c1", "c1.a", "DEPRECATED")
    a.add("c1", "c1.b", "OVERRULED")
    b = LinkBook()
    b.add("c1", "c1.a", "DEPRECATED")
    b.add("c2", "c2.z", "REVOKED")
    anti_entropy_round(a, b)
    assert a.effective_statuses() == b.effective_statuses()
    # second round: nothing left to transfer
    cost2 = anti_entropy_round(a, b)
    assert cost2.links_pulled_into_a == 0
    assert cost2.links_pulled_into_b == 0


def test_digest_bytes_are_a_compact_wire_form():
    # The digest serializes to a deterministic, order-independent wire blob; its
    # size is the per-round cost the experiment reports.
    from smp.sync.anti_entropy import LinkBook, claim_digest, digest_wire_bytes
    book = LinkBook()
    book.add("c1", "c1.a", "DEPRECATED")
    book.add("c1", "c1.b", "OVERRULED")
    blob1 = digest_wire_bytes(claim_digest(book))
    book2 = LinkBook()
    book2.add("c1", "c1.b", "OVERRULED")
    book2.add("c1", "c1.a", "DEPRECATED")
    blob2 = digest_wire_bytes(claim_digest(book2))
    assert blob1 == blob2                          # order-independent on the wire
    assert isinstance(blob1, bytes) and len(blob1) > 0
