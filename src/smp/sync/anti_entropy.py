"""Anti-entropy digest for the status-link CRDT (closes the reviewer-flagged SEC
gap: per-claim status convergence is CONDITIONAL on eventually-complete delivery,
but under lossy semantic routing some status links are silently dropped and never
reconciled -- the digest detects + repairs those losses).

A brain's reconcilable state is its grow-only set of status links: each link is a
content-addressed (stable ``link_id``, carried ``effect``, target ``claim_key``).
The DIGEST is a per-claim, order-independent summary of that set -- for each
claim the sorted tuple of link ids plus a set-hash of them. Because it is a pure
function of the converged link set, two brains with the same links produce the
SAME digest (byte-identical wire form), so a mismatch is exactly a missing link.

An anti-entropy ROUND between brains A,B exchanges digests, computes the per-claim
SYMMETRIC DIFFERENCE of link ids (which ids each side lacks), and each side pulls
the missing links and applies them to its grow-only set. Re-applying is safe (the
set is a join-semilattice: idempotent union), so a round drives both brains to the
SAME link set -> the SAME effective per-claim status, REGARDLESS of routing recall.
This makes per-claim status convergence UNCONDITIONAL on delivery completeness.

Layered cleanly on the existing status CRDT (``smp.merge.status_crdt``): a LinkBook
is a per-claim ``StatusState`` map; the digest/round operate over link ids only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Set, Tuple

from smp.merge.status_crdt import StatusState, effective_status, merge

# A single status link as carried on the wire when pulled: its target claim, its
# stable content-addressed id, and the lifecycle effect it asserts.
LinkRef = Tuple[str, str, str]  # (claim_key, link_id, effect)


class LinkBook:
    """A brain's grow-only status-link set, keyed by claim.

    ``_claims`` maps claim_key -> StatusState (the per-claim OR-set of links).
    ``_effects`` records each link_id's carried effect so a peer can pull the
    full link (id + effect), not just learn the id is missing. Both are grow-only;
    ``add`` is idempotent (the OR-set add) and order-independent."""

    def __init__(self) -> None:
        self._claims: Dict[str, StatusState] = {}
        self._effects: Dict[str, str] = {}

    def add(self, claim_key: str, link_id: str, effect: str) -> None:
        st = self._claims.get(claim_key, StatusState())
        self._claims[claim_key] = st.add(link_id, effect)
        # carried effect for the wire pull; on a (rare) id collision keep the
        # lattice-join so the book stays a true join-semilattice (mirrors the CRDT).
        self._effects[link_id] = self._claims[claim_key].links[link_id]

    def apply_link(self, link: LinkRef) -> None:
        """Apply a pulled link (idempotent union)."""
        self.add(*link)

    def link(self, claim_key: str, link_id: str) -> LinkRef:
        return (claim_key, link_id, self._effects[link_id])

    def claim_keys(self) -> Set[str]:
        return set(self._claims)

    def state(self, claim_key: str) -> StatusState:
        return self._claims.get(claim_key, StatusState())

    def effective_statuses(self) -> Dict[str, str]:
        """The effective per-claim status map (lattice-join of each claim's links)."""
        return {ck: effective_status(st) for ck, st in self._claims.items()}

    def merge_book(self, other: "LinkBook") -> None:
        """Union another book into this one (used to validate CRDT idempotence)."""
        for ck in other.claim_keys():
            self._claims[ck] = merge(self.state(ck), other.state(ck))
        self._effects.update(other._effects)


@dataclass(frozen=True)
class ClaimDigest:
    """Per-claim digest entry: the sorted link ids + a set-hash over them.

    The set-hash lets two peers cheaply detect equality without comparing id
    lists; the explicit ``link_ids`` keep the symmetric-difference exact (and the
    wire form simple). Both are pure functions of the converged link-id SET, so
    they are order-independent."""

    link_ids: Tuple[str, ...]
    set_hash: str


def _set_hash(link_ids: Tuple[str, ...]) -> str:
    """A deterministic, order-independent hash of a set of link ids."""
    h = hashlib.sha256()
    for lid in link_ids:                      # link_ids is pre-sorted -> canonical
        h.update(lid.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def claim_digest(book: LinkBook) -> Dict[str, ClaimDigest]:
    """Compute the per-claim digest of a link book. Order-independent: the same
    link set yields the same digest regardless of the order links were applied."""
    out: Dict[str, ClaimDigest] = {}
    for ck in book.claim_keys():
        ids = tuple(sorted(book.state(ck).links.keys()))
        out[ck] = ClaimDigest(link_ids=ids, set_hash=_set_hash(ids))
    return out


def digest_wire_bytes(digest: Dict[str, ClaimDigest]) -> bytes:
    """Serialize a digest to a deterministic, order-independent wire blob. Its size
    is the per-round digest cost the loss-injection experiment reports."""
    canonical = {ck: {"ids": list(cd.link_ids), "h": cd.set_hash}
                 for ck, cd in sorted(digest.items())}
    return json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode("utf-8")


def symmetric_difference(
    digest_a: Dict[str, ClaimDigest], digest_b: Dict[str, ClaimDigest]
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Per-claim symmetric difference of link ids.

    Returns ``(a_missing, b_missing)`` where ``a_missing[ck]`` is the set of ids
    present in B but absent from A (A must pull them) and vice versa. Claims with
    no difference are omitted. A matching ``set_hash`` short-circuits the per-claim
    comparison (no id-list diff needed when the sets are equal)."""
    a_missing: Dict[str, Set[str]] = {}
    b_missing: Dict[str, Set[str]] = {}
    for ck in set(digest_a) | set(digest_b):
        ca = digest_a.get(ck)
        cb = digest_b.get(ck)
        if ca is not None and cb is not None and ca.set_hash == cb.set_hash:
            continue                               # identical link sets -> nothing to do
        ids_a = set(ca.link_ids) if ca else set()
        ids_b = set(cb.link_ids) if cb else set()
        miss_a = ids_b - ids_a
        miss_b = ids_a - ids_b
        if miss_a:
            a_missing[ck] = miss_a
        if miss_b:
            b_missing[ck] = miss_b
    return a_missing, b_missing


@dataclass(frozen=True)
class RoundCost:
    """Cost of one anti-entropy round (the experiment's digest-cost columns)."""

    bytes_exchanged: int          # both digests on the wire
    links_pulled_into_a: int      # links A pulled from B (B->A repair)
    links_pulled_into_b: int      # links B pulled from A (A->B repair)


def anti_entropy_round(book_a: LinkBook, book_b: LinkBook) -> RoundCost:
    """One anti-entropy round between two link books.

    Exchange digests, compute the per-claim symmetric difference, and have each
    side PULL + APPLY the links it is missing. Idempotent (CRDT union), so re-
    running is safe; after the round both books hold the same link set and thus
    the same effective per-claim status. Returns the round cost (digest bytes
    exchanged + links transferred each way)."""
    dig_a = claim_digest(book_a)
    dig_b = claim_digest(book_b)
    a_missing, b_missing = symmetric_difference(dig_a, dig_b)

    pulled_into_a = 0
    for ck, ids in a_missing.items():
        for lid in ids:
            book_a.apply_link(book_b.link(ck, lid))
            pulled_into_a += 1
    pulled_into_b = 0
    for ck, ids in b_missing.items():
        for lid in ids:
            book_b.apply_link(book_a.link(ck, lid))
            pulled_into_b += 1

    bytes_exchanged = len(digest_wire_bytes(dig_a)) + len(digest_wire_bytes(dig_b))
    return RoundCost(bytes_exchanged=bytes_exchanged,
                     links_pulled_into_a=pulled_into_a,
                     links_pulled_into_b=pulled_into_b)


__all__ = ["LinkBook", "LinkRef", "ClaimDigest", "RoundCost",
           "claim_digest", "digest_wire_bytes", "symmetric_difference",
           "anti_entropy_round"]
