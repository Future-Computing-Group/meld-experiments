"""TDD: the LIVE node read applies the SEC-preserving read-time status gate (§5.2).

node.status_of routes through effective_status_gated, with the stale set derived
from each status link's OWN carried fields -- a link whose carried lifecycle status
is non-active, or whose retention is below the floor. This is a pure function of the
converged link set (order-independent), so it does not disturb status-CRDT
convergence.

Default no-op: a status link applied with default carried fields (active link,
retention above floor) is never stale -> the stale set is empty -> the gated read is
byte-for-byte identical to the ungated effective_status. So all existing apply_status
callers, eval-c, and the headline are unchanged.
"""
from __future__ import annotations

from smp.sync.node import BrainNode


def _node(name):
    return BrainNode(name, encoder="hashing", nli="lexical")


def test_active_links_read_identically_to_before():
    # only active links -> empty stale set -> gated read == ungated read.
    a = _node("A")
    a.ingest("Paris is the capital of France")
    assert a.status_of("Paris is the capital of France") == "ACTIVE"
    a.apply_status("Paris is the capital of France", link_id="L1", effect="OVERRULED")
    # default carried fields (active link, retention ok) -> not stale -> still OVERRULED
    assert a.status_of("Paris is the capital of France") == "OVERRULED"


def test_stale_carried_status_link_is_gated_out_at_read():
    # a status link whose OWN carried lifecycle status is non-active is dropped at
    # read -> its effect does not count -> the claim reads ACTIVE again.
    a = _node("A")
    a.ingest("the policy is X")
    a.apply_status("the policy is X", link_id="L1", effect="OVERRULED",
                   link_status="REVOKED")
    assert a.status_of("the policy is X") == "ACTIVE"


def test_retention_below_floor_link_is_gated_out_at_read():
    # a status link whose retention is below the floor is stale -> gated out.
    a = _node("A")
    a.ingest("the policy is Y")
    a.apply_status("the policy is Y", link_id="L1", effect="OVERRULED",
                   retention=0.0, retention_floor=0.2)
    assert a.status_of("the policy is Y") == "ACTIVE"


def test_mixed_active_and_stale_links_only_active_counts():
    # one active OVERRULED link + one stale (revoked) REVOKED link: only the
    # active one counts at read -> OVERRULED, not REVOKED.
    a = _node("A")
    a.ingest("the policy is Z")
    a.apply_status("the policy is Z", link_id="L1", effect="OVERRULED")
    a.apply_status("the policy is Z", link_id="L2", effect="REVOKED",
                   link_status="REVOKED")
    assert a.status_of("the policy is Z") == "OVERRULED"
