"""Task 1.2 — the Context gate.

Two claims are Context-compatible iff they share a cell/scope frame AND their
validity intervals overlap AND neither is staler than the other beyond
tolerance. The load-bearing case (Context-gate-false-merge): a high-sigma pair
in DIFFERENT frames (Paris-the-city vs Paris-mythology) must be INCOMPATIBLE so
the classifier RELATEs rather than MERGEs them.
"""
from __future__ import annotations

from smp.merge.context_gate import Context, context_compatible


def _ctx(cell="agent.public", valid=(0, 10), fresh=True):
    return Context(cell=cell, valid_from=valid[0], valid_until=valid[1], is_fresh=fresh)


def test_same_cell_overlapping_validity_is_compatible():
    assert context_compatible(_ctx(cell="agent.public", valid=(0, 10)),
                              _ctx(cell="agent.public", valid=(5, 15)))


def test_different_frame_is_incompatible():
    # Paris-the-city (geo) vs Paris-mythology: same surface term, different frame
    assert not context_compatible(_ctx(cell="geo"), _ctx(cell="mythology"))


def test_disjoint_validity_is_incompatible():
    assert not context_compatible(_ctx(valid=(0, 5)), _ctx(valid=(10, 20)))


def test_open_ended_validity_overlaps():
    # valid_until None means open-ended; overlaps any later interval in same cell
    assert context_compatible(Context(cell="agent.public", valid_from=0, valid_until=None),
                              _ctx(cell="agent.public", valid=(100, 200)))
