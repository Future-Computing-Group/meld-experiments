"""Task 1.6 — the six protocol primitives + the Patch (the NEW wire object).

Five primitives map to vendored proto types (Concept->MemoryUnit, Claim->Claim,
Evidence->GROUNDS-link, Context->Endpoint+StalenessTag, Mapping->typed
MemoryLink). The sixth, Patch (a MergeResult), is NEW: append-only, signed,
versioned, recording which gates fired -- the only thing that mutates state.
"""
from __future__ import annotations

import pytest

from smp.proto import Endpoint
from smp.protocol.messages import Patch


def _ep():
    return Endpoint(node_id=b"n", claim_key=b"k")


def test_patch_is_signed_and_records_gates():
    p = Patch(decision="MERGE", target=_ep(), emitted=["u1"],
              gates_that_fired=["kappa", "context"], provenance="agent-x",
              version=1, signature=b"sig")
    assert p.signature == b"sig"
    assert p.version == 1
    assert "kappa" in p.gates_that_fired


def test_patch_is_append_only_immutable():
    p = Patch(decision="MERGE", target=_ep(), version=1, signature=b"s")
    with pytest.raises(Exception):
        p.decision = "REJECT"          # frozen / append-only


def test_patch_decision_must_be_valid():
    with pytest.raises(ValueError):
        Patch(decision="NONSENSE", target=_ep(), version=1, signature=b"s")


def test_patch_roundtrips():
    p = Patch(decision="CONFLICT", target=_ep(), gates_that_fired=["chi"],
              version=2, signature=b"x", provenance="brain-7")
    back = Patch.from_dict(p.to_dict())
    assert back.decision == "CONFLICT"
    assert back.gates_that_fired == ["chi"]
    assert back.version == 2
