"""Round 5, issue 1: status links ride a Patch of their own operation kind
(STATUS), and materialization drops unauthorized links deterministically."""
from meld.brain.store import Brain
from meld.merge.status_crdt import StatusState, effective_status_gated
from meld.proto import Claim, MemoryUnit
from meld.protocol.messages import DECISIONS, Endpoint, Patch


def test_status_is_a_patch_operation_kind():
    assert "STATUS" in DECISIONS


def test_store_applies_status_patch_as_append():
    b = Brain()
    u = MemoryUnit(id=b"u1", relevance_magnitude=1.0, status="ACTIVE",
                   claims=[Claim(claim_key=b"ck", text="t", claim_status="ACTIVE")])
    b.apply(Patch(decision="INSERT", target=Endpoint(claim_key=b"ck"), emitted=[u],
                  gates_that_fired=["novel"], version=1, signature=b"s"))
    b.apply(Patch(decision="STATUS", target=Endpoint(claim_key=b"ck"),
                  emitted=[("STATUS", b"ck", ("l1", "OVERRULED"))],
                  gates_that_fired=["status-link"], version=1, signature=b"s"))
    st = b.claim_status(b"ck")
    assert effective_status_gated(st) == "OVERRULED"


def test_unauthorized_link_recorded_but_not_materialized():
    st = StatusState().add("l1", "OVERRULED")
    # recorded in the set either way
    assert "l1" in st.links
    # authorized: materializes
    assert effective_status_gated(st) == "OVERRULED"
    # unauthorized: same converged set, deterministic read-time drop
    assert effective_status_gated(st, unauthorized=frozenset({"l1"})) == "ACTIVE"
