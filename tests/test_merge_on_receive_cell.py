"""TDD: merge_on_receive derives the Context cell from MemoryUnit.cell on the LIVE
path (not a hardcoded constant), so two high-similarity claims in DIFFERENT cells
RELATE (same meaning, different frame) instead of MERGEing -- the §A.2 Context gate
actually fires on the federation path. Default cell ("agent.public") on both units
preserves all current behavior (M-2/headline unchanged).
"""
from smp.config import Thresholds
from smp.merge.signals import EmbeddingSimilarity
from smp.proto import Claim, MemoryUnit
from smp.protocol.merge_on_receive import merge_on_receive

CFG = Thresholds(theta_merge=0.5, sigma_lo=0.12, r_min=0.2, tau=0.5)


def _es():
    return EmbeddingSimilarity(encoder="hashing")


def _unit(text, key, cell="agent.public", uid=b"u"):
    return MemoryUnit(id=uid, relevance_magnitude=1.0, status="ACTIVE", cell=cell,
                      claims=[Claim(claim_key=key, text=text, claim_status="ACTIVE")])


A = "Paris is the capital city of France"
B = "Paris is the capital of France"


def test_cross_cell_high_sigma_pair_relates_on_live_path():
    inc = _unit(A, b"ki", cell="geo", uid=b"inc")
    loc = _unit(B, b"kl", cell="mythology", uid=b"loc")
    p = merge_on_receive(inc, [loc], cfg=CFG, es=_es(), nli="lexical")[0]
    assert p.decision == "RELATE", p.decision


def test_same_cell_high_sigma_pair_merges_unchanged():
    inc = _unit(A, b"ki", cell="agent.public", uid=b"inc")
    loc = _unit(B, b"kl", cell="agent.public", uid=b"loc")
    p = merge_on_receive(inc, [loc], cfg=CFG, es=_es(), nli="lexical")[0]
    assert p.decision == "MERGE", p.decision


def test_default_cell_preserves_merge():
    # cell defaults to "agent.public" on both -> same frame -> MERGE (unchanged).
    inc = MemoryUnit(id=b"inc", relevance_magnitude=1.0, status="ACTIVE",
                     claims=[Claim(claim_key=b"ki", text=A, claim_status="ACTIVE")])
    loc = MemoryUnit(id=b"loc", relevance_magnitude=1.0, status="ACTIVE",
                     claims=[Claim(claim_key=b"kl", text=B, claim_status="ACTIVE")])
    p = merge_on_receive(inc, [loc], cfg=CFG, es=_es(), nli="lexical")[0]
    assert p.decision == "MERGE", p.decision
