"""TDD: merge_on_receive computes epsilon from Claim.grounding and passes it to
classify, so shared grounding can promote a borderline-similarity pair to MERGE.
Ungrounded claims (grounding=()) give epsilon=0 -> path inert (HotpotQA unaffected).
"""
from smp.config import Thresholds
from smp.merge.signals import EmbeddingSimilarity
from smp.proto import Claim, MemoryUnit
from smp.protocol.merge_on_receive import merge_on_receive


def _unit(text, grounding=()):
    return MemoryUnit(id=text.encode(), relevance_magnitude=1.0, status="ACTIVE",
                      claims=[Claim(claim_key=(text + "::k").encode(), text=text,
                                    confidence=0.9, claim_status="ACTIVE",
                                    grounding=grounding)])


def _band_cfg_for(a, b, eps_hi):
    """Bracket theta/sigma_lo around the actual hashing sigma so the pair is
    deterministically in the RELATE band [sigma_lo, theta_merge)."""
    es = EmbeddingSimilarity(encoder="hashing")
    s = es.sigma(a, b)
    assert 0.0 < s < 1.0, s
    sigma_lo = max(0.0, s - 0.05)
    theta = min(1.0, s + 0.05)
    return es, Thresholds(theta_merge=theta, sigma_lo=sigma_lo, r_min=0.2,
                          tau=0.5, eps_hi=eps_hi)


A = "alpha beta gamma delta finding one"
B = "alpha beta gamma epsilon zeta finding two"


def test_shared_grounding_promotes_to_merge():
    es, cfg = _band_cfg_for(A, B, eps_hi=0.5)
    inc = _unit(A, grounding=("doc1:3",))
    loc = _unit(B, grounding=("doc1:3",))   # shares the grounding endpoint
    patches = merge_on_receive(inc, [loc], cfg=cfg, es=es, nli="lexical")
    assert any(p.decision == "MERGE" for p in patches), [p.decision for p in patches]


def test_no_shared_grounding_stays_relate():
    es, cfg = _band_cfg_for(A, B, eps_hi=0.5)
    inc = _unit(A, grounding=("doc1:3",))
    loc = _unit(B, grounding=("doc9:9",))   # disjoint grounding -> epsilon=0
    patches = merge_on_receive(inc, [loc], cfg=cfg, es=es, nli="lexical")
    assert all(p.decision != "MERGE" for p in patches), [p.decision for p in patches]


def test_ungrounded_inert_when_eps_hi_zero():
    es, cfg = _band_cfg_for(A, B, eps_hi=0.0)  # epsilon path disabled
    inc = _unit(A, grounding=("doc1:3",))
    loc = _unit(B, grounding=("doc1:3",))
    patches = merge_on_receive(inc, [loc], cfg=cfg, es=es, nli="lexical")
    assert all(p.decision != "MERGE" for p in patches), [p.decision for p in patches]
