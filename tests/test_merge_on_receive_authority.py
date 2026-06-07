"""TDD: merge_on_receive computes authority_ok from the incoming (source) vs local
(target) unit authority fields and passes it to classify (§5.3).

A high-sigma SIMILARITY merge from a LOW-authority source into a more-canonical
(higher-authority) local target is admitted as a candidate RELATE, not an automatic
MERGE -- so a canonical brain is never silently rewritten by a low-authority peer.
Equal default authority (both OBSERVATION) -> no downgrade -> MERGE (M-2-preserving).
An exact-key (kappa=1) match is the same claim -> still MERGE regardless of authority.
"""
from smp.config import Thresholds
from smp.merge.signals import EmbeddingSimilarity
from smp.proto import Claim, MemoryUnit
from smp.protocol.merge_on_receive import merge_on_receive

CFG = Thresholds(theta_merge=0.5, sigma_lo=0.12, r_min=0.2, tau=0.5)
# theta_merge low enough that a near-identical paraphrase pair lands on the
# SIMILARITY-merge branch under the hashing encoder.


def _es():
    return EmbeddingSimilarity(encoder="hashing")


def _unit(text, key, authority="OBSERVATION", uid=b"u"):
    return MemoryUnit(id=uid, relevance_magnitude=1.0, status="ACTIVE",
                      authority=authority,
                      claims=[Claim(claim_key=key, text=text, claim_status="ACTIVE")])


PARA_A = "Paris is the capital city of France"
PARA_B = "Paris is the capital of France"


def test_low_authority_similarity_merge_into_canonical_relates():
    # distinct keys (kappa=0), high sigma, source SPECULATION < local PRECEDENT
    # -> downgrade the auto-merge to a candidate RELATE.
    inc = _unit(PARA_A, b"ki", authority="SPECULATION", uid=b"inc")
    loc = _unit(PARA_B, b"kl", authority="PRECEDENT", uid=b"loc")
    p = merge_on_receive(inc, [loc], cfg=CFG, es=_es(), nli="lexical")[0]
    assert p.decision == "RELATE", p.decision


def test_equal_default_authority_still_merges():
    # both default OBSERVATION -> equal -> authority_ok True -> MERGE (M-2-preserving).
    inc = _unit(PARA_A, b"ki", authority="OBSERVATION", uid=b"inc")
    loc = _unit(PARA_B, b"kl", authority="OBSERVATION", uid=b"loc")
    p = merge_on_receive(inc, [loc], cfg=CFG, es=_es(), nli="lexical")[0]
    assert p.decision == "MERGE", p.decision


def test_exact_key_from_low_authority_still_merges():
    # same claim key -> kappa=1 -> no rewrite risk -> MERGE regardless of authority.
    inc = _unit(PARA_A, b"same", authority="SPECULATION", uid=b"inc")
    loc = _unit(PARA_A, b"same", authority="PRECEDENT", uid=b"loc")
    p = merge_on_receive(inc, [loc], cfg=CFG, es=_es(), nli="lexical")[0]
    assert p.decision == "MERGE", p.decision


def test_high_authority_source_into_low_authority_local_merges():
    # source PRECEDENT >= local OBSERVATION -> authority_ok True -> MERGE.
    inc = _unit(PARA_A, b"ki", authority="PRECEDENT", uid=b"inc")
    loc = _unit(PARA_B, b"kl", authority="OBSERVATION", uid=b"loc")
    p = merge_on_receive(inc, [loc], cfg=CFG, es=_es(), nli="lexical")[0]
    assert p.decision == "MERGE", p.decision
