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


def test_cross_boundary_identical_text_relates_not_merges():
    # D2 (S1 memo sec 2): the claim key hashes canonical content TOGETHER WITH
    # discrete scope, so the very same sentence published under two different
    # jurisdictions is a DIFFERENT claim (kappa=0), not a kappa=1 exact-key hit.
    # It falls through to the similarity path, where sigma=1.0 (identical text)
    # meets theta_merge but the cells are Context-incompatible -> R5 fails ->
    # R6 RELATE. This is the running-example message-2 regression (headline
    # finding): today the content-only key gives kappa=1 and merges silently
    # across the jurisdiction boundary.
    from smp.agent.memory import _cid, claim_key
    text = "The unemployment rate is 4.2 percent"
    us_cell, eu_cell = "jurisdiction.us", "jurisdiction.eu"
    inc = MemoryUnit(id=_cid(text, us_cell), cell=us_cell, relevance_magnitude=1.0,
                     status="ACTIVE",
                     claims=[Claim(claim_key=claim_key(text, us_cell), text=text,
                                   confidence=0.9, claim_status="ACTIVE")])
    loc = MemoryUnit(id=_cid(text, eu_cell), cell=eu_cell, relevance_magnitude=1.0,
                     status="ACTIVE",
                     claims=[Claim(claim_key=claim_key(text, eu_cell), text=text,
                                   confidence=0.9, claim_status="ACTIVE")])
    p = merge_on_receive(inc, [loc], cfg=CFG, es=_es(), nli="lexical")[0]
    assert p.decision == "RELATE", p.decision


def test_context_ineligible_candidate_does_not_shadow_admissible_merge():
    # Reviewer round 4, issue 1 (candidate shadowing): the byte-identical text in
    # an INCOMPATIBLE cell has the highest similarity (sigma=1.0) but fails the
    # Context gate; a paraphrase in the SAME cell satisfies the merge-admission
    # gates. Pre-fix, the shadower was selected and the pair RELATEd; the fix
    # makes selection gate-aware, so the admissible candidate MERGEs.
    inc = _unit(A, b"ki", cell="eu", uid=b"inc")
    shadow = _unit(A, b"ks", cell="us", uid=b"us1")          # sigma 1.0, ctx fails
    eligible = _unit(B, b"ke", cell="eu", uid=b"eu1")        # paraphrase, ctx ok
    p = merge_on_receive(inc, [shadow, eligible], cfg=CFG, es=_es(), nli="lexical")[0]
    assert p.decision == "MERGE", (p.decision, p.gates)


def test_conflict_priority_survives_gate_aware_selection():
    # A contradicting most-similar candidate must still surface as CONFLICT;
    # gate-aware selection must not bypass it toward an admissible merge.
    # the contradiction is the MOST similar candidate (negated copy of A itself),
    # so it must be the adjudication target and CONFLICT must fire; a lower-sigma
    # admissible merge candidate must not divert it.
    neg = "Paris is not the capital city of France"
    inc = _unit(A, b"ki", cell="eu", uid=b"inc")
    contra = _unit(neg, b"kc", cell="eu", uid=b"c1")
    friendly = _unit("France has Paris as its capital", b"kf", cell="eu", uid=b"f1")
    p = merge_on_receive(inc, [contra, friendly], cfg=CFG, es=_es(), nli="lexical")[0]
    assert p.decision == "CONFLICT", (p.decision, p.gates)


def test_exact_key_incompatible_frame_does_not_shadow_admissible_merge():
    # Round 5, issue 2: an exact-key candidate whose Context fails must not
    # shadow an admissible semantic merge. The exact-key hit is the
    # provisionally selected candidate; when it relates (kappa=1, ctx fails)
    # and an admissible same-frame paraphrase exists, the paraphrase merges.
    from smp.agent.memory import _cid, claim_key
    text = "The unemployment rate is 4.2 percent"
    us, eu = "jurisdiction.us", "jurisdiction.eu"
    inc = MemoryUnit(id=_cid(text, eu), cell=eu, relevance_magnitude=1.0, status="ACTIVE",
                     claims=[Claim(claim_key=claim_key(text, eu), text=text,
                                   claim_status="ACTIVE")])
    # exact-key twin: same content, same key inputs? NO - scoped key differs by
    # cell, so craft a literal same-key shadow: same text + same cell key but the
    # UNIT carries an incompatible frame via cell mismatch at the unit level.
    shadow = MemoryUnit(id=_cid(text, us), cell=us, relevance_magnitude=1.0, status="ACTIVE",
                        claims=[Claim(claim_key=claim_key(text, eu), text=text,
                                      claim_status="ACTIVE")])
    eligible = _unit("Unemployment stands at 4.2 percent", b"ke", cell=eu, uid=b"e1")
    p = merge_on_receive(inc, [shadow, eligible], cfg=CFG, es=_es(), nli="lexical")[0]
    assert p.decision == "MERGE", (p.decision, p.gates)
