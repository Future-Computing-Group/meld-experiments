"""TDD for the authority-gated merge-admission rule in classify() (§5.3).

A SIMILARITY-merge (kappa=0, sigma>=theta_merge, chi!=contradicts, ctx_ok) from a
LOW-authority source into a more-canonical (high-authority) local brain is admitted
only as a candidate RELATE, not an automatic MERGE, so a canonical brain is never
silently rewritten by a low-authority peer. An EXACT-KEY (kappa=1) match is the SAME
claim (no rewrite risk) and is unaffected. ``authority_ok`` defaults True, so every
existing caller and the M-2 benchmark (default/equal authority) are byte-for-byte
unchanged.
"""
from smp.config import load_thresholds
from smp.merge.classifier import classify

CFG = load_thresholds()


def test_low_authority_similarity_merge_downgraded_to_relate():
    # high-sigma, non-contradicting, compatible frame, but the source is LESS
    # authoritative than the canonical local -> admit as candidate (RELATE).
    assert classify(kappa=0, sigma=0.95, chi="neutral", ctx_ok=True,
                    authority_ok=False, cfg=CFG) == "RELATE"


def test_equal_authority_similarity_merge_still_merges():
    # the M-2-preserving case: default/equal authority -> authority_ok True ->
    # the similarity merge is admitted as MERGE exactly as before.
    assert classify(kappa=0, sigma=0.95, chi="neutral", ctx_ok=True,
                    authority_ok=True, cfg=CFG) == "MERGE"
    # default authority_ok must equal the explicit-True behaviour
    assert classify(kappa=0, sigma=0.95, chi="neutral", ctx_ok=True,
                    cfg=CFG) == "MERGE"


def test_exact_key_from_low_authority_still_merges():
    # an EXACT-KEY match is the SAME claim -> no silent rewrite risk -> MERGE
    # regardless of authority.
    assert classify(kappa=1, authority_ok=False, cfg=CFG) == "MERGE"


def test_high_authority_source_into_low_authority_local_merges():
    # source at least as authoritative as the local target -> authority_ok True ->
    # the canonical update flows as a normal MERGE.
    assert classify(kappa=0, sigma=0.95, chi="neutral", ctx_ok=True,
                    authority_ok=True, cfg=CFG) == "MERGE"
