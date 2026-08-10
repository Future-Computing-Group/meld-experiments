"""Task 1.3 — the four-way merge decision procedure (architecture A.2).

Every branch of the A.2 table is hit. Thresholds come from config (CFG), never
hardcoded. The load-bearing test (false-MERGE-zero): a high-sigma + contradicts
pair must NEVER be classified MERGE -- that is the dangerous error.
"""
from __future__ import annotations

import pytest

from smp.config import load_thresholds
from smp.merge.classifier import classify

CFG = load_thresholds()


def test_exact_key_merges():
    assert classify(kappa=1, cfg=CFG) == "MERGE"


def test_high_sigma_noncontradict_compatible_merges():
    assert classify(kappa=0, sigma=0.95, chi="neutral", ctx_ok=True, cfg=CFG) == "MERGE"


def test_mid_sigma_relates():
    assert classify(kappa=0, sigma=0.6, chi="neutral", ctx_ok=True, cfg=CFG) == "RELATE"


def test_high_sigma_incompatible_context_relates():
    # same meaning, different frame -> relate, don't merge
    assert classify(kappa=0, sigma=0.95, chi="neutral", ctx_ok=False, cfg=CFG) == "RELATE"


def test_contradiction_about_same_thing_conflicts():
    assert classify(kappa=1, chi="contradicts", cfg=CFG) == "CONFLICT"


def test_high_sigma_contradiction_conflicts():
    assert classify(kappa=0, sigma=0.95, chi="contradicts", ctx_ok=True, cfg=CFG) == "CONFLICT"


def test_stale_or_overruled_rejects():
    assert classify(kappa=1, retention_below_min=True, cfg=CFG) == "REJECT"


def test_overruled_status_rejects():
    assert classify(kappa=1, status="OVERRULED", cfg=CFG) == "REJECT"


def test_low_sigma_unrelated_relates_floor():
    # below sigma_lo and no key match: nothing to merge or relate strongly ->
    # the procedure returns RELATE only within [sigma_lo, theta_merge); below
    # sigma_lo it is NO_RELATION (no actionable relation) -- NOT REJECT, which
    # is the admission gate alone (D1).
    assert classify(kappa=0, sigma=0.1, chi="neutral", ctx_ok=True, cfg=CFG) == "NO_RELATION"


def test_false_merge_zero_on_dangerous_gold_case():
    # THE dangerous error: high sigma, contradicts -> must NOT merge.
    assert classify(kappa=0, sigma=0.97, chi="contradicts", ctx_ok=True, cfg=CFG) != "MERGE"


def test_reject_dominates_even_with_exact_key():
    # a stale exact-key match still rejects (gate precedence)
    assert classify(kappa=1, retention_below_min=True, chi="neutral", cfg=CFG) == "REJECT"


# --- Gate CONFLICT on sigma_lo (the relatedness floor), not theta_merge, so
# lexically-divergent contradictions are surfaced rather than dropped. ---
def test_midsigma_lexically_divergent_contradiction_conflicts():
    # A clearly contradictory pair (chi=contradicts) that is related but NOT
    # near-identical in surface form (sigma in [sigma_lo, theta_merge)). Gating
    # CONFLICT on theta_merge would route these to REJECT and drop them silently
    # (the conflict-recall blind spot). Gating CONFLICT on sigma_lo flags them.
    sigma = (CFG.sigma_lo + CFG.theta_merge) / 2.0
    assert classify(kappa=0, sigma=sigma, chi="contradicts", ctx_ok=True,
                    cfg=CFG) == "CONFLICT"


def test_below_floor_pair_is_no_relation():
    # D1 (S1 memo sec 1): the classifier's below-floor verdict is NO_RELATION,
    # never REJECT -- REJECT is the admission gate alone. The protocol layer
    # (merge_on_receive) maps an unreached NO_RELATION to INSERT.
    assert classify(kappa=0, sigma=0.05, chi="neutral", ctx_ok=True,
                    cfg=CFG) == "NO_RELATION"


def test_unrelated_spurious_contradiction_still_no_relation():
    # Below sigma_lo there is no actionable relation; even a (spurious)
    # contradiction verdict must not manufacture a CONFLICT out of nothing.
    assert classify(kappa=0, sigma=CFG.sigma_lo / 2.0, chi="contradicts",
                    ctx_ok=True, cfg=CFG) == "NO_RELATION"


def test_midsigma_noncontradiction_routing_unchanged():
    # The fix must not change non-contradiction routing: mid-band neutral still RELATEs.
    sigma = (CFG.sigma_lo + CFG.theta_merge) / 2.0
    assert classify(kappa=0, sigma=sigma, chi="neutral", ctx_ok=True,
                    cfg=CFG) == "RELATE"
