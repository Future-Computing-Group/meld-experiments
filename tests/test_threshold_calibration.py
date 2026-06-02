"""Fix #2 — calibrate the merge thresholds on a labeled claim-pair set.

The merge thresholds (theta_merge, sigma_lo) are CALIBRATED, not eyeballed
(manuscript Section 4, calibrated in M-2). The bug they replace: _calibrated_cfg
hardcoded theta_merge=0.5 < sigma_lo=0.55, an INVERTED pair that makes the RELATE
band [sigma_lo, theta_merge) empty (every candidate forced to MERGE -> false
merges, which fix #1 now turns into real recall loss).

These tests pin two properties: (a) on an encoder whose similarity separates the
classes (minilm), calibration recovers a properly-ordered, high-accuracy threshold
pair with a REACHABLE RELATE band; (b) the hashing encoder's similarity does NOT
separate genuine paraphrases (MERGE) from related-distinct pairs (RELATE), so no
threshold on hashing can be correct -- documenting WHY minilm is required.
"""
from __future__ import annotations

import pytest

from smp.merge.classifier import classify
from smp.merge.signals import EmbeddingSimilarity


def _require_minilm():
    try:
        es = EmbeddingSimilarity(encoder="minilm")
        es.sigma("warmup", "warmup")   # force the lazy sentence_transformers import
        return es
    except Exception as e:  # pragma: no cover - offline / no model
        pytest.skip(f"minilm encoder unavailable: {type(e).__name__}: {e}")


def _accuracy(cfg, pairs, encoder):
    es = EmbeddingSimilarity(encoder=encoder)
    return sum(
        classify(cfg=cfg, kappa=0, sigma=es.sigma(a, b), chi="neutral", ctx_ok=True) == gold
        for gold, a, b in pairs
    ) / len(pairs)


def test_calibration_on_minilm_is_ordered_accurate_and_relate_reachable():
    from smp.merge.calibration import CALIBRATION_PAIRS, calibrate_thresholds
    _require_minilm()
    cfg = calibrate_thresholds(CALIBRATION_PAIRS, encoder="minilm", nli="lexical",
                               r_min=0.1, tau=0.5)
    assert cfg.sigma_lo < cfg.theta_merge                 # invariant + reachable band
    assert _accuracy(cfg, CALIBRATION_PAIRS, "minilm") >= 0.88
    # the RELATE band is actually exercised by the fixture (not a dead branch)
    es = EmbeddingSimilarity(encoder="minilm")
    decisions = {classify(cfg=cfg, kappa=0, sigma=es.sigma(a, b), chi="neutral",
                          ctx_ok=True) for gold, a, b in CALIBRATION_PAIRS}
    assert "RELATE" in decisions


def test_thresholds_reject_inverted_ordering():
    # the bug: theta_merge=0.5 < sigma_lo=0.55 makes the RELATE band empty. The
    # config must reject any non-(sigma_lo < theta_merge) pair at construction.
    from smp.config import Thresholds
    with pytest.raises(ValueError):
        Thresholds(theta_merge=0.5, sigma_lo=0.55, r_min=0.1, tau=0.5)
    with pytest.raises(ValueError):
        Thresholds(theta_merge=0.5, sigma_lo=0.5, r_min=0.1, tau=0.5)  # equal: band empty
    # a properly-ordered pair constructs fine
    Thresholds(theta_merge=0.9, sigma_lo=0.12, r_min=0.1, tau=0.5)


def test_calibrated_minilm_config_matches_calibration():
    # the shipped minilm config IS the calibration output (reproducible provenance)
    from smp.agent.run_hotpot_gate import _calibrated_cfg
    from smp.merge.calibration import CALIBRATION_PAIRS, calibrate_thresholds
    _require_minilm()
    derived = calibrate_thresholds(CALIBRATION_PAIRS, encoder="minilm", nli="lexical",
                                   r_min=0.1, tau=0.5)
    shipped = _calibrated_cfg("minilm")
    assert abs(shipped.theta_merge - derived.theta_merge) < 1e-9
    assert abs(shipped.sigma_lo - derived.sigma_lo) < 1e-9


def test_calibrated_hashing_config_matches_calibration():
    # the shipped hashing config IS the calibration output for the hashing encoder
    from smp.agent.run_hotpot_gate import _calibrated_cfg
    from smp.merge.calibration import CALIBRATION_PAIRS, calibrate_thresholds
    derived = calibrate_thresholds(CALIBRATION_PAIRS, encoder="hashing", nli="lexical",
                                   r_min=0.1, tau=0.5)
    shipped = _calibrated_cfg("hashing")
    assert abs(shipped.theta_merge - derived.theta_merge) < 1e-9
    assert abs(shipped.sigma_lo - derived.sigma_lo) < 1e-9


def test_hashing_cannot_perfectly_separate_paraphrase_from_related():
    # the engineer's finding, encoded: hashing sigma puts a true paraphrase BELOW a
    # related-distinct pair, so the best possible accuracy on hashing is < 1.0.
    from smp.merge.calibration import CALIBRATION_PAIRS, calibrate_thresholds
    cfg_h = calibrate_thresholds(CALIBRATION_PAIRS, encoder="hashing", nli="lexical",
                                 r_min=0.1, tau=0.5)
    assert _accuracy(cfg_h, CALIBRATION_PAIRS, "hashing") < 1.0
