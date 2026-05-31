"""Threshold config loader — NO silent defaults.

Thresholds (theta_merge, sigma_lo, r_min, tau) are calibrated, not derived;
they must be sourced from config, never hardcoded. The loader requires every
key to be present: a missing key raises, it does not fall back to a default.
"""
from __future__ import annotations

import pytest

from smp.config import Thresholds, load_thresholds


def test_loads_tracked_default():
    cfg = load_thresholds()
    assert isinstance(cfg, Thresholds)
    assert 0.0 < cfg.sigma_lo < cfg.theta_merge <= 1.0
    assert 0.0 < cfg.r_min < 1.0
    assert 0.0 < cfg.tau <= 1.0


def test_missing_key_raises_no_silent_default(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("theta_merge: 0.85\nsigma_lo: 0.55\n")  # r_min, tau missing
    with pytest.raises(KeyError):
        load_thresholds(str(bad))


def test_thresholds_is_immutable():
    cfg = load_thresholds()
    with pytest.raises((AttributeError, Exception)):
        cfg.theta_merge = 0.1
