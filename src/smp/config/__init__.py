"""Merge-threshold config loader. NO silent defaults.

Thresholds are calibrated, not derived; the loader requires every key to be
present in the YAML (a missing key raises ``KeyError``, never a fallback). The
tracked default lives at ``thresholds.yaml`` next to this module; M-2 writes
calibrated values to a separate gitignored config under ``data/``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "thresholds.yaml")

_REQUIRED = ("theta_merge", "sigma_lo", "r_min", "tau")


@dataclass(frozen=True)
class Thresholds:
    theta_merge: float
    sigma_lo: float
    r_min: float
    tau: float
    # epsilon (evidence-overlap) promotion threshold. OPT-IN: eps_hi=0.0 disables
    # the epsilon-corroborated MERGE path entirely (the ungrounded default, so
    # HotpotQA / M-2 are unaffected). On grounded data it is calibrated, not
    # derived, like the other thresholds.
    eps_hi: float = 0.0

    def __post_init__(self) -> None:
        # The RELATE band is [sigma_lo, theta_merge); it is only reachable when
        # sigma_lo < theta_merge. An inverted or equal pair (the _calibrated_cfg
        # 0.5/0.55 bug) leaves RELATE dead and forces every candidate to MERGE.
        if not (0.0 <= self.sigma_lo < self.theta_merge <= 1.0):
            raise ValueError(
                f"thresholds must satisfy 0 <= sigma_lo ({self.sigma_lo}) < "
                f"theta_merge ({self.theta_merge}) <= 1; an inverted/equal pair "
                "leaves the RELATE band [sigma_lo, theta_merge) empty (Section 4)"
            )
        if not (0.0 <= self.eps_hi <= 1.0):
            raise ValueError(
                f"eps_hi ({self.eps_hi}) must be in [0, 1]; it is an evidence-"
                "overlap (Jaccard) threshold. eps_hi=0 disables epsilon promotion."
            )


def load_thresholds(path: str | None = None) -> Thresholds:
    path = path or DEFAULT_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise KeyError(
            f"threshold config {path} missing required keys {missing}; "
            "no silent defaults -- every threshold must be sourced from config"
        )
    return Thresholds(
        theta_merge=float(raw["theta_merge"]),
        sigma_lo=float(raw["sigma_lo"]),
        r_min=float(raw["r_min"]),
        tau=float(raw["tau"]),
        # eps_hi is OPTIONAL (default 0.0 = epsilon promotion off); not in
        # _REQUIRED so existing 4-key configs load unchanged and stay ungrounded.
        eps_hi=float(raw.get("eps_hi", 0.0)),
    )


__all__ = ["Thresholds", "load_thresholds", "DEFAULT_PATH"]
