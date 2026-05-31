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
    )


__all__ = ["Thresholds", "load_thresholds", "DEFAULT_PATH"]
