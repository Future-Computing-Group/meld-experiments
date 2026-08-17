"""E2(b): merge decision-procedure robustness under degraded signals.

The decision procedure consumes two upstream signals: the dense similarity
``sigma`` (an embedding cosine) and the NLI verdict ``chi``. The procedure is
otherwise evaluated only on clean inputs; this module stresses it under
degraded signals: inject Gaussian noise into sigma (a degraded / out-of-domain
encoder) and randomly flip a fraction of chi verdicts (a degraded NLI), and
report how macro-F1 and the safety-critical false-merge-rate respond. Pure
functions over pre-scored (gold, sigma, chi) rows; deterministic in ``seed``.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from meld.merge.m2_eval import DECISIONS, classification_metrics, decisions_full

_CHI = ("contradicts", "entails", "neutral")


def _perturb_sigma(scored, sd: float, rng: random.Random):
    """Add N(0, sd) noise to each sigma, clipped to [0,1]."""
    out = []
    for g, s, c in scored:
        ns = s + rng.gauss(0.0, sd)
        ns = 0.0 if ns < 0.0 else 1.0 if ns > 1.0 else ns
        out.append((g, ns, c))
    return out


def _perturb_chi(scored, flip_rate: float, rng: random.Random):
    """With probability flip_rate, replace each chi with a different random label."""
    out = []
    for g, s, c in scored:
        if rng.random() < flip_rate:
            alt = [x for x in _CHI if x != c]
            c = alt[rng.randrange(len(alt))]
        out.append((g, s, c))
    return out


def _avg_metrics(trial_metrics: List[Dict]) -> Dict:
    keys = ("macro_f1", "false_merge_rate")
    n = len(trial_metrics)
    return {k: sum(m[k] for m in trial_metrics) / n for k in keys}


def robustness_sweep(scored, *, theta: float, sigma_lo: float,
                     noise_sds: List[float], flip_rates: List[float],
                     n_trials: int = 20, seed: int = 0) -> Dict:
    """Two independent one-axis sweeps (noise on sigma; flips on chi).

    Returns {"noise": {sd: {macro_f1, false_merge_rate}},
             "flip":  {rate: {macro_f1, false_merge_rate}}}, each level averaged
    over ``n_trials`` perturbation draws (a level of 0.0 is the clean point, run
    once -- it is deterministic). Deterministic in ``seed``."""
    gold = [g for g, _, _ in scored]

    def _eval(rows):
        return classification_metrics(gold, decisions_full(rows, theta=theta,
                                                            sigma_lo=sigma_lo))

    noise_out: Dict[float, Dict] = {}
    for sd in noise_sds:
        if sd == 0.0:
            noise_out[sd] = {k: _eval(scored)[k]
                             for k in ("macro_f1", "false_merge_rate")}
            continue
        rng = random.Random((seed << 16) ^ int(sd * 1000))
        trials = [_eval(_perturb_sigma(scored, sd, rng)) for _ in range(n_trials)]
        noise_out[sd] = _avg_metrics(trials)

    flip_out: Dict[float, Dict] = {}
    for fr in flip_rates:
        if fr == 0.0:
            flip_out[fr] = {k: _eval(scored)[k]
                            for k in ("macro_f1", "false_merge_rate")}
            continue
        rng = random.Random((seed << 16) ^ 0x9E37 ^ int(fr * 1000))
        trials = [_eval(_perturb_chi(scored, fr, rng)) for _ in range(n_trials)]
        flip_out[fr] = _avg_metrics(trials)

    return {"noise": noise_out, "flip": flip_out,
            "theta": theta, "sigma_lo": sigma_lo, "n_trials": n_trials}


__all__ = ["robustness_sweep"]
