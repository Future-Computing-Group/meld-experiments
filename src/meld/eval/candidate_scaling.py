"""S8: claim-count scaling of candidate selection.

An external reviewer noted that the 0.23 ms candidate-scan result (supplement
Appendix M, ``supp:order``) is exercised on only a 16-claim workload (median
candidate-set size 4 -- most local claims there are not ACTIVE by the time
each call runs). This measures the SAME ``_best_candidate`` scan (same
behaviour-preserving monkeypatch ``eval.candidate_cost`` uses) at controlled,
FIXED store sizes N_local in {16, 64, 256, 1024, 4096}, so the O(N_local)
linear-scan claim and its crossover against NLI cost are grounded in a real
sweep rather than one small workload.

The store is a fixed snapshot: N_local distinct ACTIVE ``MemoryUnit``s built
directly (no ``Brain``/``apply`` bookkeeping -- nothing here mutates the
store), so N_local does not drift across the K deliveries scored against it.
Every incoming text is constructed to differ from every held text (paraphrases
reword, novel claims are a disjoint vocabulary), so ``_best_candidate`` never
takes the kappa==1 early return and always scans the full store: candidate-set
size scanned equals N_local by construction, deterministically.

Per-delta admission overhead is split into the scan (recorded by
``instrument_best_candidate``) and everything else in ``merge_on_receive``
(embed of the incoming text plus, when a candidate is found, the NLI call --
the two dominant remaining costs) by timing each whole ``merge_on_receive``
call and subtracting the recorded scan latency. Local + incoming texts are
embedded once before timing starts, so the "everything else" bucket isolates
NLI cost rather than mixing in a one-time cold-embed spike.
"""
from __future__ import annotations

import random
import statistics
import time
from collections import Counter
from typing import Any, Dict, List, Sequence

from meld.agent.memory import _unit_from_text
from meld.config import Thresholds
from meld.eval.candidate_cost import CandidateCostRecorder, instrument_best_candidate
from meld.merge.signals import EmbeddingSimilarity
from meld.protocol.merge_on_receive import merge_on_receive
from meld.proto import MemoryUnit

# Two disjoint vocabularies -- held/paraphrase facts describe industrial
# monitoring hardware; novel facts describe unrelated hobby/travel topics --
# so real embeddings separate "reworded" from "unrelated" without hand-tuning
# a similarity target.
_HELD_SUBJECTS = ("sensor", "server", "valve", "turbine", "pump", "beacon", "relay", "gateway")
_HELD_LOCATIONS = ("the north wing", "the south dock", "the control room", "the loading bay",
                   "the roof array", "the basement", "the east corridor", "the west annex")
_NOVEL_TOPICS = ("a chess opening", "a folk song", "a hiking trail", "a pasta recipe",
                 "a constellation", "a board game rule", "a river crossing", "a coral reef")


def _held_fact(i: int) -> str:
    subj, loc = _HELD_SUBJECTS[i % len(_HELD_SUBJECTS)], _HELD_LOCATIONS[i % len(_HELD_LOCATIONS)]
    return f"{subj.capitalize()} unit {i} is installed in {loc} and reports every {5 + i % 55} minutes."


def _paraphrase_fact(i: int) -> str:
    subj, loc = _HELD_SUBJECTS[i % len(_HELD_SUBJECTS)], _HELD_LOCATIONS[i % len(_HELD_LOCATIONS)]
    return f"{loc.capitalize()} houses {subj} unit {i}, which sends a report roughly every {5 + i % 55} minutes."


def _novel_fact(i: int) -> str:
    topic = _NOVEL_TOPICS[i % len(_NOVEL_TOPICS)]
    return f"Entry {i} describes {topic} first documented in the year {1900 + i % 120}."


def build_local_store(n: int) -> List[MemoryUnit]:
    """N distinct ACTIVE claims (one per unit), the fixed store snapshot."""
    return [_unit_from_text(_held_fact(i)) for i in range(n)]


def build_incoming_stream(n_local: int, k: int, seed: int) -> List[MemoryUnit]:
    """K incoming claims, a seeded 50/50 mix of paraphrases of held claims
    (index drawn from the held range) and novel, off-topic claims."""
    rng = random.Random(seed)
    stream = []
    for i in range(k):
        if rng.random() < 0.5:
            text = _paraphrase_fact(rng.randrange(n_local))
        else:
            text = _novel_fact(i)
        stream.append(_unit_from_text(text))
    return stream


def _warm(es: EmbeddingSimilarity, *unit_lists: Sequence[MemoryUnit]) -> None:
    """Embed every text once before timing starts, so the recorded scan
    latency reflects the O(N_local) dot-product/kappa loop, not a one-time
    cold-embed spike on the incoming text."""
    for units in unit_lists:
        for u in units:
            for c in u.claims:
                es.sigma(c.text, c.text)


def _median(xs: List[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def _ols(xs: List[float], ys: List[float]) -> Dict[str, float]:
    """Plain least-squares fit, ``y = a + b*x`` (stdlib only)."""
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return {"intercept_ms": a, "slope_ms_per_claim": b, "r2": r2}


def _fit(per_size: List[Dict[str, Any]]) -> Dict[str, float]:
    xs = [row["n_local"] for row in per_size]
    ys = [row["candidate_cost"]["scan_latency_ms_median"] for row in per_size]
    fit = _ols(xs, ys)
    fit["ms_per_1k_claims"] = fit["slope_ms_per_claim"] * 1000.0
    nli_samples = [row["other_ms_matched_median"] for row in per_size
                  if row["other_ms_matched_median"] == row["other_ms_matched_median"]]  # drop NaN
    nli_ref = statistics.mean(nli_samples) if nli_samples else float("nan")
    fit["nli_reference_ms"] = nli_ref
    b, a = fit["slope_ms_per_claim"], fit["intercept_ms"]
    fit["crossover_n_claims"] = (nli_ref - a) / b if b > 0 and nli_ref == nli_ref else float("nan")
    return fit


def run_candidate_scaling(store_sizes: Sequence[int], *, k_per_size: int, seed: int,
                          encoder: str, nli: str, cfg: Thresholds,
                          cache_dir: str | None = None) -> Dict[str, Any]:
    """For each N_local in ``store_sizes``, deliver ``k_per_size`` incoming
    claims (seeded mix of paraphrase/novel) through the real
    ``merge_on_receive`` path against a FIXED N_local-claim store, recording
    per-call candidate-set size, scan latency, and the non-scan ("other",
    embed+NLI-dominated) remainder of each call."""
    es = EmbeddingSimilarity(encoder=encoder, cache_dir=cache_dir)
    per_size: List[Dict[str, Any]] = []

    for n in store_sizes:
        local = build_local_store(n)
        incoming = build_incoming_stream(n, k_per_size, seed=seed)
        _warm(es, local, incoming)

        rec = CandidateCostRecorder()
        decisions: Counter = Counter()
        other_ms_matched: List[float] = []
        other_ms_unmatched: List[float] = []
        with instrument_best_candidate(rec):
            for unit in incoming:
                t0 = time.perf_counter()
                patches = merge_on_receive(unit, local, cfg=cfg, es=es, nli=nli)
                total_ms = (time.perf_counter() - t0) * 1000.0
                scan_ms = rec.scan_latencies_ms[-1]
                other_ms = total_ms - scan_ms
                p = patches[0]
                decisions[p.decision] += 1
                if "novel" in p.gates_that_fired:   # no local candidate -> no NLI call
                    other_ms_unmatched.append(other_ms)
                else:
                    other_ms_matched.append(other_ms)

        cc_summary = rec.summary()
        per_size.append({
            "n_local": n,
            "candidate_cost": cc_summary,
            "decision_counts": dict(decisions),
            "n_matched": len(other_ms_matched),
            "n_unmatched": len(other_ms_unmatched),
            "other_ms_matched_median": _median(other_ms_matched),
            "other_ms_matched_p95": _percentile(other_ms_matched, 0.95),
            "other_ms_unmatched_median": _median(other_ms_unmatched),
        })

    return {
        "store_sizes": list(store_sizes), "k_per_size": k_per_size, "seed": seed,
        "encoder": encoder, "nli": nli, "per_size": per_size, "fit": _fit(per_size),
    }


__all__ = ["run_candidate_scaling", "build_local_store", "build_incoming_stream"]
