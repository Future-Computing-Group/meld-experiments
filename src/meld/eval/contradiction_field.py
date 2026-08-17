"""R3: field contradiction-miss (NLI false-negative) rate on a naturalistic sample.

The merge safety guarantee is "never silently resolves a DETECTED contradiction", so the
deployment risk is the NLI MISS rate -- a real contradiction the NLI fails to flag, which
then silently merges. This module (a) builds a labelled sample of REAL HotpotQA sentences
(more naturalistic in form than the M-2 template gold) with controlled value-swap
contradictions and related-but-non-contradicting neighbours, and (b) scores the NLI's
contradiction recall, precision, and the missed-contradiction (false-negative) rate.

contradiction_metrics is a pure function (tested); running the real NLI over the sample is a
smoke-ladder step.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from meld.agent.mem_conditions import _gold_sentences, _make_conflicting


def contradiction_metrics(gold: List[str], pred: List[str]) -> Dict:
    """Over labelled pairs (gold/pred each a 3-way NLI label), report contradiction recall,
    the missed-contradiction (false-negative) rate, contradiction precision, and counts."""
    assert len(gold) == len(pred)
    pos = [i for i, g in enumerate(gold) if g == "contradicts"]
    tp = sum(1 for i in pos if pred[i] == "contradicts")
    fn = len(pos) - tp
    recall = tp / len(pos) if pos else 0.0
    missed = fn / len(pos) if pos else 0.0
    pred_pos = [i for i, p in enumerate(pred) if p == "contradicts"]
    tp_p = sum(1 for i in pred_pos if gold[i] == "contradicts")
    precision = tp_p / len(pred_pos) if pred_pos else 0.0
    return {"contradiction_recall": round(recall, 4),
            "missed_contradiction_rate": round(missed, 4),
            "contradiction_precision": round(precision, 4),
            "n_contradictions": len(pos), "tp": tp, "fn": fn, "n": len(gold)}


def build_field_sample(tasks, *, max_pairs: int = 80) -> List[Tuple[str, str, str]]:
    """Labelled (label, a, b) pairs from REAL HotpotQA gold sentences:
      * a real sentence + its value-swapped variant -> a genuine 'contradicts' pair;
      * two distinct real gold sentences (different facts) -> a 'neutral' pair.
    Real sentences give a naturalistic-form sample with controlled ground truth."""
    contra: List[Tuple[str, str, str]] = []
    neutral: List[Tuple[str, str, str]] = []
    for task in tasks:
        sents = [s for _, s in _gold_sentences(task)]
        for s in sents:
            sw = _make_conflicting(s)
            if sw is not None and sw != s:
                contra.append(("contradicts", s, sw))
        for i in range(len(sents) - 1):
            if sents[i] != sents[i + 1]:
                neutral.append(("neutral", sents[i], sents[i + 1]))
    half = max_pairs // 2
    return contra[:half] + neutral[:half]


__all__ = ["contradiction_metrics", "build_field_sample"]
