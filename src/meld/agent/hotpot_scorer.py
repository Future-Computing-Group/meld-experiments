"""HotpotQA answer scorer (EM + token-F1) + the competence gate.

The competence benchmark is PIVOTED from GAIA (gated on Hugging Face; no access token is assumed, so it has
no HF token) to **HotpotQA distractor** (open, CC BY-SA 4.0). HotpotQA fits MELD
better than GAIA: each item's context paragraphs are fed THROUGH the MELD brain as
merged observations, so the gate exercises recall + multi-hop reasoning over
brain-resident evidence, not tool use.

This module reimplements the OFFICIAL HotpotQA answer metrics
(``hotpot_evaluate_v1.py``, verified at source; Yang et al. EMNLP 2018,
arXiv 1809.09600):

  normalize_answer = lowercase -> drop string.punctuation -> drop articles
                     (a|an|the) -> collapse whitespace.
  EM  = normalized exact equality.
  F1  = token-level 2*P*R/(P+R) over the multiset (Counter) intersection of the
        normalized whitespace-tokens, with the yes/no/noanswer special case that
        zeroes all metrics on a non-exact match.

The competence gate adapts Sasu's "GAIA Level-2 > 60% to be in the game" to a
HotpotQA bar (see the campaign plan for the justification): **token-F1 >= 0.60 on
the medium+hard strata**. An empty medium+hard slice is NOT a pass (no evidence is
not a pass), mirroring the GAIA gate's empty-L2 rule.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional

# the harder strata that the gate is measured on (HotpotQA levels are textual)
GATE_STRATA = ["medium", "hard"]
DEFAULT_BAR = 0.60

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_YESNO = {"yes", "no", "noanswer"}


def hotpot_normalize(s: Optional[str]) -> str:
    """The official HotpotQA ``normalize_answer``: lowercase, drop punctuation,
    drop articles, collapse whitespace."""
    s = (s or "").lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLE_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def em_score(prediction: Optional[str], gold: Optional[str]) -> float:
    """Normalized exact-match (1.0 / 0.0)."""
    return 1.0 if hotpot_normalize(prediction) == hotpot_normalize(gold) else 0.0


def f1_score(prediction: Optional[str], gold: Optional[str]) -> float:
    """Official HotpotQA token-level F1 (the answer F1, not the supporting-fact
    F1). The yes/no/noanswer special case zeroes a non-exact match."""
    npred = hotpot_normalize(prediction)
    ngold = hotpot_normalize(gold)
    # yes/no/noanswer special case (mirrors hotpot_evaluate_v1.update_answer)
    for tok in _YESNO:
        if (npred == tok or ngold == tok) and npred != ngold:
            return 0.0
    pred_toks = npred.split()
    gold_toks = ngold.split()
    common = Counter(pred_toks) & Counter(gold_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_toks)
    recall = n_common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def score_predictions(preds: List[Dict]) -> Dict:
    """``preds``: list of {task_id, level, prediction, gold}. Returns per-level
    EM/F1 (stratified by HotpotQA ``level``) and overall EM/F1 (macro over items).
    """
    by_level: Dict[str, Dict[str, float]] = {}
    sum_em = sum_f1 = 0.0
    for r in preds:
        lvl = r["level"]
        em = em_score(r.get("prediction") or "", r["gold"])
        f1 = f1_score(r.get("prediction") or "", r["gold"])
        b = by_level.setdefault(lvl, {"em": 0.0, "f1": 0.0, "n": 0,
                                      "_em_sum": 0.0, "_f1_sum": 0.0})
        b["n"] += 1
        b["_em_sum"] += em
        b["_f1_sum"] += f1
        sum_em += em
        sum_f1 += f1
    for b in by_level.values():
        n = b["n"]
        b["em"] = b.pop("_em_sum") / n if n else 0.0
        b["f1"] = b.pop("_f1_sum") / n if n else 0.0
    n = len(preds)
    return {"by_level": by_level, "overall": {
        "em": (sum_em / n) if n else 0.0,
        "f1": (sum_f1 / n) if n else 0.0,
        "n": n}}


def competence_gate(preds: List[Dict], bar: float = DEFAULT_BAR,
                    strata: Optional[List[str]] = None) -> Dict:
    """The G3 competence gate (HotpotQA pivot): PASS iff there are items in the
    harder strata (default medium+hard) AND their mean token-F1 >= ``bar``
    (default 0.60). An empty gate slice is NOT a pass."""
    strata = strata or GATE_STRATA
    gate_preds = [r for r in preds if r["level"] in strata]
    f1s = [f1_score(r.get("prediction") or "", r["gold"]) for r in gate_preds]
    gate_f1 = (sum(f1s) / len(f1s)) if f1s else 0.0
    passed = len(f1s) > 0 and gate_f1 >= bar
    scored = score_predictions(preds)
    return {
        "passed": bool(passed),
        "bar": bar,
        "gate_f1": gate_f1,
        "gate_n": len(f1s),
        "gate_strata": list(strata),
        "by_level": scored["by_level"],
    }


__all__ = ["hotpot_normalize", "em_score", "f1_score", "score_predictions",
           "competence_gate", "GATE_STRATA", "DEFAULT_BAR"]
