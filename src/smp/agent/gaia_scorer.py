"""GAIA scorer (quasi-exact-match) + the competence gate (GAIA L2 >= 60%).

Verified at source (Mialon et al. 2311.12983, Sec. 3.2): GAIA evaluation is a
quasi-exact match between the model's answer and the single ground-truth answer,
"up to some normalization that is tied to the type of the ground truth"; the
answer is a string, a number, or a comma-separated list. The model finishes with
"FINAL ANSWER: <answer>". This module reimplements that type-tied normalization
(it does not depend on the gated GAIA dataset).

The competence gate is Sasu's bar: GAIA Level-2 pass-rate >= 60% to "be in the
game" (GAIA v1 is a canary, not a long-horizon eval). ``competence_gate``
returns pass/fail + the L2 rate; an empty L2 slice is NOT a pass (no evidence is
not a pass).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# Accept the GAIA template with a space or underscore (open models emit both
# "FINAL ANSWER:" and "FINAL_ANSWER:") and an optional bold/markdown wrapper.
_FINAL_RE = re.compile(r"FINAL[ _]ANSWER\s*:\s*(.*?)\s*$",
                       re.IGNORECASE | re.MULTILINE)
_NUM_RE = re.compile(r"^[+-]?\d{1,3}(,\d{3})*(\.\d+)?$|^[+-]?\d+(\.\d+)?$")
_ARTICLES = {"a", "an", "the"}

# A leading "(the) answer is[:]" preamble an instruct model often prepends; the
# only over-verbose pattern we can strip WITHOUT guessing (see tighten below).
_PREAMBLE_RE = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*", re.IGNORECASE)

# A bare action-verb wrapper a model emits instead of a clean FINAL ANSWER line
# on a forced-synthesis turn, e.g. "final_answer[no]" / "Action: recall[...]".
# The intended answer is the bracketed payload (caught on the HotpotQA ext-smoke,
# react/qwen). Matches the whole string (anchored) so a genuine bracketed list
# answer that is NOT a verb wrapper is left intact.
_VERB_WRAP_RE = re.compile(
    r"^\s*(?:action\s*:\s*)?(?:final_answer|recall|remember|web_search|"
    r"read_file|calculator)\s*\[(.*)\]\s*$", re.IGNORECASE | re.DOTALL)

# A model sometimes emits the answer and THEN keeps going with ReAct scaffolding
# on a following line ("no\nAction: recall[query]"); the answer is the span
# BEFORE the first Thought/Action/Observation marker. Cutting there is safe for a
# short-answer task (a genuine HotpotQA/GAIA answer never contains these markers),
# and we only cut when there is content before the marker (start>0) so a string
# that IS wholly an action wrapper still flows to _VERB_WRAP_RE. Caught on the
# HotpotQA memdelta run (qwen fixed_recall: pollution rose with recalled-context
# richness -> differential token-F1 drag across conditions).
_REACT_TAIL_RE = re.compile(r"(?:^|\n)\s*(?:thought|action|observation)\s*:",
                            re.IGNORECASE)

# The prompt's template placeholder echoed verbatim ("[YOUR FINAL ANSWER]",
# "[your answer here]") -- a non-answer; normalize to empty so the caller's
# fallback drops to a recalled fact rather than returning the literal template.
_TEMPLATE_ECHO_RE = re.compile(
    r"^\s*(?:final\s+answer\s*:\s*)?\[\s*your\s+(?:final\s+)?answer"
    r"(?:\s+here)?\s*\]\s*$", re.IGNORECASE)


def tighten_gaia_answer(answer: Optional[str]) -> str:
    """Normalize a model's final-answer text toward the GAIA short form, doing
    ONLY the safe, unambiguous strips (never guessing the short answer out of a
    sentence -- that would create false positives; the prompt enforces brevity).

    Strips: surrounding whitespace, a leading "the answer is[:]" preamble, and a
    single trailing sentence period. Leaves everything else verbatim so the
    quasi-exact-match scorer sees what the model meant.
    """
    s = (answer or "").strip()
    if not s:
        return ""
    # strip a trailing ReAct continuation ("no\nAction: recall[...]") -> keep the
    # answer span before the first marker (only when there IS a preceding span).
    mtail = _REACT_TAIL_RE.search(s)
    if mtail and mtail.start() > 0:
        s = s[:mtail.start()].strip()
        if not s:
            return ""
    # drop a verbatim template-placeholder echo (a non-answer) -> empty.
    if _TEMPLATE_ECHO_RE.match(s):
        return ""
    # unwrap a bare action-verb wrapper to its bracketed payload (the model meant
    # the payload, not the verb call); recurse once so the payload is tightened.
    m = _VERB_WRAP_RE.match(s)
    if m:
        return tighten_gaia_answer(m.group(1))
    s = _PREAMBLE_RE.sub("", s).strip()
    # strip a single trailing period that is sentence punctuation, but keep a
    # decimal point inside a number (e.g. "3.14") and an ellipsis untouched.
    if s.endswith(".") and not s.endswith("..") and not _is_number(s):
        s = s[:-1].strip()
    return s


def extract_final_answer(raw: str) -> Optional[str]:
    """Return the LAST 'FINAL ANSWER: ...' payload (tightened to the GAIA short
    form), or None if absent."""
    matches = _FINAL_RE.findall(raw or "")
    if not matches:
        return None
    return tighten_gaia_answer(matches[-1])


def _norm_string(s: str) -> str:
    toks = [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _ARTICLES]
    return " ".join(toks)


def _is_number(s: str) -> bool:
    return bool(_NUM_RE.match(s.strip()))


def _norm_number(s: str) -> Optional[float]:
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _norm_list(s: str) -> List[str]:
    return [p for p in (_norm_one(x) for x in s.split(",")) if p != ""]


def _norm_one(s: str) -> str:
    s = s.strip()
    if _is_number(s):
        v = _norm_number(s)
        return ("%g" % v) if v is not None else s.lower()
    return _norm_string(s)


def quasi_exact_match(prediction: str, gold: str) -> bool:
    """Type-tied quasi-exact match. Numbers compare numerically (thousands
    separators stripped); comma-lists compare element-wise after normalization;
    strings compare case/article/whitespace-insensitively."""
    if prediction is None or gold is None:
        return False
    p, g = prediction.strip(), gold.strip()
    # number type
    if _is_number(p) and _is_number(g):
        return _norm_number(p) == _norm_number(g)
    # list type (either side has a separator)
    if "," in p or "," in g:
        return _norm_list(p) == _norm_list(g)
    # string type
    return _norm_string(p) == _norm_string(g)


def score_predictions(preds: List[Dict]) -> Dict:
    """``preds``: list of {task_id, level, prediction, gold}. Returns per-level
    correct/n/rate and an overall rate (quasi-exact-match)."""
    by_level: Dict[str, Dict[str, float]] = {}
    overall_correct = 0
    for r in preds:
        lvl = r["level"]
        ok = quasi_exact_match(r.get("prediction") or "", r["gold"])
        b = by_level.setdefault(lvl, {"correct": 0, "n": 0, "rate": 0.0})
        b["n"] += 1
        if ok:
            b["correct"] += 1
            overall_correct += 1
    for b in by_level.values():
        b["rate"] = b["correct"] / b["n"] if b["n"] else 0.0
    n = len(preds)
    return {"by_level": by_level, "overall": {
        "correct": overall_correct, "n": n,
        "rate": (overall_correct / n) if n else 0.0}}


def competence_gate(preds: List[Dict], bar: float = 0.60) -> Dict:
    """The G3 competence gate: PASS iff there are L2 tasks AND the L2
    quasi-exact-match rate >= ``bar`` (Sasu's bar, default 0.60)."""
    scored = score_predictions(preds)
    l2 = scored["by_level"].get("L2", {"correct": 0, "n": 0, "rate": 0.0})
    passed = l2["n"] > 0 and l2["rate"] >= bar
    return {
        "passed": bool(passed),
        "bar": bar,
        "l2_rate": l2["rate"],
        "l2_correct": l2["correct"],
        "l2_n": l2["n"],
        "by_level": scored["by_level"],
    }


__all__ = ["extract_final_answer", "tighten_gaia_answer", "quasi_exact_match",
           "score_predictions", "competence_gate"]
