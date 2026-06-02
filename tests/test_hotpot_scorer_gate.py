"""G3 (HotpotQA pivot) — HotpotQA scorer (EM + token-F1) + competence gate.

The competence benchmark is PIVOTED from GAIA (gated; PI has no HF token) to
HotpotQA (distractor; open, CC BY-SA 4.0). These tests pin the OFFICIAL HotpotQA
answer metrics (hotpot_evaluate_v1.py, verified at source):

  normalize_answer = lowercase -> drop string.punctuation -> drop articles
                     (a|an|the) -> collapse whitespace.
  EM  = normalized exact equality.
  F1  = token-level 2PR/(P+R) over the Counter intersection; the yes/no/noanswer
        special case zeroes all metrics on a non-exact match.

The HotpotQA competence gate adapts Sasu's "GAIA L2 > 60% to be in the game" to a
HotpotQA bar: token-F1 >= 0.60 on the harder strata (medium + hard). An empty
medium+hard slice is NOT a pass.
"""
from __future__ import annotations

from smp.agent.hotpot_scorer import (
    competence_gate, em_score, f1_score, hotpot_normalize, score_predictions,
)


# -- normalization (the four official steps) --------------------------------

def test_normalize_lowercases_drops_articles_punct_whitespace():
    assert hotpot_normalize("The   Quick, Brown FOX!") == "quick brown fox"
    assert hotpot_normalize("a An the") == ""
    assert hotpot_normalize("U.S.A.") == "usa"


def test_normalize_handles_none_and_empty():
    assert hotpot_normalize(None) == ""
    assert hotpot_normalize("") == ""


# -- exact match ------------------------------------------------------------

def test_em_normalized_equality():
    assert em_score("The Eiffel Tower", "eiffel tower") == 1.0
    assert em_score("Paris.", "paris") == 1.0
    assert em_score("London", "Paris") == 0.0


# -- token F1 ---------------------------------------------------------------

def test_f1_perfect_and_zero():
    assert f1_score("quick brown fox", "the quick brown fox") == 1.0  # article dropped
    assert f1_score("cat", "dog") == 0.0


def test_f1_partial_overlap():
    # pred has 2 tokens, gold has 3, 2 common -> P=1.0 R=2/3 -> F1=0.8
    f1 = f1_score("brown fox", "quick brown fox")
    assert abs(f1 - 0.8) < 1e-9


def test_f1_yesno_special_case_zeroes_non_exact():
    # if either side is yes/no/noanswer and not exactly equal -> 0
    assert f1_score("yes", "no") == 0.0
    assert f1_score("yes the answer", "yes") == 0.0
    assert f1_score("yes", "yes") == 1.0


# -- aggregate scoring + stratification by level ----------------------------

def _preds():
    return [
        {"task_id": "a", "level": "easy", "prediction": "paris", "gold": "Paris"},
        {"task_id": "b", "level": "medium", "prediction": "brown fox",
         "gold": "quick brown fox"},  # EM 0, F1 0.8
        {"task_id": "c", "level": "hard", "prediction": "wrong", "gold": "right"},
    ]


def test_score_predictions_stratifies_by_level_with_em_and_f1():
    sc = score_predictions(_preds())
    assert sc["by_level"]["easy"]["em"] == 1.0
    assert sc["by_level"]["easy"]["f1"] == 1.0
    assert sc["by_level"]["medium"]["em"] == 0.0
    assert abs(sc["by_level"]["medium"]["f1"] - 0.8) < 1e-9
    assert sc["by_level"]["hard"]["em"] == 0.0
    assert sc["by_level"]["hard"]["f1"] == 0.0
    # overall averages
    assert abs(sc["overall"]["em"] - (1 / 3)) < 1e-9
    assert abs(sc["overall"]["f1"] - (1.8 / 3)) < 1e-9
    assert sc["overall"]["n"] == 3


# -- the competence gate ----------------------------------------------------

def test_gate_passes_when_medium_hard_f1_meets_bar():
    preds = [
        {"task_id": "m1", "level": "medium", "prediction": "quick brown fox",
         "gold": "quick brown fox"},   # F1 1.0
        {"task_id": "h1", "level": "hard", "prediction": "brown fox",
         "gold": "quick brown fox"},   # F1 0.8
    ]
    g = competence_gate(preds, bar=0.60)
    assert g["passed"] is True
    assert abs(g["gate_f1"] - 0.9) < 1e-9
    assert g["gate_n"] == 2
    assert g["gate_strata"] == ["medium", "hard"]


def test_gate_fails_below_bar():
    preds = [
        {"task_id": "m1", "level": "medium", "prediction": "x", "gold": "y"},
        {"task_id": "h1", "level": "hard", "prediction": "brown", "gold": "quick brown fox"},
    ]
    g = competence_gate(preds, bar=0.60)
    assert g["passed"] is False
    assert g["gate_f1"] < 0.60


def test_empty_hard_stratum_is_not_a_pass():
    # only easy tasks -> no medium/hard evidence -> NOT a pass (no evidence != pass)
    preds = [{"task_id": "e", "level": "easy", "prediction": "p", "gold": "p"}]
    g = competence_gate(preds, bar=0.60)
    assert g["passed"] is False
    assert g["gate_n"] == 0
