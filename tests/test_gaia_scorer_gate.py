"""G3 — GAIA scorer (quasi-exact-match) + the competence gate (L2 >= 60%).

GAIA scoring is verified at source (Mialon et al. 2311.12983, Sec. 3.2):
"evaluation is done via quasi exact match between a model's answer and the
ground truth (up to some normalization that is tied to the type of the ground
truth). There is only one correct answer." The model must finish with
"FINAL ANSWER: <answer>".

The competence gate is Sasu's bar: GAIA Level-2 pass-rate >= 60%. These tests
pin the scorer's normalization behaviour and the gate's pass/fail logic on a
TINY OFFLINE fixture; they do NOT produce a real GAIA score (that is the gated
CSC run).
"""
from __future__ import annotations

from meld.agent.gaia_scorer import (
    extract_final_answer,
    quasi_exact_match,
    score_predictions,
    competence_gate,
)


# --------------------------------------------------------------------------
# FINAL ANSWER extraction (the GAIA prompt contract)
# --------------------------------------------------------------------------


def test_extract_final_answer_from_template():
    raw = "Thought: it is Paris.\nFINAL ANSWER: Paris"
    assert extract_final_answer(raw) == "Paris"


def test_extract_final_answer_takes_last_occurrence():
    raw = "FINAL ANSWER: draft\nactually no\nFINAL ANSWER: 42"
    assert extract_final_answer(raw) == "42"


def test_extract_final_answer_missing_returns_none():
    assert extract_final_answer("I am not sure.") is None


# --------------------------------------------------------------------------
# quasi-exact-match normalization (type-tied)
# --------------------------------------------------------------------------


def test_qem_string_case_and_article_insensitive():
    assert quasi_exact_match("The Paris", "paris")
    assert quasi_exact_match("PARIS", "Paris")


def test_qem_number_normalizes_thousands_separators():
    assert quasi_exact_match("1,000", "1000")
    assert quasi_exact_match("3.0", "3")


def test_qem_list_order_and_whitespace():
    assert quasi_exact_match("a, b ,c", "a,b,c")


def test_qem_rejects_wrong_answer():
    assert not quasi_exact_match("Lyon", "Paris")
    assert not quasi_exact_match("41", "42")


# --------------------------------------------------------------------------
# scoring a set of predictions
# --------------------------------------------------------------------------


def test_score_predictions_computes_per_level_rate():
    preds = [
        {"task_id": "1", "level": "L2", "prediction": "Paris", "gold": "paris"},
        {"task_id": "2", "level": "L2", "prediction": "Lyon", "gold": "Paris"},
        {"task_id": "3", "level": "L1", "prediction": "42", "gold": "42"},
    ]
    res = score_predictions(preds)
    assert res["by_level"]["L2"]["n"] == 2
    assert res["by_level"]["L2"]["correct"] == 1
    assert abs(res["by_level"]["L2"]["rate"] - 0.5) < 1e-9
    assert abs(res["by_level"]["L1"]["rate"] - 1.0) < 1e-9


# --------------------------------------------------------------------------
# the competence gate: GAIA L2 >= 60%
# --------------------------------------------------------------------------


def test_competence_gate_fails_below_bar():
    preds = [
        {"task_id": "1", "level": "L2", "prediction": "Paris", "gold": "paris"},
        {"task_id": "2", "level": "L2", "prediction": "Lyon", "gold": "Paris"},
    ]
    gate = competence_gate(preds, bar=0.60)
    assert gate["passed"] is False
    assert abs(gate["l2_rate"] - 0.5) < 1e-9


def test_competence_gate_passes_at_or_above_bar():
    preds = [
        {"task_id": str(i), "level": "L2",
         "prediction": "Paris" if i < 7 else "wrong", "gold": "Paris"}
        for i in range(10)
    ]
    gate = competence_gate(preds, bar=0.60)
    assert gate["passed"] is True
    assert abs(gate["l2_rate"] - 0.7) < 1e-9


def test_competence_gate_requires_l2_tasks():
    """An empty L2 slice must not silently 'pass' — no evidence is not a pass."""
    gate = competence_gate([{"task_id": "1", "level": "L1",
                             "prediction": "x", "gold": "x"}], bar=0.60)
    assert gate["passed"] is False
    assert gate["l2_n"] == 0
