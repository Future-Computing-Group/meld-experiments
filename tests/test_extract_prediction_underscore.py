"""Regression: extract_prediction must honor the underscore FINAL_ANSWER form.

Caught on the M-1 memory-delta local real-model run: qwen2.5:7b emits
'FINAL_ANSWER: British' (underscore), but extract_prediction only checked for the
spaced 'FINAL ANSWER', so the whole 'FINAL_ANSWER: British' string leaked through
tighten and scored F1=0.67 instead of 1.0 (across ALL conditions equally, so the
deltas held, but the absolute numbers were wrong).
"""
from meld.agent.run_hotpot_gate import extract_prediction


def test_extract_prediction_handles_underscore_template():
    assert extract_prediction("FINAL_ANSWER: British") == "British"
    assert extract_prediction("FINAL_ANSWER: London") == "London"


def test_extract_prediction_still_handles_spaced_template():
    assert extract_prediction("FINAL ANSWER: Paris") == "Paris"


def test_extract_prediction_tightens_bare_text():
    # no template at all -> the safe tighten path
    assert extract_prediction("The answer is: dog.") == "dog"
