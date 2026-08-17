"""R3: field contradiction-miss (NLI false-negative) metric (TDD).

The merge safety guarantee is "never silently resolves a DETECTED contradiction"; the
deployment-relevant risk is therefore the NLI MISS rate on real claim pairs. This metric,
over labelled pairs, is the pure function tested here;
running the real NLI over a naturalistic sample is the smoke-ladder step. Run:
  PYTHONPATH=src python3 -m pytest tests/test_contradiction_field.py -q
"""
from meld.eval.contradiction_field import contradiction_metrics, build_field_sample


def test_build_field_sample_makes_genuine_swap_contradictions_and_distinct_neutrals():
    from meld.agent.hotpot_loader import load_hotpot_fixture
    sample = build_field_sample(load_hotpot_fixture(n=4), max_pairs=20)
    contras = [(a, b) for lab, a, b in sample if lab == "contradicts"]
    neutrals = [(a, b) for lab, a, b in sample if lab == "neutral"]
    assert contras, "no contradiction pairs were built from the gold sentences"
    # a contradiction pair is a real sentence and its value-swapped variant -> they DIFFER
    for a, b in contras:
        assert a != b
    # a neutral pair is two DISTINCT real sentences
    for a, b in neutrals:
        assert a != b


def test_missed_contradiction_rate_and_recall():
    # 4 gold contradictions, NLI catches 3 (one missed -> silent-merge risk); 1 false alarm
    gold = ["contradicts", "contradicts", "contradicts", "contradicts", "neutral", "entails"]
    pred = ["contradicts", "contradicts", "neutral", "contradicts", "contradicts", "entails"]
    m = contradiction_metrics(gold, pred)
    assert m["n_contradictions"] == 4
    assert m["tp"] == 3 and m["fn"] == 1
    assert abs(m["contradiction_recall"] - 0.75) < 1e-9
    assert abs(m["missed_contradiction_rate"] - 0.25) < 1e-9
    # precision: 4 predicted-contradicts, 3 correct -> 0.75
    assert abs(m["contradiction_precision"] - 0.75) < 1e-9


def test_no_contradictions_is_well_defined():
    m = contradiction_metrics(["neutral", "entails"], ["neutral", "entails"])
    assert m["n_contradictions"] == 0
    assert m["missed_contradiction_rate"] == 0.0 and m["contradiction_recall"] == 0.0
