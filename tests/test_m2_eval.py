"""M-2 classification metrics + (model-gated) end-to-end gold-set evaluation."""
from __future__ import annotations

import pytest

from meld.merge.m2_eval import classification_metrics


def test_classification_metrics_and_false_merge_rate():
    gold = ["MERGE", "MERGE", "RELATE", "CONFLICT", "NO_RELATION", "NO_RELATION"]
    # idx5: NO_RELATION gold -> predicted MERGE = a FALSE MERGE; idx1: MERGE -> RELATE miss
    pred = ["MERGE", "RELATE", "RELATE", "CONFLICT", "NO_RELATION", "MERGE"]
    m = classification_metrics(gold, pred)
    assert abs(m["per_class"]["MERGE"]["precision"] - 0.5) < 1e-9   # 1 of 2 pred-MERGE correct
    assert abs(m["per_class"]["MERGE"]["recall"] - 0.5) < 1e-9      # 1 of 2 gold-MERGE found
    # non-MERGE gold = 4 (RELATE,CONFLICT,NO_RELATION,NO_RELATION); 1 predicted MERGE -> 0.25
    assert abs(m["false_merge_rate"] - 0.25) < 1e-9
    assert m["n_false_merges"] == 1


def test_perfect_classification_zero_false_merge():
    labels = ["MERGE", "RELATE", "CONFLICT", "NO_RELATION"]
    m = classification_metrics(labels, labels)
    assert m["false_merge_rate"] == 0.0
    assert abs(m["macro_f1"] - 1.0) < 1e-9


