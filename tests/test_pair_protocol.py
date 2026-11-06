"""MELD through the baselines' pair protocol (PH-7) and the predicate diagnostic (PH-12)."""
from __future__ import annotations

import json
import os
import sys

import pytest

from meld.agent.memory import DEFAULT_POL
from meld.config import save_epoch
from meld.eval import calibration as C
from meld.eval import pair_protocol as P
from meld.protocol.objects import FIXED_ONE, INF16

from test_agent_meld_memory import agent_epoch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

KEY = b"k" * 32
SAME = ("The cat sat on the red mat today", "The cat sat on the red mat today .")
NEAR = ("The cat sat on the red mat today", "The cat sat on the red mat yesterday")


def _pairs():
    return [{"pair_id": "p1", "stratum": "S1_paws_para", "label": "MERGE", "text_a": SAME[0],
             "text_b": SAME[1], "source_item_id": "paws:test:1"},
            {"pair_id": "p2", "stratum": "S2_paws_nonpara", "label": "CONFLICT", "text_a": NEAR[0],
             "text_b": NEAR[1], "source_item_id": "paws:test:2"},
            {"pair_id": "p3", "stratum": "S6_scifact_conflict", "label": "RELATE", "text_a": NEAR[0],
             "text_b": "Dogs bark at night in the city", "source_item_id": "scifact:9"}]


def _paws_epoch(theta):
    pol = C.policy("paws", DEFAULT_POL["bootstrap_selector"])
    return agent_epoch("hashing", "lexical", theta=theta, pol=pol, scope="paws"), pol


def test_submitted_predicate_is_the_v106_classify():
    s = P.submitted
    assert s(sigma=0.95, chi="entails", identical=False) == "MERGE"
    assert s(sigma=0.95, chi="neutral", identical=False) == "MERGE"           # non-contradiction suffices
    assert s(sigma=0.95, chi="contradicts", identical=False) == "CONFLICT"
    assert s(sigma=0.20, chi="contradicts", identical=False) == "CONFLICT"    # floor is sigma_lo
    assert s(sigma=0.10, chi="contradicts", identical=False) == "NO_RELATION"
    assert s(sigma=0.50, chi="entails", identical=False) == "RELATE"
    assert s(sigma=0.10, chi="neutral", identical=False) == "NO_RELATION"
    assert s(sigma=0.0, chi="contradicts", identical=True) == "CONFLICT"     # kappa=1 contradicting
    assert s(sigma=0.0, chi="neutral", identical=True) == "MERGE"


def test_derive_reproduces_admission_at_the_epoch_threshold_and_abstains_without_one():
    for theta in (0.0, 0.3, 0.6, 0.9):
        epoch, pol = _paws_epoch(theta)
        rows = P.meld_rows(_pairs()[:2], epochs={"paws": epoch}, encoder="hashing", nli="lexical", key=KEY)
        for r in rows:
            assert P.derive(r, round(theta * FIXED_ONE)) == r["decision"], (theta, r)
            assert P.derive(r, INF16) in ("RELATE", "CONFLICT", "INSERT", "MERGE")
            if r["event"] != "EXACT_COPY":
                assert P.derive(r, INF16) != "MERGE"


def test_meld_rows_never_overwrite_and_carry_signals_and_stamps():
    epoch, _ = _paws_epoch(0.0)
    rows = P.meld_rows(_pairs(), epochs={"paws": epoch}, encoder="hashing", nli="lexical", key=KEY)
    assert [r["pair_id"] for r in rows] == ["p1", "p2", "p3"]
    for r in rows:
        assert r["a_live"] and not r["silent_overwrite"] and not r["claim_lost"]
        assert {"sigma", "chi_ba", "s_q", "n_prior", "bin", "gates", "epoch_id", "git_sha", "m_pol"} <= set(r)
    assert rows[0]["epoch_id"] is not None and rows[0]["bin"] == 0 and rows[0]["n_prior"] == 1
    assert rows[2]["epoch_id"] is None and rows[2]["domain"] == "scifact"      # no epoch: abstains
    assert rows[2]["decision"] != "MERGE"


def test_meld_rows_refuse_an_epoch_of_another_policy():
    epoch = agent_epoch("hashing", "lexical", theta=0.5, scope="paws")         # DEFAULT_POL's M_pol
    with pytest.raises(ValueError):
        P.meld_rows(_pairs()[:1], epochs={"paws": epoch}, encoder="hashing", nli="lexical", key=KEY)


def test_score_rates_per_stratum_and_overall():
    rows = [{"stratum": "A", "label": "CONFLICT", "a_live": True, "d": "RELATE"},
            {"stratum": "A", "label": "RELATE", "a_live": True, "d": "MERGE"},
            {"stratum": "B", "label": "MERGE", "a_live": True, "d": "MERGE"}]
    s = P.score(rows, "d")
    assert s["overall"]["false_merge"]["k"] == 1 and s["overall"]["false_merge"]["n"] == 2
    assert s["stratum"]["A"]["missed_contradiction"]["k"] == 1
    assert s["stratum"]["B"]["merge_recall"]["rate"] == 1.0
    assert s["overall"]["silent_overwrite"]["k"] == 0 and s["overall"]["claim_lost"]["k"] == 0


def test_runner_and_analysis_end_to_end(tmp_path):
    import run_meld_pairs
    import analyze_meld_pairs
    epoch, _ = _paws_epoch(0.3)
    (tmp_path / "key").write_bytes(KEY)
    save_epoch(epoch, KEY, str(tmp_path / "paws.epoch.json"))
    pf = tmp_path / "pairs.jsonl"
    pf.write_text("".join(json.dumps(r) + "\n" for r in _pairs()))
    out = tmp_path / "out"
    run_meld_pairs.main(["--pairs", str(pf), "--epoch", f"paws={tmp_path / 'paws.epoch.json'}",
                         "--key-file", str(tmp_path / "key"), "--encoder", "hashing", "--nli", "lexical",
                         "--out-dir", str(out)])
    rows = [json.loads(x) for x in open(out / "rows.jsonl")]
    meta = json.load(open(out / "meta.json"))
    assert len(rows) == 3 and meta["epochs"]["paws"]["table"][0][3] == round(0.3 * FIXED_ONE)
    rep = analyze_meld_pairs.main(["--run", str(out), "--out", str(tmp_path / "an.json")])
    assert set(rep["arms"]) >= {"meld_literal", "meld_maxcal", "submitted"}
    assert rep["arms"]["meld_literal"]["overall"]["false_merge"]["n"] == 2
    assert rep["git_sha"] and rep["rows_sha256"]
