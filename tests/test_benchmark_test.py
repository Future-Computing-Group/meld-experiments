"""E2-B3 benchmark test runner (meld.eval.benchmark_test, scripts/run_benchmark_test.py): tiny
synthetic manifest, hashing encoder, scripted NLI."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest
from scipy.stats import beta, binom

from meld.config import save_epoch
from meld.eval import benchmark_test as B
from meld.eval import calibration as C
from meld.merge.conformal import quantile_threshold
from meld.protocol.objects import FIXED_ONE, INF16, canonical_bytes, object_id

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import run_benchmark_test  # noqa: E402

KEY = b"test-only-benchmark-key"
SEL = (round(0.5 * FIXED_ONE), round(0.5 * FIXED_ONE), 3)


def _row(i, stratum, split, label, prov, mergeable, tag=""):
    return {"pair_id": f"{stratum}-{split}-{i}", "text_a": f"{tag}w{i}a w{i}b w{i}c w{i}d echo",
            "text_b": f"{tag}w{i}a w{i}b w{i}c w{i}d foxtrot", "source": "syn", "source_item_id": f"syn:{split}:{i}",
            "stratum": stratum, "label": label, "label_provenance": prov, "mergeable": mergeable,
            "subtype": None, "split": split}


def _manifest(pending=0):
    rows = []
    for split, tag in (("cal", "CAL "), ("test", "")):
        rows += [_row(i, "S1_paws_para", split, "MERGE", "source", True, tag) for i in range(15)]
        rows += [_row(i, "S2_paws_nonpara", split, None, "to_annotate", False, tag) for i in range(15, 75)]
    rows += [_row(i, "S2_paws_nonpara", "test", None, "source_filtered", True) for i in range(75, 75 + pending)]
    return {"seed": 0, "pairs": rows}


class Scripted:
    """Deterministic NLI: p_e from a hash of the direction; CONTRA texts contradict."""

    def __init__(self):
        self.seen = set()

    def __call__(self, p, h):
        self.seen.update((p, h))
        if "CONTRA" in p or "CONTRA" in h:
            return (FIXED_ONE, 0, 0)
        pe = int.from_bytes(hashlib.sha256(f"{p}\x00{h}".encode()).digest()[:2], "big")
        return (0, pe, FIXED_ONE - pe)


def _epoch(manifest, provider=None):
    rows = C.cal_rows(manifest, "paws")
    return C.calibrate(rows, scope="paws", encoder="hashing", nli="lexical", provider=provider or Scripted(),
                       key=KEY, alpha=100_000, B_0=32, selector=SEL, hw="cpu", max_unresolved=None)


def _eval(manifest, ep, provider=None, **kw):
    kw.setdefault("max_unresolved", None)       # synthetic cross-item pairs carry no label
    return B.evaluate(C.split_rows(manifest, "paws", "test"), ep, key=KEY, encoder="hashing", nli="lexical",
                      provider=provider or Scripted(), hw="cpu", **kw)


# --- Vovk 2012 Prop. 2b order statistic --------------------------------------------------------

@pytest.mark.parametrize("n", [5, 19, 50, 200, 1000])
@pytest.mark.parametrize("alpha,delta", [(0.05, 0.05), (0.1, 0.1), (0.02, 0.01)])
def test_tc_index_is_the_smallest_order_statistic_meeting_prop_2b(n, alpha, delta):
    k = B.tc_index(n, alpha, delta)
    # P(FM > alpha) = P(F(s_(k)) < 1 - alpha) = P(Beta(k, n-k+1) < 1 - alpha) for a continuous score
    pac = lambda k: beta.cdf(1 - alpha, k, n - k + 1)  # noqa: E731
    if k is None:
        assert binom.cdf(0, n, alpha) > delta
        return
    assert pac(k) <= delta + 1e-12
    assert k == 1 or pac(k - 1) > delta
    assert binom.cdf(n - k, n, alpha) <= delta          # Vovk (7) with j = n - k
    # never below the marginal ceil((n+1)(1-alpha))-th order statistic
    s = list(range(n))
    marg = quantile_threshold(s, round(alpha * 1_000_000))
    assert marg == INF16 or k - 1 >= marg


def test_tc_index_disables_a_bin_too_small_for_the_guarantee():
    assert B.tc_index(19, 0.05, 0.05) is None          # 0.95^19 = 0.377 > 0.05
    assert B.tc_index(59, 0.05, 0.05) == 59            # 0.95^59 = 0.0485 <= 0.05


def test_clopper_pearson():
    assert B.cp(0, 0) == (None, None)
    lo, hi = B.cp(0, 300)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** (1 / 300))) < 1e-9
    lo, hi = B.cp(5, 100)
    assert abs(lo - beta.ppf(0.025, 5, 96)) < 1e-12 and abs(hi - beta.ppf(0.975, 6, 95)) < 1e-12


def test_tc_epoch_thresholds_are_at_least_marginal_and_from_the_cal_scores():
    res = _epoch(_manifest())
    ep = res["epoch"]
    tc = B.tc_epoch(ep, res["cal"], 0.1)
    for (g, n, _, th, _), (g2, n2, _, th2, _) in zip(ep.table, tc.table):
        assert (g, n) == (g2, n2) and th2 >= th
        s = sorted(k.s_q for k in res["cal"] if k.bin == g)
        k = B.tc_index(n, 0.1, 0.1)
        assert th2 == (INF16 if k is None else s[k - 1])


# --- the test-split evaluation -----------------------------------------------------------------

def test_evaluate_folds_only_test_texts_under_the_given_epoch_and_counts_by_hand():
    m = _manifest()
    ep = _epoch(m)["epoch"]
    p = Scripted()
    ev = _eval(m, ep, provider=p)
    assert p.seen and not any(t.startswith("CAL ") for t in p.seen)
    us = ev["units"]
    assert us
    nm = [u for u in us if u["mergeable"] is False]
    fm = ev["summary"]["marginal"]["false_merge"]
    assert fm["n"] == len(nm) and fm["k"] == sum(u["decision"] == "MERGE" for u in nm)
    assert sum(s["false_merge"]["n"] for s in ev["summary"]["stratum"].values()) == fm["n"]
    assert sum(s["false_merge"]["k"] for s in ev["summary"]["bin"].values()) == fm["k"]
    assert {u["stratum"] for u in us} <= {"S1_paws_para", "S2_paws_nonpara", "tested_same_item", "tested_cross_item"}


def test_evaluate_refuses_the_cal_split_and_a_foreign_epoch():
    m = _manifest()
    ep = _epoch(m)["epoch"]
    with pytest.raises(ValueError, match="test split only"):
        B.evaluate(C.split_rows(m, "paws", "cal"), ep, key=KEY, encoder="hashing", nli="lexical",
                   provider=Scripted(), hw="cpu")
    import dataclasses
    with pytest.raises(ValueError, match="did not read the given epoch"):
        _eval(m, dataclasses.replace(ep, author="someone-else"))
    with pytest.raises(ValueError, match="M_pol"):
        _eval(m, dataclasses.replace(ep, m_pol=b"\0" * len(ep.m_pol)))


def test_evaluate_refuses_unresolved_test_pairs_beyond_the_tolerance():
    m = _manifest(pending=4)
    ep = _epoch(m)["epoch"]
    ev = _eval(m, ep)
    with pytest.raises(ValueError, match="unresolved"):
        _eval(m, ep, max_unresolved=ev["unresolved"] - 1)
    assert ev["unresolved"] >= 1
    assert ev["summary"]["marginal"]["false_merge"]["n"] + ev["summary"]["marginal"]["merge_recall"]["n"] \
        + ev["unresolved"] == len(ev["units"])


def test_verdicts_contradiction_miss_and_relate_precision():
    us = [{"stratum": "s", "class": "manifest", "bin": 0, "decision": d, "mergeable": mg, "provenance": "manifest",
           "verdict": v} for d, mg, v in [("MERGE", False, "CONFLICT"), ("RELATE", False, "CONFLICT"),
                                         ("CONFLICT", False, "CONFLICT"), ("RELATE", False, "RELATE"),
                                         ("RELATE", True, "MERGE"), ("RELATE", False, None)]]
    m = B.metrics(us)
    assert m["contradiction_miss"]["k"] == 2 and m["contradiction_miss"]["n"] == 3
    assert m["contradiction_miss"]["to_merge"] == 1 and m["contradiction_miss"]["to_relate"] == 1
    assert m["relate_precision"]["k"] == 1 and m["relate_precision"]["n"] == 3
    assert m["false_merge"] == {**m["false_merge"], "k": 1, "n": 5}


def test_verdict_of_order_and_consistency(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps({"pair_id": "x", "verdict": "CONFLICT", "mergeable": False}) + "\n"
                 + json.dumps({"text_a": "A b", "text_b": "c D", "verdict": "NON_MERGE", "mergeable": False}) + "\n")
    v = B.load_verdicts(str(p))
    r = {"pair_id": "x", "label": None}
    assert B.verdict_of(r, "a", "b", "i", "i", source="s", mergeable=False, prov="manifest", verdicts=v) == "CONFLICT"
    assert B.verdict_of({"pair_id": "x", "label": "RELATE"}, "a", "b", "i", "i", source="s", mergeable=False,
                        prov="manifest", verdicts=v) == "RELATE"           # manifest beats annotators
    assert B.verdict_of(None, "a b", "C d", "i", "j", source="s", mergeable=False, prov=None, verdicts=v) is None
    assert B.verdict_of(None, "q", "r", "i", "j", source="s", mergeable=True, prov="source", verdicts=v) == "MERGE"
    assert B.verdict_of(r, "a", "b", "i", "i", source="s", mergeable=True, prov="x", verdicts=v) is None  # disagrees


# --- test streams partitioned to the calibration stream's length ------------------------------

def _long_manifest():
    """Test split three times the cal split: one test stream outgrows every bin the cal stream reached."""
    rows = []
    for split, n, tag in (("cal", 75, "CAL "), ("test", 225, "")):
        rows += [_row(i, "S1_paws_para", split, "MERGE", "source", True, tag) if i % 5 == 0 else
                 _row(i, "S2_paws_nonpara", split, "NO_MERGE", "source", False, tag) for i in range(n)]
    return {"seed": 0, "pairs": rows}


def _ntexts(rows):
    return len({t for r in rows for t in (r["text_a"], r["text_b"])})


def test_chunk_rows_partitions_whole_source_items_by_seed_and_keeps_the_last_partial_chunk():
    rows = C.split_rows(_long_manifest(), "paws", "test")
    ch = B.chunk_rows(rows, 200, seed=3)
    assert ch == B.chunk_rows(rows, 200, seed=3) and ch != B.chunk_rows(rows, 200, seed=4)
    assert [_ntexts(c) for c in ch] == [200, 200, 50]
    assert sorted(r["pair_id"] for c in ch for r in c) == sorted(r["pair_id"] for r in rows)
    items = [{r["source_item_id"] for r in c} for c in ch]
    assert sum(map(len, items)) == len(set().union(*items))


def test_a_test_stream_longer_than_cal_reaches_a_disabled_bin_and_the_chunked_one_does_not():
    m = _long_manifest()
    ep = _epoch(m)["epoch"]
    n_cal = _ntexts(C.cal_rows(m, "paws"))
    single, chunked = _eval(m, ep), _eval(m, ep, chunk=n_cal)
    cal_bins = {g for g, *_ in ep.table}
    assert max(u["bin"] for u in single["units"]) > max(cal_bins)
    assert single["coverage"]["test_disabled_share"] > 0
    assert {u["bin"] for u in chunked["units"]} <= cal_bins
    assert chunked["coverage"]["test_disabled_share"] < single["coverage"]["test_disabled_share"]
    assert chunked["coverage"]["cal_per_bin"] == {str(g): n for g, n, *_ in ep.table}
    assert [c["texts"] for c in chunked["chunks"]] == [n_cal, n_cal, n_cal]


def test_pooled_metrics_are_the_sum_over_chunks_each_folded_alone_and_deterministic():
    m = _long_manifest()
    ep = _epoch(m)["epoch"]
    n_cal = _ntexts(C.cal_rows(m, "paws"))
    ev = _eval(m, ep, chunk=n_cal)
    s = ev["summary"]
    assert set(s["chunk"]) == {"0", "1", "2"}
    for k in B.METRICS:
        for f in ("k", "n"):
            assert sum(c[k][f] for c in s["chunk"].values()) == s["marginal"][k][f]
    assert s["marginal"]["merge_recall"]["n"] and s["marginal"]["false_merge"]["n"]
    for i, part in enumerate(B.chunk_rows(C.split_rows(m, "paws", "test"), n_cal, seed=0)):
        alone = B.evaluate(part, ep, key=KEY, encoder="hashing", nli="lexical", provider=Scripted(), hw="cpu",
                           max_unresolved=None)
        assert alone["summary"]["marginal"] == s["chunk"][str(i)]
    assert _eval(m, ep, chunk=n_cal)["summary"] == s
    rows = B.spread(C.cal_rows(m, "paws"), ev, epoch=ep, draws=[0], delta=0.1, key=KEY, encoder="hashing",
                    nli="lexical", provider=Scripted(), hw="cpu")
    assert rows[0]["is_given_epoch"] and rows[0]["marginal"]["summary"] == s
    assert rows[0]["marginal"]["coverage"] == ev["coverage"]


# --- calibration-draw spread -------------------------------------------------------------------

def test_spread_redraws_reproduces_the_given_epoch_and_applies_both_variants():
    m = _manifest()
    ep = _epoch(m)["epoch"]
    ev = _eval(m, ep)
    rows = B.spread(C.cal_rows(m, "paws"), ev, epoch=ep, draws=[0, 1, 2], delta=0.1, key=KEY, encoder="hashing",
                    nli="lexical", provider=Scripted(), hw="cpu")
    assert [r["draw"] for r in rows] == [0, 1, 2]
    assert rows[0]["is_given_epoch"]
    assert rows[0]["marginal"]["summary"] == ev["summary"]
    for r in rows:
        for a, b in zip(r["marginal"]["table"], r["training_conditional"]["table"]):
            assert b[3] >= a[3]
        assert r["training_conditional"]["summary"]["marginal"]["false_merge"]["k"] <= \
            r["marginal"]["summary"]["marginal"]["false_merge"]["k"]
    s = B.spread_summary(rows, 0.1)
    assert s["marginal"]["marginal"]["draws"] == 3


# --- the runner --------------------------------------------------------------------------------

def test_runner_reads_the_key_file_and_stamps_every_row(tmp_path):
    m = _manifest()
    ep = _epoch(m)["epoch"]
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    (tmp_path / "key").write_bytes(KEY + b"\n")
    save_epoch(ep, KEY, str(tmp_path / "paws.epoch.json"))
    argv = ["--manifest", str(tmp_path / "manifest.json"), "--domain", "paws",
            "--epoch", f"paws={tmp_path / 'paws.epoch.json'}", "--encoder", "hashing", "--nli", "lexical",
            "--key-file", str(tmp_path / "key"), "--no-source-labels", "--draws", "2", "--delta", "0.1",
            "--selector", *map(str, SEL), "--max-unresolved", "100000", "--out-dir", str(tmp_path / "out")]
    rep = run_benchmark_test.main(argv, provider=Scripted())
    eid = object_id(canonical_bytes(ep)).hex()
    rows = rep["rows"]
    assert rows and all(r["git_sha"] and r["epoch_id"] and r["m_nli"] and r["m_pol"] for r in rows)
    assert {r["variant"] for r in rows} == {"marginal", "training_conditional"}
    assert {r["epoch_id"] for r in rows if r["draw"] is None and r["variant"] == "marginal"} == {eid}
    assert any(r["group"] == "stratum" for r in rows) and any(r["group"] == "bin" for r in rows)
    assert rep["domains"][0]["unresolved"] >= 0 and rep["domains"][0]["spread_summary"]
    dom = rep["domains"][0]
    assert dom["chunk_size"] == 150 and dom["chunks"] and "test_disabled_share" in dom["coverage"]
    assert dom["group_purity"]["nodes"] == dom["nodes"]
    assert any(r["group"] == "chunk" for r in rows)
    assert json.loads((tmp_path / "out" / "benchmark-test-report.json").read_text())["rows"]
    # a wrong key refuses before folding
    (tmp_path / "bad").write_bytes(b"other-key")
    with pytest.raises(ValueError, match="MAC"):
        run_benchmark_test.main(argv[:argv.index("--key-file") + 1] + [str(tmp_path / "bad")]
                                + argv[argv.index("--key-file") + 2:], provider=Scripted())


def test_the_label_queue_classifies_the_tested_pairs_of_the_chunked_test_streams(monkeypatch):
    from meld.eval import label_queue as Q
    m = _long_manifest()
    ep = _epoch(m)["epoch"]
    n_cal = _ntexts(C.cal_rows(m, "paws"))
    seen, units = [], B.units

    def spy(decisions, store, *args, chunk=0, **kw):
        text = {d.claim_key: d.text for _, d in store.nodes().values()}
        seen.extend((chunk, frozenset((text[a], text[b]))) for a, b in (d.tested for d in decisions if d.tested))
        return units(decisions, store, *args, chunk=chunk, **kw)

    monkeypatch.setattr(B, "units", spy)
    _eval(m, ep, chunk=n_cal)
    test = C.split_rows(m, "paws", "test")
    got = Q.classify(test, scope="paws", encoder="hashing", selector=SEL, chunk=n_cal)
    assert sorted(seen, key=str) == sorted(((r["chunk"], frozenset((r["text_a"], r["text_b"]))) for r in got), key=str)
    assert {r["chunk"] for r in got} == {0, 1, 2}
    whole = {frozenset((r["text_a"], r["text_b"])) for r in Q.classify(test, scope="paws", encoder="hashing", selector=SEL)}
    assert whole != {p for _, p in seen}
