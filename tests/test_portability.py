"""Evaluation-only runners: portability grid and transfer (E2-P1), calibration burden (E2-P2),
recalibration under shift (E2-B4). Synthetic manifest, hashing encoder, scripted providers."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

from meld.eval import calibration as C
from meld.eval import portability as P
from meld.merge.conformal import n_min, threshold, verify_epoch
from meld.protocol.fold import readjudicate
from meld.protocol.objects import FIXED_ONE, INF16, canonical_bytes, contradicts, signed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import run_portability  # noqa: E402
import run_calibration_burden  # noqa: E402
import run_recalibration_shift  # noqa: E402

KEY = b"test-only-portability-key"
SEL = (FIXED_ONE // 2, FIXED_ONE // 2, 3)
ALPHA, B_0 = 100_000, 16


def _row(i, stratum, split, mergeable, tag):
    return {"pair_id": f"{tag}-{split}-{i}", "text_a": f"{tag}{i}a {tag}{i}b {tag}{i}c {tag}{i}d echo",
            "text_b": f"{tag}{i}a {tag}{i}b {tag}{i}c {tag}{i}d {'fox' if mergeable else 'wolf'}", "source": tag,
            "source_item_id": f"{tag}:{split}:{i}", "stratum": stratum, "label": "MERGE" if mergeable else "NO",
            "label_provenance": "source", "mergeable": mergeable, "subtype": None, "split": split}


def _manifest(n_cal=60, n_test=40):
    rows = []
    for split, n in (("cal", n_cal), ("test", n_test)):
        rows += [_row(i, "S1_paws_para" if i % 3 == 0 else "S2_paws_nonpara", split, i % 3 == 0, "pw")
                 for i in range(n)]
        rows += [_row(i, "S8_sick_para" if i % 2 == 0 else "S8_sick_contra", split, i % 2 == 0, "sk")
                 for i in range(n)]
    return {"seed": 0, "pairs": rows}


class Scripted:
    """p_e from a hash of the direction; mergeable-looking texts ('echo' vs 'fox' only) score higher."""

    def __init__(self):
        self.calls = 0

    def __call__(self, p, h):
        self.calls += 1
        u = int.from_bytes(hashlib.sha256(f"{p}\x00{h}".encode()).digest()[:2], "big") / 65536
        same = p.split()[0][:-1] == h.split()[0][:-1] and "wolf" not in p + h
        pe = 0.5 + 0.5 * u if same else 0.5 * u
        return (0, round(pe * FIXED_ONE), FIXED_ONE - round(pe * FIXED_ONE))


def _cal(domain="paws", provider=None, **kw):
    rows = C.cal_rows(_manifest(), domain)
    return C.calibrate(rows, scope=domain, encoder="hashing", nli="lexical", provider=provider or Scripted(),
                       key=KEY, alpha=ALPHA, B_0=B_0, selector=SEL, hw="cpu", max_unresolved=None, **kw)


def _test(domain="paws", scope=None, provider=None, **kw):
    rows = C.split_rows(_manifest(), domain, "test")
    return P.test_stream(rows, scope=scope or domain, encoder="hashing", nli="lexical", selector=SEL,
                         key=KEY, provider=provider or Scripted(), hw="cpu", **kw)


def _eval(base, ep, domain="paws", **kw):
    return P.evaluate(base, ep, source=domain, **kw)


# ---- the evaluator

def test_evaluate_equals_a_fresh_fold_with_the_epoch_and_reads_no_new_signal():
    res = _cal()
    base = _test()
    out = _eval(base, res["epoch"])
    assert out["nli_calls"] == 0 and out["misses"] == 0
    # reference: a fresh stream holding the epoch, folded with the provider from scratch
    store, pol, item, pair = C.stream(C.split_rows(_manifest(), "paws", "test"), scope="paws",
                                      encoder="hashing", selector=SEL, key=KEY)
    store.apply(*signed(res["epoch"], KEY))
    ref = readjudicate(store, "paws", m_nli=res["epoch"].m_nli, provider=Scripted(), issuer="x", hw="cpu")
    merged = {frozenset(d.tested) for d in ref.decisions if d.decision == "MERGE"}
    want = {"mergeable": [0, 0], "non_mergeable": [0, 0]}
    for d in ref.decisions:
        r = pair.get(frozenset(d.tested)) if d.tested else None
        if r is not None:
            want["mergeable" if r["mergeable"] else "non_mergeable"][frozenset(d.tested) in merged] += 1
    m = out["by_class"]["manifest"]
    assert (m["mergeable"], m.get("merged_mergeable", 0)) == (sum(want["mergeable"]), want["mergeable"][1])
    assert (m["non_mergeable"], m.get("false_merges", 0)) == (sum(want["non_mergeable"]), want["non_mergeable"][1])
    assert out["recall"] == pytest.approx(out["merged_mergeable"] / out["mergeable"])
    assert out["fm"] == pytest.approx(out["false_merges"] / out["non_mergeable"])
    assert out["merged_mergeable"] > 0          # the scripted signals separate the classes


def test_without_an_epoch_nothing_merges_and_unresolved_pairs_are_counted():
    out = _eval(_test(), None)
    assert out["merged_mergeable"] == out["false_merges"] == 0
    assert out["unresolved"] == sum(c.get("unresolved", 0) for c in out["by_class"].values()) > 0
    assert set(out["by_stratum"]) >= {"S1_paws_para", "S2_paws_nonpara"}


def test_items_restrict_the_evaluated_pairs():
    base = _test()
    ep = _cal()["epoch"]
    full = _eval(base, ep)
    half = set(sorted(set(base["item"].values()))[::2])
    a, b = _eval(base, ep, items=half), _eval(base, ep, items=set(base["item"].values()) - half)
    for k in ("mergeable", "non_mergeable", "merged_mergeable", "false_merges", "unresolved"):
        assert a[k] + b[k] == full[k]


def test_transfer_applies_the_thresholds_of_one_domain_to_another_stream():
    ep = _cal("paws")["epoch"]
    base = _test("sick")
    moved = P.transfer(ep, base["pol"], "sick")
    assert moved.table == ep.table and moved.scope == "sick"
    out = _eval(base, moved, domain="sick")
    assert out["epoch_scope"] == "sick" and out["nli_calls"] == 0
    assert out["thetas_used"] and all(t == threshold(ep, g) for g, t in out["thetas_used"].items())


# ---- test streams partitioned to the calibration stream's length

def _texts(rows):
    return {t for r in rows for t in (r["text_a"], r["text_b"])}


def _counts(ev):
    return {k: ev[k] for k in ("mergeable", "non_mergeable", "merged_mergeable", "false_merges", "unresolved")}


def test_a_test_stream_longer_than_cal_reaches_disabled_bins_and_chunking_keeps_it_in_cal_bins():
    m = _manifest(30, 90)                        # test stream three times the cal stream
    ep = C.calibrate(C.cal_rows(m, "paws"), scope="paws", encoder="hashing", nli="lexical", provider=Scripted(),
                     key=KEY, alpha=ALPHA, B_0=B_0, selector=SEL, hw="cpu", max_unresolved=None)["epoch"]
    rows = C.split_rows(m, "paws", "test")
    n_cal = len(_texts(C.cal_rows(m, "paws")))
    kw = dict(scope="paws", encoder="hashing", nli="lexical", selector=SEL, key=KEY, hw="cpu")
    single = _eval(P.test_stream(rows, provider=Scripted(), **kw), ep)
    chunked = _eval(P.test_stream(rows, provider=Scripted(), chunk=n_cal, **kw), ep)
    cal_bins = {g for g, *_ in ep.table}
    assert max(map(int, single["by_bin"])) > max(cal_bins)
    assert single["coverage"]["test_disabled_share"] > chunked["coverage"]["test_disabled_share"]
    assert {int(g) for g in chunked["by_bin"]} <= cal_bins
    assert chunked["coverage"]["cal_per_bin"] == {str(g): n for g, n, *_ in ep.table}
    # pooled counts are the sum over chunks, and each chunk is the chunk folded alone
    parts = P.chunk_rows(rows, n_cal, seed=0)
    assert len(parts) == len(chunked["by_chunk"]) == 3
    for k in ("mergeable", "non_mergeable", "merged_mergeable", "false_merges"):
        assert sum(c[k] for c in chunked["by_chunk"]) == chunked[k]
        assert sum(b[k] for b in chunked["by_bin"].values()) == chunked[k]
    for part, c in zip(parts, chunked["by_chunk"]):
        alone = _eval(P.test_stream(part, provider=Scripted(), **kw), ep)
        assert _counts(alone) == {k: c[k] for k in _counts(alone)} and c["nodes"] == len(_texts(part))
    # deterministic by seed
    again = _eval(P.test_stream(rows, provider=Scripted(), chunk=n_cal, **kw), ep)
    assert {k: v for k, v in again.items() if k != "eval_seconds"} == \
        {k: v for k, v in chunked.items() if k != "eval_seconds"}


def test_chunk_rows_partitions_whole_source_items_by_seed_and_keeps_the_last_partial_chunk():
    rows = C.split_rows(_manifest(30, 90), "paws", "test")
    ch = P.chunk_rows(rows, 70, seed=1)
    assert ch == P.chunk_rows(rows, 70, seed=1) and ch != P.chunk_rows(rows, 70, seed=2)
    assert [len(_texts(c)) for c in ch] == [70, 70, 40]
    assert sorted(r["pair_id"] for c in ch for r in c) == sorted(r["pair_id"] for r in rows)
    items = [{r["source_item_id"] for r in c} for c in ch]
    assert sum(map(len, items)) == len(set().union(*items))


def test_runners_partition_the_test_stream_to_the_cal_stream_and_report_coverage(tmp_path):
    files = _files(tmp_path, _manifest(30, 90))
    rep = run_portability.main([*files, "--encoders", "hashing", "--nlis", "lexical", "--domains", "paws", "sick",
                                "--max-unresolved", "100000", "--out-dir", str(tmp_path / "p1")],
                               providers=_providers)
    for r in rep["rows"]:
        assert r["status"] == "ok" and r["chunk_size"] == 60 and len(r["by_chunk"]) == 3
        assert {int(g) for g in r["by_bin"]} <= {int(g) for g in r["coverage"]["cal_per_bin"]}
        assert 0 <= r["coverage"]["test_disabled_share"] <= 1
    rep = run_recalibration_shift.main([*files, "--encoder", "hashing", "--nli", "lexical", "--shift", "paws:sick",
                                        "--degraded", "paws", "--max-unresolved", "100000",
                                        "--out-dir", str(tmp_path / "b4")], providers=_providers)
    assert all(len(r["by_chunk"]) == 3 and "coverage" in r for r in rep["rows"] if r["status"] == "ok")
    assert any(r["phase"] == "after" and r["status"] == "ok" for r in rep["rows"])
    rep = run_calibration_burden.main([*files, "--encoders", "hashing", "--nli", "lexical", "--domains", "paws",
                                       "--m", "5", "--reps", "1", "--max-unresolved", "100000",
                                       "--out-dir", str(tmp_path / "p2")], providers=_providers)
    assert all(len(r["by_chunk"]) == 3 and "coverage" in r for r in rep["rows"])


# ---- epochs over a subset of Cal

def test_reepoch_of_the_full_draw_reproduces_the_epoch():
    res = _cal()
    again = P.reepoch(res["epoch"], res["cal"], res["sig"])
    assert canonical_bytes(again) == canonical_bytes(res["epoch"])


def test_subsample_caps_each_bin_and_is_seeded():
    res = _cal()
    sub = P.subsample(res["cal"], 4, seed=1)
    per = {}
    for k in res["cal"]:
        per[k.bin] = per.get(k.bin, 0) + 1
    got = {}
    for k in sub:
        got[k.bin] = got.get(k.bin, 0) + 1
    assert got == {g: min(n, 4) for g, n in per.items()}
    assert sub == P.subsample(res["cal"], 4, seed=1) and set(sub) <= set(res["cal"])
    ep = P.reepoch(res["epoch"], sub, res["sig"], seq=2)
    assert ep.seq == 2 and verify_epoch(ep, sub, res["sig"])
    assert all(th == INF16 for _, n, _, th, _ in ep.table if n < n_min(ALPHA))


# ---- degraded signals (test time only)

def test_corrupt_provider_is_deterministic_and_changes_the_label_at_the_rate():
    clean = lambda p, h: (100, 60000, 5436)            # noqa: E731
    assert P.corrupt(clean, 0.0)("a", "b") == clean("a", "b")
    full = P.corrupt(clean, 1.0, seed=3)
    pairs = [(f"p{i}", f"h{i}") for i in range(400)]
    assert all(max(range(3), key=full(*x).__getitem__) != 1 for x in pairs)
    half = P.corrupt(clean, 0.5, seed=3)
    hit = sum(half(*x) != clean(*x) for x in pairs) / len(pairs)
    assert 0.4 < hit < 0.6 and [half(*x) for x in pairs] == [P.corrupt(clean, 0.5, seed=3)(*x) for x in pairs]
    assert sorted(half("p1", "h1")) == sorted(clean("p1", "h1"))     # a relabelling, not new mass


def test_embedding_noise_keeps_keys_and_order_and_moves_q():
    clean, noisy = _test(), _test(noise=0.3, noise_seed=1)
    assert set(clean["item"]) == set(noisy["item"])
    qa = {d.claim_key: d.q for _, d in clean["store"].nodes().values()}
    qb = {d.claim_key: d.q for _, d in noisy["store"].nodes().values()}
    assert qa.keys() == qb.keys() and sum(qa[k] != qb[k] for k in qa) > len(qa) // 2
    assert [d.claim_key for _, d in clean["store"].nodes().values()] == \
        [d.claim_key for _, d in noisy["store"].nodes().values()]
    assert noisy["noise"]["sigma"] == 0.3 and noisy["noise"]["cos_shift_sd"] > 0
    again = _test(noise=0.3, noise_seed=1)
    assert {d.claim_key: d.q for _, d in again["store"].nodes().values()} == qb


# ---- in-domain ablation provider

def test_min_provider_is_min_of_entailment_and_mean_paraphrase_and_keeps_the_gate():
    nli = {("a", "b"): (0.1, 0.9, 0.0), ("b", "a"): (0.1, 0.8, 0.1), ("c", "d"): (0.7, 0.2, 0.1),
           ("e", "f"): (0.45, 0.5, 0.05)}
    para = {("a", "b"): 0.6, ("b", "a"): 0.4, ("c", "d"): 0.9, ("d", "c"): 0.9, ("e", "f"): 0.1, ("f", "e"): 0.1}
    f = P.min_provider(lambda p, h: nli[p, h], lambda p, h: para[p, h])
    ab, ba = f("a", "b"), f("b", "a")
    assert ab[1] == round(0.5 * FIXED_ONE) and ba[1] == round(0.5 * FIXED_ONE)
    assert contradicts(f("c", "d")) and not contradicts(f("e", "f"))   # gate as the NLI decides it


def test_pawsval_rows_exclude_the_selection_sample_and_group_items_by_shared_text(tmp_path):
    import pandas as pd
    s = [f"sentence {i} here ." for i in range(40)]
    df = pd.DataFrame({"id": range(700), "sentence1": [s[i % 40] for i in range(700)],
                       "sentence2": [f"other {i} ." for i in range(700)], "label": [i % 2 for i in range(700)]})
    p = tmp_path / "validation.parquet"
    df.to_parquet(p)
    rows = P.pawsval_rows(str(p), n_per_class=40, seed=0)
    assert len(rows) == 80 and {r["split"] for r in rows} == {"cal"}
    used = P.selection_sample(df)
    assert not {r["pair_id"] for r in rows} & {f"pawsval:{int(df.id.iloc[i])}" for i in used}
    item = {}
    for r in rows:
        for t in (r["text_a"], r["text_b"]):
            assert item.setdefault(t, r["source_item_id"]) == r["source_item_id"]
    assert {r["stratum"] for r in rows} == {"S1_paws_para", "S2_paws_nonpara"}


# ---- runners

def _files(tmp_path, manifest=None):
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps(manifest or _manifest()))
    k = tmp_path / "key"
    k.write_bytes(KEY)
    return ["--manifest", str(m), "--key-file", str(k), "--alpha", "0.10", "--B0", str(B_0),
            "--selector", *map(str, SEL), "--no-source-labels"]


def _providers(name):
    return Scripted()


def test_portability_runner_writes_stamped_cells_transfer_and_refuses_unlabelled_cells(tmp_path):
    out = tmp_path / "p1"
    rep = run_portability.main([*_files(tmp_path), "--encoders", "hashing", "--nlis", "lexical",
                                "--domains", "paws", "sick", "--max-unresolved", "0",
                                "--out-dir", str(out)], providers=_providers)
    cells = [r for r in rep["rows"] if r["kind"] == "cell"]
    assert {(r["cal_domain"], r["test_domain"]) for r in cells} == {("paws", "paws"), ("sick", "sick")}
    for r in rep["rows"]:
        for f in ("git_sha", "encoder", "nli", "alpha", "B_0", "manifest_sha256", "status"):
            assert f in r, f
    # the synthetic stream leaves cross-item pairs unlabelled: every cell is refused and listed
    assert all(r["status"] == "refused" and "fm" not in r for r in rep["rows"])
    assert sorted(rep["needs_labels"]) == sorted({f"hashing|lexical|{d}" for d in ("paws", "sick")})
    lax = run_portability.main([*_files(tmp_path), "--encoders", "hashing", "--nlis", "lexical",
                                "--domains", "paws", "sick", "--max-unresolved", "100000",
                                "--out-dir", str(tmp_path / "p1b")], providers=_providers)
    tr = [r for r in lax["rows"] if r["kind"] == "transfer"]
    assert {(r["cal_domain"], r["test_domain"]) for r in tr} == {("paws", "sick"), ("sick", "paws")}
    assert all(r["status"] == "ok" and 0 <= r["fm"] <= 1 for r in lax["rows"])
    assert (tmp_path / "p1b" / "rows.jsonl").read_text().count("\n") == len(lax["rows"])


def test_burden_runner_reports_fm_recall_per_m_with_the_n_min_line(tmp_path):
    rep = run_calibration_burden.main([*_files(tmp_path), "--encoders", "hashing", "--nli", "lexical",
                                       "--domains", "paws", "--m", "5", "10", "--reps", "2",
                                       "--max-unresolved", "100000", "--out-dir", str(tmp_path / "p2")],
                                      providers=_providers)
    rows = rep["rows"]
    assert {(r["m"], r["rep"]) for r in rows} == {(5, 0), (5, 1), (10, 0), (10, 1), ("all", 0)}
    assert all(r["n_min"] == n_min(ALPHA) and "fm" in r and "recall" in r for r in rows)
    assert all(max(r["cal_per_bin"].values()) <= r["m"] for r in rows if r["m"] != "all")


def test_shift_runner_reports_before_and_after_on_held_out_items(tmp_path):
    rep = run_recalibration_shift.main([*_files(tmp_path), "--encoder", "hashing", "--nli", "lexical",
                                        "--shift", "paws:sick", "--degraded", "paws", "--audit-frac", "0.5",
                                        "--max-unresolved", "100000", "--out-dir", str(tmp_path / "b4")],
                                       providers=_providers)
    rows = rep["rows"]
    arms = {(r["arm"], r["phase"]) for r in rows}
    assert arms == {(a, p) for a in ("domain_shift", "degraded") for p in ("before", "after")}
    for arm in ("domain_shift", "degraded"):
        b, a = [next(r for r in rows if r["arm"] == arm and r["phase"] == p) for p in ("before", "after")]
        assert b["eval_items"] == a["eval_items"] and b["mergeable"] == a["mergeable"]
        assert a["audit_items"] and not set(a["audit_item_ids"]) & set(a["eval_item_ids"])
    d = next(r for r in rows if r["arm"] == "degraded")
    assert d["noise"]["sigma"] == 0.3 and d["nli_corruption"] == 0.5


def test_replay_answers_every_read_direction_with_the_stream_s_own_signal():
    base = _test()
    f, ref = P.replay(base), Scripted()
    text = {d.claim_key: d.text for _, d in base["store"].nodes().values()}
    reads = [(text[p], text[h]) for r in base["store"].objects.iter("Sig", "paws")
             for (p, h), pay in (((r.c_x, r.c_y), r.pay_xy), ((r.c_y, r.c_x), r.pay_yx)) if pay is not None]
    assert reads and all(f(p, h) == ref(p, h) for p, h in reads)
