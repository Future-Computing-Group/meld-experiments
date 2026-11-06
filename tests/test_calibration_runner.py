"""T10: the calibration runner that folds a domain's cal split and writes signed SEED epochs
(design r3 §4.1-4.3, §4.6, §7.4). Tiny synthetic manifest, hashing encoder, scripted NLI."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

from meld.config import read_epoch
from meld.eval import calibration as C
from meld.merge.conformal import n_min, verify_epoch
from meld.protocol import fold as fold_mod
from meld.protocol.objects import FIXED_ONE, INF16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import run_calibration  # noqa: E402

KEY = b"test-only-calibration-key"


def _row(i, stratum, split, label, prov, mergeable, a=None, b=None):
    a = a or f"w{i}a w{i}b w{i}c w{i}d echo"
    b = b or f"w{i}a w{i}b w{i}c w{i}d foxtrot"
    return {"pair_id": f"{stratum}-{split}-{i}", "text_a": a, "text_b": b, "source": "syn",
            "source_item_id": f"syn:{i}", "stratum": stratum, "label": label,
            "label_provenance": prov, "mergeable": mergeable, "subtype": None, "split": split}


def _manifest():
    rows = [_row(i, "S1_paws_para", "cal", "MERGE", "source", True) for i in range(10)]
    rows += [_row(i, "S2_paws_nonpara", "cal", None, "to_annotate", False) for i in range(10, 60)]
    # dev and test rows carry poison texts that must never reach the fold
    rows += [_row(i, "S2_paws_nonpara", s, None, "to_annotate", False, a=f"POISON {s} {i} x y",
                  b=f"POISON {s} {i} x z") for s in ("dev", "test") for i in range(60, 70)]
    rows += [_row(i, "S3_vitc_nonfactual", "cal", "MERGE", "source_filtered", True)
             for i in range(70, 75)]
    rows += [_row(i, "S4_vitc_factual", "cal", None, "to_annotate", False) for i in range(75, 80)]
    return {"seed": 0, "pairs": rows}


class Scripted:
    """Deterministic NLI: p_e from a hash of the direction, never a contradiction."""

    def __init__(self):
        self.seen = set()

    def __call__(self, p, h):
        self.seen.update((p, h))
        pe = int.from_bytes(hashlib.sha256(f"{p}\x00{h}".encode()).digest()[:2], "big")
        return (0, pe, FIXED_ONE - pe)


def _cal(manifest=None, provider=None, **kw):
    rows = C.cal_rows(manifest or _manifest(), "paws")
    args = dict(scope="paws", encoder="hashing", nli="lexical", provider=provider or Scripted(),
                key=KEY, alpha=100_000, B_0=48, selector=(round(0.5 * FIXED_ONE),
                                                         round(0.5 * FIXED_ONE), 3), hw="cpu",
                max_unresolved=None)
    return C.calibrate(rows, **{**args, **kw})


def test_cal_rows_read_only_the_cal_split_of_the_domain():
    rows = C.cal_rows(_manifest(), "paws")
    assert rows and {r["split"] for r in rows} == {"cal"}
    assert {r["stratum"] for r in rows} == {"S1_paws_para", "S2_paws_nonpara"}


def test_dev_and_test_texts_never_reach_the_fold():
    p = Scripted()
    res = _cal(provider=p)
    assert p.seen and not any("POISON" in t for t in p.seen)
    assert not any("POISON" in d.text for _, d in res["store"].nodes().values())


def test_calibrate_refuses_a_row_outside_the_cal_split():
    rows = C.cal_rows(_manifest(), "paws")
    bad = [r for r in _manifest()["pairs"] if r["split"] == "test"][:1]
    with pytest.raises(ValueError, match="cal split"):
        C.calibrate(rows + bad, scope="paws", encoder="hashing", nli="lexical",
                    provider=Scripted(), key=KEY, alpha=100_000, B_0=48,
                    selector=(32768, 32768, 3), hw="cpu")


def test_refuses_a_domain_whose_cal_truth_is_not_final():
    with pytest.raises(ValueError, match="S3_vitc_nonfactual.*source_filtered"):
        C.cal_rows(_manifest(), "vitaminc")
    m = _manifest()
    for r in m["pairs"]:
        if r["stratum"] == "S2_paws_nonpara":
            r["mergeable"] = None                       # to_annotate with no source mergeability
    with pytest.raises(ValueError, match="S2_paws_nonpara.*to_annotate"):
        C.cal_rows(m, "paws")


def test_cal_records_are_tested_pairs_of_the_fold_selector_with_manifest_truth():
    res = _cal()
    drawn = {tuple(sorted((a, b))): n for a, b, n in
             fold_mod.tested_pairs(res["store"], "paws", sigma_lo=32768, m=3)}
    assert res["cal"]
    for k in res["cal"]:
        assert drawn[(k.c_x, k.c_y)] == k.n_prior
        assert k.source == "SEED" and k.obs == 1 and k.hw == "cpu" and k.label == "NON_MERGEABLE"
    # truth: a manifest pair by its mergeable flag (MERGE -> never Cal); without annotations no
    # other tested pair has a label, so every Cal record is a manifest non-mergeable pair
    rows = {r["pair_id"]: r for r in _manifest()["pairs"]}
    why = [res["truth"][k.c_x, k.c_y] for k in res["cal"]]
    assert all(w["provenance"] == "manifest" and rows[w["pair_id"]]["mergeable"] is False for w in why)
    assert res["labels"]["manifest"]["mergeable"] > 0          # S1 pairs were tested, none is Cal
    assert len({k.item for k in res["cal"]}) == len(res["cal"])


def _tested(**kw):
    from meld.eval import label_queue as Q
    return Q.classify(C.cal_rows(_manifest(), "paws"), scope="paws", encoder="hashing",
                      selector=(32768, 32768, 3), **kw)


def _annotate(tmp_path, tested, mergeable):
    from meld.eval import label_queue as Q
    p = tmp_path / "ann.jsonl"
    p.write_text("".join(json.dumps({"text_a": t["text_a"], "text_b": t["text_b"], "mergeable": mergeable,
                                     "label_provenance": "panel"}) + "\n" for t in tested))
    return Q.load_annotations(str(p))


def test_cross_item_pairs_are_never_labelled_by_construction(tmp_path):
    cross = [t for t in _tested() if t["class"] == "cross_item"]
    assert cross
    res = _cal()
    assert res["labels"]["cross_item"] == {"unresolved": len(cross)}
    assert res["stats"]["unresolved"] == sum(v.get("unresolved", 0) for v in res["labels"].values())
    assert all(w["class"] == "manifest" for w in res["truth"].values())
    pos = _cal(annotations=_annotate(tmp_path, cross, True), max_unresolved=0)
    assert pos["labels"]["cross_item"] == {"mergeable": len(cross)}
    assert all(w["class"] == "manifest" for w in pos["truth"].values())
    neg = _cal(annotations=_annotate(tmp_path, cross, False), max_unresolved=0)
    assert neg["labels"]["cross_item"] == {"non_mergeable": len(cross)}
    assert any(w == {"class": "cross_item", "provenance": "panel", "pair_id": None} for w in neg["truth"].values())


def test_source_labels_enter_cal_and_rank_below_the_manifest():
    tested = _tested()
    cross = {frozenset((t["text_a"], t["text_b"])) for t in tested if t["class"] == "cross_item"}
    man = {frozenset((t["text_a"], t["text_b"])) for t in tested if t["class"] == "manifest"}
    src = lambda x, y: False if frozenset((x, y)) in cross else True if frozenset((x, y)) in man else None  # noqa: E731
    res = _cal(source_label=src, max_unresolved=0)
    assert res["labels"]["source_labelled"] == {"non_mergeable": len(cross)}
    assert res["labels"]["manifest"] == _cal()["labels"]["manifest"]      # the manifest wins


def test_refuses_an_epoch_while_tested_pairs_are_unresolved(tmp_path):
    n = _cal()["stats"]["unresolved"]
    assert n > 0
    with pytest.raises(ValueError, match=f"{n} tested pairs.*unresolved"):
        _cal(max_unresolved=0)
    with pytest.raises(ValueError, match="unresolved"):
        _cal(max_unresolved=n - 1)
    assert _cal(max_unresolved=n)["epoch"]
    cross = [t for t in _tested() if t["class"] == "cross_item"]
    a = _cal(annotations=_annotate(tmp_path, cross, False), max_unresolved=0)
    b = _cal(annotations=_annotate(tmp_path, cross, False), max_unresolved=0)
    assert a["cal"] == b["cal"] and a["epoch"] == b["epoch"]


def test_epoch_is_seed_mondrian_off_and_verifies_by_recomputation():
    res = _cal()
    ep = res["epoch"]
    assert ep.policy == "SEED" and not ep.mondrian_c and ep.bins == (48, "doubling")
    assert verify_epoch(ep, res["cal"], res["sig"])


@pytest.mark.parametrize("B_0", [32, 48])        # 32 puts n_g = 10 = n_min + 1 in bin 1
def test_short_bins_are_reported_disabled(B_0):
    res = _cal(B_0=B_0)
    rows = {g: n for g, n, *_ in res["epoch"].table}
    nm = n_min(100_000)
    assert res["disabled_bins"] == sorted(g for g in res["populated_bins"] if rows.get(g, 0) < nm)
    assert res["disabled_bins"] and set(res["populated_bins"]) - set(res["disabled_bins"])
    for g, n, _, th, _ in res["epoch"].table:
        assert (th == INF16) == (g in res["disabled_bins"])


def _run(tmp_path, *extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(_manifest()))
    kpath = tmp_path / "key"
    kpath.write_bytes(KEY)
    out = tmp_path / "out"
    rep = run_calibration.main(["--manifest", str(mpath), "--domain", "paws", "--alpha", "0.10",
                                "--encoder", "hashing", "--nli", "lexical", "--B0", "48",
                                "--key-file", str(kpath), "--out-dir", str(out), "--no-source-labels",
                                *(extra or ("--allow-unresolved",))],
                               provider=Scripted())
    return rep, out


def test_runner_writes_a_signed_verified_epoch_and_a_stamped_report(tmp_path):
    rep, out = _run(tmp_path)
    e = rep["epochs"][0]
    for f in ("git_sha", "m_nli", "m_pol", "epoch_id", "seq", "bins", "disabled_bins", "nli_model",
              "encoder", "manifest_sha256", "alpha", "B_0", "verified"):
        assert f in e, f
    assert e["verified"] is True and e["seq"] == 1
    assert all({"g", "n_g", "theta_q"} <= set(b) for b in e["bins"])
    ep = read_epoch(os.path.join(out, e["epoch_file"]), KEY)
    assert ep.m_nli.hex() == e["m_nli"] and ep.m_pol.hex() == e["m_pol"]
    assert json.loads((out / "calibration-report.json").read_text()) == rep


def test_runner_is_deterministic(tmp_path):
    rep1, out1 = _run(tmp_path / "a")
    rep2, out2 = _run(tmp_path / "b")
    f = rep1["epochs"][0]["epoch_file"]
    assert (out1 / f).read_bytes() == (out2 / f).read_bytes()


def test_runner_refuses_without_a_key_and_never_writes_an_unverified_epoch(tmp_path, monkeypatch):
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(_manifest()))
    monkeypatch.delenv("SMP_GROUP_KEY", raising=False)
    with pytest.raises(SystemExit, match="key"):
        run_calibration.main(["--manifest", str(mpath), "--domain", "paws", "--alpha", "0.10",
                              "--encoder", "hashing", "--nli", "lexical",
                              "--out-dir", str(tmp_path / "o")], provider=Scripted())
    monkeypatch.setattr(C, "verify_epoch", lambda *a: False)
    with pytest.raises(ValueError, match="verify"):
        _run(tmp_path)
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_runner_refuses_unresolved_pairs_unless_allowed_and_then_says_so(tmp_path, capsys):
    with pytest.raises(ValueError, match="unresolved"):
        _run(tmp_path / "strict", "--seed", "0")
    assert not (tmp_path / "strict" / "out").exists()
    rep, out = _run(tmp_path / "dev")
    e = rep["epochs"][0]
    assert e["development"] is True and e["unresolved"] == e["stats"]["unresolved"] > 0
    assert e["labels"]["cross_item"]["unresolved"] > 0
    assert e["epoch_file"].endswith("-dev.epoch.json") and (out / e["epoch_file"]).exists()
    assert f"{e['unresolved']} tested pairs UNRESOLVED" in capsys.readouterr().err


# ---- labels from the annotation panel: cal rows resolved by the store, AMBIGUOUS policy

def test_cal_rows_accept_rows_the_annotations_label_and_refuse_the_rest():
    m = _manifest()
    s3 = [r for r in m["pairs"] if r["stratum"] == "S3_vitc_nonfactual" and r["split"] == "cal"]
    ann = {r["pair_id"]: (False, "adjudicated") for r in s3}
    rows = C.cal_rows(m, "vitaminc", annotations=ann)
    assert {r["stratum"] for r in rows} == {"S3_vitc_nonfactual", "S4_vitc_factual"}
    del ann[s3[0]["pair_id"]]
    with pytest.raises(ValueError, match=r"S3_vitc_nonfactual', 'source_filtered'\), 1\)"):
        C.cal_rows(m, "vitaminc", annotations=ann)
    ann[s3[0]["pair_id"]] = (None, "adjudicated_ambiguous")      # decided AMBIGUOUS, excluded
    assert C.cal_rows(m, "vitaminc", annotations=ann) == rows


def _ambiguous(tmp_path, cross):
    from meld.eval import label_queue as Q
    p = tmp_path / "labels.jsonl"
    ids = [q["pair_id"] for q in Q.queue_rows(cross, source="paws")]
    p.write_text("".join(json.dumps({"pair_id": i, "verdict": "AMBIGUOUS", "path": "adjudicated"}) + "\n"
                         for i in ids) + json.dumps({"pair_id": "x", "verdict": "MERGE"}) + "\n")
    return str(p), len(ids)


def test_ambiguous_items_are_non_mergeable_by_default_or_excluded(tmp_path):
    from meld.eval import label_queue as Q
    cross = [t for t in _tested() if t["class"] == "cross_item"]
    path, n = _ambiguous(tmp_path, cross)
    assert Q.load_ambiguous(path) == {q["pair_id"] for q in Q.queue_rows(cross, source="paws")}
    neg = _cal(annotations=Q.with_ambiguous({}, Q.load_ambiguous(path)), max_unresolved=0)
    assert neg["labels"]["cross_item"] == {"non_mergeable": len(cross), "ambiguous": len(cross)}
    assert {"class": "cross_item", "provenance": Q.AMBIGUOUS, "pair_id": None} in neg["truth"].values()
    exc = _cal(annotations=Q.with_ambiguous({}, Q.load_ambiguous(path), exclude=True), max_unresolved=0)
    assert exc["labels"]["cross_item"] == {"excluded": len(cross), "ambiguous": len(cross)}
    assert exc["stats"]["unresolved"] == 0
    assert all(w["provenance"] == "manifest" for w in exc["truth"].values())
    with pytest.raises(ValueError, match="both"):
        Q.with_ambiguous({next(iter(Q.load_ambiguous(path))): (True, "panel")}, Q.load_ambiguous(path))


def test_runner_reports_the_ambiguous_policy_and_counts(tmp_path):
    cross = [t for t in _tested() if t["class"] == "cross_item"]
    path, _ = _ambiguous(tmp_path, cross)
    ann = tmp_path / "annotations.jsonl"
    ann.write_text("")
    rep, _ = _run(tmp_path / "d", "--annotations", str(ann))       # labels.jsonl beside it
    e = rep["epochs"][0]
    assert e["unresolved"] == 0 and e["development"] is False
    assert e["ambiguous"] == {"policy": "non-mergeable", "provenance": "adjudicated_ambiguous",
                              "labels_file": path, "by_class": {"cross_item": len(cross)}}
    rep, _ = _run(tmp_path / "x", "--annotations", str(ann), "--ambiguous", "exclude")
    e = rep["epochs"][0]
    assert e["ambiguous"]["policy"] == "exclude" and e["labels"]["cross_item"]["excluded"] == len(cross)
    assert e["epoch_file"] == "paws-seed-a0.10-ambx.epoch.json"
