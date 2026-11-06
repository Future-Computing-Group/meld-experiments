"""Blinded annotation view, key file, panel queue, audit-UI repeats, and the audit drawer."""
import json
import random
import re
from collections import Counter

import pytest

from meld.bench.export import (annotation_export, audit_export, draw_audit, ht_mean, inclusion_probabilities, panel_export,
                               panel_queue, reversed_view, unswap, with_repeats, write_jsonl)
from meld.bench.schema import LABELS, PROVENANCE, STRATA, SUBTYPES, make_pair

VIEW_KEYS = {"pair_id", "text_a", "text_b", "badges_a", "badges_b"}


def _rows(n=60, strata=STRATA, splits=("dev", "cal", "test")):
    out = []
    for i in range(n):
        st = strata[i % len(strata)]
        lab = dict(label=None, label_provenance="to_annotate", mergeable=False) if (i // len(strata)) % 2 \
            else dict(label="CONFLICT", label_provenance="source", mergeable=False, subtype="SUPERSESSION")
        p = make_pair(text_a=f"the river was {i} km long", text_b=f"the river is {i + 1} km long",
                      source=("paws", "vitaminc", "longmemeval", "scifact")[i % 4], source_item_id=f"item-{i}",
                      stratum=st, badges_a={"time": "2023/05/23"} if i % 3 == 0 else {},
                      badges_b={"time": "2023/05/30"} if i % 3 == 0 else {}, **lab)
        out.append(dict(p.to_dict(), split=splits[(i // len(strata)) % len(splits)]))
    return out


def test_annotation_view_leaks_no_label_stratum_source_or_badge(tmp_path):
    view, key = annotation_export(_rows(), seed=1)
    f = tmp_path / "view.jsonl"
    write_jsonl(str(f), view)
    raw = f.read_text()
    lines = [json.loads(l) for l in raw.splitlines()]
    assert all(set(r) == VIEW_KEYS for r in lines)
    assert all(r["badges_a"] == {} and r["badges_b"] == {} for r in lines)     # badges stripped for every stratum
    for needle in (*LABELS, *SUBTYPES, *STRATA, *PROVENANCE, "paws", "vitaminc", "longmemeval", "scifact",
                   "item-", "stratum", "label", "split", "swapped", "mergeable", "source", "2023/"):
        assert needle not in raw, needle
    assert {r["pair_id"] for r in key} == {r["pair_id"] for r in view}
    assert all({"swapped", "label", "subtype", "stratum", "split", "source_item_id", "badges_a", "badges_b"} <= set(r)
               for r in key)


def test_side_order_randomised_recorded_and_invertible():
    rows = _rows(400)
    view, key = annotation_export(rows, seed=2)
    assert (view, key) == annotation_export(rows, seed=2)
    k = {r["pair_id"]: r for r in key}
    orig = {r["pair_id"]: r for r in rows}
    assert 150 < sum(r["swapped"] for r in key) < 250
    for v in view:
        o = orig[v["pair_id"]]
        assert unswap(v, k[v["pair_id"]]) == (o["text_a"], o["text_b"])
        assert (k[v["pair_id"]]["badges_a"], k[v["pair_id"]]["badges_b"]) == (o["badges_a"], o["badges_b"])
        if k[v["pair_id"]]["swapped"]:
            assert v["text_a"] == o["text_b"]


def test_stream_order_is_shuffled_and_seed_dependent():
    rows = _rows(200)
    ids = [r["pair_id"] for r in annotation_export(rows, seed=2)[0]]
    assert ids != sorted(ids) and ids != [r["pair_id"] for r in rows]
    assert ids != [r["pair_id"] for r in annotation_export(rows, seed=3)[0]]


def test_reversed_view_flips_every_pair_and_reshuffles():
    view, _ = annotation_export(_rows(200), seed=2)
    rev = reversed_view(view, seed=5)
    by = {r["pair_id"]: r for r in view}
    assert sorted(r["pair_id"] for r in rev) == sorted(by)
    assert all((r["text_a"], r["text_b"]) == (by[r["pair_id"]]["text_b"], by[r["pair_id"]]["text_a"]) for r in rev)
    assert [r["pair_id"] for r in rev] != [r["pair_id"] for r in view]


def test_panel_views_and_key_invert_to_manifest_order_in_both_orientations():
    rows = _rows(300)
    ab, ba, key = panel_export(rows, seed=4)
    k = {r["pair_id"]: r for r in key}
    orig = {r["pair_id"]: r for r in rows}
    assert {"S8_sick_para", "S8_sick_oneway", "S8_sick_contra"} <= {r["stratum"] for r in rows}
    assert {r["pair_id"] for r in ab} == {r["pair_id"] for r in ba} == set(k) == set(orig)
    for v in ab:
        o = orig[v["pair_id"]]
        assert unswap(v, k[v["pair_id"]]) == (o["text_a"], o["text_b"])
    for v in ba:
        o = orig[v["pair_id"]]
        assert unswap(v, dict(k[v["pair_id"]], swapped=k[v["pair_id"]]["swapped_ba"])) == (o["text_a"], o["text_b"])
    assert all(r["swapped_ba"] == (not r["swapped"]) for r in key)


def test_repeat_ids_are_keyed_not_derivable_from_visible_ids():
    view, _ = annotation_export(_rows(300), seed=2)
    a, ra = with_repeats(view, n=60, min_gap=50, seed=1, secret="s1")
    b, rb = with_repeats(view, n=60, min_gap=50, seed=1, secret="s2")
    assert [r["pair_id"] for r in ra] == [r["pair_id"] for r in rb]
    assert not ({r["repeat_id"] for r in ra} & {r["repeat_id"] for r in rb})
    import hashlib
    assert not {r["repeat_id"] for r in ra} & {hashlib.sha256(("repeat:" + r["pair_id"]).encode()).hexdigest()[:16]
                                               for r in ra}


def test_unannounced_repeats_are_far_apart_and_opaque():
    view, _ = annotation_export(_rows(300), seed=2)
    out, rep = with_repeats(view, n=60, min_gap=50, seed=1, secret="k")
    assert len(out) == 360 and len(rep) == 60
    pos = {r["pair_id"]: i for i, r in enumerate(out)}
    assert len(pos) == 360                                                     # repeat ids do not collide
    for r in rep:
        assert re.fullmatch(r"[0-9a-f]{16}", r["repeat_id"])
        assert pos[r["repeat_id"]] - pos[r["pair_id"]] >= 50
        a, b = out[pos[r["pair_id"]]], out[pos[r["repeat_id"]]]
        assert (a["text_a"], a["text_b"]) == (b["text_a"], b["text_b"])
    assert with_repeats(view, n=60, min_gap=50, seed=1, secret="k") == (out, rep)
    with pytest.raises(ValueError):
        with_repeats(view[:40], n=60, min_gap=50, seed=1, secret="k")


def test_panel_queue_scope():
    rows = _rows(700)
    q = panel_queue(rows, seed=0)
    c = Counter((r["stratum"], r["split"]) for r in q)
    full = Counter((r["stratum"], r["split"]) for r in rows)
    for st in ("S3_vitc_nonfactual", "S4_vitc_factual"):
        assert c[(st, "dev")] == 0 and c[(st, "cal")] == full[(st, "cal")] and c[(st, "test")] == full[(st, "test")]
    for sp in ("dev", "cal", "test"):
        assert c[("S6_scifact_conflict", sp)] == full[("S6_scifact_conflict", sp)]
        assert c[("S7_organic", sp)] == full[("S7_organic", sp)]
        assert c[("S0_gold", sp)] == 0
        for st in ("S1_paws_para", "S2_paws_nonpara", "S8_sick_para", "S8_sick_oneway", "S8_sick_contra"):
            assert full[(st, sp)] and c[(st, sp)] == round(0.1 * full[(st, sp)])
    assert len([r for r in q if r["stratum"] == "S5_lme_ku"]) == len([r for r in rows if r["stratum"] == "S5_lme_ku"])
    assert q == panel_queue(rows, seed=0) and q != panel_queue(rows, seed=1)



def test_panel_validation_sample_of_one_group_never_moves_another():
    rows = _rows(700)
    grown = rows + [dict(r, pair_id=f"extra{i}", split="cal") for i, r in enumerate(rows)
                    if r["stratum"] == "S1_paws_para"]
    key = lambda q: {r["pair_id"] for r in q if (r["stratum"], r["split"]) != ("S1_paws_para", "cal")}  # noqa: E731
    assert key(panel_queue(rows, seed=0)) == key(panel_queue(grown, seed=0))

# ------------------------------------------------------------------ audit drawer

def test_inclusion_probabilities_sum_to_n_and_respect_weights():
    w = [1.0] * 90 + [4.0] * 10 + [100.0]
    pi = inclusion_probabilities(w, 20)
    assert abs(sum(pi) - 20) < 1e-9 and all(0 < p <= 1 for p in pi)
    assert pi[-1] == 1.0                                   # capped
    assert abs(pi[90] / pi[0] - 4.0) < 1e-9
    assert inclusion_probabilities([1.0, 4.0], 5) == [1.0, 1.0]          # N <= n: take every item
    for bad in ([0.0, 1.0], [-1.0, 2.0]):
        with pytest.raises(ValueError, match="weight"):
            inclusion_probabilities(bad, 1)


def test_exact_inclusion_frequencies_small_design():
    # w = [1, 1, 4], n = 1: pi = [1/6, 1/6, 2/3]; a fixed systematic start would always pick the big item
    panel = [{"pair_id": "a", "group": "g", "labels": ["X"]}, {"pair_id": "b", "group": "g", "labels": ["X"]},
             {"pair_id": "c", "group": "g", "labels": ["X", "Y"]}]
    R = 6000
    f = Counter(r["pair_id"] for s in range(R) for r in draw_audit(panel, {"g": 1}, seed=s)["sample"])
    for pid, p in (("a", 1 / 6), ("b", 1 / 6), ("c", 2 / 3)):
        assert abs(f[pid] / R - p) < 0.02


def test_drawer_refusals_and_reports():
    rows = [{"pair_id": f"p{i}", "group": "g", "labels": ["X"]} for i in range(5)]
    with pytest.raises(ValueError, match="duplicate"):
        draw_audit(rows + [rows[0]], {"g": 1}, seed=0)
    with pytest.raises(ValueError, match="weight"):
        draw_audit(rows, {"g": 1}, seed=0, disagreement_weight=0)
    a = draw_audit(rows + [{"pair_id": "q", "group": "h", "labels": ["X"]}], {"g": 9, "z": 2}, seed=0)
    assert len(a["sample"]) == 5 and all(r["pi"] == 1.0 for r in a["sample"])
    assert a["unallocated"] == {"h": 1} and a["short"] == {"g": [5, 9], "z": [0, 2]}
    e = draw_audit([{"pair_id": "x", "group": "g", "labels": []}, {"pair_id": "y", "group": "g", "labels": ["X"]}],
                   {"g": 1}, seed=0)
    assert [r["pi"] for r in e["population"]] == [0.5, 0.5]           # empty label list is not a disagreement


def _panel(seed=0):
    rng = random.Random(seed)
    rows, y = [], {}
    for g, n, p_dis in (("S4", 200, 0.2), ("S2", 120, 0.1), ("S7", 100, 0.4)):
        for i in range(n):
            dis = rng.random() < p_dis
            labels = ["CONFLICT", "RELATE", "CONFLICT"] if dis else ["RELATE"] * 3
            pid = f"{g}-{i}"
            rows.append({"pair_id": pid, "group": g, "labels": labels})
            y[pid] = 1.0 if rng.random() < (0.6 if dis else 0.1) else 0.0     # panel error, correlated with disagreement
    return rows, y


def test_audit_sample_has_exact_allocation_and_logged_probabilities():
    rows, _ = _panel()
    alloc = {"S4": 80, "S2": 40, "S7": 80}
    a = draw_audit(rows, alloc, seed=0)
    assert a == draw_audit(rows, alloc, seed=0)
    assert Counter(r["group"] for r in a["sample"]) == Counter(alloc)
    pop = {r["pair_id"]: r for r in a["population"]}
    assert len(pop) == len(rows)
    for g, n in alloc.items():
        assert abs(sum(r["pi"] for r in pop.values() if r["group"] == g) - n) < 1e-9
    dis = next(r for r in rows if r["group"] == "S4" and len(set(r["labels"])) > 1)["pair_id"]
    una = next(r for r in rows if r["group"] == "S4" and len(set(r["labels"])) == 1)["pair_id"]
    assert pop[dis]["pi"] > pop[una]["pi"]                 # disagreements over-sampled
    assert all(abs(r["weight"] * r["pi"] - 1) < 1e-12 for r in a["sample"])


def test_inverse_probability_weights_recover_population_mean():
    rows, y = _panel()
    alloc = {"S4": 40, "S2": 20, "S7": 30}
    truth = sum(y.values()) / len(y)
    pis = {r["pair_id"]: r["pi"] for r in draw_audit(rows, alloc, seed=0)["population"]}
    freq, hts, naive = Counter(), [], []
    R = 3000
    for s in range(R):
        smp = draw_audit(rows, alloc, seed=s)["sample"]
        freq.update(r["pair_id"] for r in smp)
        hts.append(ht_mean(smp, y, len(rows)))
        naive.append(sum(y[r["pair_id"]] for r in smp) / len(smp))
    assert max(abs(freq[p] / R - pis[p]) for p in pis) < 0.04       # realised inclusion = logged pi
    ht, nv = sum(hts) / R, sum(naive) / R
    assert abs(ht - truth) < 0.005
    assert nv - truth > 0.03                                          # the unweighted mean is biased


def test_every_pair_of_items_can_be_drawn_together():
    # positive joint inclusion probabilities (needed for design-based variance): the
    # systematic order must be shuffled, else neighbours in a fixed order never co-occur
    panel = [{"pair_id": f"p{i}", "group": "g", "labels": ["X"]} for i in range(4)]
    seen = {frozenset(r["pair_id"] for r in draw_audit(panel, {"g": 2}, seed=s)["sample"]) for s in range(400)}
    assert len(seen) == 6


def test_audit_export_view_has_repeats_and_private_mapping():
    rows = _rows(800)
    panel = [{"pair_id": r["pair_id"], "group": r["stratum"][:2], "labels": ["X", "Y"] if i % 5 == 0 else ["X"]}
             for i, r in enumerate(rows)]
    view, mapping = audit_export(rows, panel, {"S2": 60, "S4": 70, "S7": 70}, seed=0, secret="sec")
    drawn = {r["pair_id"] for r in mapping["draw"]["sample"]}
    assert len(drawn) == 200 and len(view) == 260
    assert drawn <= {r["pair_id"] for r in view}
    assert mapping["secret"] == "sec" and len(mapping["repeats"]) == 60
    assert {k["pair_id"] for k in mapping["key"]} == drawn
    assert all(set(r) == VIEW_KEYS for r in view)
