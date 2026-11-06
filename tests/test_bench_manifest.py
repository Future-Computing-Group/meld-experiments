"""Claim-pair benchmark manifest: splits disjoint by source item, deterministic, licensed."""
import json
from collections import Counter

import pytest

from meld.bench.manifest import SOURCES, build_manifest, manifest_hash, record_identity
from meld.bench.schema import make_pair

SIZES = {
    "S3_vitc_nonfactual": {"dev": 5, "cal": 5, "test": 6, "cap": 2},
    "S4_vitc_factual": {"dev": 5, "cal": 5, "test": 6, "cap": 2},
    "S5_lme_ku": {"dev": 2, "cal": 0, "test": None, "cap": 2},
}


def _vit(item, k, stratum, origin):
    lab = dict(label="MERGE", label_provenance="source", mergeable=True) if stratum == "S3_vitc_nonfactual" \
        else dict(label=None, label_provenance="to_annotate", mergeable=False)
    return make_pair(text_a=f"page {item} sentence {k} {stratum} version a", text_b=f"page {item} sentence {k} version b",
                     source="vitaminc", source_item_id=f"vitaminc:page:{item}", stratum=stratum,
                     origin_split=origin, **lab)


def _pool():
    ps = []
    for i in range(40):            # official dev pool; each page carries both strata, 3 pairs each
        ps += [_vit(f"d{i}", k, s, "dev") for k in range(3) for s in ("S3_vitc_nonfactual", "S4_vitc_factual")]
    for i in range(20):            # official test pool
        ps += [_vit(f"t{i}", k, s, "test") for k in range(3) for s in ("S3_vitc_nonfactual", "S4_vitc_factual")]
    for i in range(12):            # no official split
        ps.append(make_pair(text_a=f"old value {i} was here", text_b=f"new value {i} is here", source="longmemeval",
                            source_item_id=f"lme:q{i}", stratum="S5_lme_ku", label="CONFLICT",
                            subtype="SUPERSESSION", label_provenance="source", mergeable=False))
    return ps


def _by_split(m):
    out = {}
    for p in m["pairs"]:
        out.setdefault(p["split"], []).append(p)
    return out


def test_splits_disjoint_by_source_item_across_strata():
    m = build_manifest(_pool(), seed=0, sizes=SIZES)
    items = {s: {p["source_item_id"] for p in ps} for s, ps in _by_split(m).items()}
    assert items["dev"] and items["cal"] and items["test"]
    assert not (items["dev"] & items["cal"]) and not (items["dev"] & items["test"]) \
        and not (items["cal"] & items["test"])


def test_official_splits_respected_and_sizes_and_cap():
    m = build_manifest(_pool(), seed=0, sizes=SIZES)
    c = Counter((p["split"], p["stratum"]) for p in m["pairs"])
    for st in ("S3_vitc_nonfactual", "S4_vitc_factual"):
        assert (c[("dev", st)], c[("cal", st)], c[("test", st)]) == (5, 5, 6)
    assert c[("dev", "S5_lme_ku")] == 2 and c[("test", "S5_lme_ku")] == 10 and c[("cal", "S5_lme_ku")] == 0
    for p in m["pairs"]:
        if p["origin_split"] == "test":
            assert p["split"] == "test"
        if p["origin_split"] == "dev":
            assert p["split"] in ("dev", "cal")
    per_item = Counter((p["split"], p["stratum"], p["source_item_id"]) for p in m["pairs"])
    assert max(per_item.values()) <= 2


def test_manifest_is_deterministic_given_seed():
    a, b = build_manifest(_pool(), seed=7, sizes=SIZES), build_manifest(list(reversed(_pool())), seed=7, sizes=SIZES)
    assert a == b and manifest_hash(a) == manifest_hash(b)
    assert manifest_hash(build_manifest(_pool(), seed=8, sizes=SIZES)) != manifest_hash(a)


def test_test_pair_sharing_a_text_with_dev_or_cal_is_dropped():
    ps = _pool()
    m0 = build_manifest(ps, seed=0, sizes=SIZES)
    dev_text = next(p["text_a"] for p in m0["pairs"] if p["split"] == "dev" and p["source"] == "vitaminc")
    leak = make_pair(text_a=dev_text, text_b="a test-side rewrite", source="vitaminc",
                     source_item_id="vitaminc:page:leak", stratum="S4_vitc_factual", origin_split="test",
                     label=None, label_provenance="to_annotate", mergeable=False)
    m = build_manifest(ps + [leak], seed=0, sizes={**SIZES, "S4_vitc_factual": {**SIZES["S4_vitc_factual"], "test": None}})
    held = {p["text_a"] for p in m["pairs"] if p["split"] in ("dev", "cal")}
    assert dev_text in held
    assert leak.pair_id not in {p["pair_id"] for p in m["pairs"]}


def test_licences_recorded_and_unlicensed_source_refused():
    m = build_manifest(_pool(), seed=0, sizes=SIZES)
    assert set(m["sources"]) == {"vitaminc", "longmemeval"}
    assert all(m["sources"][s]["licence"] and m["sources"][s]["url"] for s in m["sources"])
    assert {"paws", "vitaminc", "longmemeval", "scifact", "meld_gold"} <= set(SOURCES)
    stray = make_pair(text_a="x is y here", text_b="x equals y here", source="mrpc", source_item_id="mrpc:1",
                      stratum="S1_paws_para", label="MERGE", label_provenance="source", mergeable=True)
    with pytest.raises(ValueError, match="licence"):
        build_manifest(_pool() + [stray], seed=0, sizes=SIZES)
    with pytest.raises(ValueError, match="licence"):
        build_manifest(_pool(), seed=0, sizes=SIZES, licences={**SOURCES, "vitaminc": {"licence": "", "url": "u"}})


def test_record_identity_merges_into_provenance(tmp_path):
    prov = tmp_path / "benchmark_provenance.json"
    prov.write_text(json.dumps({"longmemeval": {"name": "LongMemEval"}}))
    m = build_manifest(_pool(), seed=0, sizes=SIZES)
    entry = record_identity(m, str(prov), manifest_path="data/bench/manifest.json", files={"f": "abc"})
    data = json.loads(prov.read_text())
    assert data["longmemeval"] == {"name": "LongMemEval"}
    assert data["meld_claimpair"] == entry and entry["manifest_sha256"] == manifest_hash(m)
    assert entry["seed"] == 0 and entry["n_pairs"] == len(m["pairs"]) and entry["files"] == {"f": "abc"}


def test_item_partly_taken_into_dev_never_spills_into_test():
    # three pairs per item, cap two: the untaken third pair must not reach test
    ps = [make_pair(text_a=f"item {i} old fact {k}", text_b=f"item {i} new fact {k}", source="longmemeval",
                    source_item_id=f"lme:q{i}", stratum="S5_lme_ku", label="CONFLICT", subtype="SUPERSESSION",
                    label_provenance="source", mergeable=False) for i in range(6) for k in range(3)]
    m = build_manifest(ps, seed=0, sizes=SIZES)
    sp = _by_split(m)
    assert not ({p["source_item_id"] for p in sp["dev"]} & {p["source_item_id"] for p in sp["test"]})


def test_leftover_official_dev_items_never_reach_an_open_test_split():
    sizes = {**SIZES, "S3_vitc_nonfactual": {"dev": 5, "cal": 5, "test": None, "cap": 2},
             "S4_vitc_factual": {"dev": 5, "cal": 5, "test": None, "cap": 2}}
    m = build_manifest(_pool(), seed=0, sizes=sizes)
    test = [p for p in m["pairs"] if p["split"] == "test" and p["source"] == "vitaminc"]
    assert test and all(p["origin_split"] == "test" for p in test)


def _leak_run(text_a, text_b):
    ps = _pool()
    sizes = {**SIZES, "S4_vitc_factual": {**SIZES["S4_vitc_factual"], "test": None}}
    m0 = build_manifest(ps, seed=0, sizes=sizes)
    dev = next(p for p in m0["pairs"] if p["split"] == "dev" and p["source"] == "vitaminc")
    leak = make_pair(text_a=text_a(dev), text_b=text_b(dev), source="vitaminc", source_item_id="vitaminc:page:leak",
                     stratum="S4_vitc_factual", origin_split="test", label=None, label_provenance="to_annotate",
                     mergeable=False)
    m = build_manifest(ps + [leak], seed=0, sizes=sizes)
    return leak.pair_id in {p["pair_id"] for p in m["pairs"]}


def test_held_text_filter_checks_the_b_side():
    assert not _leak_run(lambda d: "an unrelated new sentence here", lambda d: d["text_b"])


def test_held_text_filter_is_case_and_whitespace_insensitive():
    assert not _leak_run(lambda d: "  " + d["text_a"].upper().replace(" ", "   "), lambda d: "a fresh other side")


def test_item_order_is_shuffled_by_seed():
    dev = lambda s: {p["source_item_id"] for p in build_manifest(_pool(), seed=s, sizes=SIZES)["pairs"]
                     if p["split"] == "dev"}
    assert dev(0) != dev(1)


def test_manifest_carries_design_notes():
    notes = build_manifest(_pool(), seed=0, sizes=SIZES)["notes"]
    for k in ("power", "lme_haystack", "badges", "release_licences", "s3_filter", "s3_scope", "s5_rule", "panel"):
        assert notes[k]
    assert SOURCES["vitaminc"]["release_licence"] == "CC BY-SA 3.0"
    from meld.bench.manifest import SIZES as DEFAULT
    assert DEFAULT["S5_lme_ku"]["dev"] == 0 and DEFAULT["S5_lme_ku"]["cal"] == 0     # S5 is never tuned on


# ------------------------------------------------------------------------- SICK (S8)

S8_SIZES = {st: {"dev": 0, "cal": 6, "test": 20, "cap": None}
            for st in ("S8_sick_para", "S8_sick_oneway", "S8_sick_contra")}
_S8_LAB = {"S8_sick_para": ("A_entails_B", "B_entails_A"), "S8_sick_oneway": ("A_entails_B", "B_neutral_A"),
           "S8_sick_contra": ("A_contradicts_B", "B_contradicts_A")}


def _sick_pool(tmp_path):
    """60 scenes; each scene's pairs reuse its sentences (as SICK does), one pair per class."""
    from meld.bench.loaders import load_sick

    head = ("pair_ID\tsentence_A\tsentence_B\tentailment_label\trelatedness_score\tentailment_AB\tentailment_BA\t"
            "sentence_A_original\tsentence_B_original\tsentence_A_dataset\tsentence_B_dataset\tSemEval_set")
    lines, k = [head], 0
    for s in range(60):
        base = f"Scene {s} shows a man playing"
        for st, other in (("S8_sick_para", f"Scene {s} shows a guy playing"),
                          ("S8_sick_oneway", f"Scene {s} shows a person"),
                          ("S8_sick_contra", f"Scene {s} shows nobody playing")):
            k += 1
            lines.append("\t".join([str(k), base, other, "X", "3", *_S8_LAB[st], base, other, "FLICKR", "FLICKR", "TEST"]))
    f = tmp_path / "SICK.txt"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_sick(str(f))


def _sentence_leaks(m):
    seen = {}
    for p in m["pairs"]:
        for t in (p["text_a"], p["text_b"]):
            seen.setdefault(" ".join(t.lower().split()), set()).add(p["split"])
    return {t for t, s in seen.items() if len(s) > 1}


def test_sick_splits_are_sentence_disjoint_and_a_leak_is_detected(tmp_path):
    ps = _sick_pool(tmp_path)
    m = build_manifest(ps, seed=0, sizes=S8_SIZES)
    c = Counter((p["split"], p["stratum"]) for p in m["pairs"])
    assert all(c[("cal", st)] == 6 and c[("test", st)] == 20 and c[("dev", st)] == 0 for st in S8_SIZES)
    assert not _sentence_leaks(m)
    # mutant: a split that moves one cal pair into test shares its sentences across splits and is caught
    cal = next(p for p in m["pairs"] if p["split"] == "cal")
    assert _sentence_leaks(dict(m, pairs=m["pairs"] + [dict(cal, split="test")]))


def test_sick_build_is_deterministic(tmp_path):
    a = build_manifest(_sick_pool(tmp_path), seed=3, sizes=S8_SIZES)
    b = build_manifest(list(reversed(_sick_pool(tmp_path))), seed=3, sizes=S8_SIZES)
    assert manifest_hash(a) == manifest_hash(b)


def test_s8_defaults_licence_and_notes():
    from meld.bench.manifest import NOTES, SIZES as DEFAULT

    for st in ("S8_sick_para", "S8_sick_oneway", "S8_sick_contra"):
        assert (DEFAULT[st]["dev"], DEFAULT[st]["cal"], DEFAULT[st]["test"]) == (100, 300, 1000)
    assert SOURCES["sick"]["licence"].startswith("CC BY-NC-SA 3.0") and SOURCES["sick"]["release_licence"] == "CC BY-NC-SA 3.0"
    assert "zenodo.org/records/2787612" in SOURCES["sick"]["url"]
    for word in ("hypernym", "captions, not factual claims", "S1", "one-way", "CC BY-NC-SA 3.0"):
        assert word in NOTES["s8_sick"], word


def test_adding_a_source_never_moves_pairs_of_another(tmp_path):
    sizes = {**SIZES, **S8_SIZES}
    split = lambda m: {(p["pair_id"], p["split"]) for p in m["pairs"] if p["source"] != "sick"}  # noqa: E731
    assert split(build_manifest(_pool(), seed=0, sizes=sizes)) == \
        split(build_manifest(_pool() + _sick_pool(tmp_path), seed=0, sizes=sizes))


def test_sick_dev_is_carved_from_test_and_leaves_cal_and_other_sources_unchanged(tmp_path):
    sizes = {**SIZES, **S8_SIZES}
    with_dev = {**SIZES, **{st: dict(v, dev=4) for st, v in S8_SIZES.items()}}
    a = build_manifest(_pool() + _sick_pool(tmp_path), seed=0, sizes=sizes)
    b = build_manifest(_pool() + _sick_pool(tmp_path), seed=0, sizes=with_dev)
    c = Counter((p["split"], p["stratum"]) for p in b["pairs"])
    assert all(c[("dev", st)] == 4 and c[("cal", st)] == 6 and c[("test", st)] == 20 for st in S8_SIZES)
    sick = lambda m, s: {p["pair_id"] for p in m["pairs"] if p["source"] == "sick" and p["split"] == s}  # noqa: E731
    assert sick(b, "cal") == sick(a, "cal")                       # cal unchanged
    assert sick(b, "dev") and not sick(b, "dev") & sick(a, "cal")  # dev comes out of the old test pool
    assert not _sentence_leaks(dict(b, pairs=[p for p in b["pairs"] if p["source"] == "sick"]))
    other = lambda m: {(p["pair_id"], p["split"]) for p in m["pairs"] if p["source"] != "sick"}  # noqa: E731
    assert other(a) == other(b)


def test_paws_cal_defaults_enlarged_dev_and_test_kept():
    from meld.bench.manifest import SIZES as DEFAULT

    for st in ("S1_paws_para", "S2_paws_nonpara"):
        assert (DEFAULT[st]["dev"], DEFAULT[st]["cal"], DEFAULT[st]["test"]) == (300, 1000, 1000)


def test_enlarging_a_dev_first_cal_keeps_dev_test_and_other_sources(tmp_path):
    # the PAWS case on the synthetic dev-first source: cal grows from the unused official dev pool only
    sizes = {**SIZES, **S8_SIZES}
    big = {**sizes, **{st: dict(sizes[st], cal=20) for st in ("S3_vitc_nonfactual", "S4_vitc_factual")}}
    a = build_manifest(_pool() + _sick_pool(tmp_path), seed=0, sizes=sizes)
    b = build_manifest(_pool() + _sick_pool(tmp_path), seed=0, sizes=big)
    c = Counter((p["split"], p["stratum"]) for p in b["pairs"])
    assert c[("cal", "S3_vitc_nonfactual")] == c[("cal", "S4_vitc_factual")] == 20
    ids = lambda m, s, src="vitaminc": {p["pair_id"] for p in m["pairs"] if p["split"] == s and p["source"] == src}  # noqa: E731
    assert ids(b, "dev") == ids(a, "dev") and ids(b, "test") == ids(a, "test")
    assert ids(a, "cal") < ids(b, "cal")
    other = lambda m: {(p["pair_id"], p["split"]) for p in m["pairs"] if p["source"] != "vitaminc"}  # noqa: E731
    assert other(a) == other(b)
