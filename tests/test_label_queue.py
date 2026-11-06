"""Label queue over the tested pairs the fold produces on a split's stream: every tested pair is a
manifest pair, a source-labelled pair or a pair for the annotation panel. Hashing encoder, no NLI."""
from __future__ import annotations

import hashlib
import json
import os
import sys

from meld.bench.export import panel_export
from meld.eval import calibration as C
from meld.eval import label_queue as Q
from meld.protocol.objects import FIXED_ONE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import build_label_queue  # noqa: E402

SEL = (round(0.3 * FIXED_ONE), round(0.5 * FIXED_ONE), 3)


def _row(item, k, stratum, split, mergeable, prov, a, b):
    return {"pair_id": f"{split}-{item}-{k}", "text_a": a, "text_b": b, "source": "vitaminc",
            "source_item_id": f"vit:{split}:{item}", "stratum": stratum, "label": None,
            "label_provenance": prov, "mergeable": mergeable, "subtype": None, "split": split,
            "origin_split": "test" if split == "test" else "dev", "badges_a": {}, "badges_b": {}}


def _manifest():
    """Per item three related sentences, only two of them paired by the manifest; items share a
    topic in twos, so the selector also reaches across items. Cal and test use disjoint words."""
    rows = []
    for split in ("cal", "test"):
        for i in range(12):
            w, x = f"{split} topic{i // 2}", f"alpha{i} beta{i} gamma{i}"
            s = [f"{w} {x} delta", f"{w} {x} epsilon", f"{w} {x} delta epsilon"]
            rows.append(_row(i, 0, "S4_vitc_factual", split, False, "to_annotate", s[0], s[1]))
            rows.append(_row(i, 1, "S3_vitc_nonfactual", split, True, "source_filtered", s[1], s[2]))
    return {"seed": 0, "pairs": rows}


def _classify(split, source_label=None):
    return Q.classify(C.split_rows(_manifest(), "vitaminc", split), scope="vitaminc", encoder="hashing",
                      selector=SEL, source_label=source_label)


def _src(ta, tb):
    """A raw-source lookup that knows the unpaired pairs of items 0-3 only."""
    return False if any(all(f"alpha{i} " in x for x in (ta, tb)) for i in range(4)) else None


def test_every_class_appears_and_rows_are_consistent():
    t = _classify("cal", _src)
    got = {r["class"] for r in t}
    assert got == set(Q.CLASSES) - {"copy"}, got          # copy: recall workloads only
    rows = {r["pair_id"]: r for r in _manifest()["pairs"]}
    for r in t:
        if r["class"] in ("manifest", "manifest_pending"):
            m = rows[r["pair_id"]]
            assert {r["text_a"], r["text_b"]} == {m["text_a"], m["text_b"]}
            assert (r["class"] == "manifest_pending") == (m["label_provenance"] == "source_filtered")
        elif r["class"] == "source_labelled":
            assert r["mergeable"] is False and r["item_a"] == r["item_b"]
        else:
            assert r["mergeable"] is None and (r["item_a"] == r["item_b"]) == (r["class"] == "same_item")


def test_tested_pairs_are_the_calibration_folds_whatever_the_nli():
    rows = C.split_rows(_manifest(), "vitaminc", "cal")
    rows = [dict(r, label_provenance="source") for r in rows]          # calibrate wants final truth
    t = Q.classify(rows, scope="vitaminc", encoder="hashing", selector=SEL)
    want = {frozenset((r["text_a"], r["text_b"])) for r in t}
    for salt in ("x", "y"):
        def nli(p, h, salt=salt):
            pe = int.from_bytes(hashlib.sha256(f"{salt}{p}\x00{h}".encode()).digest()[:2], "big")
            return (0, pe, FIXED_ONE - pe)
        res = C.calibrate(rows, scope="vitaminc", encoder="hashing", nli="lexical", provider=nli,
                          key=b"another-key", alpha=100_000, B_0=8, selector=SEL, hw="cpu",
                          max_unresolved=None)
        text = {d.claim_key: d.text for _, d in res["store"].nodes().values()}
        assert want == {frozenset((text[a], text[b])) for a, b in
                        (d.tested for d in _decisions(res))}


def _decisions(res):
    from meld.protocol.fold import readjudicate
    again = readjudicate(res["store"], "vitaminc", m_nli=res["epoch"].m_nli, provider=None)
    return [d for d in again.decisions if d.tested is not None]


def test_deterministic_and_split_pure():
    cal, test = _classify("cal", _src), _classify("test", _src)
    assert cal == _classify("cal", _src)
    assert {r["split"] for r in cal} == {"cal"} and {r["split"] for r in test} == {"test"}
    assert not any(x.startswith("test") for r in cal for x in (r["text_a"], r["text_b"]))
    assert not any(x.startswith("cal") for r in test for x in (r["text_a"], r["text_b"]))


def test_queue_rows_take_only_needs_annotation_deduplicated_against_the_panel_queue():
    t = _classify("cal", _src) + _classify("test", _src)
    need = [r for r in t if r["class"] in Q.NEEDS]
    existing = [{"pair_id": "p0", "text_a": need[0]["text_b"].upper(), "text_b": need[0]["text_a"]}]
    q = Q.queue_rows(t + t, source="vitaminc", existing=existing)
    assert len(q) == len(need) - 1 and len({r["pair_id"] for r in q}) == len(q)
    assert all(r["stratum"] in ("tested_same_item", "tested_cross_item") and r["label"] is None
               and r["label_provenance"] == "to_annotate" for r in q)
    assert {r["split"] for r in q} == {"cal", "test"}
    view, view_ba, key = panel_export(q, seed=0)                         # the annotation export format
    assert len(view) == len(view_ba) == len(key) == len(q)
    assert set(view[0]) == {"pair_id", "text_a", "text_b", "badges_a", "badges_b"}
    assert q == Q.queue_rows(t + t, source="vitaminc", existing=existing)


def test_script_writes_blinded_views_key_and_counts(tmp_path):
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(_manifest()))
    pv = tmp_path / "panel_view_ab.jsonl"
    pv.write_text("")
    out = tmp_path / "out"
    rep = build_label_queue.main(["--manifest", str(mpath), "--domain", "vitaminc", "--encoder", "hashing",
                                  "--selector", *map(str, SEL), "--panel-view", str(pv),
                                  "--no-source-labels", "--out-dir", str(out)])
    lines = lambda f: [json.loads(x) for x in (out / f).read_text().splitlines()]  # noqa: E731
    view, key = lines("label_queue_view_ab.jsonl"), lines("label_queue_key.jsonl")
    assert len(view) == len(key) == len(lines("label_queue_view_ba.jsonl")) == rep["new_rows"] > 0
    assert all("stratum" not in v and "label" not in v for v in view)
    assert {"swapped", "swapped_ba", "split", "stratum"} <= set(key[0])
    assert set(rep["counts"]) == {"vitaminc/cal", "vitaminc/test"}
    assert len(lines("tested_pairs.jsonl")) == sum(sum(c.values()) for c in rep["counts"].values())



def test_script_chunked_folds_the_test_split_in_streams_of_the_cal_streams_length(tmp_path):
    m = _manifest()
    m["pairs"] = [r for r in m["pairs"] if r["split"] == "test" or int(r["source_item_id"].split(":")[2]) < 6]
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(m))
    pv = tmp_path / "panel_view_ab.jsonl"
    pv.write_text("")
    out = tmp_path / "out"
    rep = build_label_queue.main(["--manifest", str(mpath), "--domain", "vitaminc", "--encoder", "hashing",
                                  "--selector", *map(str, SEL), "--panel-view", str(pv), "--chunked",
                                  "--no-source-labels", "--out-dir", str(out)])
    assert rep["chunk_size"] == {"vitaminc": 18}
    tested = [json.loads(x) for x in (out / "tested_pairs.jsonl").read_text().splitlines()]
    want = Q.classify(C.split_rows(m, "vitaminc", "test"), scope="vitaminc", encoder="hashing", selector=SEL, chunk=18)
    assert [r for r in tested if r["split"] == "test"] == [dict(r, domain="vitaminc") for r in want]
    assert {r["chunk"] for r in want} == {0, 1} and all("chunk" not in r for r in tested if r["split"] == "cal")

def _ann(tmp_path, rows):
    p = tmp_path / "ann.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return Q.load_annotations(str(p))


def test_resolve_precedence_manifest_then_source_then_annotator(tmp_path):
    m = _row(0, 0, "S4_vitc_factual", "cal", False, "source", "Alpha one.", "Beta two.")
    ann = _ann(tmp_path, [{"text_a": "beta  TWO.", "text_b": "alpha one.", "mergeable": True,
                           "label_provenance": "panel"}])
    yes = lambda x, y: True  # noqa: E731
    args = ("Alpha one.", "Beta two.", "i0", "i1")
    assert Q.resolve(m, *args, source="vitaminc", source_label=yes, annotations=ann) == ("manifest", False, "manifest")
    assert Q.resolve(None, *args, source="vitaminc", source_label=lambda x, y: False,
                     annotations=ann) == ("source_labelled", False, "source")
    assert Q.resolve(None, *args, source="vitaminc", annotations=ann) == ("cross_item", True, "panel")
    assert Q.resolve(None, *args, source="vitaminc") == ("cross_item", None, None)
    assert Q.resolve(None, "Alpha one.", "Beta two.", "i0", "i0", source="vitaminc") == ("same_item", None, None)


def test_resolve_by_pair_id_for_pending_manifest_and_queue_pairs(tmp_path):
    pend = _row(0, 0, "S3_vitc_nonfactual", "cal", True, "source_filtered", "A x.", "B y.")
    qid = Q.queue_id("vitaminc", "i1", "i0", "B y.", "A x.")
    ann = _ann(tmp_path, [{"pair_id": pend["pair_id"], "mergeable": False, "label_provenance": "panel"},
                          {"pair_id": qid, "mergeable": False, "label_provenance": "adjudicated"}])
    assert Q.resolve(pend, "A x.", "B y.", "vit:cal:0", "vit:cal:0", source="vitaminc") == \
        ("manifest_pending", None, None)
    assert Q.resolve(pend, "A x.", "B y.", "vit:cal:0", "vit:cal:0", source="vitaminc", annotations=ann) == \
        ("manifest_pending", False, "panel")
    assert Q.resolve(None, "A x.", "B y.", "i0", "i1", source="vitaminc", annotations=ann) == \
        ("cross_item", False, "adjudicated")
    q = Q.queue_rows([{"class": "cross_item", "mergeable": None, "text_a": "A x.", "text_b": "B y.",
                       "item_a": "i0", "item_b": "i1", "split": "cal", "origin_split": "dev"}], source="vitaminc")
    assert q[0]["pair_id"] == qid


def test_load_annotations_refuses_malformed_or_conflicting_rows(tmp_path):
    import pytest
    for bad in ([{"text_a": "x", "mergeable": True, "label_provenance": "panel"}],
                [{"pair_id": "p", "mergeable": None, "label_provenance": "panel"}],
                [{"pair_id": "p", "mergeable": True}],
                [{"pair_id": "p", "mergeable": True, "label_provenance": "panel"},
                 {"pair_id": "p", "mergeable": False, "label_provenance": "panel"}]):
        with pytest.raises(ValueError):
            _ann(tmp_path, bad)


def test_classify_carries_annotations_and_the_queue_skips_annotated_pairs(tmp_path):
    t = _classify("cal", _src)
    need = [r for r in t if r["class"] in Q.NEEDS]
    ann = _ann(tmp_path, [{"text_a": need[0]["text_a"], "text_b": need[0]["text_b"], "mergeable": False,
                           "label_provenance": "panel"}])
    t2 = Q.classify(C.split_rows(_manifest(), "vitaminc", "cal"), scope="vitaminc", encoder="hashing",
                    selector=SEL, source_label=_src, annotations=ann)
    hit = [r for r in t2 if r["provenance"] == "panel"]
    assert len(hit) == 1 and hit[0]["mergeable"] is False and hit[0]["class"] == need[0]["class"]
    assert len(Q.queue_rows(t2, source="vitaminc")) == len(Q.queue_rows(t, source="vitaminc")) - 1
