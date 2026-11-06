"""Annotation harness: blinded deterministic batches, strict output validation with one retry,
backends behind one call, aggregation over orientations and annotators, adjudication routing,
label-store output that calibration resolves, agreement and accuracy statistics, audit draw.
Fake backends only; nothing here calls a model."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter

import pytest

from meld.annotate import aggregate as A
from meld.annotate import batch as B
from meld.bench.export import panel_export, panel_queue
from meld.eval import calibration as C
from meld.eval import label_queue as Q
from meld.protocol.objects import FIXED_ONE

SEL = (round(0.3 * FIXED_ONE), round(0.5 * FIXED_ONE), 3)
SECRET = "test-secret"


def _row(item, k, stratum, split, mergeable, prov, a, b, label=None):
    return {"pair_id": f"{split}-{item}-{k}", "text_a": a, "text_b": b, "source": "vitaminc",
            "source_item_id": f"vit:{split}:{item}", "stratum": stratum, "label": label,
            "label_provenance": prov, "mergeable": mergeable, "subtype": None, "split": split,
            "origin_split": "test" if split == "test" else "dev", "badges_a": {}, "badges_b": {}}


def _manifest():
    rows = []
    for split in ("cal", "test"):
        for i in range(12):
            w, x = f"{split} topic{i // 2}", f"alpha{i} beta{i} gamma{i}"
            s = [f"{w} {x} delta", f"{w} {x} epsilon", f"{w} {x} delta epsilon"]
            rows.append(_row(i, 0, "S4_vitc_factual", split, False, "to_annotate", s[0], s[1]))
            rows.append(_row(i, 1, "S3_vitc_nonfactual", split, True, "source_filtered", s[1], s[2], "MERGE"))
    return {"seed": 0, "pairs": rows}


def _queues():
    """Both queues in the export format: bench (panel_queue) and fold (label_queue rows)."""
    m = _manifest()
    bench = panel_export(panel_queue(m["pairs"], seed=0), seed=0)
    t = []
    for sp in ("cal", "test"):
        t += Q.classify(C.split_rows(m, "vitaminc", sp), scope="vitaminc", encoder="hashing", selector=SEL)
    fold = panel_export(Q.queue_rows(t, source="vitaminc", existing=bench[0]), seed=0)
    return m, {"bench": bench, "fold": fold}


def _diff(a, b):
    return len(set(a.split()) ^ set(b.split()))


def rule_ok(it):
    """Order-invariant: MERGE when the word sets differ by at most one word, else RELATE."""
    if _diff(it["text_a"], it["text_b"]) <= 1:
        return {"verdict": "MERGE", "subtype": None, "overlap_flag": False}
    return {"verdict": "RELATE", "subtype": "ELABORATION", "overlap_flag": False}


def _views(queues):
    return {q: (v[0], v[1]) for q, v in queues.items()}


def _keys(queues):
    return {q: v[2] for q, v in queues.items()}


# ------------------------------------------------------------------ batches and blinding

def test_batches_are_deterministic_blinded_and_cover_both_orientations():
    _, qs = _queues()
    b1, items1 = B.make_batches(_views(qs), secret=SECRET, seed=0, size=7, n_repeats=3, min_gap=4)
    b2, items2 = B.make_batches(_views(qs), secret=SECRET, seed=0, size=7, n_repeats=3, min_gap=4)
    assert b1 == b2 and items1 == items2
    pids = {r["pair_id"] for v in qs.values() for r in v[0]}
    blob = json.dumps(b1)
    for b in b1:
        assert set(b) == {"batch_id", "verdict_order", "items"} and len(b["items"]) <= 7
        assert sorted(b["verdict_order"]) == sorted(B.VERDICTS)
        for it in b["items"]:
            assert set(it) == {"item_id", "text_a", "text_b"}
    assert not any(p in blob for p in pids)                     # no pair id reaches an annotator
    assert not re.search(r"stratum|source|label|split|swapped|S\d_|vitaminc", blob)
    real = [m for m in items1.values() if not m["repeat"]]
    for o in ("ab", "ba"):                                       # every pair once per orientation
        got = sorted(m["pair_id"] for m in real if m["orientation"] == o)
        assert got == sorted(pids)
        assert sum(m["repeat"] and m["orientation"] == o for m in items1.values()) == 3
    ids = [it["item_id"] for b in b1 for it in b["items"]]
    assert len(ids) == len(set(ids)) == len(items1)
    ab_texts = {it["item_id"]: (it["text_a"], it["text_b"]) for b in b1 for it in b["items"]}
    for iid, m in items1.items():                                 # ba shows the reversed order of ab
        if m["orientation"] == "ba" and not m["repeat"]:
            iab = next(k for k, n in items1.items() if n == dict(m, orientation="ab"))
            assert ab_texts[iid] == ab_texts[iab][::-1]


def test_other_secret_gives_other_ids_same_layout():
    _, qs = _queues()
    b1, _ = B.make_batches(_views(qs), secret="s1", seed=0, size=7, n_repeats=0)
    b2, _ = B.make_batches(_views(qs), secret="s2", seed=0, size=7, n_repeats=0)
    strip = lambda bs: [[(i["text_a"], i["text_b"]) for i in b["items"]] for b in bs]  # noqa: E731
    assert strip(b1) == strip(b2) and b1 != b2


def test_pilot_draws_dev_bench_and_cal_fold_items_without_repeats():
    qs = {"bench": ([{"pair_id": f"b{i}", "text_a": f"x{i}", "text_b": f"y{i}"} for i in range(10)], [],
                    [{"pair_id": f"b{i}", "split": "dev" if i < 6 else "test"} for i in range(10)]),
          "fold": ([{"pair_id": f"f{i}", "text_a": f"u{i}", "text_b": f"v{i}"} for i in range(10)], [],
                   [{"pair_id": f"f{i}", "split": "cal" if i % 2 else "test"} for i in range(10)])}
    b, items = B.pilot_batch({q: (v[0], v[2]) for q, v in qs.items()}, n={"bench": 4, "fold": 3}, secret=SECRET,
                             seed=0)
    assert len(b["items"]) == 7 and not any(m["repeat"] for m in items.values())
    got = sorted(m["pair_id"] for m in items.values())
    assert all(int(p[1:]) < 6 for p in got if p[0] == "b") and all(int(p[1:]) % 2 for p in got if p[0] == "f")
    assert b == B.pilot_batch({q: (v[0], v[2]) for q, v in qs.items()}, n={"bench": 4, "fold": 3},
                              secret=SECRET, seed=0)[0]


def test_prompt_carries_rubric_menu_and_items_only():
    batch = {"batch_id": "b0001", "verdict_order": ["RELATE", "NO_RELATION", "MERGE", "CONFLICT"],
             "items": [{"item_id": "i1", "text_a": "Cats purr.", "text_b": "Cats meow."}]}
    p = B.build_prompt(batch)
    assert "Step 1, MERGE" in p and "Worked examples" in p and "Adjudication" not in p
    assert "RELATE, NO_RELATION, MERGE, CONFLICT" in p and '"Cats purr."' in p and "<!--" not in p
    adj = dict(batch, items=[dict(batch["items"][0], votes=[{"verdict": "MERGE"}])])
    assert "Adjudication" in B.build_prompt(adj) and "AMBIGUOUS" in B.build_prompt(adj)


# ------------------------------------------------------------------ validation, backends, retry

BATCH = {"batch_id": "b0001", "verdict_order": list(B.VERDICTS),
         "items": [{"item_id": "i1", "text_a": "a b", "text_b": "a b"},
                   {"item_id": "i2", "text_a": "a b c d", "text_b": "x y"}]}


def _ok(**kw):
    e = {"item_id": "i1", "verdict": "MERGE", "subtype": None, "overlap_flag": False, "confidence": 5,
         "rationale": "Same."}
    return [dict(e, **kw), dict(e, item_id="i2", verdict="NO_RELATION")]


def test_validate_accepts_fenced_json_and_rejects_every_schema_breach():
    labels, err = B.validate("```json\n" + json.dumps({"labels": _ok()}) + "\n```", BATCH)
    assert not err and labels["i1"]["verdict"] == "MERGE"
    bad = ["not json", json.dumps([]), json.dumps({"labels": _ok()[:1]}),
           json.dumps({"labels": _ok() + _ok()[:1]}),
           json.dumps({"labels": _ok(item_id="i9")}),
           json.dumps({"labels": _ok(verdict="SAME")}),
           json.dumps({"labels": _ok(verdict="AMBIGUOUS")}),             # only an adjudicator may
           json.dumps({"labels": _ok(subtype="CORRECTION")}),            # MERGE has no subtype
           json.dumps({"labels": _ok(verdict="CONFLICT", subtype="ELABORATION")}),
           json.dumps({"labels": _ok(verdict="RELATE")}),                # RELATE needs a subtype
           json.dumps({"labels": _ok(overlap_flag=True)}),               # never with MERGE
           json.dumps({"labels": _ok(confidence=7)}),
           json.dumps({"labels": _ok(rationale="")}),
           json.dumps({"labels": _ok(extra=1)})]
    for raw in bad:
        labels, err = B.validate(raw, BATCH)
        assert labels is None and err, raw


class Scripted:
    def __init__(self, *outs):
        self.outs, self.prompts = list(outs), []

    def complete(self, batch, prompt, attempt):
        self.prompts.append(prompt)
        return self.outs[attempt]

    def describe(self):
        return {"backend": "scripted"}


def test_one_retry_with_the_errors_then_give_up_raw_responses_kept():
    good = json.dumps({"labels": _ok()})
    be = Scripted("garbage", good)
    rec = B.run_batch(BATCH, be, annotator="A1")
    assert rec["labels"]["i2"]["verdict"] == "NO_RELATION" and rec["raw"] == ["garbage", good]
    assert len(be.prompts) == 2 and "not valid" in be.prompts[1] and rec["errors"] == []
    rec = B.run_batch(BATCH, Scripted("garbage", "still garbage"), annotator="A1")
    assert rec["labels"] is None and rec["errors"] and rec["raw"] == ["garbage", "still garbage"]
    assert rec["prompt_sha256"] == hashlib.sha256(B.build_prompt(BATCH).encode()).hexdigest()


def test_subagent_backend_only_writes_prompts_and_reads_responses(tmp_path):
    be = B.SubagentBackend(str(tmp_path))
    assert B.run_batch(BATCH, be, annotator="A1") is None                     # pending
    assert (tmp_path / "b0001.prompt0.txt").read_text() == B.build_prompt(BATCH)
    (tmp_path / "b0001.response0.json").write_text("oops")
    assert B.run_batch(BATCH, be, annotator="A1") is None                     # retry pending
    assert "not valid" in (tmp_path / "b0001.prompt1.txt").read_text()
    (tmp_path / "b0001.response1.json").write_text(json.dumps({"labels": _ok()}))
    rec = B.run_batch(BATCH, be, annotator="A1")
    assert rec["labels"]["i1"]["verdict"] == "MERGE" and rec["raw"][0] == "oops"


def test_http_backend_posts_an_openai_compatible_request(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

    def fake_urlopen(req, timeout):
        seen["url"], seen["body"], seen["auth"] = req.full_url, json.loads(req.data), req.get_header("Authorization")
        return Resp()

    monkeypatch.setattr(B.urllib.request, "urlopen", fake_urlopen)
    be = B.HttpBackend("http://host:8000/v1/", "qwen", temperature=0.0, seed=3, api_key="k")
    assert be.complete(BATCH, "PROMPT", 0) == "{}"
    assert seen["url"] == "http://host:8000/v1/chat/completions" and seen["auth"] == "Bearer k"
    body = seen["body"]
    assert body["model"] == "qwen" and body["temperature"] == 0.0 and body["seed"] == 3
    assert body["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert body["response_format"]["type"] == "json_schema"
    # the schema pins the batch: one entry per item, in batch order (models drop, repeat or miscopy ids)
    labels = body["response_format"]["json_schema"]["schema"]["properties"]["labels"]
    ids = [it["item_id"] for it in BATCH["items"]]
    assert labels["minItems"] == labels["maxItems"] == len(ids) and labels["items"] is False
    assert [e["properties"]["item_id"] for e in labels["prefixItems"]] == [{"enum": [i]} for i in ids]
    assert labels["prefixItems"][0]["properties"]["verdict"] == B.RESPONSE_SCHEMA["properties"]["labels"]["items"]["properties"]["verdict"]
    assert "minItems" not in B.RESPONSE_SCHEMA["properties"]["labels"]  # the shared schema is not mutated
    assert be.describe()["model"] == "qwen" and be.describe()["schema"] == "per-batch"


# ------------------------------------------------------------------ aggregation

def _v(verdict, subtype=None, flag=False):
    return {"verdict": verdict, "subtype": subtype, "overlap_flag": flag, "confidence": 3, "rationale": "r"}


def _both(verdict, subtype=None, ba=None, flag=False):
    return {"ab": _v(verdict, subtype, flag), "ba": _v(*(ba or (verdict, subtype)), flag=flag)}


def test_aggregation_precedence_and_adjudication_routing():
    M, R, X, N = "MERGE", "RELATE", "CONFLICT", "NO_RELATION"
    votes = {
        "unan": {"A1": _both(M), "A2": _both(M), "A3": _both(M)},
        "maj": {"A1": _both(M), "A2": _both(M), "A3": _both(N)},
        "maj_incons": {"A1": _both(N), "A2": _both(N), "A3": _both(M, ba=(N,))},
        "split3": {"A1": _both(M), "A2": _both(N), "A3": _both(R, "SCOPE")},
        "two_incons": {"A1": _both(M, ba=(N,)), "A2": _both(N, ba=(M,)), "A3": _both(M)},
        "missing": {"A1": _both(M), "A2": {"ab": _v(M)}, "A3": {"ab": _v(N), "ba": _v(N)}},
        "subtie": {"A1": _both(X, "CORRECTION"), "A2": _both(X, "SUPERSESSION"), "A3": _both(X, "SCOPE")},
        "submaj": {"A1": _both(X, "CORRECTION"), "A2": _both(X, "CORRECTION"), "A3": _both(X, "SUPERSESSION")},
    }
    agg = A.aggregate(votes, ["A1", "A2", "A3"])
    assert (agg["unan"]["path"], agg["unan"]["verdict"]) == ("unanimous", M)
    assert (agg["maj"]["path"], agg["maj"]["verdict"]) == ("majority", M)
    assert (agg["maj_incons"]["path"], agg["maj_incons"]["verdict"]) == ("majority", N)
    assert agg["split3"] == {"path": "adjudicate", "reason": "disagreement"}
    assert agg["two_incons"] == {"path": "adjudicate", "reason": "orientation"}
    assert agg["missing"] == {"path": "adjudicate", "reason": "disagreement"}  # one consistent vote left
    assert agg["subtie"] == {"path": "adjudicate", "reason": "subtype"}
    assert (agg["submaj"]["verdict"], agg["submaj"]["subtype"]) == (X, "CORRECTION")
    assert agg["unan"]["subtype"] is None


def test_mergeability_resolution_decides_the_merge_flag_only():
    M, R, X, N = "MERGE", "RELATE", "CONFLICT", "NO_RELATION"
    votes = {
        "unan_non": {"A1": _both(R, "SCOPE"), "A2": _both(N), "A3": _both(R, "ELABORATION", ba=(N,))},
        "unan_merge": {"A1": _both(M), "A2": _both(M), "A3": _both(M)},
        "maj_merge": {"A1": _both(M, flag=True), "A2": _both(M, flag=True), "A3": _both(X, "SCOPE")},
        "maj_incons": {"A1": _both(N), "A2": _both(R, "SCOPE"), "A3": _both(M, ba=(R, "SCOPE"))},
        "split": {"A1": _both(M), "A2": _both(R, "SCOPE"), "A3": _both(M, ba=(N,))},
        "two_incons": {"A1": _both(M, ba=(N,)), "A2": _both(N, ba=(M,)), "A3": _both(M)},
        "missing": {"A1": _both(M), "A2": {"ab": _v(M)}, "A3": _both(N)},
    }
    ann = ["A1", "A2", "A3"]
    agg = A.aggregate(votes, ann, resolve="mergeability")
    got = {p: (a["path"], a.get("verdict")) for p, a in agg.items()}
    assert got == {"unan_non": ("mergeability_unanimous", "NON_MERGE"),
                   "unan_merge": ("mergeability_unanimous", M),
                   "maj_merge": ("mergeability_majority", M),
                   "maj_incons": ("mergeability_majority", "NON_MERGE"),
                   "split": ("adjudicate", None), "two_incons": ("adjudicate", None), "missing": ("adjudicate", None)}
    assert agg["split"]["reason"] == "disagreement" and agg["missing"]["reason"] == "disagreement"
    assert agg["two_incons"] == {"path": "adjudicate", "reason": "orientation"}
    assert all(a["subtype"] is None for a in agg.values() if a["path"] != "adjudicate")
    assert agg["maj_merge"]["overlap_flag"] is True and agg["unan_non"]["overlap_flag"] is False
    # the default stays the full verdict: the non-merge pair split on relation type goes to adjudication there
    assert A.aggregate(votes, ann) == A.aggregate(votes, ann, resolve="verdict")
    assert A.aggregate(votes, ann)["unan_non"] == {"path": "adjudicate", "reason": "disagreement"}


def test_collect_maps_items_back_and_separates_repeats():
    items = {"i1": {"pair_id": "p", "queue": "bench", "orientation": "ab", "repeat": False},
             "i2": {"pair_id": "p", "queue": "bench", "orientation": "ba", "repeat": False},
             "i3": {"pair_id": "p", "queue": "bench", "orientation": "ab", "repeat": True}}
    recs = [{"annotator": "A1", "labels": {"i1": _v("MERGE"), "i2": _v("MERGE"), "i3": _v("RELATE", "SCOPE")}},
            {"annotator": "A2", "labels": None}]
    votes, repeats = A.collect(recs, items)
    assert votes == {"p": {"A1": {"ab": _v("MERGE"), "ba": _v("MERGE")}}}
    assert repeats == {"p": {"A1": {"ab": _v("RELATE", "SCOPE")}}}


# ------------------------------------------------------------------ end to end

def test_end_to_end_fake_annotators_to_calibration(tmp_path):
    m, qs = _queues()
    batches, items = B.make_batches(_views(qs), secret=SECRET, seed=0, size=5, n_repeats=2, min_gap=3)

    def flip_to(verdict, gt):        # order-sensitive on RELATE items: two such annotators -> adjudication
        def rule(it):
            lab = rule_ok(it)
            if lab["verdict"] == "RELATE" and (it["text_a"] > it["text_b"]) == gt:
                return {"verdict": verdict, "subtype": None, "overlap_flag": False}
            return lab
        return rule

    recs = []
    for name, rule in (("A1", rule_ok), ("A2", flip_to("NO_RELATION", True)), ("A3", flip_to("NO_RELATION", False))):
        be = B.FakeBackend(rule)
        recs += [B.run_batch(b, be, annotator=name) for b in batches]
    assert all(r["labels"] for r in recs)
    votes, repeats = A.collect(recs, items)
    agg = A.aggregate(votes, ["A1", "A2", "A3"])
    paths = {a["path"] for a in agg.values()}
    assert {"unanimous", "adjudicate"} <= paths

    views_ab = {q: v[0] for q, v in qs.items()}
    adj, adj_items = A.adjudication_batches(agg, votes, views_ab, secret=SECRET, seed=0, size=4)
    assert sum(len(b["items"]) for b in adj) == sum(a["path"] == "adjudicate" for a in agg.values())
    for b in adj:
        for it in b["items"]:
            assert set(it) == {"item_id", "text_a", "text_b", "votes"}
            for v in it["votes"]:
                assert set(v) == {"verdict", "subtype", "overlap_flag", "rationale", "order"}
    blob = json.dumps(adj)
    assert not re.search(r"A1|A2|A3|stratum|label_provenance|mergeable", blob)

    adj_recs = [B.run_batch(b, B.FakeBackend(rule_ok), annotator="A4") for b in adj]
    final = A.finalize(agg, A.adjudicated(adj_recs, adj_items))
    assert all(f["path"] in ("unanimous", "majority", "adjudicated") for f in final.values())

    keys = _keys(qs)
    rows = A.annotation_rows(final, keys, views_ab)
    p = tmp_path / "annotations.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ann = Q.load_annotations(str(p))
    assert len(rows) == len(final)

    # texts come back in the manifest's order (unswap), whatever the side order annotators saw
    man = {r["pair_id"]: r for r in m["pairs"]}
    for r in rows:
        if r["pair_id"] in man:
            assert (r["text_a"], r["text_b"]) == (man[r["pair_id"]]["text_a"], man[r["pair_id"]]["text_b"])
        assert r["mergeable"] == (r["verdict"] == "MERGE") and r["contradiction"] == (r["verdict"] == "CONFLICT")

    def nli(p_, h):
        pe = int.from_bytes(hashlib.sha256(f"{p_}\x00{h}".encode()).digest()[:2], "big")
        return (0, pe, FIXED_ONE - pe)

    rows_cal = C.split_rows(m, "vitaminc", "cal")
    res = C.calibrate(rows_cal, scope="vitaminc", encoder="hashing", nli="lexical", provider=nli, key=b"k",
                      alpha=100_000, B_0=8, selector=SEL, hw="cpu", annotations=ann, max_unresolved=0)
    assert res["epoch"] is not None

    rep = A.report(final, votes, repeats, keys, ["A1", "A2", "A3"])
    assert rep["agreement"]["S4_vitc_factual"]["n"] > 0
    assert set(rep["repeat_consistency"]) == {"A1", "A2", "A3"} and rep["repeat_consistency"]["A1"]["agree"] == 1.0
    acc = rep["accuracy"]["S3_vitc_nonfactual"]
    assert acc["aggregate"]["n"] > 0 and acc["A1"]["n"] > 0


# ------------------------------------------------------------------ statistics and audit

def test_agreement_statistics_hand_values():
    # Krippendorff nominal: units (a,a),(a,b),(b,b),(b,b) -> 1 - 2 / (30/7) = 8/15
    assert A.krippendorff_alpha([["a", "a"], ["a", "b"], ["b", "b"], ["b", "b"]]) == pytest.approx(8 / 15)
    assert A.krippendorff_alpha([["a", "a", None], ["b", "b", "b"]]) == pytest.approx(1.0)
    # Cohen: po = 0.75, pe = 0.5 -> 0.5
    assert A.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == pytest.approx(0.5)
    lo, hi = A.wilson(9, 10)
    assert lo == pytest.approx(0.5958, abs=1e-3) and hi == pytest.approx(0.9821, abs=1e-3)


def test_cohen_kappa_matches_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    x, y = list("aabbcabcab"), list("abbbcaccab")
    assert A.cohen_kappa(x, y) == pytest.approx(skm.cohen_kappa_score(x, y))


def test_audit_draw_reuses_the_export_and_oversamples_disagreement():
    rows, votes = [], {}
    for i in range(300):
        pid = f"p{i:03d}"
        st = "S4_vitc_factual" if i < 200 else "S2_paws_nonpara"
        rows.append(_row(i, 0, st, "test", False, "to_annotate", f"x{i} one", f"x{i} two") | {"pair_id": pid})
        dis = i % 10 == 0
        votes[pid] = {"A1": _both("CONFLICT", "CORRECTION"), "A2": _both("CONFLICT", "CORRECTION"),
                      "A3": _both("RELATE", "SCOPE") if dis else _both("CONFLICT", "CORRECTION")}
    keys = {"bench": [{k: r[k] for k in ("pair_id", "stratum")} for r in rows]}
    view, private = A.audit_draw(votes, rows, keys, allocation={"S4": 40, "S2": 20}, seed=0, secret=SECRET,
                                 n_repeats=10, min_gap=20)
    pi = {r["pair_id"]: r["pi"] for r in private["draw"]["population"]}
    assert pi["p000"] > pi["p001"]                                   # disagreement over-sampled
    assert len(private["draw"]["sample"]) == 60 and len(view) == 70
    assert all(set(v) == {"pair_id", "text_a", "text_b", "badges_a", "badges_b"} for v in view)


# ------------------------------------------------------------------ the rubric's examples on real data

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "bench")


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "manifest.json")), reason="benchmark data not built")
def test_rubric_examples_are_dev_pairs_outside_every_queue():
    with open(os.path.join(DATA, "manifest.json"), encoding="utf-8") as fh:
        man = {r["pair_id"]: r for r in json.load(fh)["pairs"]}
    with open(os.path.join(DATA, "panel_key.jsonl"), encoding="utf-8") as fh:
        queued = {json.loads(x)["pair_id"] for x in fh if x.strip()}
    rubric = open(B.RUBRIC_PATH, encoding="utf-8").read()
    refs = re.findall(r"dev:([0-9a-f]{16})(?::([ab]))?", rubric)
    assert len(refs) >= 9
    for pid, side in refs:
        r = man[pid]
        assert r["split"] == "dev" and pid not in queued
        for t in ([r["text_" + side]] if side else [r["text_a"], r["text_b"]]):
            assert f'"{t}"' in rubric


# ------------------------------------------------------------------ the script

def test_script_batches_run_aggregate_and_audit(tmp_path):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import annotate as S
    _, qs = _queues()
    for q, (d, prefix) in S.QUEUES.items():
        os.makedirs(tmp_path / d, exist_ok=True)
        for f, rows in zip(("view_ab", "view_ba", "key"), qs[q]):
            (tmp_path / d / f"{prefix}{f}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = str(tmp_path / "run")
    res = S.main(["batches", "--out", out, "--root", str(tmp_path), "--size", "5", "--repeats", "2",
                  "--min-gap", "3", "--secret", SECRET])
    assert res["repeats"] == 4
    always_conflict = lambda it: {"verdict": "CONFLICT", "subtype": "CORRECTION", "overlap_flag": True}  # noqa: E731
    for a, rule in (("A1", rule_ok), ("A3", always_conflict)):            # subagents, answered by rules
        assert S.main(["run", "--dir", out, "--annotator", a, "--backend", "subagent"])["pending"] == res["batches"]
        for f in os.listdir(os.path.join(out, "batches")):
            b = json.load(open(os.path.join(out, "batches", f)))
            with open(os.path.join(out, "exchange", a, f"{b['batch_id']}.response0.json"), "w") as fh:
                fh.write(B.FakeBackend(rule).complete(b, "", 0))
        assert S.main(["run", "--dir", out, "--annotator", a, "--backend", "subagent"]) == {"done": res["batches"]}
    S.main(["run", "--dir", out, "--annotator", "A2", "--backend", "fake"])
    agg = S.main(["aggregate", "--dir", out, "--annotators", "A1", "A2", "A3", "--adjudicator", "A4"])
    assert agg["pending_adjudication"] > 0 and agg["adjudication_batches"] > 0   # three-way splits
    # mergeability resolution on a copy: fewer pairs adjudicated, stale adjudication batches dropped
    import shutil
    mo = str(tmp_path / "mergeonly")
    shutil.copytree(out, mo)
    cmd = ["aggregate", "--dir", mo, "--annotators", "A1", "A2", "A3", "--adjudicator", "A4", "--size", "1"]
    assert S.main(cmd)["adjudication_batches"] == agg["pending_adjudication"]
    res_mo = S.main(cmd + ["--resolve", "mergeability"])
    assert res_mo["pending_adjudication"] < agg["pending_adjudication"]
    assert len(os.listdir(os.path.join(mo, "adjudication", "batches"))) == res_mo["adjudication_batches"]
    rep = json.load(open(os.path.join(mo, "report.json")))
    assert rep["resolve"] == "mergeability"
    assert sum(rep["paths"].values()) + sum(rep["pending_by_reason"].values()) == rep["n_pairs"]
    assert set(rep["paths"]) <= {"mergeability_unanimous", "mergeability_majority", "adjudicated"}
    ann = Q.load_annotations(os.path.join(mo, "annotations.jsonl"))
    rows = [json.loads(x) for x in open(os.path.join(mo, "annotations.jsonl"))]
    assert all(ann[r["pair_id"]] == (r["mergeable"], r["label_provenance"]) for r in rows)
    assert len(rows) == res_mo["final"] > 0
    assert {r["verdict"] for r in rows} <= {"MERGE", "NON_MERGE"}
    assert all(r["mergeable"] == (r["verdict"] == "MERGE") and r["subtype"] is None for r in rows)
    assert "resolve" not in json.load(open(os.path.join(out, "report.json")))       # default report unchanged
    S.main(["run", "--dir", os.path.join(out, "adjudication"), "--annotator", "A4", "--backend", "fake"])
    agg = S.main(["aggregate", "--dir", out, "--annotators", "A1", "A2", "A3", "--adjudicator", "A4", "--audit",
                  "--audit-alloc", '{"S4": 4}', "--audit-repeats", "1", "--audit-min-gap", "2"])
    assert agg["pending_adjudication"] == 0 and agg["final"] == agg["pairs"]
    ann = Q.load_annotations(os.path.join(out, "annotations.jsonl"))
    assert len(ann) > 0 and os.path.exists(os.path.join(out, "audit_view.jsonl"))


def test_script_batches_take_named_queue_files(tmp_path):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import annotate as S
    _, qs = _queues()
    for q, name in (("bench", "copyval"), ("fold", "substrate")):
        for f, rows in zip(("view_ab", "view_ba", "key"), qs[q]):
            (tmp_path / f"{name}_{f}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = str(tmp_path / "run")
    res = S.main(["batches", "--out", out, "--root", "/nonexistent", "--size", "5", "--repeats", "0",
                  "--secret", SECRET, "--queue", f"copyval={tmp_path}/copyval_", "--queue", f"substrate={tmp_path}/substrate_"])
    priv = json.load(open(os.path.join(out, "private.json")))
    assert set(priv["inputs"]) == {"copyval", "substrate"}
    assert priv["inputs"]["copyval"]["key"] == f"{tmp_path}/copyval_key.jsonl"
    n = {q: len(qs[q][0]) for q in qs}
    assert Counter(m["queue"] for m in priv["items"].values()) == {"copyval": 2 * n["bench"], "substrate": 2 * n["fold"]}
    assert res["items"] == 2 * (n["bench"] + n["fold"])
