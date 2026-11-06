"""Calibration substrates of the recall workloads (meld.eval.substrates): disjoint task pools, the
evaluation's own stream, every tested pair classified, refusal while annotators have not labelled.
Hashing encoder, lexical or fake NLI."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from types import SimpleNamespace

import pytest

from meld.agent.hotpot_loader import load_hotpot_fixture
from meld.agent.mem_conditions import build_condition
from meld.agent.memory import DEFAULT_SCOPE
from meld.eval import calibration as C
from meld.eval import label_queue as Q
from meld.eval import substrates as S
from meld.eval import topology
from meld.protocol.objects import FIXED_ONE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import run_nbrains  # noqa: E402
import run_r2_lme  # noqa: E402
import run_r2_overlap_k  # noqa: E402

IDS = [f"q{i:04d}" for i in range(2000)]
CAL = [i for i in IDS if S.pool(i) == "cal"]
EVAL = [i for i in IDS if S.pool(i) == "eval"]


def _lme_item(qid, sessions):
    """A LongMemEval-shaped item: sessions of user turns (assistant turns are not claims)."""
    return {"question_id": qid, "question_type": "multi-session", "question": "what about alpha?",
            "haystack_session_ids": [f"{qid}-s{i}" for i in range(len(sessions))],
            "haystack_dates": [f"2023/01/{i + 1:02d}" for i in range(len(sessions))],
            "haystack_sessions": [[t for u in s for t in ({"role": "user", "content": u},
                                                          {"role": "assistant", "content": "ok"})]
                                  for s in sessions]}


LME = _lme_item("qfix", [["alpha beta gamma delta", "alpha beta gamma epsilon"],
                         ["alpha beta gamma delta zeta", "omega psi chi"],
                         ["omega psi chi phi", "kappa lambda mu"]])


def test_pools_are_disjoint_stable_and_about_one_in_ten():
    assert 0.07 < len(CAL) / len(IDS) < 0.13
    assert all(S.pool(i + "_abs") == S.pool(i) for i in IDS[:200])
    tasks = [SimpleNamespace(task_id=i) for i in IDS]
    ev, ca = S.eval_pool(tasks), S.cal_pool(tasks)
    assert {t.task_id for t in ev} == set(EVAL) and {t.task_id for t in ca} == set(CAL)


def test_recall_runners_never_evaluate_a_cal_task(tmp_path, monkeypatch):
    tasks = [SimpleNamespace(task_id=i) for i in IDS[:300]]
    seen = {}

    def rows(ts, **kw):
        seen.setdefault("tasks", []).extend(t.task_id for t in ts)
        return []
    for mod in (run_r2_overlap_k, run_r2_lme):
        monkeypatch.setattr(mod, "recall_rows", rows)
        monkeypatch.setattr(mod, "aggregate_recall", _stop)
    monkeypatch.setattr(run_r2_overlap_k, "load_hotpot_real", lambda **kw: list(tasks))
    monkeypatch.setattr(run_r2_lme, "load_longmemeval_tasks", lambda *a, **kw: list(tasks))
    for mod, argv in ((run_r2_overlap_k, ["--n", "250"]), (run_r2_lme, ["--n", "250"])):
        with pytest.raises(_Stop):
            mod.main(argv + ["--encoder", "hashing", "--nli", "lexical", "--out", str(tmp_path / "o.json")])
    assert seen["tasks"] and not set(seen["tasks"]) & set(CAL)
    assert len(set(seen["tasks"])) == 250

    data = tmp_path / "lme.json"
    data.write_text(json.dumps([dict(LME, question_id=i) for i in IDS[:300]]))
    a = SimpleNamespace(workload="lme", data=str(data), types=["multi-session"], n_items=300, seed=0)
    got = [w for w, *_ in run_nbrains.worlds(a)]
    assert got and not set(got) & set(CAL) and set(got) == set(EVAL) & set(IDS[:300])
    monkeypatch.setattr("meld.agent.hotpot_loader.load_hotpot_real", lambda **kw: list(tasks))
    a = SimpleNamespace(workload="hotpot", data=None, n_items=300, per_store=1, overlap=0.5)
    monkeypatch.setattr(run_nbrains, "pool_tasks", lambda ts, k: [(t, [t]) for t in ts])
    got = [w for w, *_ in run_nbrains.worlds(a)]
    assert got and not set(got) & set(CAL)


class _Stop(Exception):
    pass


def _stop(rows):
    raise _Stop


def _ids(store):
    objs = store.objects
    return sorted((objs.get("D", d).created_at, d) for d in objs.ids("D", DEFAULT_SCOPE))


def test_hotpot_cal_stream_is_the_evaluation_sink_stream():
    task = load_hotpot_fixture(n=1)[0]
    (wid, plan, item, src), = S.hotpot_worlds([task], n_brains=3, dup_degree=2, overlap=0.5)
    assert wid == task.task_id and any(src[x] != x for x in src)          # copies were placed
    store = S.world_store(plan, encoder="hashing")
    merge = build_condition(task, "distributed_merge", encoder="hashing", nli="lexical",
                            n_brains=3, overlap_fraction=0.5, dup_degree=2)
    assert plan.per_brain == merge.split.per_brain
    assert _ids(store) == _ids(merge.brain.store)
    star = topology.build(plan, "exact_dedup", "star", encoder="hashing", nli="lexical")
    assert _ids(store) == _ids(star["memory"].store)
    # the N-brains runner's pooled hotpot world, for the same tasks
    tasks = load_hotpot_fixture(n=4)
    a = SimpleNamespace(workload="hotpot", data="fixture", n_items=4, per_store=2, overlap=0.5)
    for (w, split, _), (w2, p2, *_) in zip(run_nbrains.worlds(a),
                                           S.hotpot_worlds(tasks, n_brains=3, dup_degree=2, per_store=2)):
        assert w == w2 and split(3, 2).per_brain == p2.per_brain


def test_lme_cal_stream_is_the_evaluation_sink_stream(tmp_path, monkeypatch):
    (wid, plan, item, src), = S.lme_worlds([LME], n_brains=3, dup_degree=2)
    store = S.world_store(plan, encoder="hashing")
    star = topology.build(plan, "exact_dedup", "star", encoder="hashing", nli="lexical")
    assert _ids(store) == _ids(star["memory"].store)
    data = tmp_path / "lme.json"
    data.write_text(json.dumps([LME]))
    a = SimpleNamespace(workload="lme", data=str(data), types=["multi-session"], n_items=1, seed=0)
    monkeypatch.setattr(S, "pool", lambda i: "eval")                    # the fixture id's pool
    (w, split, _), = run_nbrains.worlds(a)
    assert split(3, 2).per_brain == plan.per_brain and split(3, 2).when == plan.when
    assert item[plan.per_brain[1][0]] == "qfix-s0" and set(item.values()) == {f"qfix-s{i}" for i in range(3)}


def test_classification_counts_on_a_small_world():
    t = Q.classify_substrate("lme", S.lme_worlds([LME], n_brains=3, dup_degree=2), encoder="hashing")
    n = Counter(r["class"] for r in t)
    assert n == {"copy": 5, "same_item": 1, "cross_item": 3}, n
    for r in t:
        assert r["mergeable"] is None and r["provenance"] is None and r["world"] == "qfix"
        assert (r["item_a"] == r["item_b"]) == (r["class"] in ("copy", "same_item"))
    rows = Q.queue_rows(t, source="lme")
    assert len(rows) == len(t) and {r["stratum"] for r in rows} == {f"tested_{c}" for c in n}
    from meld.bench.export import panel_export
    view, view_ba, key = panel_export(rows, seed=0)
    assert len(view) == len(view_ba) == len(key) == len(t)
    # labelled pairs leave the queue
    ann = {r["pair_id"]: (False, "panel") for r in rows[:3]}
    t2 = Q.classify_substrate("lme", S.lme_worlds([LME], n_brains=3, dup_degree=2), encoder="hashing",
                              annotations=ann)
    assert len(Q.queue_rows(t2, source="lme")) == len(t) - 3


def _nli(p, h):
    return (0, FIXED_ONE // 2, FIXED_ONE // 2)


def _cal(**kw):
    return C.calibrate_substrate("lme", S.lme_worlds([LME], n_brains=3, dup_degree=2), encoder="hashing",
                                 nli="lexical", provider=_nli, alpha=100_000, B_0=4, hw="cpu", **kw)


def test_strict_calibration_refuses_until_annotators_have_labelled():
    with pytest.raises(ValueError, match="unresolved"):
        _cal()
    dev = _cal(max_unresolved=None)
    assert dev["epoch"] is None and dev["cal"] == [] and dev["stats"]["unresolved"] == 9
    assert dev["populated_bins"] and set(dev["items_by_bin"]) == set(dev["populated_bins"])
    t = Q.classify_substrate("lme", S.lme_worlds([LME], n_brains=3, dup_degree=2), encoder="hashing")
    ann = {r["pair_id"]: (r["stratum"] == "tested_copy", "panel") for r in Q.queue_rows(t, source="lme")}
    res = _cal(annotations=ann)
    ep = res["epoch"]
    assert ep.scope == DEFAULT_SCOPE and res["stats"]["unresolved"] == 0
    assert len(res["cal"]) == len({k.item for k in res["cal"]}) >= 1
    from meld.agent.memory import SMPMemory
    mem = SMPMemory(encoder="hashing", nli="lexical", epoch=ep)               # the evaluation accepts it
    from meld.protocol.fold import scope_epoch
    assert scope_epoch(mem.store, DEFAULT_SCOPE, mem.m_nli) is not None


# ---- the runners: the queue of a recall workload, strict refusal, development bins

import build_label_queue  # noqa: E402
import run_calibration  # noqa: E402


def _runner_env(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "cal_worlds", lambda d, **kw: list(S.lme_worlds([LME], n_brains=3, dup_degree=2)))
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"seed": 0, "pairs": []}))
    pv = tmp_path / "panel_view_ab.jsonl"
    pv.write_text("")
    return ["--manifest", str(m), "--encoder", "hashing"], pv


def test_label_queue_runner_queues_every_tested_pair_of_a_recall_workload(tmp_path, monkeypatch):
    base, pv = _runner_env(tmp_path, monkeypatch)
    rep = build_label_queue.main(base + ["--panel-view", str(pv), "--domain", "lme",
                                         "--out-dir", str(tmp_path / "q")])
    assert rep["counts"]["lme/cal"] == {"copy": 5, "same_item": 1, "cross_item": 3, "worlds": 1}
    assert rep["new_rows"] == 9 and rep["new_rows_by_stream"]["lme/cal/tested_copy"] == 5
    view = [json.loads(x) for x in open(tmp_path / "q" / "label_queue_view_ab.jsonl")]
    assert len(view) == 9


def test_calibration_runner_refuses_then_reports_development_bins(tmp_path, monkeypatch, capsys):
    base, pv = _runner_env(tmp_path, monkeypatch)
    key = tmp_path / "key"
    key.write_bytes(b"k" * 32)
    args = base + ["--domain", "lme", "--alpha", "0.10", "--nli", "lexical", "--B0", "4",
                   "--key-file", str(key), "--no-source-labels"]
    with pytest.raises(ValueError, match="unresolved"):
        run_calibration.main(args + ["--out-dir", str(tmp_path / "strict")], provider=_nli)
    rep = run_calibration.main(args + ["--allow-unresolved", "--out-dir", str(tmp_path / "dev")], provider=_nli)
    e = rep["epochs"][0]
    assert e["epoch_file"] is None and e["bins"] == [] and e["scope"] == DEFAULT_SCOPE
    assert e["unresolved"] == 9 and e["populated_bins"] and e["disabled_bins"] == e["populated_bins"]
    assert "no Cal and no epoch" in capsys.readouterr().err
    t = Q.classify_substrate("lme", S.lme_worlds([LME], n_brains=3, dup_degree=2), encoder="hashing")
    ann = tmp_path / "ann.jsonl"
    ann.write_text("".join(json.dumps({"pair_id": r["pair_id"], "mergeable": r["stratum"] == "tested_copy",
                                       "label_provenance": "adjudicated"}) + "\n"
                           for r in Q.queue_rows(t, source="lme")))
    rep = run_calibration.main(args + ["--annotations", str(ann), "--out-dir", str(tmp_path / "ok")],
                               provider=_nli)
    e = rep["epochs"][0]
    assert e["verified"] and e["unresolved"] == 0 and (tmp_path / "ok" / e["epoch_file"]).exists()


# ---- copies: source-labelled mergeable only behind a validated panel sample


def _validation(tmp_path, n, k, name="val.jsonl"):
    """n validated copy pairs, k judged MERGE (labels.jsonl rows of scripts/annotate.py)."""
    f = tmp_path / name
    f.write_text("".join(json.dumps({"pair_id": f"p{i}", "verdict": "MERGE" if i < k else "AMBIGUOUS",
                                     "path": "unanimous"}) + "\n" for i in range(n)))
    return str(f)


def test_copies_validated_refuses_below_the_rate_and_reports_the_wilson_bound(tmp_path):
    with pytest.raises(ValueError, match="0.97"):
        Q.copies_validated(_validation(tmp_path, 100, 97))
    st = Q.copies_validated(_validation(tmp_path, 200, 199))
    assert st["n"] == 200 and st["mergeable"] == 199 and st["rate"] == 0.995
    assert 0.97 < st["wilson_lower"] < 0.995
    with pytest.raises(ValueError):
        Q.copies_validated(_validation(tmp_path, 0, 0))


def test_validated_copies_are_source_labelled_mergeable():
    w = lambda: S.lme_worlds([LME], n_brains=3, dup_degree=2)  # noqa: E731
    t = Q.classify_substrate("lme", w(), encoder="hashing", copies_mergeable=True)
    n = Counter(r["class"] for r in t)
    assert "copy" not in n and n["source_labelled"] == 5
    assert all(r["mergeable"] is True for r in t if r["class"] == "source_labelled")
    assert {r["stratum"] for r in Q.queue_rows(t, source="lme")} == {"tested_same_item", "tested_cross_item"}
    res = C.calibrate_substrate("lme", w(), encoder="hashing", nli="lexical", provider=_nli, alpha=100_000,
                                B_0=4, hw="cpu", max_unresolved=None, copies_mergeable=True)
    assert res["stats"]["unresolved"] == 4 and res["labels"]["source_labelled"] == {"mergeable": 5}


def test_runners_take_copies_as_mergeable_only_with_a_validated_sample(tmp_path, monkeypatch):
    base, pv = _runner_env(tmp_path, monkeypatch)
    args = base + ["--panel-view", str(pv), "--domain", "lme"]
    with pytest.raises(SystemExit):
        build_label_queue.main(args + ["--out-dir", str(tmp_path / "bad"),
                                       "--copies-validated", _validation(tmp_path, 100, 90)])
    rep = build_label_queue.main(args + ["--out-dir", str(tmp_path / "ok"),
                                         "--copies-validated", _validation(tmp_path, 200, 200, "v2.jsonl")])
    assert rep["copies_validated"]["rate"] == 1.0 and rep["new_rows"] == 4
    assert "lme/cal/tested_copy" not in rep["new_rows_by_stream"]
    key = tmp_path / "key"
    key.write_bytes(b"k" * 32)
    with pytest.raises(SystemExit):
        run_calibration.main(base + ["--domain", "lme", "--alpha", "0.10", "--nli", "lexical", "--B0", "4",
                                     "--key-file", str(key), "--no-source-labels", "--allow-unresolved",
                                     "--out-dir", str(tmp_path / "c"),
                                     "--copies-validated", _validation(tmp_path, 100, 90, "v3.jsonl")],
                             provider=_nli)


def test_copy_validation_sample_is_a_seeded_panel_queue_of_copies(tmp_path, monkeypatch):
    base, pv = _runner_env(tmp_path, monkeypatch)
    args = base + ["--panel-view", str(pv), "--domain", "lme", "--copy-sample", "3"]
    build_label_queue.main(args + ["--out-dir", str(tmp_path / "a")])
    build_label_queue.main(args + ["--out-dir", str(tmp_path / "b")])
    rd = lambda d, f: [json.loads(x) for x in open(tmp_path / d / f)]  # noqa: E731
    key = rd("a", "copy_validation_key.jsonl")
    assert len(key) == 3 and key == rd("b", "copy_validation_key.jsonl")
    assert len(rd("a", "copy_validation_view_ab.jsonl")) == len(rd("a", "copy_validation_view_ba.jsonl")) == 3
    copies = {r["pair_id"] for r in rd("a", "label_queue_key.jsonl") if r["stratum"] == "tested_copy"}
    assert {r["pair_id"] for r in key} <= copies


def test_cal_worlds_seeded_pick_is_a_stable_subset_in_pool_order(tmp_path):
    items = [_lme_item(q, [["alpha beta gamma", "delta epsilon zeta"]]) for q in CAL[:8] + EVAL[:3]]
    p = tmp_path / "lme.json"
    p.write_text(json.dumps(items))
    ids = lambda **kw: [w[0] for w in S.cal_worlds("lme", data=str(p), **kw)]  # noqa: E731
    assert ids(n=3) == CAL[:3]
    a = ids(n=3, pick=7)
    assert a == ids(n=3, pick=7) and len(a) == 3 and set(a) <= set(CAL[:8])
    assert a == sorted(a, key=CAL.index) and a != ids(n=3, pick=8)
    assert ids(n=20, pick=7) == CAL[:8]


def test_label_queue_runner_drops_copies_dedupes_several_views_and_records_worlds(tmp_path, monkeypatch):
    base, pv = _runner_env(tmp_path, monkeypatch)
    args = base + ["--domain", "lme", "--copy-sample", "2"]
    build_label_queue.main(args + ["--panel-view", str(pv), "--out-dir", str(tmp_path / "a")])
    rd = lambda d, f: [json.loads(x) for x in open(tmp_path / d / f)]  # noqa: E731
    kept = [r["pair_id"] for r in rd("a", "label_queue_key.jsonl") if r["stratum"] != "tested_copy"]
    rest = tmp_path / "rest.jsonl"
    rest.write_text("".join(json.dumps(r) + "\n" for r in rd("a", "label_queue_view_ab.jsonl") if r["pair_id"] == kept[0]))
    rep = build_label_queue.main(args + ["--panel-view", str(pv), str(rest), "--no-copies", "--world-seed", "5",
                                         "--out-dir", str(tmp_path / "b")])
    assert [r["pair_id"] for r in rd("b", "label_queue_key.jsonl")] == kept[1:]
    assert rep["copy_sample"] == {"lme": 2} and rep["world_seed"] == 5 and rep["worlds"] == {"lme": ["qfix"]}
    assert rep["existing_panel_rows"] == 1


def test_run_calibration_passes_world_seed_to_the_world_draw(monkeypatch, tmp_path):
    """--world-seed must reach cal_worlds as pick, so calibration folds exactly the worlds the
    label queue drew (build_label_queue --world-seed)."""
    seen = {}

    def fake(domain, **kw):
        seen[domain] = kw.get("pick")
        raise SystemExit(0)

    monkeypatch.setattr(run_calibration.S, "cal_worlds", fake)
    monkeypatch.setenv("SMP_GROUP_KEY", "00" * 32)
    with pytest.raises(SystemExit):
        run_calibration.main(["--domain", "hotpot", "--alpha", "0.05", "--encoder", "hashing", "--nli", "lexical",
                              "--n-cal", "3", "--world-seed", "7", "--allow-unresolved",
                              "--out-dir", str(tmp_path)])
    assert seen == {"hotpot": 7}


# ---- world specs: calibration worlds of several shapes, named task by task


def _spec_data(tmp_path):
    shared = ["alpha beta gamma", "delta epsilon zeta"]
    items = [_lme_item(q, [shared, [f"{q} kappa lambda", f"{q} mu nu xi"]]) for q in CAL[:4] + EVAL[:1]]
    items[1]["haystack_session_ids"][0] = items[0]["haystack_session_ids"][0]     # one shared session
    p = tmp_path / "lme.json"
    p.write_text(json.dumps(items))
    return str(p)


def test_cal_worlds_by_ids_keep_the_given_order_and_refuse_evaluation_tasks(tmp_path):
    data = _spec_data(tmp_path)
    ids = lambda **kw: [w[0] for w in S.cal_worlds("lme", data=data, **kw)]  # noqa: E731
    assert ids(ids=[CAL[2], CAL[0]]) == [CAL[2], CAL[0]]
    with pytest.raises(ValueError, match="cal pool"):
        S.cal_worlds("lme", data=data, ids=[CAL[0], EVAL[0]])


def test_spec_worlds_build_each_shape_and_report_items_in_several_worlds(tmp_path):
    data = _spec_data(tmp_path)
    spec = {"lme": [{"n_brains": 4, "dup": 4, "ids": [CAL[0], CAL[2]]}, {"n_brains": 8, "dup": 2, "ids": [CAL[1]]}]}
    w = S.spec_worlds(spec, data=data)["lme"]
    assert [x[0] for x in w] == [CAL[0], CAL[2], CAL[1]]
    assert [len(x[1].per_brain) for x in w] == [4, 4, 8]
    rep = S.repeated_items(w)
    assert rep == {f"{CAL[0]}-s0": sorted([CAL[0], CAL[1]])}
    assert S.repeated_items(w[:2]) == {}


def test_label_queue_and_calibration_runners_take_a_world_spec(tmp_path, monkeypatch):
    data = _spec_data(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"lme": [{"n_brains": 3, "dup": 2, "ids": [CAL[0], CAL[1]]}]}))
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"seed": 0, "pairs": []}))
    pv = tmp_path / "pv.jsonl"
    pv.write_text("")
    base = ["--manifest", str(m), "--encoder", "hashing", "--lme-data", data, "--domain", "lme",
            "--world-spec", str(spec)]
    rep = build_label_queue.main(base + ["--panel-view", str(pv), "--out-dir", str(tmp_path / "q")])
    assert rep["worlds"] == {"lme": [CAL[0], CAL[1]]} and rep["repeated_items"] == {"lme": 1}
    assert rep["world_spec"]["sha256"] and rep["counts"]["lme/cal"]["worlds"] == 2
    key = tmp_path / "key"
    key.write_bytes(b"k" * 32)
    rep = run_calibration.main(base + ["--alpha", "0.10", "--nli", "lexical", "--B0", "4", "--key-file", str(key),
                                       "--no-source-labels", "--allow-unresolved", "--out-dir", str(tmp_path / "c")],
                               provider=_nli)
    e = rep["epochs"][0]
    assert e["stats"]["worlds"] == 2 and e["substrate"]["world_spec"]["sha256"] and e["repeated_items"] == 1
