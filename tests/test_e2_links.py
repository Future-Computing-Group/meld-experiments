"""E2 link evaluation (eval/e2_links.py): read modes for typed-link value (E2-T2), the stale-fact
surfacing proxy, paired contrasts, and the structural epochs and seed-cluster CIs of the
missed-link diagnostic (E2-T1). Fake signal providers (hashing encoder, lexical NLI) only.
Run:  PYTHONPATH=src python -m pytest tests/test_e2_links.py -q
"""
import importlib.util
import json
import os
import random
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from meld.agent.memory import SMPMemory
from meld.config import save_epoch
from meld.eval import e2_links as L
from meld.sync.wire_delta import group_key

from test_agent_meld_memory import agent_epoch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXTS = ["The river Thames flows through London", "London is the capital of England",
         "Paris lies on the Seine", "Berlin has many museums", "The Seine flows through Paris"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name[:-3], os.path.join(REPO, "scripts", name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mem():
    m = SMPMemory(encoder="hashing", nli="lexical", epoch=agent_epoch(theta=0.5))
    for t in TEXTS:
        m.write(t)
    return m


def _key(mem, text):
    return next(d.claim_key for _, d in mem.store.nodes().values() if d.text == text)


def test_typed_mode_is_the_store_read_and_node_only_is_beta_graph_zero():
    mem = _mem()
    store = mem.store
    q = "Which river flows through London?"
    assert L.ModeReader(mem, "typed").recall(q, k=5) == mem.recall(q, k=5)
    texts = mem.texts()
    zero = [texts[u] for u, _ in store.graph_walk_read(q, encoder="hashing", k=5, beta_graph=0.0)]
    assert L.ModeReader(mem, "node_only").recall(q, k=5) == zero
    assert "resolved_links" not in vars(store)            # the read leaves the store untouched


def test_a_link_moves_rank_only_in_the_modes_that_carry_it():
    mem = _mem()
    src, dst = _key(mem, TEXTS[0]), _key(mem, "Berlin has many museums")
    mem.store.derived.links = [("CONTRADICTS", src, dst, 1.0)]     # test fixture link
    q = "Which river flows through London?"
    typed = L.ModeReader(mem, "typed").recall(q, k=5)
    node = L.ModeReader(mem, "node_only").recall(q, k=5)
    assert typed.index("Berlin has many museums") < node.index("Berlin has many museums")
    # one link: its type-erased weight is its own weight, so types-ignored equals typed here
    assert L.ModeReader(mem, "untyped").recall(q, k=5) == typed


def test_rewire_preserves_degrees_types_and_weights_without_self_loops():
    keys = [bytes([i]) * 16 for i in range(8)]
    rng = random.Random(1)
    links = [(rng.choice(["ASSOCIATES", "CONTRADICTS"]), *rng.sample(keys, 2), rng.random())
             for _ in range(30)]
    out = L.rewire(links, random.Random(7))
    assert Counter((t, s, w) for t, s, _, w in out) == Counter((t, s, w) for t, s, _, w in links)
    assert Counter(d for _, _, d, _ in out) == Counter(d for _, _, d, _ in links)
    assert all(s != d for _, s, d, _ in out)
    assert out != links and out == L.rewire(links, random.Random(7))


def test_untyped_erases_types_and_keeps_total_mass():
    links = [("ASSOCIATES", b"a", b"b", 0.2), ("CONTRADICTS", b"b", b"c", 1.0)]
    store = SimpleNamespace(resolved_links=lambda: links)
    out = L.mode_links(store, "untyped")
    assert {t for t, *_ in out} == {"LINK"} and [x[1:3] for x in out] == [x[1:3] for x in links]
    assert sum(x[3] for x in out) == pytest.approx(1.2)
    assert L.mode_links(store, "node_only") == [] and L.mode_links(store, "typed") == links
    with pytest.raises(ValueError):
        L.mode_links(store, "bogus")


def _ku(old, new):
    return {"question_type": "knowledge-update", "question": "Where do I live now?",
            "haystack_dates": ["2023/01/01", "2023/06/01"],
            "haystack_sessions": [[{"role": "user", "content": old, "has_answer": True}],
                                  [{"role": "user", "content": new, "has_answer": True}]]}


def test_stale_proxy_flags_only_through_a_typed_contradicts_link():
    mem = SMPMemory(encoder="hashing", nli="lexical")
    old, new = "I live in Oulu", "I moved to Helsinki"
    for t in (old, new, "I like tea"):
        mem.publish(t)
    ko, kn = _key(mem, old), _key(mem, new)
    item = _ku(old, new)
    reader = SimpleNamespace(mem=mem, recall=lambda q, k: [old, "I like tea"][:k])
    contra = [("CONTRADICTS", kn, ko, 1.0)]
    flagged = L.stale_proxy(reader, item, 2, {}, contra)
    assert flagged == {"unflagged_stale": 0.0, "stale_preferred": 1.0, "flagged": 1.0}
    bare = L.stale_proxy(reader, item, 2, {}, [("LINK", kn, ko, 1.0)])
    assert bare == {"unflagged_stale": 1.0, "stale_preferred": 1.0, "flagged": 0.0}
    both = SimpleNamespace(mem=mem, recall=lambda q, k: [new, old])
    assert L.stale_proxy(both, item, 2, {}, []) == {"unflagged_stale": 0.0, "stale_preferred": 0.0,
                                                     "flagged": 0.0}
    assert L.stale_proxy(reader, {**item, "question_type": "multi-session"}, 2, {}, []) is None


def test_paired_contrast_bootstrap_and_dz():
    rows = [{"typed": 0.5 + 0.1 * (i % 2), "node_only": 0.5} for i in range(20)]
    c = L.paired(rows, "typed", "node_only")
    assert c["n"] == 20 and c["delta"] == pytest.approx(0.05)
    assert c["lo"] <= c["delta"] <= c["hi"] and c["dz"] == pytest.approx(0.05 / 0.0513, rel=1e-2)
    assert L.paired([{"typed": None, "node_only": 1.0}], "typed", "node_only")["n"] == 0


def test_structural_epochs_vary_only_m_and_name_distinct_dev_roots():
    from meld.eval.order_shuffle import e2t1, gold_workload
    from meld.protocol.fold import nli_provider
    from meld.sync.node import BrainNode
    lo = round(0.3 * 65536)
    src = agent_epoch(theta=0.5, sel=[lo, lo, 3])
    eps = L.structural_epochs(src, (1, 3, 5))
    assert [e.m for e in eps] == [1, 3, 5] and len({e.cal_root for e in eps}) == 3
    assert all(replace(e, m=3, cal_root=src.cal_root) == src for e in eps)
    deltas, roles = gold_workload("hashing", n_seeds=3)
    out = e2t1(deltas, roles, eps, scope="agent", make_node=lambda ep: BrainNode(
        "sink", encoder="hashing", nli="lexical", epoch=ep, provider=nli_provider("lexical")))
    ci = L.seed_cluster_ci(out[1]["per_seed"], n_boot=200)
    for typ in ("MERGE", "CONFLICT", "RELATE"):
        assert ci[typ]["miss_rate"] == pytest.approx(out[1]["miss_rate"][typ])
        assert 0.0 <= ci[typ]["lo"] <= ci[typ]["miss_rate"] <= ci[typ]["hi"] <= 1.0


def test_e2t2_runner_smoke_hotpot_fixture(tmp_path):
    ep = tmp_path / "ep.json"
    save_epoch(agent_epoch(theta=0.5), group_key({}), str(ep))
    out = tmp_path / "e2t2.json"
    rep = _load("run_e2t2_links.py").main(["--substrate", "hotpot", "--data", "fixture", "--n", "2",
                                           "--ks", "1", "5", "--encoder", "hashing", "--nli", "lexical",
                                           "--epoch", str(ep), "--out", str(out)])
    assert json.loads(out.read_text())["n_worlds"] == 2 == len(rep["rows"])
    for k in ("1", "5"):
        for mode in L.MODES:
            assert 0.0 <= rep["summary"][k]["recall"][mode] <= 1.0
        assert set(rep["summary"][k]["contrasts"]) == {"typed-node_only", "typed-rewired", "typed-untyped"}
    assert rep["epoch_id"] and rep["rows"][0]["links"] is not None


def test_e2t2_runner_smoke_lme_reports_the_stale_proxy(tmp_path):
    from meld.eval.substrates import pool
    items = []
    for i in (i for i in range(20) if pool(f"q{i}") == "eval"):
        it = _ku(f"My cat is named Tom {i}", f"My cat is now named Max {i}")
        it.update(question_id=f"q{i}", haystack_session_ids=["s0", "s1"])
        items.append(it)
    data = tmp_path / "lme.json"
    data.write_text(json.dumps(items))
    rep = _load("run_e2t2_links.py").main(["--substrate", "lme", "--data", str(data), "--n", "3",
                                           "--ks", "1", "--encoder", "hashing", "--nli", "lexical",
                                           "--out", str(tmp_path / "o.json")])
    s = rep["summary"]["1"]
    assert s["stale"]["n"] > 0 and set(s["stale"]["unflagged_stale"]) == set(L.MODES)


def test_e2t1_runner_smoke(tmp_path):
    src = tmp_path / "src.json"
    lo = round(0.3 * 65536)
    save_epoch(agent_epoch(theta=0.5, sel=[lo, lo, 3]), group_key({}), str(src))
    rep = _load("run_e2t1_structural.py").main(["--source-epoch", str(src), "--ms", "1", "3",
                                                "--encoder", "hashing", "--nli", "lexical",
                                                "--n-seeds", "2", "--n-boot", "100",
                                                "--out-dir", str(tmp_path / "e2t1")])
    assert sorted(rep["by_m"]) == [1, 3] and rep["theta_source_epoch_id"]
    assert set(rep["by_m"][1]["ci"]) == {"MERGE", "CONFLICT", "RELATE"}
    assert len(list((tmp_path / "e2t1").glob("*.epoch.json"))) == 2
