"""Result runners: each refuses to start without explicit models (no default model on a result
path; the lexical NLI and hashing encoder are test fixtures, passed explicitly here), and every
output row carries the provenance stamps of eval/topology.stamps (git sha, M_nli, M_pol, epoch id
and seq, NLI and encoder ids), per-decision rows also the epoch bin. Plus the top-k coverage of the
gold workload in the order-shuffle and lossy harnesses."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

from meld.config import save_epoch
from meld.eval import topology
from meld.sync.wire_delta import group_key

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from test_agent_meld_memory import agent_epoch  # noqa: E402

ENC, NLI = "hashing", "lexical"
STAMP = {"git_sha", "m_nli", "m_pol", "epoch_id", "seq", "nli_id", "encoder"}


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_stamps", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stamped(row, nli=NLI, enc=ENC, epoch=None):
    assert STAMP <= set(row), STAMP - set(row)
    assert row["git_sha"] and row["nli_id"] == nli and row["encoder"] == enc
    assert row["epoch_id"] == (topology.stamps(ENC, NLI, epoch)["epoch_id"] if epoch else None)
    assert row["seq"] == (epoch.seq if epoch else None)


def _epoch_file(tmp_path, epoch=None):
    f = tmp_path / "epoch.json"
    save_epoch(epoch or agent_epoch(ENC, NLI), group_key({}), str(f))
    return str(f)


# -- no default model on a result path -------------------------------------------------------------

MODELS = ["--encoder", ENC, "--nli", NLI]
RUNNERS = {
    "run_order_shuffle": ["--workload", "running", *MODELS],
    "run_anti_entropy_loss": MODELS,
    "run_r2_overlap_k": ["--smoke", *MODELS],
    "run_r2_lme": ["--smoke", *MODELS],
    "run_index_scaling": ["--corpus", "synthetic", "--sizes", "10", *MODELS],   # stamps: test_candidate_index
    "run_wire_bytes": MODELS,
    "run_baseline_pilot": ["--policy", "graphiti", "--fake", *MODELS],
    "run_h4_determinism": ["deltas", "--key-file", "k", "--out", "o", *MODELS],
    "run_nbrains": ["--out", "o", *MODELS],
    "run_sink_load": ["--smoke", *MODELS],
    "run_scale_sweep": ["--smoke", "--encoder", ENC],
    "run_m2_edge_cost": MODELS,
    "run_calibration": ["--domain", "paws", "--alpha", "0.05", "--out-dir", "o", *MODELS],
    "run_mem0_fidelity": ["--rescore", "x", "--out", "o", "--embedder", "e"],
}


def _without(argv, flag):
    i = argv.index(flag)
    return argv[:i] + argv[i + 2:]


@pytest.mark.parametrize("name", sorted(RUNNERS))
def test_runner_refuses_to_start_without_explicit_models(name, capsys):
    argv = RUNNERS[name]
    for flag in ("--encoder", "--nli", "--embedder"):
        if flag in argv:
            with pytest.raises(SystemExit) as e:
                _load(name).main(_without(argv, flag))
            assert e.value.code == 2 and flag in capsys.readouterr().err


def test_no_result_runner_hard_codes_the_lexical_stand_in():
    for name in RUNNERS:
        src = (ROOT / "scripts" / f"{name}.py").read_text()
        assert '"lexical"' not in src and "'lexical'" not in src, name
    assert not (ROOT / "scripts" / "run_memdelta.py").exists()


# -- stamps per runner -----------------------------------------------------------------------------

def test_order_shuffle_stamps_and_gold_topk(tmp_path):
    from dataclasses import replace
    ep = agent_epoch(ENC, NLI, theta=0.7)
    e2 = [replace(ep, m=m, cal_root=bytes([m]) * 16) for m in (1, 3)]
    files = []
    for e in e2:
        files.append(str(tmp_path / f"e{e.m}.json"))
        save_epoch(e, group_key({}), files[-1])
    rep = _load("run_order_shuffle").main(["--workload", "gold", "--k", "2", *MODELS,
                                            "--epoch", _epoch_file(tmp_path, ep), "--e2t1-epochs", *files])
    _stamped(rep, epoch=ep)
    for e in e2:
        _stamped(rep["e2t1"][e.m], epoch=e)
    from meld.eval.order_shuffle import gold_queries
    assert len(gold_queries()) == 26 and rep["topk_distinct"] == dict.fromkeys(gold_queries(), 1)


def test_order_shuffle_detects_a_topk_divergence_on_the_gold_workload():
    import random
    from meld.eval.order_shuffle import gold_queries, gold_workload, run_order_shuffle
    from meld.protocol.objects import FIXED_ONE as F
    from meld.sync.node import BrainNode
    deltas, _ = gold_workload(ENC)
    epoch = agent_epoch(ENC, NLI, theta=0.5)

    def jitter(seed):           # a nondeterministic provider: merges differ from order to order
        rng = random.Random(seed)

        def p(a, b):
            e = rng.choice((int(0.45 * F), int(0.55 * F)))
            return (0, e, F - e)
        return p
    rep = run_order_shuffle([], deltas, k=6, seed=7, scope="agent", probe_queries=gold_queries(),
                            make_node=lambda i: BrainNode("sink", encoder=ENC, nli=NLI, epoch=epoch,
                                                          provider=jitter(i)))
    assert rep["distinct_fingerprints"] > 1 and max(rep["topk_distinct"].values()) > 1


def test_lossy_harness_reads_topk_and_detects_a_divergence():
    from meld.eval.anti_entropy_eval import lossy_cell
    from meld.eval.order_shuffle import gold_queries, gold_workload
    from meld.sync.experiment import node_factory
    deltas, roles = gold_workload(ENC)
    make = node_factory(encoder=ENC, nli=NLI)
    kw = dict(make_node=make, n=3, loss=0.3, seed=0, scope="agent", probe_queries=gold_queries())
    healed = lossy_cell(deltas, **kw)
    assert healed["topk_distinct"] == dict.fromkeys(gold_queries(), 1)
    lossy = lossy_cell(deltas, anti_entropy=False, **kw)       # replicas hold different deltas
    assert max(lossy["topk_distinct"].values()) > 1


def test_anti_entropy_loss_rows_are_stamped(tmp_path, monkeypatch):
    ae = _load("run_anti_entropy_loss")
    monkeypatch.setattr(ae, "LOSSES", [0.05])
    monkeypatch.setattr(ae, "SEEDS", [0])
    ep = agent_epoch(ENC, NLI)
    rows = ae.main([*MODELS, "--n", "3", "--epoch", _epoch_file(tmp_path, ep)])["rows"]
    _stamped(rows[0], epoch=ep)
    assert rows[0]["with_ae_topk_distinct_max"] == 1


def test_r2_overlap_k_cells_are_stamped(tmp_path):
    out = tmp_path / "r2.json"
    _load("run_r2_overlap_k").main(["--smoke", *MODELS, "--out", str(out), "--cache-dir", str(tmp_path / "c")])
    rep = json.loads(out.read_text())
    assert rep["cells"] and all(_stamped(c) is None for c in rep["cells"].values())


def test_r2_overlap_k_writes_stamped_per_task_rows_that_reproduce_the_cells(tmp_path):
    """PH-4: per-task rows (one per overlap, condition, task, k) beside the aggregate, so paired
    intervals and d_z can be formed; their per-cell means are the cell means."""
    out = tmp_path / "r2.json"
    _load("run_r2_overlap_k").main(["--smoke", *MODELS, "--out", str(out), "--cache-dir", str(tmp_path / "c")])
    rep = json.loads(out.read_text())
    rows = [json.loads(x) for x in (tmp_path / "r2.rows.jsonl").read_text().splitlines()]
    assert len(rows) == rep["n_tasks"] * len(rep["overlaps"]) * len(rep["conditions"]) * len(rep["ks"])
    assert all(_stamped(r) is None for r in rows)
    for key, cell in rep["cells"].items():
        sub = [r for r in rows if f"{r['overlap']}|{r['condition']}|{r['k']}" == key]
        assert len({r["task_id"] for r in sub}) == len(sub) == cell["n"]
        assert round(sum(r["recall"] for r in sub) / len(sub), 4) == cell["mean_recall"]
        assert round(sum(r["live_units"] for r in sub) / len(sub), 2) == cell["mean_live_units"]


def test_r2_lme_cells_are_stamped(tmp_path):
    item = {"question_id": "q", "question": "Where do I live?", "answer": "Oslo",
            "question_type": "single-session-user", "haystack_session_ids": ["s1", "s2"],
            "haystack_sessions": [[{"role": "user", "content": "I live in Oslo now", "has_answer": True}],
                                  [{"role": "user", "content": "I like tea"}]]}
    data = tmp_path / "lme.json"
    data.write_text(json.dumps([dict(item, question_id=f"q{i}") for i in range(3)]))
    out = tmp_path / "r2.json"
    _load("run_r2_lme").main(["--smoke", *MODELS, "--data", str(data), "--out", str(out),
                              "--cache-dir", str(tmp_path / "c")])
    rep = json.loads(out.read_text())
    assert rep["cells"] and all(_stamped(c) is None for c in rep["cells"].values())


def test_wire_bytes_sections_are_stamped():
    rep = _load("run_wire_bytes").main([*MODELS, "--n", "2", "--losses", "0.05"])
    for row in (rep["running_example"], rep["gold"], rep["anti_entropy"], *rep["anti_entropy"]["lossy"]):
        _stamped(row)


def test_baseline_pilot_rows_are_stamped_with_a_bin():
    res = _load("run_baseline_pilot").main(["--policy", "graphiti", "--fake", "--n", "3", *MODELS])
    for r in res["rows"]:
        _stamped(r)
        assert "bin" in r and r["bin"] is None          # the baseline memory holds no epoch


def test_mem0_fidelity_rows_are_stamped(tmp_path):
    rows = [{"new": "Alice lives in Lyon", "old": [], "facts": [], "prompt": None, "results": [],
             "replies": ['{"facts": []}']}]
    src = tmp_path / "in.json"
    src.write_text(json.dumps({"summary": {}, "rows": rows}))
    out = tmp_path / "out.json"
    _load("run_mem0_fidelity").main(["--rescore", str(src), "--out", str(out), "--embedder", "e5"])
    rep = json.loads(out.read_text())
    for r in (rep["summary"], *rep["rows"]):     # Mem0's own embedder; no MELD NLI or policy
        assert STAMP <= set(r) and r["encoder"] == "e5" and r["nli_id"] is None and r["m_pol"] is None


def test_h4_records_are_stamped_and_decisions_carry_a_bin(tmp_path):
    import test_h4_determinism as T
    H = T.H
    ref = T._ref()
    ep = H.epoch_of(ref, "gold")
    _stamped(ref["gold"], epoch=ep)
    cells = {"a": T._cell(ref, "a")}
    _stamped(cells["a"]["gold"], epoch=ep)
    assert {d[-1] for d in cells["a"]["gold"]["decisions1"]} == {0}      # 104 nodes, all below B_0
    p2 = T._p2(ref, cells, ["a"])
    _stamped(p2["a"]["gold"], epoch=ep)
    assert {d[-1] for d in p2["a"]["gold"]["decisions2"]} == {0}
    _stamped(H.report(ref, cells, p2, "a", key=T.KEY, encoder=T.ENC, nli_name=T.NLI_NAME)["gold"], epoch=ep)


def test_sink_load_rows_are_stamped(tmp_path):
    out = tmp_path / "s.json"
    _load("run_sink_load").main(["--smoke", *MODELS, "--out", str(out), "--cache-dir", str(tmp_path / "c")])
    assert all(_stamped(r) is None for r in json.loads(out.read_text())["rows"])


def test_scale_sweep_rows_are_stamped(tmp_path):
    out = tmp_path / "s.json"
    _load("run_scale_sweep").main(["--smoke", "--encoder", ENC, "--per-domain", "2", "--out", str(out)])
    for r in json.loads(out.read_text())["rows"]:
        assert STAMP <= set(r) and r["encoder"] == ENC and r["nli_id"] is None


def test_m2_edge_cost_is_stamped(tmp_path):
    out = tmp_path / "m.json"
    _load("run_m2_edge_cost").main([*MODELS, "--n", "3", "--out", str(out)])
    _stamped(json.loads(out.read_text()))
