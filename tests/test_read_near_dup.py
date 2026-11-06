"""Near-duplicate-suppressing read (wp6c-nd, r1.13): every recall metric walks the store's full
fused ranking, skips a unit whose replicated int8 embedding q is a near-duplicate of an already
listed unit's (exact integer test dot^2 >= tau^2 |a|^2 |b|^2, dot >= 0; no encoder, no float), and
cuts at k. tau = 0.95 is fixed in advance; tau None is the reported wp6c read
(the store's top k, mapped to sources). The rule is the same for MELD, the ablations, the E2-T2
read modes and the baselines, because all of them read through recall_metrics.recalled_sources."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib

import pytest

from meld.agent.memory import SMPMemory
import numpy as np

from meld.agent.recall_metrics import READ_SIM, READ_TAU, near_dup, recalled_sources
from meld.baselines.memory import BaselineMemory, Verdict
from meld.eval.e2_links import ModeReader

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENC, NLI = "hashing", "lexical"

FACT = ("the orbital survey mapped crater ridges basalt plains lava tubes dust dunes polar ice "
        "caps impact basins canyon walls volcanic vents and sulfur deposits")
COPY = "Note: " + FACT                         # a surface paraphrase copy (cosine about 0.98)
OTHER = "the orbital survey mapped crater ridges near the equator last spring"
THIRD = "a separate mission sampled basalt plains and lava tubes"
QUERY = "what did the orbital survey map: crater ridges basalt plains lava tubes dust dunes"
SOURCE = {COPY: FACT}


class KeepAll:
    """A baseline policy that keeps every arriving claim (no merge, no retirement)."""
    name, k = "keep-all", 3

    def __call__(self, text, retrieved):
        return Verdict("INSERT", "ADD", add=(text,))


def _fill(mem, via="publish"):
    for t in (FACT, COPY, OTHER, THIRD):
        getattr(mem, via)(t)
    return mem


def _memories():
    yield "smp", _fill(SMPMemory(encoder=ENC, nli=NLI))
    yield "baseline", _fill(BaselineMemory(KeepAll(), encoder=ENC, nli=NLI), "write")
    yield "e2t2-typed", ModeReader(_fill(SMPMemory(encoder=ENC, nli=NLI)), "typed")


def test_default_tau_is_fixed_in_advance():
    assert READ_TAU == 0.95


@pytest.mark.parametrize("label,mem", list(_memories()), ids=lambda x: x if isinstance(x, str) else "")
def test_copies_crowd_the_old_read_and_suppression_lists_the_next_distinct_fact(label, mem):
    top2 = mem.recall(QUERY, k=2)
    assert set(top2) == {FACT, COPY}                                   # both copies take the slots
    assert recalled_sources(mem, QUERY, 2, SOURCE, tau=None) == [FACT]  # one distinct fact in k = 2
    new = recalled_sources(mem, QUERY, 2, SOURCE)                      # default tau 0.95
    assert new[0] == FACT and len(new) == 2 and new[1] == mem.recall(QUERY, k=3)[2]
    assert recalled_sources(mem, QUERY, 2, SOURCE, tau=0.95) == new


@pytest.mark.parametrize("label,mem", list(_memories()), ids=lambda x: x if isinstance(x, str) else "")
def test_tau_none_or_above_one_is_the_old_read(label, mem):
    for k in (1, 2, 3, 4, 10):
        old = list(dict.fromkeys(SOURCE.get(t, t) for t in mem.recall(QUERY, k=k)))
        assert recalled_sources(mem, QUERY, k, SOURCE, tau=None) == old
        assert recalled_sources(mem, QUERY, k, SOURCE, tau=1.01) == old


def test_suppression_never_lists_more_than_k_and_fills_k_when_distinct_units_exist():
    mem = _fill(SMPMemory(encoder=ENC, nli=NLI))
    for k in (1, 2, 3):
        assert len(recalled_sources(mem, QUERY, k, SOURCE)) == k
    assert len(recalled_sources(mem, QUERY, 10, SOURCE)) == 3          # 4 units, 3 distinct


# -- runners: tau stamped, tau None reproduces the reported read exactly ------------------------


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_nd", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lme_item(qid):
    turn = lambda c, ans=False: {"role": "user", "content": c, "has_answer": ans}  # noqa: E731
    return {"question_id": qid, "question_type": "knowledge-update",
            "question": "What is my 5K personal best time?", "answer": "25 minutes",
            "haystack_session_ids": ["a", "b", "c", "d"],
            "haystack_dates": ["2023/05/02 (Tue) 12:00", "2023/05/01 (Mon) 09:00",
                               "2023/05/02 (Tue) 09:00", "2023/05/03 (Wed) 09:00"],
            "haystack_sessions": [[turn("My 5K personal best time is now 25 minutes.", True)],
                                  [turn("I bought new running shoes for my 5K today.")],
                                  [turn("My 5K personal best time is 27 minutes.", True)],
                                  [turn("I want to bake bread this weekend."), turn("My best time to bake is 5 pm.")]]}


def _runs(tmp_path, extra):
    """Recall values of run_nbrains (LongMemEval toy worlds, HotpotQA fixture) and run_r2_overlap_k
    (smoke) under ``extra`` read arguments; returns (digest of the values, the rows)."""
    data = tmp_path / "lme.json"
    data.write_text(json.dumps([_lme_item("q1"), _lme_item("q2")]))
    nb = _load("run_nbrains")
    common = ["--encoder", ENC, "--nli", NLI, "--dev-theta", "0.5", "--ks", "1", "2", "5", "10"]
    lme = nb.main(["--data", str(data), "--ns", "3", "8", "--ds", "1", "2", "8", *common, *extra,
                   "--cache-dir", str(tmp_path / "c"), "--out", str(tmp_path / f"lme{len(extra)}.jsonl")])
    hot = nb.main(["--workload", "hotpot", "--data", "fixture", "--per-store", "2", "--n-items", "4",
                   "--ns", "3", "--ds", "1", "2", *common, *extra,
                   "--cache-dir", str(tmp_path / "c"), "--out", str(tmp_path / f"hot{len(extra)}.jsonl")])
    out = tmp_path / f"r2{len(extra)}.json"
    _load("run_r2_overlap_k").main(["--smoke", "--encoder", ENC, "--nli", NLI, *extra,
                                    "--out", str(out), "--cache-dir", str(tmp_path / "c")])
    r2 = [json.loads(x) for x in (tmp_path / f"r2{len(extra)}.rows.jsonl").read_text().splitlines()]
    vals = sorted([[r["world"], r["n_brains"], r["dup"], r["condition"], r["topology"],
                    {k: v for k, v in r.items() if k.startswith(("recall@", "stale@"))}]
                   for r in [*lme.values(), *hot.values()]], key=json.dumps)
    vals += sorted([[r["overlap"], r["condition"], r["task_id"], r["k"], r["recall"]] for r in r2],
                   key=json.dumps)
    rows = [*lme.values(), *hot.values()], json.loads(out.read_text())["cells"], r2
    return hashlib.sha256(json.dumps(vals).encode()).hexdigest(), rows


# digest of the same runs at 51dc162 (before the read change), i.e. the reported wp6c read
OLD_READ_DIGEST = "2f22cb48d7d63ac358579a8f03a8bc9bb16f5a17d247893ced7e2aa9a7836b24"


def test_runners_with_tau_none_reproduce_the_reported_read(tmp_path):
    digest, _ = _runs(tmp_path, ["--read-tau", "none"])
    assert digest == OLD_READ_DIGEST


def test_runners_stamp_tau_and_default_to_the_fixed_value(tmp_path):
    _, (nb_rows, cells, r2_rows) = _runs(tmp_path, [])
    assert all(r["read_tau"] == 0.95 and r["read_sim"] == READ_SIM for r in nb_rows)
    assert all(c["read_tau"] == 0.95 and c["read_sim"] == READ_SIM for c in cells.values())
    assert all(r["read_tau"] == 0.95 and r["read_sim"] == READ_SIM for r in r2_rows)
    (tmp_path / "none").mkdir()
    _, (nb_rows, cells, r2_rows) = _runs(tmp_path / "none", ["--read-tau", "none"])
    assert all(r["read_tau"] is None for r in [*nb_rows, *cells.values(), *r2_rows])


def test_e2t2_and_r2_lme_stamp_tau(tmp_path):
    rep = _load("run_e2t2_links").main(["--substrate", "hotpot", "--data", "fixture", "--n", "2",
                                        "--ks", "1", "5", "--encoder", ENC, "--nli", NLI, "--n-boot", "10",
                                        "--cache-dir", str(tmp_path / "c"), "--out", str(tmp_path / "e.json")])
    assert rep["read_tau"] == 0.95 and rep["read_sim"] == READ_SIM
    data = tmp_path / "lme.json"
    data.write_text(json.dumps([_lme_item(f"q{i}") for i in range(3)]))
    out = tmp_path / "r2.json"
    _load("run_r2_lme").main(["--smoke", "--encoder", ENC, "--nli", NLI, "--data", str(data),
                              "--read-tau", "none", "--out", str(out), "--cache-dir", str(tmp_path / "c")])
    assert all(c["read_tau"] is None and c["read_sim"] == READ_SIM
               for c in json.loads(out.read_text())["cells"].values())


# -- r1.13: the decision reads replicated q only, exactly ----------------------------------------


class _Poison:
    def __getattr__(self, name):
        raise AssertionError(f"the read touched the memory's encoder ({name})")


@pytest.mark.parametrize("label,mem", list(_memories()), ids=lambda x: x if isinstance(x, str) else "")
def test_decision_reads_replicated_q_not_the_encoder(label, mem):
    before = [recalled_sources(mem, QUERY, k, SOURCE) for k in (1, 2, 3, 10)]
    inner = getattr(mem, "mem", mem)
    inner._es = _Poison()
    assert [recalled_sources(mem, QUERY, k, SOURCE) for k in (1, 2, 3, 10)] == before


def _q(*v):
    return np.array(v, dtype=np.int8).tobytes()


def test_boundary_is_exact_integer():
    a = _q(20, 0, 0, 0, 0)
    b = _q(19, 5, 3, 2, 1)                       # |b| = 20, dot = 380: cosine exactly 19/20
    assert 400 * 380 ** 2 == 361 * 400 * 400
    assert near_dup(a, b, 0.95) and near_dup(b, a, 0.95)
    assert not near_dup(a, _q(19, 5, 3, 2, 2), 0.95)   # one unit off: |b|^2 = 403, below
    assert not near_dup(a, _q(-20, 0, 0, 0, 0), 0.95)  # dot < 0 never counts
    assert not near_dup(a, _q(0, 0, 0, 0, 0), 0.95)    # a zero q has no direction
    assert near_dup(a, a, 1.0) and not near_dup(a, b, 1.0)


def test_read_sim_constant():
    assert READ_SIM == "int8-q"


def test_reader_without_replicated_q_is_refused():
    from types import SimpleNamespace
    brain = SimpleNamespace(recall=lambda q, k: ["x", "y"][:k])
    with pytest.raises(TypeError, match="replicated"):
        recalled_sources(brain, QUERY, 2)
    assert recalled_sources(brain, QUERY, 2, tau=None) == ["x", "y"]
