"""Shared runner arguments: the group key file and the receive-path baseline arms."""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

from meld.agent.llm import FakeChatJSON
from meld.config import read_epoch, save_epoch
from meld.eval.runner_args import read_key_file, runner_key
from meld.sync.wire_delta import group_key
from test_agent_meld_memory import agent_epoch
from test_baselines import _addall

ENC, NLI = "hashing", "lexical"
BINARY_KEY = bytes(range(256))[::7] + b"\xff\xfe"           # not UTF-8: cannot travel in the env
SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _keyed_epoch(tmp_path):
    kf = tmp_path / "group.key"
    kf.write_bytes(BINARY_KEY + b"\n")
    ep = tmp_path / "e.epoch.json"
    save_epoch(agent_epoch(ENC, NLI), BINARY_KEY, str(ep))
    return str(kf), str(ep)


def test_binary_key_file_is_read_stripped_and_verifies_an_epoch(tmp_path, monkeypatch):
    monkeypatch.delenv("SMP_GROUP_KEY", raising=False)
    kf, ep = _keyed_epoch(tmp_path)
    assert read_key_file(kf) == BINARY_KEY
    assert read_epoch(ep, runner_key(kf)).seq == 1
    assert runner_key(None) == group_key({})                 # no key file: the previous default


def test_missing_empty_and_doubled_keys_are_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("SMP_GROUP_KEY", raising=False)
    with pytest.raises(SystemExit, match="key file"):
        read_key_file(str(tmp_path / "absent.key"))
    (tmp_path / "empty.key").write_bytes(b" \n")
    with pytest.raises(SystemExit, match="empty"):
        read_key_file(str(tmp_path / "empty.key"))
    kf, _ = _keyed_epoch(tmp_path)
    monkeypatch.setenv("SMP_GROUP_KEY", "abc")
    with pytest.raises(SystemExit, match="mutually exclusive") as e:
        runner_key(kf)
    assert BINARY_KEY.hex() not in str(e.value)


RUNNERS = {
    "run_anti_entropy_loss": ["--encoder", ENC, "--nli", NLI, "--epoch", "x"],
    "run_order_shuffle": ["--encoder", ENC, "--nli", NLI, "--workload", "gold", "--epoch", "x"],
    "run_sink_load": ["--encoder", ENC, "--nli", NLI, "--epoch", "x"],
    "run_nbrains": ["--encoder", ENC, "--nli", NLI, "--data", "x", "--out", "x", "--epoch", "x"],
    "run_r2_overlap_k": ["--encoder", ENC, "--nli", NLI, "--epoch", "x"],
    "run_r2_lme": ["--encoder", ENC, "--nli", NLI, "--epoch", "x"],
    "run_calibration": ["--manifest", "x", "--domain", "paws", "--alpha", "0.10", "--encoder", ENC,
                        "--nli", NLI, "--out-dir", "x"],
    "run_h4_determinism": ["deltas", "--encoder", ENC, "--nli", NLI, "--out", "x"],
}


@pytest.mark.parametrize("name", sorted(RUNNERS))
def test_every_keyed_runner_takes_key_file_and_refuses_a_missing_one(name, tmp_path, monkeypatch):
    monkeypatch.delenv("SMP_GROUP_KEY", raising=False)
    with pytest.raises(SystemExit, match="key file"):
        _script(name).main(RUNNERS[name] + ["--key-file", str(tmp_path / "absent")])


def test_r2_overlap_runs_a_baseline_arm_under_a_binary_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SMP_GROUP_KEY", raising=False)
    kf, ep = _keyed_epoch(tmp_path)
    out = tmp_path / "r2.json"
    llm = FakeChatJSON(_addall)
    _script("run_r2_overlap_k").main(
        ["--smoke", "--encoder", ENC, "--nli", NLI, "--epoch", ep, "--key-file", kf,
         "--baseline", "mem0-update", "--llm-model", "m-x", "--cache-dir", str(tmp_path / "c"),
         "--out", str(out)], llm=llm)
    res = json.loads(out.read_text())
    assert "mem0-update" in res["conditions"] and llm.calls
    base = [v for k, v in res["cells"].items() if k.split("|")[1] == "mem0-update"]
    assert base and all(v["baseline"] == "mem0-update" and v["llm_model"] == "m-x" for v in base)
    assert all("baseline" not in v for k, v in res["cells"].items() if k.split("|")[1] != "mem0-update")
    assert all(v["seq"] == 1 for v in base)                     # the epoch verified under the file key
    assert BINARY_KEY.hex() not in capsys.readouterr().out


def test_r2_baseline_needs_an_endpoint(tmp_path):
    with pytest.raises(SystemExit, match="llm-base-url"):
        _script("run_r2_lme").main(["--smoke", "--encoder", ENC, "--nli", NLI, "--baseline", "graphiti",
                                    "--out", str(tmp_path / "o.json")])


def test_nbrains_runs_a_baseline_arm_on_the_star(tmp_path):
    from test_nbrains import _item
    data = tmp_path / "lme.json"
    data.write_text(json.dumps([_item()]))
    llm = FakeChatJSON(_addall)
    rows = list(_script("run_nbrains").main(
        ["--data", str(data), "--encoder", ENC, "--nli", NLI, "--ns", "3", "--ds", "1",
         "--conditions", "meld", "--ks", "5", "--dev-theta", "0.5", "--baseline", "mem0-update",
         "--llm-model", "m-x", "--cache-dir", str(tmp_path / "c"), "--out", str(tmp_path / "o.jsonl")],
        llm=llm).values())
    base = [r for r in rows if r["condition"] == "mem0-update"]
    assert len(base) == 1 and base[0]["topology"] == "star" and llm.calls
    assert base[0]["baseline"] == "mem0-update" and base[0]["llm_model"] == "m-x"
    assert base[0]["stored"] >= 1
    assert sum(base[0]["decisions"].values()) >= 1 and "recall@5" in base[0]
    assert all("baseline" not in r for r in rows if r["condition"] == "meld")
