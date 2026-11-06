"""Smoke tests for the generic edge-testbed orchestrator scaffolding (edge/).

The orchestrator ships as generic, config-driven scripts + a committed, sanitized
example topology; the real topology (edge/testbed.yaml) is gitignored. These tests pin:
the example config loads through the real loader, exposes the expected convenience maps,
uses only placeholder hosts/IPs, and the scripts are importable / syntactically valid.
"""
from __future__ import annotations

import importlib.util
import py_compile
import shutil
import subprocess
from pathlib import Path

import pytest

EDGE = Path(__file__).resolve().parents[1] / "edge"


def _load_config_module():
    spec = importlib.util.spec_from_file_location("edge_config", EDGE / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_example_topology_loads_and_builds_maps():
    cfg = _load_config_module().load_testbed(EDGE / "testbed.example.yaml")
    assert cfg["ids"], "no nodes parsed"
    # every node has an SSH target, an IP, and a derived brain URL
    for nid in cfg["ids"]:
        assert nid in cfg["host"] and nid in cfg["ip"]
        assert cfg["url"][nid].startswith("http://")
    # the grid + partition cells the orchestrators consume are present
    assert {"rtt_ms", "n_nodes", "routings", "seeds"} <= set(cfg["grid"])
    assert {"mode", "seeds"} <= set(cfg["partition"])
    assert cfg["orchestrator"] in cfg["ids"]


def test_example_topology_is_placeholders_only():
    cfg = _load_config_module().load_testbed(EDGE / "testbed.example.yaml")
    for nid in cfg["ids"]:
        host, ip = cfg["host"][nid], cfg["ip"][nid]
        assert host.endswith(".example.invalid"), f"non-placeholder host: {host}"
        assert ip.startswith("10.0.0."), f"non-placeholder ip: {ip}"
    # (the internal-token scan over all tracked files is the hygiene guard's job.)


@pytest.mark.parametrize("script", ["config.py", "run_campaign.py", "run_eval_c.py"])
def test_python_orchestrators_compile(script):
    py_compile.compile(str(EDGE / script), doraise=True)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_deploy_script_is_valid_bash():
    r = subprocess.run(["bash", "-n", str(EDGE / "deploy.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --- one deploy path: the start command rendered from the config (deploy.sh and the drivers) ---

def _fixture_cfg(tmp_path, **over):
    import yaml
    cfg = {"iface": "eth0", "brain_port": 8080, "watchdog_port": 8090,
           "images": {"smoke": "meld-brain:smoke", "campaign": "meld-brain:minilm"},
           "nli": "mnli-cross-encoder", "key_file": "~/run/run.key",
           "epoch_file": "~/run/epoch/hotpot.group.epoch.json", "http_timeout_s": 2,
           "grid": {"sigma_lo": 0.12, "top_k": 1},
           "nodes": [{"id": "A", "host": "a.example.invalid", "ip": "10.0.0.1"},
                     {"id": "B", "host": "b.example.invalid", "ip": "10.0.0.2"}]}
    cfg.update(over)
    p = tmp_path / "testbed.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return _load_config_module().load_testbed(p)


def test_start_command_carries_the_deployed_models_key_epoch_and_timeout(tmp_path):
    mod = _load_config_module()
    cmd = mod.render_start(_fixture_cfg(tmp_path), "A", routing="semantic", encoder="minilm")
    for want in ("SMP_NLI=mnli-cross-encoder", "SMP_ENCODER=minilm", "SMP_EPOCH=/run/meld/epoch.json",
                 "SMP_GROUP_KEY_FILE=/run/meld/key", "SMP_HTTP_TIMEOUT=2", "SMP_TOP_K=1",
                 "SMP_SIGMA_LO=0.12", "SMP_PEERS=http://10.0.0.2:8080", "--name meld-brain",
                 "meld-brain:minilm", '"$HOME"/run/run.key:/run/meld/key:ro'):
        assert want in cmd, (want, cmd)
    assert "SMP_GROUP_KEY=" not in cmd                  # the key never travels as an env value


def test_lexical_is_refused_on_a_result_path_and_the_smoke_rung_runs_without_an_epoch(tmp_path):
    import pytest
    mod = _load_config_module()
    with pytest.raises(ValueError, match="lexical"):
        mod.render_start(_fixture_cfg(tmp_path, nli="lexical"), "A", routing="broadcast", encoder="minilm")
    with pytest.raises(ValueError, match="epoch"):
        mod.render_start(_fixture_cfg(tmp_path, epoch_file=None), "A", routing="broadcast", encoder="minilm")
    with pytest.raises(ValueError, match="key"):
        mod.render_start(_fixture_cfg(tmp_path, key_file=None), "A", routing="broadcast", encoder="minilm")
    smoke = mod.render_start(_fixture_cfg(tmp_path), "A", routing="broadcast", encoder="hashing")
    assert "SMP_NLI=lexical" in smoke and "SMP_EPOCH" not in smoke and "meld-brain:smoke" in smoke
    ok = mod.render_start(_fixture_cfg(tmp_path, nli="lexical", smoke=True, epoch_file=None), "A",
                          routing="broadcast", encoder="minilm")
    assert "SMP_NLI=lexical" in ok


def test_driver_reaches_brains_through_the_watchdog_and_brains_reach_peers_directly(tmp_path):
    cfg = _fixture_cfg(tmp_path)
    assert cfg["url"]["B"] == "http://10.0.0.2:8090/brain"
    assert cfg["peer_url"]["B"] == "http://10.0.0.2:8080"
    assert cfg["wd_url"]["B"] == "http://10.0.0.2:8090"


def test_deploy_script_starts_brains_with_the_rendered_command():
    text = (EDGE / "deploy.sh").read_text()
    assert "render_start" in text or "start-cmd" in text
    assert "SMP_NLI=lexical" not in text


def test_service_reads_the_group_key_from_a_file(tmp_path):
    from meld.sync.service import build_service_from_env
    (tmp_path / "k").write_text("a-testbed-run-key\n")
    svc, _ = build_service_from_env({"SMP_NODE_ID": "A", "SMP_PORT": "8800", "SMP_ENCODER": "hashing",
                                     "SMP_NLI": "lexical", "SMP_GROUP_KEY_FILE": str(tmp_path / "k")})
    assert svc.node.key == b"a-testbed-run-key"
    import pytest
    (tmp_path / "k").write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        build_service_from_env({"SMP_NODE_ID": "A", "SMP_PORT": "8800", "SMP_ENCODER": "hashing",
                                "SMP_NLI": "lexical", "SMP_GROUP_KEY_FILE": str(tmp_path / "k")})
