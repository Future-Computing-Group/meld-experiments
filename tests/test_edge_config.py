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
