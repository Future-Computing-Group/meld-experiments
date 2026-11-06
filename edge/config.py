"""Load the edge-testbed topology + grid from `testbed.yaml`.

All host aliases, IPs, SSH targets, ports, image tags, and the netem RTT /
partition grid live in the (gitignored) config file — never in the orchestrator
code. Copy `testbed.example.yaml` -> `testbed.yaml` and fill in your nodes.
"""
from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit("pyyaml is required to read testbed.yaml: pip install pyyaml") from exc

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "testbed.yaml"


def load_testbed(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Read and lightly validate the testbed config.

    Returns the parsed dict plus convenience maps: ``ids`` (logical node ids in
    order), ``host`` (id -> SSH target), ``ip`` (id -> address), ``url`` (id ->
    brain http base).
    """
    cfg_path = Path(path) if path else Path(os.environ.get("SMP_TESTBED", DEFAULT_CONFIG))
    if not cfg_path.is_file():
        raise SystemExit(
            f"testbed config not found: {cfg_path}\n"
            "  cp testbed.example.yaml testbed.yaml and fill in your nodes."
        )
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    nodes = cfg.get("nodes") or []
    if not nodes:
        raise SystemExit(f"{cfg_path}: at least one entry under `nodes` is required")
    port = int(cfg.get("brain_port", 8080))

    wd_port = int(cfg.setdefault("watchdog_port", 8090))
    cfg["ids"] = [n["id"] for n in nodes]
    cfg["host"] = {n["id"]: n["host"] for n in nodes}
    cfg["ip"] = {n["id"]: n["ip"] for n in nodes}
    # brains reach their peers directly on the brain port (the port a partition drops); the
    # driver reaches each brain through that host's watchdog (/brain/<path>), which stays up
    cfg["peer_url"] = {n["id"]: f"http://{n['ip']}:{port}" for n in nodes}
    cfg["wd_url"] = {n["id"]: n.get("watchdog_url", f"http://{n['ip']}:{wd_port}") for n in nodes}
    cfg["url"] = {n["id"]: n.get("url", cfg["wd_url"][n["id"]] + "/brain") for n in nodes}
    cfg.setdefault("orchestrator", cfg["ids"][0])
    cfg.setdefault("iface", "eth0")
    cfg.setdefault("images", {})
    cfg.setdefault("grid", {})
    cfg.setdefault("partition", {})
    return cfg


def _host_path(p: str) -> str:
    """A path on the brain host for the remote shell: ``~/`` becomes "$HOME"/ (docker -v needs an
    absolute path), the rest is quoted."""
    return '"$HOME"/' + shlex.quote(p[2:]) if p.startswith("~/") else shlex.quote(p)


def render_start(cfg: Dict[str, Any], node_id: str, *, routing: str, encoder: str) -> str:
    """The shell command that starts ``node_id``'s brain (deploy.sh runs it on the host).

    The hashing encoder is the smoke rung: smoke image, lexical NLI, no epoch. Any other encoder
    is a result path: the campaign image, the configured ``nli`` (lexical refused unless the
    config says ``smoke: true``) and the configured ``epoch_file``, mounted read-only. Every
    brain gets the run key as a read-only file (``key_file``, never an env value) and the
    configured ``http_timeout_s``. Semantic routing takes ``grid.top_k`` and ``grid.sigma_lo``."""
    if routing not in ("broadcast", "semantic"):
        raise ValueError(f"unknown routing {routing!r}")
    node = next(n for n in cfg["nodes"] if n["id"] == node_id)
    port, smoke = int(cfg.get("brain_port", 8080)), encoder == "hashing"
    nli = "lexical" if smoke else cfg.get("nli")
    if not nli or (nli == "lexical" and not smoke and cfg.get("smoke") is not True):
        raise ValueError("nli: a result path needs the deployed NLI model, not lexical (set smoke: true for a smoke run)")
    if not cfg.get("key_file"):
        raise ValueError("key_file is required: the testbed run key, never the default test key")
    image = cfg.get("images", {}).get("smoke" if smoke else "campaign",
                                      "meld-brain:smoke" if smoke else "meld-brain:minilm")
    env = {"SMP_NODE_ID": node_id, "SMP_PORT": port, "SMP_ENCODER": encoder, "SMP_NLI": nli,
           "SMP_ROUTING_MODE": routing, "SMP_GROUP_KEY_FILE": "/run/meld/key",
           "SMP_HTTP_TIMEOUT": cfg.get("http_timeout_s", 2),
           "SMP_PEERS": ",".join(cfg["peer_url"][p] for p in cfg["ids"] if p != node_id)}
    vols = [f"{_host_path(cfg['key_file'])}:/run/meld/key:ro"]
    if not smoke:
        if cfg.get("epoch_file"):
            vols.append(f"{_host_path(cfg['epoch_file'])}:/run/meld/epoch.json:ro")
            env["SMP_EPOCH"] = "/run/meld/epoch.json"
        elif cfg.get("smoke") is not True:
            raise ValueError("epoch_file is required on a result path")
    if routing == "semantic":
        grid = cfg.get("grid", {})
        if "sigma_lo" not in grid:
            raise ValueError("grid.sigma_lo is required for semantic routing (no silent default)")
        env["SMP_TOP_K"], env["SMP_SIGMA_LO"] = grid.get("top_k", 1), grid["sigma_lo"]
    name = node.get("container", "meld-brain")
    net = cfg.get("docker_network", "host")
    netargs = (["--network=host"] if net == "host" else
               [f"--network {net}", f"--ip {node['ip']}", "--cap-add NET_ADMIN"]
               + ([f"-p 127.0.0.1:{node['publish']}:{port}"] if node.get("publish") else []))
    return (f"docker rm -f {name} >/dev/null 2>&1 || true; docker run -d --name {name} "
            + " ".join(netargs + [f"-v {v}" for v in vols]
                       + [f"-e {k}={shlex.quote(str(v))}" for k, v in env.items()]) + f" {image}")


if __name__ == "__main__":      # python config.py start-cmd NODE ROUTING ENCODER
    if sys.argv[1:2] != ["start-cmd"] or len(sys.argv) != 5:
        raise SystemExit("usage: config.py start-cmd NODE ROUTING ENCODER")
    print(render_start(load_testbed(), sys.argv[2], routing=sys.argv[3], encoder=sys.argv[4]))
