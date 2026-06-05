"""Load the edge-testbed topology + grid from `testbed.yaml`.

All host aliases, IPs, SSH targets, ports, image tags, and the netem RTT /
partition grid live in the (gitignored) config file — never in the orchestrator
code. Copy `testbed.example.yaml` -> `testbed.yaml` and fill in your nodes.
"""
from __future__ import annotations

import os
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

    cfg["ids"] = [n["id"] for n in nodes]
    cfg["host"] = {n["id"]: n["host"] for n in nodes}
    cfg["ip"] = {n["id"]: n["ip"] for n in nodes}
    cfg["url"] = {n["id"]: f"http://{n['ip']}:{port}" for n in nodes}
    cfg.setdefault("orchestrator", cfg["ids"][0])
    cfg.setdefault("iface", "eth0")
    cfg.setdefault("watchdog_port", 8090)
    cfg.setdefault("images", {})
    cfg.setdefault("grid", {})
    cfg.setdefault("partition", {})
    return cfg
