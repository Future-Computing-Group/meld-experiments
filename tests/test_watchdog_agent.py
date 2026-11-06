"""The per-host watchdog as the partition and shaping agent (no inner ssh between hosts).

The driver asks each host's own watchdog over HTTP to install a partition (POST /isolate), a netem
delay (POST /shape) or to heal (POST /heal); the daemon applies tc locally through the tested
PartitionController / WanShaper, then reads the installed u32 filters back and refuses (500) unless
they are exactly the requested peer /32s. The stale-heartbeat teardown also clears a shaped host.
The container the watchdog removes is the one the deploy path starts.
"""
from __future__ import annotations

import ipaddress
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


class FakeTc:
    """tc state on one host: root qdisc, one netem band (loss or delay), u32 dst filters."""

    def __init__(self, extra_filter=None):
        self.calls, self.root, self.netem, self.filters = [], "fq_codel", "", []
        self.extra_filter = extra_filter

    def __call__(self, argv):
        self.calls.append(list(argv))
        cmd = " ".join(argv)
        ok = SimpleNamespace(returncode=0, stdout="", stderr="")
        if "qdisc del" in cmd and "root" in cmd:
            self.root, self.netem, self.filters = None, "", []
        elif "qdisc add" in cmd and "prio" in cmd:
            if self.root is not None:
                return SimpleNamespace(returncode=2, stdout="", stderr="File exists")
            self.root = "prio"
        elif "qdisc add" in cmd and "netem" in cmd:
            self.netem = " ".join(argv[argv.index("netem") + 1:])
        elif "filter add" in cmd:
            self.filters.append(argv[argv.index("dst") + 1].split("/")[0])
            if self.extra_filter:
                self.filters.append(self.extra_filter)
        elif "qdisc show" in cmd:
            ok.stdout = f"qdisc {self.root or 'noqueue'} 1: root " + (f"qdisc netem 30: parent 1:3 {self.netem}" if self.netem else "")
        elif "filter show" in cmd:
            ok.stdout = "".join(f"filter parent 1: protocol ip pref 3 u32 chain 0 fh 800::80{i} order 2048 key ht 800 "
                                f"bkt 0 flowid 1:3\n  match {int(ipaddress.ip_address(ip)):08x}/ffffffff at 16\n"
                                for i, ip in enumerate(self.filters))
        return ok


def _daemon(tc, clk, downs):
    from meld.sync.partition import PartitionController
    from meld.sync.wan import WanShaper
    from meld.sync.watchdog import Watchdog, WatchdogDaemon
    wd = Watchdog(iface="eth0", heartbeat_timeout_s=30, runner=tc, clock=lambda: clk[0],
                  compose_down=lambda: downs.append(True))
    host_runner = lambda _host, argv: tc(argv)          # noqa: E731 - local tc only
    return WatchdogDaemon(wd, partition=PartitionController(host_runner, iface="eth0"),
                          shaper=WanShaper(host_runner, iface="eth0"))


def _post(d, path, body):
    s, b = d.handle("POST", path, json.dumps(body).encode())
    return s, json.loads(b)


def test_isolate_installs_loss_toward_exactly_the_requested_peers_and_heal_clears_it():
    tc, clk, downs = FakeTc(), [0.0], []
    d = _daemon(tc, clk, downs)
    s, b = _post(d, "/isolate", {"peer_ips": ["10.0.0.3", "10.0.0.4"]})
    assert s == 200 and sorted(b["filters"]) == ["10.0.0.3", "10.0.0.4"]
    assert "loss 100%" in tc.netem
    s, b = _post(d, "/heal", {})
    assert s == 200 and tc.root is None and tc.filters == []
    assert downs == []                                 # a heal never removes the brain


def test_shape_installs_delay_toward_every_requested_peer_under_one_root():
    tc, clk, downs = FakeTc(), [0.0], []
    d = _daemon(tc, clk, downs)
    s, b = _post(d, "/shape", {"peer_ips": ["10.0.0.2", "10.0.0.3"], "delay_ms": 25})
    assert s == 200 and sorted(b["filters"]) == ["10.0.0.2", "10.0.0.3"]
    assert "delay 25" in tc.netem
    assert sum("del" in c and "root" in c for c in tc.calls) == 1


def test_a_filter_beyond_the_requested_peers_is_refused_and_cleared():
    tc, clk, downs = FakeTc(extra_filter="10.9.9.9"), [0.0], []
    d = _daemon(tc, clk, downs)
    s, b = _post(d, "/isolate", {"peer_ips": ["10.0.0.3"]})
    assert s == 500 and tc.root is None and tc.filters == []


@pytest.mark.parametrize("body", [{"peer_ips": "10.0.0.3"}, {"peer_ips": ["10.0.0.3; reboot"]},
                                  {"peer_ips": ["10.0.0.0/8"]}, {"peer_ips": []},
                                  {}])
def test_malformed_requests_are_400_and_touch_nothing(body):
    tc, clk, downs = FakeTc(), [0.0], []
    d = _daemon(tc, clk, downs)
    for path in ("/isolate", "/shape"):
        assert _post(d, path, body)[0] == 400
    assert _post(d, "/shape", {"peer_ips": ["10.0.0.3"], "delay_ms": "x"})[0] == 400
    assert tc.calls == []


def test_stale_heartbeat_clears_a_shaped_host_then_removes_the_brain_and_refuses_new_rules():
    tc, clk, downs = FakeTc(), [0.0], []
    d = _daemon(tc, clk, downs)
    d.handle("POST", "/heartbeat")
    assert _post(d, "/shape", {"peer_ips": ["10.0.0.2"], "delay_ms": 1})[0] == 200
    clk[0] = 31.0
    assert d.check_once() is True
    assert tc.root is None and tc.filters == [] and downs == [True]
    assert _post(d, "/isolate", {"peer_ips": ["10.0.0.2"]})[0] == 409     # no rules after teardown


def test_watchdog_removes_the_container_the_deploy_path_starts():
    import importlib.util
    from meld.sync.watchdog import DEFAULT_CONTAINER
    spec = importlib.util.spec_from_file_location("edge_config", REPO / "edge" / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.load_testbed(REPO / "edge" / "testbed.example.yaml")
    started = {m for n in cfg["ids"] for enc in ("hashing", "minilm")
               for m in re.findall(r"--name ([\w-]+)", mod.render_start(cfg, n, routing="broadcast", encoder=enc))}
    stopped = set(re.findall(r"docker rm -f ([\w-]+)", (REPO / "edge" / "deploy.sh").read_text()))
    assert started == stopped == {DEFAULT_CONTAINER} == {"meld-brain"}


def test_isolate_scopes_the_partition_to_the_brain_port():
    tc, clk, downs = FakeTc(), [0.0], []
    d = _daemon(tc, clk, downs)
    d.brain_port = 8080
    assert _post(d, "/isolate", {"peer_ips": ["10.0.0.3"]})[0] == 200
    assert any("dport" in c and "8080" in c for c in tc.calls if "filter" in c)


def test_brain_requests_are_forwarded_to_the_local_brain_on_the_unpartitioned_port():
    tc, clk, downs = FakeTc(), [0.0], []
    d = _daemon(tc, clk, downs)
    seen = []
    d.forward = lambda method, path, body: seen.append((method, path, body)) or (200, b'{"ok": 1}')
    assert d.handle("POST", "/brain/patch", b"x") == (200, b'{"ok": 1}')
    assert d.handle("GET", "/brain/metrics") == (200, b'{"ok": 1}')
    assert seen == [("POST", "/patch", b"x"), ("GET", "/metrics", b"")]


_PRIO_ROOT = ["tc", "qdisc", "add", "dev", "eth0", "root", "handle", "1:", "prio",
              "bands", "3", "priomap", *["0"] * 16]


@pytest.mark.parametrize("path,body,netem", [
    ("/isolate", {"peer_ips": ["10.0.0.3", "10.0.0.4"]}, ["netem", "loss", "100%"]),
    ("/shape", {"peer_ips": ["10.0.0.3", "10.0.0.4"], "delay_ms": 25}, ["netem", "delay", "25ms"]),
])
def test_prio_root_sends_every_unfiltered_packet_to_the_unshaped_band(path, body, netem):
    # The default prio priomap (1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1) puts TC_PRIO_BULK traffic in band 2
    # = 1:3, the netem band, so unfiltered traffic (management, proxy, other hosts) would be
    # shaped too. The root must carry an all-zeros priomap: only the /32 u32 filters reach 1:3.
    tc, clk, downs = FakeTc(), [0.0], []
    d = _daemon(tc, clk, downs)
    s, _b = _post(d, path, body)
    assert s == 200
    adds = [c for c in tc.calls if c[1] in ("qdisc", "filter") and c[2] == "add"]
    assert adds[0] == _PRIO_ROOT
    assert adds[1] == ["tc", "qdisc", "add", "dev", "eth0", "parent", "1:3", "handle", "30:", *netem]
    assert all(c[1] == "filter" and c[-2:] == ["flowid", "1:3"] for c in adds[2:]) and len(adds) == 4
    assert d.wd._shaping_present()                     # the read-back still sees the netem band
    assert _post(d, "/heal", {})[0] == 200 and not d.wd._shaping_present()
