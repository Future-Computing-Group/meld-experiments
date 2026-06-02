"""eval-s WAN-RTT regimes: tc netem DELAY toward a peer IP (the edge<->cloud crossover).

Mirrors the validated PartitionController tc pattern (setup-wan.sh): del-root-first
(the VMs carry a default fq_codel root), prio root, netem delay band, u32 dst filter,
verify + raise. clear() del-root's back to the kernel default. The FakeRunner
models the fq_codel root so this catches the add-without-del bug a container can't.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


class FakeRunner:
    def __init__(self, fail_substr=None, initial_root="fq_codel"):
        self.calls = []
        self.fail_substr = fail_substr
        self._initial_root = initial_root
        self.root = {}
        self.delay = {}

    def _root(self, host):
        return self.root.get(host, self._initial_root)

    def __call__(self, host, argv):
        self.calls.append((host, list(argv)))
        cmd = " ".join(argv)
        if self.fail_substr and self.fail_substr in cmd:
            return SimpleNamespace(returncode=2, stdout="", stderr="tc: forced fail")
        if "qdisc" in cmd and "del" in cmd and "root" in cmd:
            self.root[host] = None; self.delay[host] = False
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "qdisc" in cmd and "add" in cmd and "root" in cmd and "prio" in cmd:
            if self._root(host) is not None:
                return SimpleNamespace(returncode=2, stdout="", stderr="File exists")
            self.root[host] = "prio"
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "add" in cmd and "parent" in cmd and "delay" in cmd:
            if self._root(host) != "prio":
                return SimpleNamespace(returncode=2, stdout="", stderr="Parent missing")
            self.delay[host] = True
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "filter" in cmd and "add" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "show" in cmd:
            r = self._root(host)
            out = ("qdisc prio 1: root " if r == "prio" else f"qdisc {r or 'noqueue'} 0: root ")
            if self.delay.get(host):
                out += "qdisc netem 30: ... delay 50.0ms "
            return SimpleNamespace(returncode=0, stdout=out, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def _seq(calls, host):
    return [" ".join(argv) for h, argv in calls if h == host]


def test_shape_clears_root_then_adds_delay_toward_peer():
    from smp.sync.wan import WanShaper
    r = FakeRunner(initial_root="fq_codel")
    w = WanShaper(r, iface="enp1s0")
    w.shape("hostA", "10.0.0.2", delay_ms=50, jitter_ms=5)
    a = _seq(r.calls, "hostA")
    del_i = next(i for i, c in enumerate(a) if "del" in c and "root" in c)
    add_i = next(i for i, c in enumerate(a) if "add" in c and "root" in c and "prio" in c)
    assert del_i < add_i                                  # del-root-first (fq_codel)
    assert any("delay 50ms" in c for c in a)              # configured delay present
    assert any("5ms" in c for c in a)                     # jitter present
    assert any("10.0.0.2" in c for c in a)                # toward the peer
    assert r.delay["hostA"] is True
    assert all(argv[0] != "iptables" for _h, argv in r.calls)   # tc only


def test_shape_raises_on_tc_failure():
    from smp.sync.wan import WanShaper
    w = WanShaper(FakeRunner(fail_substr="delay"), iface="enp1s0")
    with pytest.raises(RuntimeError):
        w.shape("hostA", "10.0.0.2", delay_ms=25)


def test_clear_del_roots_and_verifies():
    from smp.sync.wan import WanShaper
    r = FakeRunner(initial_root="fq_codel")
    w = WanShaper(r, iface="enp1s0")
    w.shape("hostA", "10.0.0.2", delay_ms=25)
    w.clear("hostA")
    assert r.delay["hostA"] is False and r.root["hostA"] is None
    assert any("del" in c and "root" in c for c in _seq(r.calls, "hostA"))


def test_clear_tolerates_default_only_root_del_failure():
    # C2: on an UNSHAPED host (kernel-default fq_codel root, no explicit root), `tc qdisc
    # del dev X root` returns nonzero ("No such file"). clear() must treat a verify-clean
    # iface as success (verify is authoritative), NOT raise -- else every RTT=0 cell fails
    # and aborts cleanup, leaking shaping on later cells.
    from smp.sync.wan import WanShaper

    class DefaultOnlyRunner(FakeRunner):
        def __call__(self, host, argv):
            cmd = " ".join(argv)
            if "del" in cmd and "root" in cmd:
                return SimpleNamespace(returncode=2, stdout="",
                                       stderr="RTNETLINK answers: No such file or directory")
            return super().__call__(host, argv)

    w = WanShaper(DefaultOnlyRunner(initial_root="fq_codel"), iface="enp1s0")
    w.clear("hostA")        # must NOT raise (no delay present -> verify clean)


def test_verify_delay_reads_tc_show():
    from smp.sync.wan import WanShaper
    r = FakeRunner()
    r.root["hostA"] = "prio"; r.delay["hostA"] = True
    w = WanShaper(r, iface="enp1s0")
    assert w.verify_delay("hostA") is True
    assert w.verify_delay("hostB") is False
