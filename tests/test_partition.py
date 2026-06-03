"""C-1 multi-host: a REAL, healable network partition via tc (not iptables).

Avoids a weak "partition" that is a single-broker `docker
stop` (not an isolation) that swallows failures. This controller installs a
symmetric `tc netem loss 100%` toward the PEER IP only (never blanket loss -- that
would also kill the orchestrator's own SSH), verifies the rule landed, heals by
removing it, and RAISES on any failure (inject-must-raise). iptables is
off-limits on the testbed; tc only.

Mirrors a proven tc pattern (setup-wan.sh / multi_vm_runner): the VMs' enp1s0
carries the kernel-default `fq_codel` root, so a prio root cannot simply be ADDED on
top -- the root must be `del`'d first, then `add`'d, and heal restores the default by
`del root`. The FakeRunner models that fq_codel root faithfully so this test catches
the add-without-del bug that a (noqueue-root) container smoke cannot.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


class FakeRunner:
    """Models tc state on a real iface: a default `fq_codel` root that must be del'd
    before a `prio` root can be added; a netem loss band needs the prio root present."""
    def __init__(self, fail_substr=None, initial_root="fq_codel"):
        self.calls = []
        self.fail_substr = fail_substr
        self._initial_root = initial_root
        self.root = {}      # host -> "fq_codel" | "prio" | None
        self.loss = {}      # host -> bool
        self.filters = {}   # host -> set of dst IPs currently dropped (del root clears them)

    def _root(self, host):
        return self.root.get(host, self._initial_root)

    def __call__(self, host, argv):
        self.calls.append((host, list(argv)))
        cmd = " ".join(argv)
        if self.fail_substr and self.fail_substr in cmd:
            return SimpleNamespace(returncode=2, stdout="", stderr="tc: forced fail")
        if "qdisc" in cmd and "del" in cmd and "root" in cmd:
            self.root[host] = None          # del root -> back to no qdisc (kernel reinstalls default)
            self.loss[host] = False
            self.filters[host] = set()      # del root removes ALL u32 filters at once
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "qdisc" in cmd and "add" in cmd and "root" in cmd and "prio" in cmd:
            if self._root(host) is not None:    # a root already exists -> add fails
                return SimpleNamespace(returncode=2, stdout="", stderr="RTNETLINK: File exists")
            self.root[host] = "prio"
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "add" in cmd and "parent" in cmd and "loss 100%" in cmd:
            if self._root(host) != "prio":      # netem band needs the prio root present
                return SimpleNamespace(returncode=2, stdout="", stderr="Parent qdisc missing")
            self.loss[host] = True
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "filter" in cmd and "add" in cmd:
            if "dst" in argv:                       # record the dropped dst IP (strip /32)
                dst = argv[argv.index("dst") + 1].split("/")[0]
                self.filters.setdefault(host, set()).add(dst)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "show" in cmd:
            r = self._root(host)
            out = ("qdisc prio 1: root " if r == "prio" else f"qdisc {r or 'noqueue'} 0: root ")
            if self.loss.get(host):
                out += "qdisc netem 30: ... loss 100% "
            return SimpleNamespace(returncode=0, stdout=out, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def _seq(calls, host):
    return [" ".join(argv) for h, argv in calls if h == host]


def test_partition_clears_root_then_installs_symmetric_loss_no_iptables():
    from smp.sync.partition import PartitionController
    r = FakeRunner(initial_root="fq_codel")     # the real VM condition
    pc = PartitionController(r, iface="enp1s0")
    pc.partition("hostA", "hostB", ip_a="10.0.0.1", ip_b="10.0.0.2")

    a = _seq(r.calls, "hostA")
    # the NPS pattern: `del root` MUST precede `add root ... prio` (else add fails on fq_codel)
    del_i = next(i for i, c in enumerate(a) if "del" in c and "root" in c)
    add_i = next(i for i, c in enumerate(a) if "add" in c and "root" in c and "prio" in c)
    assert del_i < add_i
    assert any("loss 100%" in c and "add" in c for c in a)       # loss installed
    assert any("10.0.0.2" in c for c in a)                       # A targets B's ip
    assert any("10.0.0.1" in c for c in _seq(r.calls, "hostB"))  # B targets A's ip (symmetric)
    assert r.loss["hostA"] and r.loss["hostB"]                   # active on both
    assert all(argv[0] != "iptables" for _h, argv in r.calls)    # tc only, never iptables


def test_isolate_egress_one_sided_drops_toward_peer_only():
    # one-sided egress drop (VM1 -> peer) -- enough to break the bidirectional TCP of an
    # HTTP publish without needing tc on the remote host (no inter-VM ssh for the ext-smoke).
    from smp.sync.partition import PartitionController
    r = FakeRunner(initial_root="fq_codel")
    pc = PartitionController(r, iface="enp1s0")
    pc.isolate_egress("hostA", "10.0.0.2")
    a = _seq(r.calls, "hostA")
    del_i = next(i for i, c in enumerate(a) if "del" in c and "root" in c)
    add_i = next(i for i, c in enumerate(a) if "add" in c and "root" in c and "prio" in c)
    assert del_i < add_i                                  # del-root-first (fq_codel)
    assert any("loss 100%" in c for c in a) and any("10.0.0.2" in c for c in a)
    assert r.loss["hostA"] is True
    assert "hostB" not in {h for h, _ in r.calls}         # ONLY hostA touched


def test_heal_egress_restores_and_verifies():
    from smp.sync.partition import PartitionController
    r = FakeRunner(initial_root="fq_codel")
    pc = PartitionController(r, iface="enp1s0")
    pc.isolate_egress("hostA", "10.0.0.2")
    pc.heal_egress("hostA", "10.0.0.2")
    assert r.loss["hostA"] is False and r.root["hostA"] is None


def test_partition_raises_on_tc_failure():
    from smp.sync.partition import PartitionController
    pc = PartitionController(FakeRunner(fail_substr="loss 100%"), iface="enp1s0")
    with pytest.raises(RuntimeError):
        pc.partition("hostA", "hostB", ip_a="10.0.0.1", ip_b="10.0.0.2")


def test_verify_partition_reads_tc_show():
    from smp.sync.partition import PartitionController
    r = FakeRunner()
    r.root["hostA"] = "prio"; r.loss["hostA"] = True
    pc = PartitionController(r, iface="enp1s0")
    assert pc.verify_partition("hostA", "10.0.0.2") is True
    assert pc.verify_partition("hostB", "10.0.0.1") is False     # no loss on B


def test_heal_del_roots_and_restores_default():
    from smp.sync.partition import PartitionController
    r = FakeRunner(initial_root="fq_codel")
    pc = PartitionController(r, iface="enp1s0")
    pc.partition("hostA", "hostB", ip_a="10.0.0.1", ip_b="10.0.0.2")
    pc.heal("hostA", "hostB", ip_a="10.0.0.1", ip_b="10.0.0.2")
    assert r.loss["hostA"] is False and r.loss["hostB"] is False
    assert r.root["hostA"] is None and r.root["hostB"] is None   # root cleared -> kernel default
    # heal must `del root` on both hosts
    assert any("del" in c and "root" in c for c in _seq(r.calls, "hostA"))
    assert any("del" in c and "root" in c for c in _seq(r.calls, "hostB"))


def test_heal_egress_tolerates_default_only_root_del_failure():
    # C2: heal an UNSHAPED host (kernel-default root) -> `del root` returns nonzero
    # ("No such file"); heal_egress must treat verify-clean as success (verify is
    # authoritative), not raise.
    from smp.sync.partition import PartitionController

    class DefaultOnlyRunner(FakeRunner):
        def __call__(self, host, argv):
            cmd = " ".join(argv)
            if "del" in cmd and "root" in cmd:
                return SimpleNamespace(returncode=2, stdout="",
                                       stderr="RTNETLINK answers: No such file or directory")
            return super().__call__(host, argv)

    pc = PartitionController(DefaultOnlyRunner(initial_root="fq_codel"), iface="enp1s0")
    pc.heal_egress("hostA", "10.0.0.2")     # must NOT raise (no loss present -> verify clean)


def test_isolate_egress_multi_one_del_root_filters_every_peer():
    # the 2|2 split-brain case: ONE host must drop egress to SEVERAL peers at once. Repeated
    # single-peer isolate_egress would del-root each call and WIPE the prior peer's filter
    # (leaving only the last) -- the bug this primitive fixes. One del-root, N filters.
    from smp.sync.partition import PartitionController
    r = FakeRunner(initial_root="fq_codel")
    pc = PartitionController(r, iface="enp1s0")
    pc.isolate_egress_multi("hostA", ["10.0.0.3", "10.0.0.4"])
    a = _seq(r.calls, "hostA")
    assert sum(1 for c in a if "del" in c and "root" in c) == 1     # del root EXACTLY once
    assert sum(1 for c in a if "add" in c and "root" in c and "prio" in c) == 1
    assert r.filters["hostA"] == {"10.0.0.3", "10.0.0.4"}          # BOTH peers dropped
    assert r.loss["hostA"] is True


def test_repeated_single_isolate_would_wipe_but_multi_does_not():
    # regression guard documenting the bug: two single-peer isolates leave only the LAST
    # filter (each del-roots); the multi primitive retains both.
    from smp.sync.partition import PartitionController
    r1 = FakeRunner(initial_root="fq_codel")
    pc1 = PartitionController(r1, iface="enp1s0")
    pc1.isolate_egress("hostA", "10.0.0.3")
    pc1.isolate_egress("hostA", "10.0.0.4")
    assert r1.filters["hostA"] == {"10.0.0.4"}                     # only the last survived (the bug)
    r2 = FakeRunner(initial_root="fq_codel")
    pc2 = PartitionController(r2, iface="enp1s0")
    pc2.isolate_egress_multi("hostA", ["10.0.0.3", "10.0.0.4"])
    assert r2.filters["hostA"] == {"10.0.0.3", "10.0.0.4"}         # the fix keeps both


def test_heal_host_clears_whole_tree_and_verifies_clean():
    from smp.sync.partition import PartitionController
    r = FakeRunner(initial_root="fq_codel")
    pc = PartitionController(r, iface="enp1s0")
    pc.isolate_egress_multi("hostA", ["10.0.0.3", "10.0.0.4"])
    pc.heal_host("hostA")
    assert r.loss["hostA"] is False and r.root["hostA"] is None
    assert r.filters["hostA"] == set()                            # all peer filters gone
    assert any("del" in c and "root" in c for c in _seq(r.calls, "hostA"))


def test_heal_host_tolerates_unshaped_default_only_root():
    # heal a host that was never shaped (default-only root): del returns nonzero harmlessly,
    # verify is authoritative -> must NOT raise.
    from smp.sync.partition import PartitionController

    class DefaultOnlyRunner(FakeRunner):
        def __call__(self, host, argv):
            cmd = " ".join(argv)
            if "del" in cmd and "root" in cmd:
                return SimpleNamespace(returncode=2, stdout="",
                                       stderr="RTNETLINK answers: No such file or directory")
            return super().__call__(host, argv)

    pc = PartitionController(DefaultOnlyRunner(initial_root="fq_codel"), iface="enp1s0")
    pc.heal_host("hostA")           # must not raise


def test_heal_raises_if_partition_persists():
    from smp.sync.partition import PartitionController

    class StuckRunner(FakeRunner):
        def __call__(self, host, argv):
            was_loss = self.loss.get(host, False)        # only sabotage a HEAL del (loss up)
            res = super().__call__(host, argv)
            if was_loss and "del" in " ".join(argv) and "root" in " ".join(argv):
                self.root[host] = "prio"; self.loss[host] = True   # heal del did not take
            return res
    pc = PartitionController(StuckRunner(initial_root="fq_codel"), iface="enp1s0")
    pc.partition("hostA", "hostB", ip_a="10.0.0.1", ip_b="10.0.0.2")
    with pytest.raises(RuntimeError):
        pc.heal("hostA", "hostB", ip_a="10.0.0.1", ip_b="10.0.0.2")
