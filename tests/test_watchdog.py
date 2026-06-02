"""Per-VM watchdog: the shared-infra safety prerequisite for any testbed run.

Each VM runs a watchdog that listens for the orchestrator's heartbeat. If the beat
goes stale (orchestrator crashed / lost contact mid-run), the watchdog AUTONOMOUSLY
restores the host: it clears ALL tc shaping on the measured iface FIRST (so no stale
`netem loss`/`delay` lingers on the shared enp1s0 -- the blast-radius concern), VERIFIES
the iface is clean and RAISES if shaping persists (the safety guarantee must be
real, not best-effort-and-swallowed), and only THEN brings the compose stack down.

This is the mechanism that makes unattended runs admissible under the shared-infra rule
(no automated rollback -> do not run unattended). Built + tested before the testbed gate so
it is ready the moment the infra owner signs off. ``runner(argv) -> result`` (local tc exec) and
``compose_down()`` are the seams; mocked here, real at local-docker smoke.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


class FakeRunner:
    """Models tc shaping state; `qdisc del root` clears it (unless told otherwise)."""
    def __init__(self, cleared_after_del=True):
        self.calls = []
        self.shaping = True                 # iface starts shaped (a run is in progress)
        self.cleared_after_del = cleared_after_del

    def __call__(self, argv):
        self.calls.append(list(argv))
        cmd = " ".join(argv)
        if "del" in cmd and "root" in cmd:
            self.shaping = not self.cleared_after_del
        stdout = ""
        if "show" in cmd:
            stdout = "qdisc netem 30: ... loss 100%" if self.shaping else "qdisc pfifo_fast 0: root"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def _wd(runner, *, clock, compose_down=None, timeout=30):
    from smp.sync.watchdog import Watchdog
    return Watchdog(iface="enp1s0", heartbeat_timeout_s=timeout,
                    runner=runner, clock=clock, compose_down=compose_down or (lambda: None))


def test_not_stale_right_after_beat():
    clk = [100.0]
    w = _wd(FakeRunner(), clock=lambda: clk[0])
    w.beat()
    clk[0] = 120.0
    assert w.stale() is False                # 20s < 30s timeout


def test_stale_after_timeout_elapses():
    clk = [100.0]
    w = _wd(FakeRunner(), clock=lambda: clk[0])
    w.beat()
    clk[0] = 140.0
    assert w.stale() is True                 # 40s > 30s timeout


def test_beat_resets_the_timer():
    clk = [100.0]
    w = _wd(FakeRunner(), clock=lambda: clk[0])
    w.beat()
    clk[0] = 125.0
    w.beat()                                 # fresh beat
    clk[0] = 150.0
    assert w.stale() is False                # 25s since last beat < 30s


def test_teardown_clears_tc_and_brings_compose_down():
    r = FakeRunner(cleared_after_del=True)
    downs = []
    w = _wd(r, clock=lambda: 0.0, compose_down=lambda: downs.append(True))
    w.teardown()
    cmds = [" ".join(c) for c in r.calls]
    assert any("del" in c and "root" in c and "enp1s0" in c for c in cmds)  # tc cleared
    assert all(c[0] != "iptables" for c in r.calls)                          # tc only
    assert downs == [True]                                                   # stack down


def test_teardown_heals_tc_before_compose_down():
    events = []

    class OrderedRunner(FakeRunner):
        def __call__(self, argv):
            res = super().__call__(argv)
            if "del" in " ".join(argv) and "root" in " ".join(argv):
                events.append("tc_del")
            return res

    w = _wd(OrderedRunner(), clock=lambda: 0.0,
            compose_down=lambda: events.append("compose_down"))
    w.teardown()
    assert events.index("tc_del") < events.index("compose_down")   # heal first


def test_teardown_raises_if_tc_not_cleared():
    # the safety guarantee must be real: if shaping persists after del, RAISE
    r = FakeRunner(cleared_after_del=False)
    w = _wd(r, clock=lambda: 0.0)
    with pytest.raises(RuntimeError):
        w.teardown()


def test_teardown_does_not_bring_compose_down_if_tc_clearing_fails():
    r = FakeRunner(cleared_after_del=False)
    downs = []
    w = _wd(r, clock=lambda: 0.0, compose_down=lambda: downs.append(True))
    with pytest.raises(RuntimeError):
        w.teardown()
    assert downs == []                       # never reached compose_down (fail-loud first)


# --- WatchdogDaemon (the unattended-run gate; heartbeat over the internal net) ----------

def _daemon(runner, *, clock, compose_down, timeout=30):
    from smp.sync.watchdog import Watchdog, WatchdogDaemon
    wd = Watchdog(iface="enp1s0", heartbeat_timeout_s=timeout, runner=runner,
                  clock=clock, compose_down=compose_down)
    return WatchdogDaemon(wd)


def test_daemon_heartbeat_resets_staleness():
    clk = [100.0]
    d = _daemon(FakeRunner(), clock=lambda: clk[0], compose_down=lambda: None)
    clk[0] = 140.0                           # would be stale (40 > 30)...
    import json
    d.handle("POST", "/heartbeat")           # ...but a beat refreshes it
    clk[0] = 150.0
    _s, body = d.handle("GET", "/status")
    assert json.loads(body)["stale"] is False   # 10s since beat


def test_daemon_check_once_tears_down_when_stale_and_is_idempotent():
    clk = [100.0]
    r = FakeRunner(cleared_after_del=True)
    downs = []
    d = _daemon(r, clock=lambda: clk[0], compose_down=lambda: downs.append(True))
    d.handle("POST", "/heartbeat")           # last beat @100
    clk[0] = 200.0                           # 100s later -> stale
    assert d.check_once() is True            # fires teardown
    assert downs == [True]
    assert any("del" in " ".join(c) and "root" in " ".join(c) for c in r.calls)
    assert d.check_once() is True            # idempotent: already torn down
    assert downs == [True]                   # NOT torn down twice


def test_daemon_check_once_noop_when_fresh():
    clk = [100.0]
    downs = []
    d = _daemon(FakeRunner(), clock=lambda: clk[0], compose_down=lambda: downs.append(True))
    d.handle("POST", "/heartbeat")
    clk[0] = 110.0                           # 10s < 30s timeout
    assert d.check_once() is False
    assert downs == []                       # nothing torn down
