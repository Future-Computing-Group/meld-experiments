"""Per-VM watchdog -- the shared-infra safety mechanism for unattended testbed runs.

Listens for the orchestrator's heartbeat; when it goes stale the host self-heals:
clears ALL tc shaping on the measured iface (so no stale `netem` lingers on the shared
enp1s0), VERIFIES the iface is clean and RAISES if shaping persists, and only then
brings the compose stack down. tc is healed BEFORE the stack so the network is never
left shaped after the workload is gone. tc only -- iptables is off-limits on the testbed.

The class is the testable core (clock + runner + compose_down injected); a thin daemon
loop (not in this unit) calls ``beat()`` on each received heartbeat and, on ``stale()``,
calls ``teardown()`` once and exits.
"""
from __future__ import annotations

import json
from typing import Callable, List, Optional, Tuple


class Watchdog:
    def __init__(self, *, iface: str, heartbeat_timeout_s: float,
                 runner: Callable, clock: Callable[[], float],
                 compose_down: Optional[Callable[[], None]] = None):
        self.iface = iface
        self.timeout = heartbeat_timeout_s
        self._run = runner
        self._clock = clock
        self._compose_down = compose_down or (lambda: None)
        self._last_beat = clock()

    # -- heartbeat ----------------------------------------------------------
    def beat(self) -> None:
        self._last_beat = self._clock()

    def stale(self, now: Optional[float] = None) -> bool:
        now = self._clock() if now is None else now
        return (now - self._last_beat) > self.timeout

    # -- self-heal ----------------------------------------------------------
    def _check(self, argv: List[str]):
        r = self._run(argv)
        if getattr(r, "returncode", 0) != 0:
            raise RuntimeError(
                f"watchdog tc cmd failed: {' '.join(argv)} "
                f"(rc={getattr(r, 'returncode', '?')} {getattr(r, 'stderr', '')})")
        return r

    def _shaping_present(self) -> bool:
        r = self._run(["tc", "qdisc", "show", "dev", self.iface])
        out = getattr(r, "stdout", "") or ""
        return "netem" in out or "loss" in out

    def teardown(self) -> None:
        """Restore the host: clear tc shaping (verify + raise on failure), THEN stack-down."""
        # 1. heal the network first -- remove the whole qdisc tree on the iface.
        self._check(["tc", "qdisc", "del", "dev", self.iface, "root"])
        # 2. the safety guarantee must be real: verify, else fail loud.
        if self._shaping_present():
            raise RuntimeError(
                f"watchdog: tc shaping persists on {self.iface} after teardown")
        # 3. only now bring the workload down.
        self._compose_down()


class WatchdogDaemon:
    """Wraps a Watchdog as the per-VM unattended-run gate. Exposes a heartbeat endpoint
    (the orchestrator on VM1 beats it over the INTERNAL net, immune to laptop flakiness)
    and a stale-check the monitor loop calls: on stale it tears the host down ONCE
    (idempotent -- never a double teardown). The HTTP server + monitor loop are thin
    wrappers over ``handle`` and ``check_once`` (the tested core)."""

    def __init__(self, watchdog: Watchdog):
        self.wd = watchdog
        self._torn = False

    def handle(self, method: str, path: str, body: bytes = b"") -> Tuple[int, bytes]:
        if method == "POST" and path == "/heartbeat":
            self.wd.beat()
            return 200, json.dumps({"ok": True}).encode("utf-8")
        if method == "GET" and path == "/status":
            return 200, json.dumps(
                {"stale": self.wd.stale(), "torn_down": self._torn}).encode("utf-8")
        return 404, json.dumps({"error": "not found"}).encode("utf-8")

    def check_once(self) -> bool:
        """Called by the monitor loop. Tears down once if the heartbeat is stale;
        returns True once torn down (idempotent), False while still fresh."""
        if self._torn:
            return True
        if self.wd.stale():
            self.wd.teardown()
            self._torn = True
            return True
        return False


__all__ = ["Watchdog", "WatchdogDaemon"]
