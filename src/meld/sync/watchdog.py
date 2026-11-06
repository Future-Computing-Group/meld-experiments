"""Per-VM watchdog -- the shared-infra safety mechanism for unattended testbed runs.

Listens for the orchestrator's heartbeat; when it goes stale the host self-heals:
clears ALL tc shaping on the measured iface (so no stale `netem` lingers on the shared
enp1s0), VERIFIES the iface is clean and RAISES if shaping persists, and removes the
brain container in every case (after the tc attempt, even when tc fails). tc only -- iptables is off-limits on the testbed.

The class is the testable core (clock + runner + compose_down injected); a thin daemon
loop (not in this unit) calls ``beat()`` on each received heartbeat and, on ``stale()``,
calls ``teardown()`` once and exits.

The daemon is also the host's partition and shaping agent: the driver asks each host's own
watchdog over HTTP (POST /isolate, /shape, /heal) instead of reaching other hosts by ssh. Every
install is read back (``tc filter show``) and kept only if the u32 filters are exactly the
requested peer /32s. A partition drops only the brains' peer port, and the driver reaches the
local brain through ``/brain/<path>`` on the watchdog port, so the orchestrator (itself on a
brain host) keeps its control path to both sides during a partition. Stdlib only: it ships flat next to partition.py and wan.py.
"""
from __future__ import annotations

import ipaddress
import json
import re
from typing import Callable, List, Optional, Tuple

DEFAULT_CONTAINER = "meld-brain"     # the container edge/deploy.sh starts
_U32_DST = re.compile(r"match ([0-9a-f]{8})/ffffffff at 16")


def installed_peer_ips(filter_show: str) -> List[str]:
    """The destination /32s of the u32 filters in ``tc filter show`` output."""
    return [str(ipaddress.ip_address(int(h, 16))) for h in _U32_DST.findall(filter_show or "")]


def _peer_ips(body: dict) -> List[str]:
    ips = body.get("peer_ips")
    if type(ips) is not list or not ips or not all(type(i) is str for i in ips):
        raise ValueError("peer_ips must be a non-empty list of addresses")
    return [str(ipaddress.IPv4Address(i)) for i in ips]


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
    def _shaping_present(self) -> bool:
        r = self._run(["tc", "qdisc", "show", "dev", self.iface])
        out = getattr(r, "stdout", "") or ""
        return "netem" in out or "loss" in out

    def teardown(self) -> None:
        """Restore the host: clear tc shaping (verify + raise if it persists), THEN remove the
        brain. The brain is removed even when tc fails: a tc error is surfaced, never a reason
        to leave the workload running."""
        try:
            # 1. heal the network first -- remove the whole qdisc tree on the iface. With nothing
            # shaped the root is the default qdisc (handle 0) and del returns rc != 0: clean, not
            # a failure (as in WatchdogDaemon._heal); the read-back below is the real check.
            r = self._run(["tc", "qdisc", "del", "dev", self.iface, "root"])
            # 2. the safety guarantee must be real: verify, else fail loud.
            if self._shaping_present():
                raise RuntimeError(
                    f"watchdog: tc shaping persists on {self.iface} after teardown "
                    f"(del rc={getattr(r, 'returncode', '?')} {getattr(r, 'stderr', '')})")
        finally:
            # 3. bring the workload down regardless.
            self._compose_down()


class WatchdogDaemon:
    """Wraps a Watchdog as the per-VM unattended-run gate. Exposes a heartbeat endpoint
    (the orchestrator on VM1 beats it over the INTERNAL net, immune to laptop flakiness)
    and a stale-check the monitor loop calls: on stale it tears the host down ONCE
    (idempotent -- never a double teardown). The HTTP server + monitor loop are thin
    wrappers over ``handle`` and ``check_once`` (the tested core)."""

    def __init__(self, watchdog: Watchdog, *, partition=None, shaper=None,
                 brain_port: Optional[int] = None,
                 forward: Optional[Callable[[str, str, bytes], Tuple[int, bytes]]] = None):
        self.wd = watchdog
        self.partition, self.shaper = partition, shaper   # host-runner seams, host label "local"
        self.brain_port = brain_port      # a partition drops only this destination port
        self.forward = forward            # /brain/<path> -> the local brain (control path)
        self._torn = False

    def _heal(self) -> None:
        self.wd._run(["tc", "qdisc", "del", "dev", self.wd.iface, "root"])   # default-only root: rc != 0
        if self.wd._shaping_present():
            raise RuntimeError(f"tc shaping persists on {self.wd.iface} after heal")

    def _install(self, path: str, body: dict) -> List[str]:
        ips = _peer_ips(body)
        if path == "/isolate":
            if self.partition is None:
                raise LookupError("no partition seam")
            self.partition.isolate_egress_multi("local", ips, dport=self.brain_port)
        else:
            delay = body.get("delay_ms")
            if type(delay) not in (int, float) or not 0 <= delay <= 10_000:
                raise ValueError("delay_ms must be a number in [0, 10000]")
            if self.shaper is None:
                raise LookupError("no shaper seam")
            self.shaper.shape_multi("local", ips, delay_ms=delay)
        got = installed_peer_ips(getattr(self.wd._run(["tc", "filter", "show", "dev", self.wd.iface]),
                                         "stdout", ""))
        if sorted(got) != sorted(ips):
            self._heal()
            raise RuntimeError(f"installed filters {got} != requested {ips}; healed")
        return got

    def handle(self, method: str, path: str, body: bytes = b"") -> Tuple[int, bytes]:
        if path.startswith("/brain/") and self.forward is not None:
            return self.forward(method, path[len("/brain"):], body)
        if method == "POST" and path in ("/isolate", "/shape", "/heal"):
            try:
                req = json.loads(body or b"{}")
                if type(req) is not dict:
                    raise ValueError(req)
                if path == "/heal":
                    self._heal()
                    return 200, json.dumps({"healed": True}).encode()
                if self._torn:
                    _peer_ips(req)
                    return 409, json.dumps({"error": "torn down: no rules after teardown"}).encode()
                return 200, json.dumps({"filters": self._install(path, req)}).encode()
            except (ValueError, TypeError) as e:
                return 400, json.dumps({"error": str(e)}).encode()
            except (RuntimeError, LookupError) as e:
                return 500, json.dumps({"error": str(e)}).encode()
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


__all__ = ["Watchdog", "WatchdogDaemon", "DEFAULT_CONTAINER", "installed_peer_ips"]
