"""WAN-RTT emulation via tc netem delay (eval-s edge<->cloud regimes).

Shares the validated tc pattern with PartitionController (deploy/setup-wan.sh):
del-root-first (the VMs' enp1s0 carries a default fq_codel root), prio root, a netem
DELAY band toward the peer IP, u32 dst filter, verify + raise. clear() del-root's
back to the kernel default. The orchestrator pairs shape() with a ping read-back to
verify the ACTUAL RTT before measuring -- verify_delay() here only confirms
the rule landed. tc only; iptables is off-limits on the testbed.
"""
from __future__ import annotations

from typing import Callable, List

_BAND = "1:3"
# all-zeros priomap: every unfiltered packet goes to band 1:1 (unshaped); only the u32 /32
# filters reach the netem band. The default priomap would put TC_PRIO_BULK traffic in 1:3.
_PRIO_ROOT = ["prio", "bands", "3", "priomap", *["0"] * 16]
_HANDLE = "30:"


class WanShaper:
    def __init__(self, runner: Callable, *, iface: str = "enp1s0"):
        self._run = runner
        self.iface = iface

    def _check(self, host: str, argv: List[str]):
        r = self._run(host, argv)
        if getattr(r, "returncode", 0) != 0:
            raise RuntimeError(
                f"tc failed on {host}: {' '.join(argv)} "
                f"(rc={getattr(r, 'returncode', '?')} {getattr(r, 'stderr', '')})")
        return r

    def _del_root(self, host: str, *, check: bool) -> None:
        argv = ["tc", "qdisc", "del", "dev", self.iface, "root"]
        if check:
            self._check(host, argv)
        else:
            self._run(host, argv)        # clear-before-setup: ignore (root may be default-only)

    def shape(self, host: str, peer_ip: str, *, delay_ms: float, jitter_ms: float = 0) -> None:
        """Add one-way ``delay_ms`` (+ optional jitter) toward ``peer_ip`` on ``host``."""
        self.shape_multi(host, [peer_ip], delay_ms=delay_ms, jitter_ms=jitter_ms)

    def shape_multi(self, host: str, peer_ips: List[str], *, delay_ms: float,
                    jitter_ms: float = 0) -> None:
        """One-way delay toward EVERY peer under one prio root (one del-root, one netem band, a
        u32 filter per peer); repeated ``shape`` calls would each del-root the previous peer."""
        self._del_root(host, check=False)
        self._check(host, ["tc", "qdisc", "add", "dev", self.iface,
                           "root", "handle", "1:", *_PRIO_ROOT])
        netem = ["tc", "qdisc", "add", "dev", self.iface, "parent", _BAND,
                 "handle", _HANDLE, "netem", "delay", f"{delay_ms}ms"]
        if jitter_ms:
            netem.append(f"{jitter_ms}ms")
        self._check(host, netem)
        for peer_ip in peer_ips:
            self._check(host, ["tc", "filter", "add", "dev", self.iface,
                               "protocol", "ip", "parent", "1:0", "prio", "3",
                               "u32", "match", "ip", "dst", f"{peer_ip}/32",
                               "flowid", _BAND])
        if not self.verify_delay(host):
            raise RuntimeError(f"wan delay not active on {host} after apply")

    def verify_delay(self, host: str) -> bool:
        r = self._run(host, ["tc", "qdisc", "show", "dev", self.iface])
        return "delay" in (getattr(r, "stdout", "") or "")

    def clear(self, host: str) -> None:
        """del root -> kernel default; VERIFY is authoritative. The del is best-effort: on
        an unshaped host (default-only root) `tc qdisc del root` returns nonzero ("No such
        file"), which is NOT a failure -- there is simply nothing to remove. So we ignore
        the del's rc and raise only if delay actually persists."""
        self._del_root(host, check=False)
        if self.verify_delay(host):
            raise RuntimeError(f"wan delay still active on {host} after clear")


__all__ = ["WanShaper"]
