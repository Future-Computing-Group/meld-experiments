"""Real, healable network partition via tc (C-1 multi-host).

Installs a symmetric `tc netem loss 100%` toward the PEER IP only (a prio qdisc +
a netem-loss band + a u32 dst filter), so the two hosts cannot exchange Patches but
all OTHER traffic (incl. the orchestrator's SSH) is untouched -- never a blanket
loss. Verifies the rule landed, heals by removing it, and RAISES on any failure
(inject-must-raise; avoids a fake-partition that swallows errors).
iptables is off-limits on the testbed; tc only.

Mirrors a proven tc pattern (deploy/setup-wan.sh, multi_vm_runner): the VMs'
enp1s0 carries the kernel-default `fq_codel` root, so the root is first `del`'d (a prio
root cannot be ADDED over an existing root), then the prio root + netem band + u32
filter are added; heal `del root`s, restoring the default. ``runner(host, argv) ->
result`` with ``.returncode``/``.stdout``/``.stderr`` is the exec seam (real = ssh +
`sudo tc`; mocked in tests).
"""
from __future__ import annotations

from typing import Callable, List, Optional

_LOSS_BAND = "1:3"
# all-zeros priomap: every unfiltered packet goes to band 1:1 (unshaped); only the u32 /32
# filters reach the netem band. The default priomap would put TC_PRIO_BULK traffic in 1:3.
_PRIO_ROOT = ["prio", "bands", "3", "priomap", *["0"] * 16]
_LOSS_HANDLE = "30:"


class PartitionController:
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
            self._run(host, argv)        # clear-before-setup: ignore (|| true), root may be default-only

    def _add_loss(self, host: str, peer_ips: List[str], dport: Optional[int] = None) -> None:
        # clear the default fq_codel root ONCE, add the prio + netem tree ONCE, then add a
        # u32 dst filter per peer. del-root is done once so several peers can be dropped on the
        # same host without each call wiping the previous peer's filter (the 2|2-split case).
        self._del_root(host, check=False)
        self._check(host, ["tc", "qdisc", "add", "dev", self.iface,
                           "root", "handle", "1:", *_PRIO_ROOT])
        self._check(host, ["tc", "qdisc", "add", "dev", self.iface,
                           "parent", _LOSS_BAND, "handle", _LOSS_HANDLE,
                           "netem", "loss", "100%"])
        port = ["match", "ip", "dport", str(int(dport)), "0xffff"] if dport is not None else []
        for peer_ip in peer_ips:
            self._check(host, ["tc", "filter", "add", "dev", self.iface,
                               "protocol", "ip", "parent", "1:0", "prio", "3",
                               "u32", "match", "ip", "dst", f"{peer_ip}/32", *port,
                               "flowid", _LOSS_BAND])

    def verify_partition(self, host: str, peer_ip: str) -> bool:
        r = self._run(host, ["tc", "qdisc", "show", "dev", self.iface])
        return "loss 100%" in (getattr(r, "stdout", "") or "")

    def isolate_egress(self, host: str, peer_ip: str) -> None:
        """One-sided: drop ``host``'s egress toward ``peer_ip`` (breaks the bidirectional
        TCP of an HTTP publish without touching the remote host). Verifies + raises."""
        self._add_loss(host, [peer_ip])
        if not self.verify_partition(host, peer_ip):
            raise RuntimeError(f"egress isolation not active on {host} after apply")

    def isolate_egress_multi(self, host: str, peer_ips: List[str], dport: Optional[int] = None) -> None:
        """Drop ``host``'s egress toward EVERY ``peer_ip`` under a single prio root (one
        del-root, one netem band, one u32 filter per peer). Repeated single-peer
        ``isolate_egress`` would del-root each call and wipe the prior peer's filter -- this
        is the primitive for a 2|2 (or N/2) split where a host must be cut from several peers
        at once. ``dport`` scopes each rule to that destination port (the brains' peer port), so
        a host's other traffic to the peers (orchestration, heartbeats) stays up. Verifies the
        loss landed + raises."""
        if not peer_ips:
            return
        self._add_loss(host, peer_ips, dport)
        if not self.verify_partition(host, peer_ips[0]):
            raise RuntimeError(f"egress isolation not active on {host} after multi-apply")

    def heal_host(self, host: str) -> None:
        """Remove the whole prio tree on ``host`` (del root -> kernel default), clearing every
        peer filter at once. del is best-effort (default-only root del returns nonzero
        harmlessly); verify is authoritative -- raise only if loss actually persists."""
        self._del_root(host, check=False)
        r = self._run(host, ["tc", "qdisc", "show", "dev", self.iface])
        if "loss 100%" in (getattr(r, "stdout", "") or ""):
            raise RuntimeError(f"egress isolation still active on {host} after heal")

    def heal_egress(self, host: str, peer_ip: str) -> None:
        """Undo isolate_egress: del root -> kernel default; VERIFY is authoritative. The del
        is best-effort: on an unshaped host (default-only root) `tc qdisc del root` returns
        nonzero ("No such file"), which is not a failure (nothing to remove). Raise only if
        the loss actually persists."""
        self._del_root(host, check=False)
        if self.verify_partition(host, peer_ip):
            raise RuntimeError(f"egress isolation still active on {host} after heal")

    def partition(self, host_a: str, host_b: str, *, ip_a: str, ip_b: str) -> None:
        """Symmetrically isolate host_a and host_b (each drops egress to the other)."""
        self.isolate_egress(host_a, ip_b)
        self.isolate_egress(host_b, ip_a)

    def heal(self, host_a: str, host_b: str, *, ip_a: str, ip_b: str) -> None:
        """Remove the prio tree (del root -> kernel default) and confirm connectivity."""
        self.heal_egress(host_a, ip_b)
        self.heal_egress(host_b, ip_a)


__all__ = ["PartitionController"]
