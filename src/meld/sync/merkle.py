"""Merkle root over a set of object ids (design §2.2 DIGEST, §2.4 roots, §4.1 cal_root).

Leaves are the sorted ids; leaf and inner hashes are domain-separated (0x00 / 0x01) so no
inner node equals a leaf; an odd node is carried up unchanged. H is h16 (SHA-256 truncated
to 16 bytes). The root of the empty set is h16(b"").

Deviation from design §2.2 (accepted): DIGEST ships one flat root per (scope, kind) and,
where roots differ, the full sorted id list, with no bucketed descent. POST /resync reports
``digest_bytes`` per round so the cost can be measured.
ponytail: add bucket descent when the measured digest bytes matter.
"""
from __future__ import annotations

from typing import Iterable

from meld.proto import h16


def merkle_root(ids: Iterable[bytes]) -> bytes:
    level = [h16(b"\x00" + i) for i in sorted(set(ids))]
    if not level:
        return h16(b"")
    while len(level) > 1:
        level = [h16(b"\x01" + level[k] + level[k + 1]) if k + 1 < len(level) else level[k]
                 for k in range(0, len(level), 2)]
    return level[0]


__all__ = ["merkle_root"]
