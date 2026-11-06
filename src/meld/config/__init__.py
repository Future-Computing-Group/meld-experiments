"""Epoch loading (design §2.1, §4.6). There is no threshold config and no default operating
point: a replica merges only under the scope's active epoch, a record the scope's
calibration authority signs and publishes like any other object.

An epoch file holds that signed record as distributed: JSON ``{"received": <the record's
canonical bytes as text>, "mac": <hex HMAC>}``. ``load_epoch`` returns ``(received, mac)``
for the replica's receive gate, which verifies the MAC, the authority and M_nli/M_pol;
a missing field raises (no silent default). ``save_epoch`` writes one; ``read_epoch``
returns the MAC-verified record for an in-process memory (SMPMemory(epoch=...)), and a
script without an epoch file runs without one: no similarity merge.
"""
from __future__ import annotations

import json
from typing import Tuple

from meld.protocol.objects import EpochRecord, from_bytes, signed, verify_mac


def save_epoch(epoch, key: bytes, path: str) -> None:
    received, mac = signed(epoch, key)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"received": received.decode("ascii"), "mac": mac}, fh)


def load_epoch(path: str) -> Tuple[bytes, str]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return raw["received"].encode("ascii"), raw["mac"]


def read_epoch(path: str, key: bytes) -> EpochRecord:
    epoch = from_bytes(EpochRecord, *load_epoch(path))
    if not verify_mac(epoch, key):
        raise ValueError(f"epoch file {path}: MAC does not verify under the group key")
    return epoch


__all__ = ["save_epoch", "load_epoch", "read_epoch"]
