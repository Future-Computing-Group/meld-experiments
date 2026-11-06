"""Re-sign epoch files under another group key: each ``<name>.epoch.json`` is read and its MAC
checked under --old-key-file (config.read_epoch), the MAC is re-attached under --new-key-file, and
``<name>.group.epoch.json`` is written beside it. The record bytes (``received``) are unchanged,
so the epoch id and every recomputation check stay the same; only the MAC differs. Each
directory gets ``group-epochs.manifest.json`` (epoch id, sha256 of the record bytes, the new
key's fingerprint = first 8 bytes of its SHA-256). Key files are read as run_calibration reads
them (stripped); the key itself is never printed.

Run:  PYTHONPATH=src python scripts/resign_epoch.py --old-key-file OLD --new-key-file NEW \
          outputs/calibration-labelled/a0.05/paws-seed-a0.05.epoch.json ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from meld.config import load_epoch, read_epoch, save_epoch
from meld.eval.runner_args import read_key_file
from meld.protocol.objects import canonical_bytes, object_id

MANIFEST = "group-epochs.manifest.json"
SUFFIX = ".epoch.json"


def fingerprint(key: bytes) -> str:
    return hashlib.sha256(key).digest()[:8].hex()


def resign(path: str, old_key: bytes, new_key: bytes) -> dict:
    if not path.endswith(SUFFIX) or path.endswith(".group" + SUFFIX):
        raise ValueError(f"not a source epoch file: {path}")
    received, _ = load_epoch(path)
    epoch = read_epoch(path, old_key)                   # refuses a MAC that fails under old_key
    out = path[: -len(SUFFIX)] + ".group" + SUFFIX
    save_epoch(epoch, new_key, out)
    if load_epoch(out)[0] != received or canonical_bytes(read_epoch(out, new_key)) != received:
        raise ValueError(f"{out}: record bytes changed or MAC fails under the new key")
    return {"source": os.path.basename(path), "file": os.path.basename(out),
            "epoch_id": object_id(received).hex(), "scope": epoch.scope, "alpha": epoch.alpha,
            "body_sha256": hashlib.sha256(received).hexdigest(), "key_fingerprint": fingerprint(new_key)}


def main(argv=None) -> list:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old-key-file", required=True)
    ap.add_argument("--new-key-file", required=True)
    ap.add_argument("epochs", nargs="+")
    a = ap.parse_args(argv)
    old_key, new_key = read_key_file(a.old_key_file), read_key_file(a.new_key_file)
    if old_key == new_key:
        raise SystemExit("old and new key are the same")
    rows = [resign(p, old_key, new_key) for p in a.epochs]
    by_dir: dict = {}
    for p, r in zip(a.epochs, rows):
        by_dir.setdefault(os.path.dirname(os.path.abspath(p)), []).append(r)
    for d, rs in by_dir.items():
        with open(os.path.join(d, MANIFEST), "w", encoding="utf-8") as fh:
            json.dump({"key_fingerprint": fingerprint(new_key), "epochs": rs}, fh, indent=1)
            fh.write("\n")
    for r in rows:
        print(f"{r['file']}  {r['epoch_id']}  {r['body_sha256']}  key {r['key_fingerprint']}")
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
