"""The protocol code is fixed: evaluation changes must not touch it.

Each path below is compared, as a git blob hash, against its content at the reference commit of
the protocol (bdb44ea). A change to any of them fails the suite; so does a new module in the
protocol package.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PROTOCOL_BLOBS = {
    "src/meld/protocol/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "src/meld/protocol/admission.py": "3af9372d8d6b6b29e0cc2584091185df74fe7505",
    "src/meld/protocol/fold.py": "0e4738672126ca294ce2942d32d913f4a3edb76e",
    "src/meld/protocol/objects.py": "68c4d6593becd159a5fae7fdbc5d7bc9f106fe98",
    "src/meld/merge/classifier.py": "5222b6d089782f7e524b50c7f9964d50cea6813e",
    "src/meld/merge/conformal.py": "b11af1463c49a6159d9decb683e7850260f4ef9a",
    "src/meld/merge/signals.py": "0e45a3a99e642dc2aab17899fd8b24c7cd59189d",
    "src/meld/merge/status_crdt.py": "98348ad8c8845f3d9e4c6581a6903ec6321de66f",
    "src/meld/brain/store.py": "fe8da3ba31e36966cc93e21bb46ae0962a4e09b3",
    "src/meld/sync/anti_entropy.py": "7f601357d1e7b82b126de670652d4d99d6f916b2",
    "src/meld/sync/merkle.py": "560245aba3f7b24dcdebcd8329ab9e64e38d7e2e",
    "src/meld/sync/node.py": "28e969fd62ab1ea0cfb0500cf8ee73bc6e59d6e2",
    "src/meld/sync/objectset.py": "5250cd9bd97bd8a9964a2ad92ddb69d99ded5548",
    "src/meld/sync/wire.py": "7dc4e16a5c65bd58a685d7a0ba2f27448059ea93",
    "src/meld/sync/wire_delta.py": "5b6d8860dbd3e52ebc13aebf0bc23472f01e8ff0",
    "src/meld/agent/memory.py": "5aa23e0e06ddeac5326098437b3a80bb326f45b2",
}


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def test_protocol_files_unchanged():
    changed = [p for p, h in PROTOCOL_BLOBS.items() if _blob(REPO / p) != h]
    assert not changed, f"protocol code changed: {changed}"


def test_no_new_protocol_modules():
    present = {str(p.relative_to(REPO)) for p in (REPO / "src/meld/protocol").glob("*.py")}
    assert present <= set(PROTOCOL_BLOBS), f"new protocol modules: {sorted(present - set(PROTOCOL_BLOBS))}"
