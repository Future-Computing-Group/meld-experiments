"""Re-signing an epoch under another group key keeps the record bytes and the recomputation
verdict; only the MAC changes."""
from __future__ import annotations

import json
import os
import sys

import pytest

from meld.config import load_epoch, read_epoch, save_epoch
from meld.merge.conformal import verify_epoch
from meld.protocol.objects import verify_mac
from test_conformal import _build, _cal, _sig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import resign_epoch  # noqa: E402

OLD, NEW = b"throwaway-old-key", b"throwaway-new-key"


def test_resign_keeps_body_and_verdict_and_moves_the_mac(tmp_path):
    cals = [_cal(i, 500 * i, bin=i % 2) for i in range(120)]
    sigs = [_sig(c, c.s_q) for c in cals]
    src = str(tmp_path / "eu-seed-a0.02.epoch.json")
    save_epoch(_build(cals), OLD, src)
    (tmp_path / "old").write_bytes(OLD + b"\n")
    (tmp_path / "new").write_bytes(NEW)

    [row] = resign_epoch.main(["--old-key-file", str(tmp_path / "old"),
                               "--new-key-file", str(tmp_path / "new"), src])
    out = str(tmp_path / "eu-seed-a0.02.group.epoch.json")
    assert load_epoch(out)[0] == load_epoch(src)[0]                     # body bytes unchanged
    new = read_epoch(out, NEW)
    assert not verify_mac(read_epoch(src, OLD), NEW)                     # old MAC fails under NEW
    with pytest.raises(ValueError):
        read_epoch(src, NEW)
    with pytest.raises(ValueError):
        read_epoch(out, OLD)
    assert verify_epoch(new, cals, sigs) == verify_epoch(read_epoch(src, OLD), cals, sigs) is True
    man = json.loads((tmp_path / resign_epoch.MANIFEST).read_text())
    assert man["epochs"] == [row] and row["key_fingerprint"] == resign_epoch.fingerprint(NEW)


def test_resign_refuses_a_mac_that_fails_under_the_old_key(tmp_path):
    src = str(tmp_path / "eu-seed-a0.02.epoch.json")
    save_epoch(_build([_cal(i, 500 * i) for i in range(60)]), NEW, src)
    with pytest.raises(ValueError):
        resign_epoch.resign(src, OLD, NEW)
    assert not os.path.exists(tmp_path / "eu-seed-a0.02.group.epoch.json")
