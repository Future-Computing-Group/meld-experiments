"""Epoch loading replaces the threshold config (design §2.1, §4.6, §7.4): a replica's
operating point is the scope's epoch record, a signed object from the calibration authority,
never a config default. ``load_epoch`` returns the record's original bytes and MAC; the
replica's receive gate verifies it like any other object."""
from __future__ import annotations

import json

import pytest

import meld.config as config
from meld.agent.memory import SMPMemory
from test_step_fold import KEY, _ep


def test_epoch_file_round_trips_signed_bytes_and_the_receive_gate_verifies_them(tmp_path):
    path = tmp_path / "epoch.json"
    config.save_epoch(_ep(), KEY, str(path))
    received, mac = config.load_epoch(str(path))
    from meld.brain.store import Store
    from test_step_fold import POL
    s = Store(POL, KEY, encoder="hashing")
    from meld.protocol.objects import attach_mac
    assert s.apply(received, mac) == attach_mac(_ep(), KEY)
    raw = json.loads(path.read_text())
    raw["received"] = raw["received"].replace('"seq":1', '"seq":7')
    path.write_text(json.dumps(raw))
    assert Store(POL, KEY, encoder="hashing").apply(*config.load_epoch(str(path))) is None


def test_missing_field_raises_no_silent_default(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"received": "{}"}))
    with pytest.raises(KeyError):
        config.load_epoch(str(path))


def test_no_threshold_loader_or_default_operating_point_remains():
    assert not hasattr(config, "load_thresholds") and not hasattr(config, "Thresholds")
    mem = SMPMemory(encoder="hashing", nli="lexical")
    mem.write("the valve is open and the pump runs")
    dec = mem.write("the valve is open and the pump is running")
    assert dec[0].tested and dec[0].decision != "MERGE"          # no epoch: no similarity merge


def test_read_epoch_verifies_the_mac(tmp_path):
    path = tmp_path / "epoch.json"
    config.save_epoch(_ep(), KEY, str(path))
    from meld.protocol.objects import attach_mac
    assert config.read_epoch(str(path), KEY) == attach_mac(_ep(), KEY)
    with pytest.raises(ValueError):
        config.read_epoch(str(path), b"another-group-key")
