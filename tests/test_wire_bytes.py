"""Wire bytes per message class (sync/wire_bytes.py): every replicated object travels as one signed
envelope (canonical bytes, MAC, scope; sync/wire_delta.pack), so its bytes are the length of that
envelope under the real serializers; DIGEST and PULL are metered at the anti-entropy call seam."""
from __future__ import annotations

import json
import os
import sys

import pytest

from meld.protocol.fold import PassCertificate, publish_pass
from meld.protocol.objects import canonical_bytes, signed
from meld.sync.wire_bytes import CLASSES, envelope_bytes, message_class, meter
from meld.sync.wire_delta import pack, unpack

from test_step_fold import KEY, MNLI, R, _cal, _e, _ep, _k, _sig, A

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import run_wire_bytes  # noqa: E402


def _cert():
    return PassCertificate(scope="eu", epoch_id=b"\x01" * 16, epoch_seq=1, m_nli=MNLI,
                           selector=(1, 2, 3), roots=(b"\x02" * 16,) * 5, decisions_digest=b"\x03" * 16,
                           fingerprint=b"\x04" * 16, issuer="r1", misses=0)


def _objects():
    return [R, _e("OVERRULE", _k(R)), _sig(_k(R), _k(A), (1, 2, 3), None), _cal(_k(R), _k(A)), _ep(),
            _cert()]


def test_envelope_bytes_equal_len_of_the_packed_entry_and_roundtrip():
    for o in _objects():
        received, mac = signed(o, KEY)
        n = envelope_bytes(received, mac, "eu")
        assert n == len(pack([(received, mac, "eu")]))
        assert n > len(canonical_bytes(o)) + len(mac)            # body + MAC + envelope members
        assert unpack(pack([(received, mac, "eu")])) == [(received, mac, "eu")]


def test_every_object_type_has_its_class():
    got = [message_class(signed(o, KEY)[0]) for o in _objects()]
    assert got == ["DELTA", "EFFECT", "SIG", "CAL", "EPOCH", "PASS"]
    assert set(got) | {"DIGEST", "PULL", "PATCH"} == set(CLASSES)
    assert publish_pass(_cert(), KEY) == signed(_cert(), KEY)


def test_meter_counts_per_class():
    entries = [(*signed(o, KEY), "eu") for o in _objects()] + [(*signed(R, KEY), "eu")]
    rep = meter(entries)
    assert rep["DELTA"]["count"] == 2 and rep["SIG"]["count"] == 1
    d = envelope_bytes(*entries[0])
    assert rep["DELTA"] == {"count": 2, "bytes": 2 * d, "mean": d, "min": d, "max": d,
                            "mac": 2 * len(entries[0][1]), "body": 2 * len(entries[0][0])}
    assert sum(r["bytes"] for r in rep.values()) == sum(envelope_bytes(*e) for e in entries)


def test_run_wire_bytes_script_reports_every_class(tmp_path):
    out = tmp_path / "wire.json"
    rep = run_wire_bytes.main(["--encoder", "hashing", "--nli", "lexical", "--n", "3",
                               "--losses", "0.05", "--out", str(out)])
    assert json.loads(out.read_text()) == json.loads(json.dumps(rep))
    ex, gold = rep["running_example"]["classes"], rep["gold"]["classes"]
    assert {"DELTA", "EFFECT", "SIG", "PASS"} <= set(ex)
    assert {"DELTA", "EFFECT", "SIG", "CAL", "EPOCH", "PASS"} <= set(gold)
    assert gold["DELTA"]["count"] == 104 and ex["DELTA"]["count"] == 4
    # a hashing-encoder delta carries 256 int8 values, base64 in the canonical body
    assert gold["DELTA"]["min"] > 4 * 256 / 3
    ae = rep["anti_entropy"]
    assert ae["quiescent_round"]["/pull"] == ae["quiescent_round"]["/patch"] == 0
    assert ae["quiescent_round"]["/digest"] > 0
    row = ae["lossy"][0]
    assert row["loss"] == 0.05 and row["rounds"] >= 1
    assert row["per_round"]["/digest"] * row["rounds"] == pytest.approx(row["bytes"]["/digest"])
