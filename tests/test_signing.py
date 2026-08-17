"""M5-test: signature verification precedes the merge procedure (TDD).

The manuscript (Section 4.2 / 5.4) states every delta is signed and verified before any
signal is read, so a delta cannot be silently forged by the transport. This pins that
claim with a deterministic probe: a forged-signature delta and a
tampered delta are dropped before the decision procedure runs; a correctly-signed delta
is admitted. HMAC is the stand-in suite (the manuscript scopes the specific PKI out).
Run:  PYTHONPATH=src:cross-tier python3 -m pytest tests/test_signing.py -q
"""
from meld.trust.signing import sign_delta, verify_delta, admit_signed

KEY = b"canonical-brain-key"


def test_valid_signature_verifies_and_admits():
    d = sign_delta({"op": "publish", "claim_id": "c1", "sigma": 0.93, "chi": "entails"}, KEY)
    assert verify_delta(d, KEY) is True
    assert admit_signed(d, KEY) is d            # admitted -> returned for processing


def test_forged_signature_is_rejected_and_dropped():
    forged = {"op": "publish", "claim_id": "c1", "sigma": 0.93, "sig": "deadbeefcafe"}
    assert verify_delta(forged, KEY) is False
    assert admit_signed(forged, KEY) is None    # dropped before any signal is read


def test_tampered_content_after_signing_is_rejected():
    d = sign_delta({"op": "publish", "claim_id": "c1", "sigma": 0.93, "chi": "entails"}, KEY)
    d["sigma"] = 0.05                            # transport tampers with the payload
    assert verify_delta(d, KEY) is False
    assert admit_signed(d, KEY) is None


def test_wrong_key_is_rejected():
    d = sign_delta({"op": "overrule", "claim_id": "c2", "link_id": "L"}, KEY)
    assert verify_delta(d, b"some-other-brain-key") is False


def test_unsigned_delta_is_rejected():
    assert verify_delta({"op": "publish", "claim_id": "c1"}, KEY) is False
    assert admit_signed({"op": "publish", "claim_id": "c1"}, KEY) is None
