"""B2 risk-specified admission (protocol design r3 §4.1-4.6, Prop 5): exact-rational per-bin
conformal thresholds, store-size bins, the epoch builder's refusals and epoch verification."""
from __future__ import annotations

import math
import random
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from scipy.stats import beta

from meld.merge.conformal import (
    active_epoch, bin_of, build_epoch, cal_root, derive_B0, n_min, quantile_threshold,
    threshold, verify_epoch,
)
from meld.protocol.objects import (
    ALPHA_ONE, FIXED_ONE, INF16, CalRecord, EpochRecord, SignalRecord, canonical_bytes, h16,
    m_nli, m_pol, object_id,
)
from meld.sync.merkle import merkle_root

MNLI = m_nli("enc@1", "nli@1", "int8", "canon-1")
MPOL = m_pol({"A": 1}, {"eu": ["eu.a"]}, {}, 1)
SEL = (52000, 3, MPOL)                      # (sigma_lo_q, m, M_pol)
B0 = 256


def _ppm(a: float) -> int:
    return round(a * ALPHA_ONE)


def _pair(i: int):
    a, b = h16(b"x%d" % i), h16(b"y%d" % i)
    return min(a, b), max(a, b)


def _cal(i, s_q, *, bin=0, subtype="OTHER", scope="eu", hw="x86", obs=1, source="SEED",
         item=None, n_prior=None, sel=SEL):
    """A Cal record for pair i; n_prior defaults to the smallest store size of ``bin``."""
    cx, cy = _pair(i)
    if n_prior is None:
        n_prior = 0 if bin == 0 else B0 << (bin - 1)
    return CalRecord(scope=scope, c_x=cx, c_y=cy, bin=bin, s_q=s_q, hw=hw, obs=obs,
                     label="NON_MERGEABLE", subtype=subtype, source=source,
                     item=item if item is not None else f"it{i}", author="cal-auth",
                     created_at=0, sigma_lo_q=sel[0], m=sel[1], m_pol=sel[2], n_prior=n_prior)


def _build(cals, **over):
    kw = dict(scope="eu", seq=1, alpha=_ppm(0.02), alpha_c=_ppm(0.05), policy="SEED",
              bins=(B0, "doubling"), selector_params=SEL, m_nli=MNLI,
              sigma_c_q=60000, author="cal-auth", created_at=0)
    kw.update(over)
    return build_epoch(cals, **kw)


def _sig(cal, s, m=MNLI, hw="x86", scope="eu"):
    """A single-observation Sig record for the Cal pair with score s (both directions p_e = s)."""
    pay = (0, s, FIXED_ONE - s)
    return SignalRecord(scope=scope, c_x=cal.c_x, c_y=cal.c_y, m_nli=m, pay_xy=pay, pay_yx=pay,
                        issuer="r1", hw=hw)


# -- n_min and the quantile, exact rationals -------------------------------------------------

def test_n_min_and_quantile_threshold_exact_rational():
    audited = {0.2: 4, 0.1: 9, 0.05: 19, 0.02: 49, 0.013: 76, 0.01: 99}
    assert {a: n_min(_ppm(a)) for a in audited} == audited

    for a in (0.01, 0.013, 0.02, 0.025, 0.05, 0.1, 0.2):
        p = _ppm(a)
        for n in range(0, 5001):
            scores = list(range(n, 0, -1))           # distinct, reversed: rank = value
            k = math.ceil(Fraction(n + 1) * (1 - Fraction(str(a))))
            th = quantile_threshold(scores, p)
            # +inf exactly when the index overflows, i.e. n < n_min
            assert (th == INF16) == (k > n) == (n < n_min(p))
            if k <= n:
                assert th == k                       # k-th smallest of 1..n
            # the audit's float check: float index agrees with the exact one on this grid
            assert math.ceil((n + 1) * (1 - a)) == k

    # ties and order do not matter; fewer than n_min gives the sentinel
    assert quantile_threshold([7] * 49, _ppm(0.02)) == 7
    assert quantile_threshold([7] * 48, _ppm(0.02)) == INF16
    assert quantile_threshold([], _ppm(0.02)) == INF16
    with pytest.raises(ValueError):
        n_min(0)
    with pytest.raises(ValueError):
        quantile_threshold([1], ALPHA_ONE)
    with pytest.raises(ValueError):
        n_min(0.02)                                  # floats refused: alpha is ppm


def test_bin_of_n_prior_with_B0_and_derive_B0_at_defaults_is_256():
    assert [bin_of(n, 256) for n in (0, 1, 255)] == [0, 0, 0]
    assert [bin_of(n, 256) for n in (256, 511, 512, 1023, 1024, 10_000)] == [1, 1, 2, 2, 3, 6]
    for n in range(0, 20_000, 7):
        expect = 0 if n < 256 else 1 + math.floor(math.log2(n / 256))
        assert bin_of(n, 256) == expect
    assert derive_B0(_ppm(0.02), q=0.4, r=0.5, t=0.8) == 256
    # the bound itself is about 204 (49 / 0.24); B_0 is the next power of two
    assert derive_B0(_ppm(0.02), q=0.4, r=1, t=1) == 128   # 49 / 0.6 = 81.7
    assert derive_B0(_ppm(0.02), q=Fraction(2, 5), r="0.5", t=Fraction(4, 5)) == 256
    with pytest.raises(ValueError):
        derive_B0(_ppm(0.02), q=1, r=0.5, t=0.8)


def test_small_bins_calibratable_at_B0():
    q, r, t = Fraction(2, 5), Fraction(1, 2), Fraction(4, 5)
    assert derive_B0(_ppm(0.02), q, r, t) == B0
    # in expectation a stream of B_0 nodes yields B_0 t (1 - q) r = 61.44 >= n_min(0.02) = 49
    assert B0 * t * (1 - q) * r >= n_min(_ppm(0.02))
    # simulated streams of B_0 nodes, all in bin 0: tested (t), non-mergeable (1 - q), audited (r)
    rng = random.Random(7)
    counts = []
    for _ in range(200):
        n0 = 0
        for n_prior in range(B0):
            assert bin_of(n_prior, B0) == 0
            n0 += rng.random() < t and rng.random() >= q and rng.random() < r
        counts.append(n0)
    assert sorted(counts)[len(counts) // 2] >= n_min(_ppm(0.02))
    # B_0 is derived in expectation: a single stream is short with probability about 0.027 (binomial)
    assert sum(c >= n_min(_ppm(0.02)) for c in counts) / len(counts) >= 0.9
    n0 = max(counts)
    assert quantile_threshold([rng.randrange(FIXED_ONE) for _ in range(n0)], _ppm(0.02)) < INF16
    # the r2 bin of 64 nodes could not reach n_min(0.02) = 49 even with r = 1
    assert 64 * t * (1 - q) < n_min(_ppm(0.02))


# -- the audits' toy generator (sim_bins.py), bins (256, doubling) -----------------------------

def _draw(rng, n_arr):
    """sim_bins.draw: argmax-sigma over n prior non-mergeable nodes; the tested pair's score."""
    out = np.empty(len(n_arr))
    for i, n in enumerate(n_arr):
        z = rng.standard_normal(n)
        sig = z + 0.5 * rng.standard_normal(n)
        j = sig.argmax()
        out[i] = 1 / (1 + np.exp(-(1.5 * z[j] + rng.standard_normal() - 2.0)))
    return np.rint(out * FIXED_ONE).astype(int)


def _cp(k, n, level=0.95):
    lo = 0.0 if k == 0 else beta.ppf((1 - level) / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - (1 - level) / 2, k + 1, n - k)
    return lo, hi


def test_realised_false_merge_per_bin_within_alpha_on_synthetic_stream():
    rng = np.random.default_rng(1)
    M = 1500                                         # tested pairs (items) per bin
    ranges = [(1, 256), (256, 512), (512, 1024), (1024, 2048)]
    stream = []                                      # (n_prior, s_q), one pair per item
    for lo, hi in ranges:
        n = np.sort(rng.integers(lo, hi, size=M))
        stream += list(zip(n.tolist(), _draw(rng, n).tolist()))
    assert sorted({bin_of(n, B0) for n, _ in stream}) == [0, 1, 2, 3]

    for a in (0.05, 0.02):
        alpha = _ppm(a)
        idx = rng.permutation(len(stream))
        cal_i, test_i = set(idx[: len(idx) // 2].tolist()), idx[len(idx) // 2:].tolist()
        cals = [_cal(i, stream[i][1], bin=bin_of(stream[i][0], B0), n_prior=stream[i][0])
                for i in sorted(cal_i)]
        ep = _build(cals, alpha=alpha)
        assert verify_epoch(ep, cals, [_sig(c, c.s_q) for c in cals])

        # (b) per-bin random split: realised FM within alpha plus the CP width in every bin
        for g in range(4):
            tests = [stream[i][1] for i in test_i if bin_of(stream[i][0], B0) == g]
            k = sum(s > threshold(ep, g) for s in tests)
            lo, hi = _cp(k, len(tests))
            assert k / len(tests) <= a + (hi - lo) / 2, (a, g, k, len(tests))

        # (a) one threshold from bin 0 applied at large N fails
        bin0 = [c for c in cals if c.bin == 0]
        th0 = quantile_threshold([c.s_q for c in bin0], alpha)
        tests3 = [stream[i][1] for i in test_i if bin_of(stream[i][0], B0) == 3]
        k3 = sum(s > th0 for s in tests3)
        # realised 0.178 at alpha 0.05 and 0.057 at 0.02 under this seed: significantly above
        assert _cp(k3, len(tests3))[0] > a and k3 / len(tests3) >= 2.5 * a, (a, k3, len(tests3))


# -- epochs ------------------------------------------------------------------------------------

def test_mondrian_off_by_default_and_on_disables_short_bins():
    # bin 0: 60 records (>= n_min(0.02) = 49), 5 contradictions (< n_min(0.05) = 19)
    # bin 1: 60 records, 30 contradictions with high scores
    cals = [_cal(i, 1000 * i, bin=0, subtype="CONTRADICTION" if i < 5 else "OTHER")
            for i in range(60)]
    cals += [_cal(100 + i, 500 * i + (30000 if i < 30 else 0), bin=1,
                  subtype="CONTRADICTION" if i < 30 else "OTHER") for i in range(60)]
    off = _build(cals)
    assert off.mondrian_c is False
    rows = {r[0]: r for r in off.table}
    th0 = quantile_threshold([c.s_q for c in cals if c.bin == 0], _ppm(0.02))
    assert rows[0] == (0, 60, 5, th0, INF16)         # theta_c recorded, short -> sentinel
    assert threshold(off, 0) == th0 < INF16
    assert threshold(off, 1) == rows[1][3]
    assert threshold(off, 2) == INF16                # no records for bin 2

    on = _build(cals, mondrian_c=True)
    assert on.table == off.table
    assert threshold(on, 0) == INF16                 # contradiction-short bin disabled
    assert rows[1][4] < INF16
    assert threshold(on, 1) == max(rows[1][3], rows[1][4])
    # an epoch with no row for a bin, or no epoch at all, makes no similarity merge
    assert threshold(None, 0) == INF16


def test_epoch_rejects_mixed_sources_multi_hardware_or_multi_observation_cal_and_selector_mismatch():
    base = [_cal(i, 1000 + i) for i in range(60)]
    _build(base)                                     # the clean set builds
    bad_sets = {
        "mixed sources": base[:-1] + [_cal(59, 5, source="AUDIT")],
        "multi hardware": base[:-1] + [_cal(59, 5, hw="gh200")],
        "multi observation": base[:-1] + [_cal(59, 5, obs=2)],
        "other scope": base[:-1] + [_cal(59, 5, scope="us")],
        "two records for one item": base + [_cal(60, 5, item="it0")],
        "two records for one tested pair": base + [_cal(0, 5, item="fresh-item")],
        "carried bin disagrees with n_prior": base[:-1] + [_cal(59, 5, bin=1, n_prior=3)],
        "one record under another selector": base[:-1] + [_cal(59, 5, sel=(51000, 3, MPOL))],
        "empty": [],
    }
    for what, cals in bad_sets.items():
        with pytest.raises(ValueError):
            _build(cals)
    with pytest.raises(ValueError):                  # policy AUDIT over SEED records
        _build(base, policy="AUDIT")
    ep = _build(base)
    assert (ep.sigma_lo_q, ep.m, ep.m_pol, ep.cal_hw, ep.policy) == (52000, 3, MPOL, "x86", "SEED")


def test_epoch_verifies_by_recomputation_and_sq_against_sig():
    cals = [_cal(i, 500 * i, bin=i % 2) for i in range(120)]
    ep = _build(cals)
    sigs = [_sig(c, c.s_q) for c in cals]
    assert verify_epoch(ep, cals, sigs)
    assert verify_epoch(ep, list(reversed(cals)), sigs)          # order-independent
    assert ep.cal_root == cal_root(reversed(cals))
    # the root is the Merkle root over the Cal ids, as the Cal set's DIGEST root
    assert ep.cal_root == merkle_root(object_id(canonical_bytes(c)) for c in cals)
    # Sig joined below s_q is fine (Sig only lowers s); a joined s above s_q is not
    assert verify_epoch(ep, cals, sigs + [_sig(cals[3], cals[3].s_q - 1)])
    assert not verify_epoch(ep, cals, sigs[:3] + [_sig(cals[3], cals[3].s_q + 1)] + sigs[4:])
    assert not verify_epoch(ep, cals, sigs[1:])                  # an s_q without Sig evidence
    assert not verify_epoch(ep, cals, sigs[:1] + [_sig(cals[1], cals[1].s_q, m=b"\0" * 16)]
                            + sigs[2:])                          # Sig under another M_nli
    # tampering: a table row, the root, the Cal set, the alpha, the hardware
    g, n, nc, th, thc = ep.table[0]
    assert not verify_epoch(replace(ep, table=((g, n, nc, th - 1, thc),) + ep.table[1:]),
                            cals, sigs)
    assert not verify_epoch(replace(ep, cal_root=h16(b"x")), cals, sigs)
    assert not verify_epoch(ep, cals[1:], sigs)
    assert not verify_epoch(replace(ep, alpha=_ppm(0.05)), cals, sigs)
    assert not verify_epoch(replace(ep, cal_hw="gh200"), cals, sigs)
    assert not verify_epoch(ep, [replace(c, obs=2) if c is cals[0] else c for c in cals], sigs)


def test_active_epoch_is_max_seq():
    cals = [_cal(i, 1000 * i) for i in range(60)]
    eps = [_build(cals, seq=s) for s in (3, 1, 7, 2)] + [_build(
        [_cal(i, 1000 * i, scope="us") for i in range(60)], scope="us", seq=9)]
    assert active_epoch(eps, "eu").seq == 7
    assert active_epoch(eps, "us").seq == 9
    assert active_epoch(eps, "ap") is None
    # equal seq: a deterministic choice independent of arrival order
    twin = _build(cals, seq=7, created_at=5)
    a, b = active_epoch(eps + [twin], "eu"), active_epoch([twin] + eps, "eu")
    assert a == b and a.seq == 7
    assert isinstance(a, EpochRecord)
    assert object_id(canonical_bytes(a)) == max(object_id(canonical_bytes(e))
                                                for e in (twin, eps[2]))


def test_epoch_changing_selector_requires_new_cal_root():
    cals = [_cal(i, 1000 + i) for i in range(60)]
    sigs = [_sig(c, c.s_q) for c in cals]
    ep = _build(cals)
    assert verify_epoch(ep, cals, sigs)
    for other in ((51000, 3, MPOL), (52000, 4, MPOL), (52000, 3, m_pol({"A": 2}, {}, {}, 1))):
        with pytest.raises(ValueError):              # the same Cal set under another selector
            _build(cals, selector_params=other)
        relabelled = replace(ep, sigma_lo_q=other[0], m=other[1], m_pol=other[2])
        assert not verify_epoch(relabelled, cals, sigs)
    # a Cal set drawn under the new selector builds and verifies
    new = (51000, 3, MPOL)
    cals2 = [_cal(i, 1000 + i, sel=new) for i in range(60)]
    ep2 = _build(cals2, selector_params=new)
    assert ep2.cal_root != ep.cal_root and verify_epoch(ep2, cals2, [_sig(c, c.s_q) for c in cals2])


def test_epoch_cannot_rebin_a_reused_cal_set():
    cals = [_cal(i, 500 * i, bin=i % 3) for i in range(120)]
    sigs = [_sig(c, c.s_q) for c in cals]
    ep = _build(cals)
    assert verify_epoch(ep, cals, sigs)
    assert not verify_epoch(replace(ep, bins=(128, "doubling")), cals, sigs)
    with pytest.raises(ValueError):
        _build(cals, bins=(128, "doubling"))


def test_cal_score_is_a_single_observation_on_cal_hw():
    cals = [_cal(i, 1000 + 10 * i) for i in range(60)]
    honest = [_sig(c, c.s_q) for c in cals]
    assert verify_epoch(_build(cals), cals, honest)
    # gh200 observed a lower score for pair 3; the Cal author lowers s_q to the joined minimum
    low = _sig(cals[3], cals[3].s_q - 200, hw="gh200")
    forged = cals[:3] + [replace(cals[3], s_q=cals[3].s_q - 200)] + cals[4:]
    assert not verify_epoch(_build(forged), forged, honest + [low])
    assert verify_epoch(_build(cals), cals, honest + [low])     # the honest epoch still verifies
    # raising s_q above the observation is not an observation either
    raised = cals[:3] + [replace(cals[3], s_q=cals[3].s_q + 1)] + cals[4:]
    assert not verify_epoch(_build(raised), raised, honest)
    # the observation's score is min over BOTH directions: an asymmetric record scores lower
    c = cals[5]
    asym = replace(honest[5], pay_yx=(0, c.s_q - 100, FIXED_ONE - c.s_q + 100))
    assert not verify_epoch(_build(cals), cals, honest[:5] + [asym] + honest[6:])
    # the observation must also carry the epoch's scope
    other_scope = [_sig(c, c.s_q, scope="us") for c in cals]
    assert not verify_epoch(_build(cals), cals, other_scope)


def test_mutant_killing_inputs_from_behavioural_review():
    assert quantile_threshold(list(range(1, 1000)), 59000) == 941
    assert derive_B0(20000, 0, "0.5", "0.8") == 128        # q = 0 accepted (bound 122.5)
    assert derive_B0(200000, 0, 1, 1) == 4                 # exact power of two stays
    assert derive_B0(20000, 0, 1, "0.191") == 512          # bound 256.54
    # exactly 16 in decimal arithmetic; binary 0.1 would give 16 + eps and B_0 = 32
    assert derive_B0(100000, "0.1", 1, "0.625") == derive_B0(100000, 0.1, 1, 0.625) == 16
    cals = [_cal(i, 100 * i, bin=0) for i in range(60)] + \
        [_cal(100 + i, 100 * i, bin=2) for i in range(60)]
    ep = _build(cals)
    assert [r[0] for r in ep.table] == [0, 2] and threshold(ep, 1) == INF16
    sigs = [_sig(c, c.s_q) for c in cals]
    # a later record with a higher p_e must not raise the joined score (join, not last writer)
    assert verify_epoch(ep, cals, sigs + [_sig(cals[3], cals[3].s_q + 5)])


def test_epoch_row_at_n_min_minus_one_and_n_min_and_mondrian_with_infinite_theta_g():
    a = _ppm(0.02)
    short = _build([_cal(i, 100 * i) for i in range(n_min(a) - 1)])
    full = _build([_cal(i, 100 * i) for i in range(n_min(a))])
    assert short.table == ((0, 48, 0, INF16, INF16),) and threshold(short, 0) == INF16
    assert full.table == ((0, 49, 0, 4800, INF16),) and threshold(full, 0) == 4800
    # 30 contradiction records: theta_c finite at alpha_c = 0.05, theta_g infinite at 0.02
    ep = _build([_cal(i, 100 * i, subtype="CONTRADICTION") for i in range(30)], mondrian_c=True)
    g, n, nc, th, thc = ep.table[0]
    assert th == INF16 and thc < INF16 and threshold(ep, 0) == INF16


def test_bin_of_refuses_bad_arguments():
    for n, b0 in ((-1, 256), (5, 0), (5, -4)):
        with pytest.raises(ValueError):
            bin_of(n, b0)
