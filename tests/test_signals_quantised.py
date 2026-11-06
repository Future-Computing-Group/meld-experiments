"""Quantised signals (int8 q, exact top-m, fixed-point Sig payloads).

Each test pins a property over a class of inputs (maximal, random, adversarial ties, every thread
count), not one example.
"""
from __future__ import annotations

import math
import operator
from fractions import Fraction

import numpy as np
import pytest

from meld.merge import signals
from meld.merge.signals import (chi, int8_gram, nli_probs, quantize_int8, sigma_q_cmp, to_fixed16,
                                topm_exact)
from meld.protocol.objects import FIXED_ONE

D = 384


def _py_dot(a, b) -> int:
    return sum(map(operator.mul, (int(x) for x in a), (int(y) for y in b)))


def _exact_cos(a, b) -> Fraction:
    """Independent exact reference: sign(cos) * cos^2 as a Fraction (monotone in cos; zero norm -> 0)."""
    d, nn = _py_dot(a, b), _py_dot(a, a) * _py_dot(b, b)
    return Fraction(0) if nn == 0 else Fraction((d > 0) - (d < 0)) * Fraction(d * d, nn)


def _key(p: int) -> Fraction:
    """The same monotone transform of the threshold p / FIXED_ONE."""
    return Fraction(((p > 0) - (p < 0)) * p * p, FIXED_ONE * FIXED_ONE)


def _brute_topm(qb, Q, m, sigma_lo):
    cand = [(_exact_cos(qb, r), i) for i, r in enumerate(Q)]
    cand = [(c, i) for c, i in cand if c >= _key(sigma_lo)]
    cand.sort(key=lambda t: (-t[0], t[1]))
    return [i for _, i in cand[:m]]


# ---------------------------------------------------------------- int8 products and the GEMM
def test_int8_products_cast_before_multiply_and_float32_gemm_is_exact():
    hi = np.full((1, D), 127, dtype=np.int8)
    lo = np.full((1, D), -127, dtype=np.int8)
    # the trap the spec names: numpy int8 @ int8 stays int8 and wraps
    assert (hi[0] @ hi[0]).dtype == np.int8 and int(hi[0] @ hi[0]) == -128
    for a, b, want in ((hi, hi, 6_193_536), (lo, lo, 6_193_536), (hi, lo, -6_193_536)):
        g = int8_gram(a, b)
        assert g.dtype == np.int64 and int(g[0, 0]) == want
    # the guard: anything but int8 in is refused before a product is formed
    # the guard: anything but int8 in is refused before a product is formed (explicit raise, not
    # assert, so it survives python -O)
    with pytest.raises(TypeError):
        int8_gram(hi.astype(np.int16), hi)
    with pytest.raises(TypeError):
        int8_gram(hi, hi.astype(np.float32))
    # the 2^24 bound is tight: d = 1024 all -128 gives exactly 2^24, d = 1025 is refused
    big = np.full((1, 1024), -128, dtype=np.int8)
    assert int(int8_gram(big, big)[0, 0]) == 2**24
    with pytest.raises(ValueError):
        int8_gram(np.zeros((1, 1025), np.int8), np.zeros((1, 1025), np.int8))
    rng = np.random.default_rng(20261009)
    A = rng.integers(-128, 128, size=(5000, D), dtype=np.int8)
    B = rng.integers(-128, 128, size=(50, D), dtype=np.int8)
    G = int8_gram(A, B)
    Al, Bl = A.tolist(), B.tolist()
    for j, b in enumerate(Bl):
        col = [sum(map(operator.mul, a, b)) for a in Al]
        assert G[:, j].tolist() == col


def test_sigma_q_squared_comparisons_do_not_overflow():
    # near-maximal vectors at threshold r = FIXED_ONE: r^2 <q_x,q_y>^2 reaches ~1.65e23 > int64 max
    x = np.full(D, 127, dtype=np.int8)
    y = x.copy()
    y[0] = 126
    dot, nx, ny = _py_dot(x, y), _py_dot(x, x), _py_dot(y, y)
    assert FIXED_ONE**2 * dot**2 > np.iinfo(np.int64).max
    cos = _exact_cos(x, y)
    for p in range(FIXED_ONE - 3, FIXED_ONE + 2):
        want = (cos > _key(p)) - (cos < _key(p))
        assert sigma_q_cmp(x, y, p) == want
    assert sigma_q_cmp(x, x, FIXED_ONE) == 0          # identical vectors: exactly cosine 1
    assert sigma_q_cmp(x, -x, 0) == -1                 # sign handled before squaring
    assert sigma_q_cmp(x, y, -FIXED_ONE) == 1 and sigma_q_cmp(x, -y, FIXED_ONE // 2) == -1
    assert sigma_q_cmp(x, np.zeros(D, np.int8), 0) == 0  # zero norm -> sigma_q = 0
    with pytest.raises(TypeError):
        sigma_q_cmp(x, y, 0.5)                          # thresholds are fixed-point ints
    with pytest.raises(TypeError):
        sigma_q_cmp(x, y.astype(np.int16), 0)
    # negative cosines against negative thresholds (sign first, then reversed squares)
    for p in range(-FIXED_ONE + 3, -FIXED_ONE - 2, -1):
        want = (_exact_cos(x, -y) > _key(p)) - (_exact_cos(x, -y) < _key(p))
        assert sigma_q_cmp(x, -y, p) == want
    rng = np.random.default_rng(7)
    for _ in range(300):
        a = rng.integers(-127, 128, D, dtype=np.int8)
        b = np.clip(a.astype(np.int16) + rng.integers(-2, 3, D), -127, 127).astype(np.int8)
        if rng.random() < 0.5:
            b = -b
        c = _exact_cos(a, b)
        p = int(rng.integers(-FIXED_ONE, FIXED_ONE + 1))
        t = _key(p)
        assert sigma_q_cmp(a, b, p) == (c > t) - (c < t)


def _formula_vectors(n):
    # platform-independent by construction (no RNG): q[i, j] in [-127, 127]
    return np.array([[((i * 31 + j * 17 + i * j) % 255) - 127 for j in range(D)] for i in range(n)],
                    dtype=np.int8)


# recorded fixture (computed once by brute force; integers do not depend on platform or threads)
_FIX_TOPM = [16, 50, 33, 84, 67, 43, 28, 75]
_FIX_DOTS = [1056119, 762784, 633159, 299364, 278114, 91529, 90779, 84459]


def test_sigma_q_exact_across_thread_counts_and_platforms():
    Q = _formula_vectors(101)
    qb, Q = Q[0], Q[1:]
    want_dots = [_py_dot(qb, r) for r in Q]
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover
        pytest.skip("threadpoolctl not installed")
    import os
    results = []
    for n in (1, max(2, os.cpu_count() or 2)):
        with threadpool_limits(limits=n):
            g = int8_gram(Q, qb[None, :])[:, 0].tolist()
            results.append((g, topm_exact(qb, Q, 8, 0)))
    assert results[0] == results[1]
    assert results[0][0] == want_dots
    assert results[0][1] == _FIX_TOPM == _brute_topm(qb, Q, 8, 0)
    assert [want_dots[i] for i in _FIX_TOPM] == _FIX_DOTS


# ---------------------------------------------------------------- exact top-m
def test_topm_exact_equals_brute_force_including_ties():
    rng = np.random.default_rng(3)
    # random stores
    for trial in range(40):
        n = int(rng.integers(0, 60))
        Q = rng.integers(-127, 128, (n, D), dtype=np.int8)
        qb = rng.integers(-127, 128, D, dtype=np.int8)
        if n and trial % 2:
            Q[: n // 2] = np.clip(qb.astype(np.int16) + rng.integers(-40, 41, (n // 2, D)), -127, 127)
        m = int(rng.integers(1, 8))
        lo = int(rng.integers(-FIXED_ONE, FIXED_ONE // 2))   # negative floors rank negative cosines
        assert topm_exact(qb, Q, m, lo) == _brute_topm(qb, Q, m, lo)
    # adversarial: exact duplicates, scaled copies (equal cosine, different norm), floor exactly hit
    qb = rng.integers(-127, 128, D, dtype=np.int8)
    half = (qb // 2).astype(np.int8)
    base = (qb.astype(np.int16) // 3 * 2).astype(np.int8)  # c * (qb//3): equal cosine family
    third = (qb.astype(np.int16) // 3).astype(np.int8)
    noise = rng.integers(-127, 128, (5, D), dtype=np.int8)
    Q = np.stack([noise[0], third, qb, base, qb, half, noise[1], third, np.zeros(D, np.int8),
                  -qb, noise[2]]).astype(np.int8)
    for m in range(1, 12):
        for lo in (0, 1, FIXED_ONE // 2, FIXED_ONE - 1, FIXED_ONE):
            assert topm_exact(qb, Q, m, lo) == _brute_topm(qb, Q, m, lo), (m, lo)
    # duplicates of qb tie at cosine exactly 1 and are ordered by index; floor FIXED_ONE keeps exactly them
    assert topm_exact(qb, Q, 11, FIXED_ONE) == [2, 4]
    # exact ties that float64 cosines split: k * v for k = 1, 2, 3 have one cosine, but
    # d / sqrt(n) and 3d / sqrt(9n) can differ in the last bit; ties must still go by index
    float_split = 0
    for _ in range(30):
        v = rng.integers(-42, 43, D).astype(np.int16)
        for order in ((3, 1, 2), (2, 3, 1), (1, 3, 2)):
            Q = np.stack([k * v for k in order]).astype(np.int8)
            assert topm_exact(qb, Q, 3, -FIXED_ONE) == [0, 1, 2]
            d = [int(k * v.astype(np.int64) @ qb.astype(np.int64)) for k in order]
            n = [int((k * v.astype(np.int64)) @ (k * v.astype(np.int64))) * _py_dot(qb, qb) for k in order]
            float_split += len({x / math.sqrt(y) for x, y in zip(d, n)}) > 1
    assert float_split > 0, "fixture must contain ties that a float ranking would split"
    # near-ties that a float32 cosine cannot separate are resolved exactly
    x = np.full(D, 127, dtype=np.int8)
    near = []
    for k in range(6):
        v = x.copy()
        v[k] = 126 - (k % 2)
        near.append(v)
    Q = np.stack(near + [x]).astype(np.int8)
    for m in range(1, 8):
        assert topm_exact(x, Q, m, 0) == _brute_topm(x, Q, m, 0)
    assert topm_exact(x, Q[:0], 3, 0) == [] and topm_exact(x, Q, 0, 0) == []


def test_topm_exact_is_exact_under_any_float_error_below_half_the_band(monkeypatch):
    # The float cosine is only a pre-sort: make it wrong by up to 0.45 band, pseudo-randomly per
    # norm, and the result must still equal the exact brute force (floor, m-th cut, ties, chains).
    band = 1.0 / FIXED_ONE

    class _LyingMath:
        @staticmethod
        def sqrt(n):
            return math.sqrt(n) * (1 + 0.45 * band * (((n * 2654435761) % 1001) / 500 - 1))

    monkeypatch.setattr(signals, "math", _LyingMath)
    rng = np.random.default_rng(99)
    for _ in range(25):
        qb = rng.integers(-127, 128, D).astype(np.int16)
        Q = []
        for _ in range(60):                       # dense cluster: cosines a fraction of a band apart
            r = qb.copy()
            idx = rng.choice(D, size=int(rng.integers(0, 40)), replace=False)
            r[idx] = np.clip(r[idx] + rng.choice([-1, 1], size=len(idx)), -127, 127)
            Q.append(r)
        Q.append(qb.copy())
        Q.append(2 * (qb // 2))
        Q = np.stack(Q).astype(np.int8)
        qb = qb.astype(np.int8)
        cos = sorted((_exact_cos(qb, r) for r in Q), reverse=True)
        mid = int(math.sqrt(float(cos[len(cos) // 2])) * FIXED_ONE)
        for m in (1, 5, 20, 62):
            for lo in (0, mid - 1, mid, mid + 1, FIXED_ONE):
                assert topm_exact(qb, Q, m, lo) == _brute_topm(qb, Q, m, lo), (m, lo)


def test_topm_exact_refuses_bad_arguments():
    x = np.ones(D, np.int8)
    Q = np.ones((3, D), np.int8)
    for bad in ((x, Q, -1, 0, ValueError), (x, Q, 2, 0.5, TypeError), (x, Q, 2.0, 0, TypeError),
                (x, Q, 2, True, TypeError), (x.astype(np.int16), Q, 2, 0, TypeError),
                (x, Q.astype(np.float32), 2, 0, TypeError)):
        with pytest.raises(bad[-1]):
            topm_exact(*bad[:-1])


def test_floor_keep_predicate_decides_inside_the_band_in_integers():
    # cos = 3/5 exactly = 39321 / FIXED_ONE; a float approximation on either side of the floor
    # but within the band must not decide: the integer pair does.
    pair, p = (3, 25), 39321
    lo = p / FIXED_ONE
    eps = 0.25 / FIXED_ONE
    assert signals._keep(lo - eps, pair, p) and signals._keep(lo + eps, pair, p)
    assert not signals._keep(lo + eps, pair, p + 1)       # exact 0.6 < (p + 1) / FIXED_ONE
    assert signals._keep(lo + 2 / FIXED_ONE, (0, 1), p)    # outside the band the float decides
    assert not signals._keep(lo - 2 / FIXED_ONE, pair, p)


# ---------------------------------------------------------------- quantisation
def test_quantize_int8_is_deterministic_symmetric_and_maxabs_scaled():
    rng = np.random.default_rng(11)
    for _ in range(200):
        v = rng.normal(size=D)
        q = quantize_int8(v)
        assert q.dtype == np.int8 and q.shape == (D,)
        assert int(np.abs(q.astype(np.int16)).max()) == 127 and int(q.min()) >= -127  # never -128
        assert np.array_equal(q, quantize_int8(v.copy())) and np.array_equal(q, quantize_int8(list(v)))
        assert np.array_equal(quantize_int8(-v), -q)                  # symmetric
        assert np.array_equal(quantize_int8(v * 3.5), q)              # scale-invariant
        want = np.rint(127.0 * v / np.abs(v).max())                   # stated rule: half-to-even
        assert np.array_equal(q, want.astype(np.int8))
    assert np.array_equal(quantize_int8(np.zeros(D)), np.zeros(D, np.int8))
    assert np.array_equal(quantize_int8([0.5, -1.0, 0.25]), np.array([64, -127, 32], np.int8))
    # 127 * v / 127 = 62.5 exactly: half to even gives 62, not 63
    assert np.array_equal(quantize_int8([127.0, 62.5, -62.5, 63.5]), np.array([127, 62, -62, 64], np.int8))
    with pytest.raises(ValueError):
        quantize_int8([1.0, float("nan")])
    # huge magnitudes: an exact power-of-two pre-scale avoids overflow, rounding unchanged
    assert np.array_equal(quantize_int8([1e308, -5e307, 1.0]), np.array([127, -64, 0], np.int8))
    v = rng.normal(size=D) * 1e306
    assert np.array_equal(quantize_int8(-v), -quantize_int8(v))
    assert np.array_equal(quantize_int8(v), quantize_int8(v / 1e306 * 1e306))
    with pytest.raises(ValueError):
        quantize_int8(np.ones((2, D)))                    # one vector at a time (per-vector scale)


# ---------------------------------------------------------------- fixed point
def test_to_fixed16_range_and_rounding():
    assert to_fixed16((0.0, 1.0, 0.5)) == (0, FIXED_ONE, 32768)  # 32767.5 -> 32768 (half to even)
    halves = (1 / 131070, 5 / 131070, 9 / 131070)                # p * FIXED_ONE = 0.5, 2.5, 4.5 exactly
    assert [p * FIXED_ONE for p in halves] == [0.5, 2.5, 4.5]
    assert to_fixed16(halves) == (0, 2, 4)                         # half to even, not half up
    rng = np.random.default_rng(5)
    for p in rng.random(2000):
        (v,) = to_fixed16((float(p),))
        assert type(v) is int and 0 <= v <= FIXED_ONE and abs(v - p * FIXED_ONE) <= 0.5
    for bad in ((-0.01,), (1.01,), (1 + 1e-9,), (-1e-12,), (float("nan"),), (float("inf"),)):
        with pytest.raises(ValueError):
            to_fixed16(bad)
    with pytest.raises(TypeError):
        to_fixed16((True, 0.0, 0.0))


# ---------------------------------------------------------------- NLI probability forms
_PAIRS = [("Paris is in France", "Paris is in Italy"), ("the budget increased", "the budget decreased"),
          ("the sky is blue", "the sky is blue today"), ("cats purr", "quantum chromodynamics is hard"),
          ("the server is not active", "the server is active"), ("", "x"),
          ("the sky is blue", "the sky is blue")]


def test_lexical_probability_form_is_valid_deterministic_and_agrees_with_chi():
    order = ("contradicts", "entails", "neutral")
    for a, b in _PAIRS:
        p = nli_probs(a, b, model="lexical")
        assert len(p) == 3 and all(type(x) is float and 0.0 <= x <= 1.0 for x in p)
        assert math.isclose(sum(p), 1.0, abs_tol=1e-12)
        assert p == nli_probs(a, b, model="lexical")
        assert p == nli_probs(b, a, model="lexical")     # the lexical rung is symmetric (documented)
        assert order[int(np.argmax(p))] == chi(a, b, model="lexical")
        assert to_fixed16(p) == to_fixed16(nli_probs(a, b, model="lexical"))
    with pytest.raises(ValueError):
        nli_probs("a", "b", model="no-such-model")
    # pinned values (J = 1/3 contradiction; J = 2/3 entailment)
    assert nli_probs("Paris is in France", "Paris is in Italy", model="lexical") == (
        0.6666666666666666, 0.16666666666666669, 0.16666666666666669)
    assert nli_probs("the sky is blue", "the sky is blue today", model="lexical") == (
        0.08333333333333334, 0.8333333333333333, 0.08333333333333334)


class _FakeConfig:
    def __init__(self, id2label):
        self.id2label = id2label


class _FakeCE:
    """Offline stand-in: emits fixed logits-turned-probabilities in the model's own label order."""

    def __init__(self, id2label, probs_by_label):
        self.config = _FakeConfig(id2label)
        self._p = [probs_by_label[id2label[i].lower()] for i in range(len(id2label))]
        self.seen = []

    def predict(self, pairs, apply_softmax=False):
        assert apply_softmax
        self.seen.extend(pairs)
        return np.array([self._p])


def test_cross_encoder_triple_follows_config_label_order():
    want = {"contradiction": 0.1, "entailment": 0.7, "neutral": 0.2}
    for id2label in ({0: "contradiction", 1: "entailment", 2: "neutral"},
                     {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"},   # bart-large-mnli order
                     {0: "neutral", 1: "contradiction", 2: "entailment"}):
        ce = signals._CrossEncoderNLI("fake")
        ce._model = _FakeCE(id2label, want)
        assert ce.probs("premise", "hypothesis") == (0.1, 0.7, 0.2)
        assert ce._model.seen == [("premise", "hypothesis")]          # premise first
        assert ce("premise", "hypothesis") == "entails"
    for bad in ({0: "contradiction", 1: "entailment"}, {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
                {0: "contradiction", 1: "contradiction", 2: "neutral"}):
        ce = signals._CrossEncoderNLI("fake")
        ce._model = _FakeCE(bad, {v.lower(): 1 / 3 for v in bad.values()})
        with pytest.raises(ValueError):
            ce.probs("a", "b")


def test_cross_encoder_is_directional_specific_entails_general():
    pytest.importorskip("sentence_transformers")
    s, g = "A man is playing a guitar on stage.", "A man is playing an instrument."
    try:
        fwd = nli_probs(s, g, model="mnli-cross-encoder")
    except Exception as exc:  # pragma: no cover - weights unavailable offline
        pytest.skip(f"cross-encoder unavailable: {exc}")
    back = nli_probs(g, s, model="mnli-cross-encoder")
    assert int(np.argmax(fwd)) == 1 and int(np.argmax(back)) != 1


def test_cross_encoder_probability_form_is_softmax_triple():
    pytest.importorskip("sentence_transformers")
    try:
        p = nli_probs("A man is eating.", "A man is not eating.", model="mnli-cross-encoder")
    except Exception as exc:  # pragma: no cover - weights unavailable offline
        pytest.skip(f"cross-encoder unavailable: {exc}")
    assert len(p) == 3 and math.isclose(sum(p), 1.0, abs_tol=1e-5) and min(p) >= 0.0
    assert ("contradicts", "entails", "neutral")[int(np.argmax(p))] == chi(
        "A man is eating.", "A man is not eating.", model="mnli-cross-encoder")


def test_topm_exact_vectorised_presort_tolerates_float_error_and_cached_matrices(monkeypatch):
    # the vectorised pre-sort (signals._approx) lies by up to 0.45 band; cached float32 rows and
    # int64 norms (as the fold keeps them) give the same exact answer as the brute force
    band = 1.0 / FIXED_ONE
    real = signals._approx

    def lying(dots, nn):
        a = real(dots, nn)
        return a * (1 + 0.45 * band * (((nn.astype(np.int64) * 2654435761) % 1001) / 500 - 1))
    monkeypatch.setattr(signals, "_approx", lying)
    rng = np.random.default_rng(7)
    for _ in range(25):
        qb = rng.integers(-127, 128, D).astype(np.int16)
        Q = [np.clip(qb + rng.choice([-1, 0, 1], size=D) * (rng.random(D) < .1), -127, 127)
             for _ in range(60)] + [qb.copy(), 2 * (qb // 2)]
        Q = np.stack(Q).astype(np.int8)
        qb = qb.astype(np.int8)
        Qf, norms = Q.astype(np.float32), (Q.astype(np.int64) ** 2).sum(axis=1)
        cos = sorted((_exact_cos(qb, r) for r in Q), reverse=True)
        mid = int(math.sqrt(float(cos[len(cos) // 2])) * FIXED_ONE)
        for m in (1, 5, 20, 62):
            for lo in (0, mid - 1, mid, mid + 1, FIXED_ONE):
                want = _brute_topm(qb, Q, m, lo)
                assert topm_exact(qb, Q, m, lo) == want, (m, lo)
                assert topm_exact(qb, Q, m, lo, Qf=Qf, norms=norms) == want, (m, lo)
    with pytest.raises(ValueError):
        topm_exact(qb, Q, 3, 0, Qf=Qf[:-1], norms=norms)
