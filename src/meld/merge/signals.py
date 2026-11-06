"""The merge signals: three core (kappa / sigma / chi) plus an optional epsilon.

REUSE posture
-------------
- sigma REUSES the experiment-template embedding plumbing: embed once, cache per
  (encoder, text-hash) to disk, return a real cosine. The encoder ladder mirrors
  the h-residual template: a weak deterministic 'hashing' rung (offline, fast,
  for tests + local smoke) and a 'minilm' rung (sentence-transformers, for
  stronger sweeps). NO silent default: the encoder must be named.
- kappa is a one-line content-addressed claim-key identity over the vendored
  proto Claim.
- epsilon is a set Jaccard over shared grounding (GROUNDS) endpoints. It is an
  OPTIONAL signal, disabled by default (eps_hi=0): the three core signals drive
  the evaluated decision procedure; epsilon only participates when eps_hi>0.
- chi is a three-way NLI; the model is config-driven (no silent default). A
  deterministic 'lexical' backend keeps tests/local-smoke offline and fast; a
  cross-encoder backend ('mnli-cross-encoder') is available for stronger runs.

The numbers this module emits are real (cosine over cached embeddings, lexical
NLI heuristics); none are fabricated. Stronger encoder/NLI rungs only sharpen
the same signals.
"""
from __future__ import annotations

import hashlib
import math
from functools import cmp_to_key
import os
import re
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np

from meld.proto import Claim
from meld.protocol.objects import FIXED_ONE

# ---------------------------------------------------------------------------
# kappa: content-addressed claim-key identity
# ---------------------------------------------------------------------------


def kappa(a: Claim, b: Claim) -> int:
    """1 iff the two claims are byte-identical after canonicalisation (same
    content-addressed claim_key); else 0."""
    return int(a.claim_key == b.claim_key and a.claim_key != b"")


# ---------------------------------------------------------------------------
# epsilon: Jaccard overlap of grounding-evidence endpoint sets
# ---------------------------------------------------------------------------


def epsilon(ev_a: Set[str], ev_b: Set[str]) -> float:
    """|A ∩ B| / |A ∪ B| over grounding endpoint sets. Empty/empty -> 0.0."""
    a, b = set(ev_a), set(ev_b)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# sigma: dense cosine over cached embeddings
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _HashingEncoder:
    """Deterministic, offline, fast bag-of-tokens hashing encoder.

    Maps each token to a fixed bucket via sha1, accumulates an L2-normalised
    vector. Deterministic (no weights to download), so it is the test + local-
    smoke rung. A STRONGER synthetic/real encoder reveals more of the text;
    this rung is the cheap monotone baseline (mirrors the template's tfidf rung).
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        for tok in _tokenize(text):
            h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v


class _SentenceTransformerEncoder:
    """Stronger rung via sentence-transformers (downloads weights once).

    REUSED from the experiment-template encoder ladder. Used for stronger sweeps;
    not exercised in the offline test path.
    """

    _MODEL_IDS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2",
                  "mpnet": "sentence-transformers/all-mpnet-base-v2",
                  "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2"}

    def __init__(self, rung: str):
        self.rung = rung
        self._model = None

    def encode_one(self, text: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Pin to CPU: this is a tiny encoder, and on a shared GPU the LLM
            # (vLLM at gpu_memory_utilization~0.90) already owns almost all VRAM,
            # so loading minilm on CUDA OOMs (caught on the minilm smoke:
            # naive_union, the most units to embed, failed with 46 MiB short).
            # Honor an explicit override via SMP_ST_DEVICE if ever needed.
            import os
            device = os.environ.get("SMP_ST_DEVICE", "cpu")
            self._model = SentenceTransformer(self._MODEL_IDS[self.rung],
                                              device=device)
        emb = self._model.encode([text], normalize_embeddings=True,
                                 show_progress_bar=False)
        return np.asarray(emb[0], dtype=np.float64)


# int8 embedding length per encoder rung: a delta whose q has another length is never a node
ENCODER_DIM = {"hashing": 256, "minilm": 384, "minilm-l12": 384, "mpnet": 768}


def _make_encoder(name: str):
    if name == "hashing":
        return _HashingEncoder()
    if name in _SentenceTransformerEncoder._MODEL_IDS:
        return _SentenceTransformerEncoder(name)
    raise ValueError(f"unknown encoder rung {name!r}")


class EmbeddingSimilarity:
    """sigma provider: embed once, cache per (encoder, text-hash), real cosine.

    No silent default: ``encoder`` is required.
    """

    def __init__(self, encoder: str, cache_dir: str | None = None):
        self.encoder_name = encoder
        self.encoder = _make_encoder(encoder)
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._mem: dict[str, np.ndarray] = {}
        self.cache_hits = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]

    def _embed(self, text: str) -> np.ndarray:
        key = self._key(text)
        if key in self._mem:
            self.cache_hits += 1
            return self._mem[key]
        path = os.path.join(self.cache_dir, f"{self.encoder_name}.{key}.npy") if self.cache_dir else None
        if path and os.path.exists(path):
            v = np.load(path)
            self._mem[key] = v
            self.cache_hits += 1
            return v
        v = self.encoder.encode_one(text)
        self._mem[key] = v
        if path:
            np.save(path, v)
        return v

    def sigma(self, text_a: str, text_b: str) -> float:
        va, vb = self._embed(text_a), self._embed(text_b)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))


def sigma(text_a: str, text_b: str, *, encoder: str, cache_dir: str | None = None) -> float:
    """Functional sigma (constructs a one-shot EmbeddingSimilarity). For repeated
    calls prefer an ``EmbeddingSimilarity`` instance so the cache persists."""
    return EmbeddingSimilarity(encoder=encoder, cache_dir=cache_dir).sigma(text_a, text_b)


# ---------------------------------------------------------------------------
# sigma_q: int8 embeddings, exact products, exact top-m (r3 3.1, 3.3)
#
# Guards raise (never assert), so they hold under python -O. A zero vector has sigma_q = 0 with
# every vector, so it is a top-m candidate only when sigma_lo <= 0.
# ---------------------------------------------------------------------------


def _int8(*arrays) -> None:
    for x in arrays:
        if not isinstance(x, np.ndarray) or x.dtype != np.int8:
            raise TypeError(f"int8 array required, got {getattr(x, 'dtype', type(x))}")


def _int(*vals) -> None:
    for v in vals:
        if type(v) is not int:
            raise TypeError(f"int required, got {type(v).__name__}")


def quantize_int8(vec) -> np.ndarray:
    """One vector v -> q = rint(127 * v / max|v|) as int8; rint rounds half to even. Per-vector
    max-abs scale (cosine is scale-free, so the scale is not carried); range [-127, 127], never -128,
    so q(-v) = -q(v). v is first scaled by the exact power of two 2^-e (e = frexp exponent of max|v|),
    which cannot overflow and leaves the quotient bit-identical. The zero vector maps to zeros;
    non-finite or non-1-D input is refused. Wire form: ``.tobytes()``; back: ``np.frombuffer(q, np.int8)``."""
    v = np.asarray(vec, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError("quantize_int8 takes one vector (per-vector scale)")
    if not np.isfinite(v).all():
        raise ValueError("non-finite embedding")
    mx = np.abs(v).max() if v.size else 0.0
    if mx == 0.0:
        return np.zeros(v.shape, dtype=np.int8)
    e = int(np.frexp(mx)[1])
    return np.rint(127.0 * np.ldexp(v, -e) / np.ldexp(mx, -e)).astype(np.int8)


def int8_gram(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Exact A @ B.T for int8 rows, as int64. numpy int8 @ int8 stays int8 and wraps, so the inputs
    must be int8 and are cast to float32 before the product; the float32 GEMM is exact because every
    partial sum is an integer of magnitude <= d * 128^2 <= 1024 * 128^2 = 2^24, whatever the BLAS
    order or thread count."""
    _int8(A, B)
    if A.shape[-1] != B.shape[-1] or A.shape[-1] > 1024:
        raise ValueError("dimensions must match and d <= 1024 (2^24 exactness bound)")
    return (A.astype(np.float32) @ B.astype(np.float32).T).astype(np.int64)


def _cos_cmp(x: Tuple[int, int], y: Tuple[int, int]) -> int:
    """Sign of dx/sqrt(nx) - dy/sqrt(ny) for Python ints with nx, ny > 0: sign first, then the
    cross-multiplied squares (these exceed int64, hence Python ints)."""
    (dx, nx), (dy, ny) = x, y
    sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
    if sx != sy:
        return (sx > sy) - (sx < sy)
    lhs, rhs = dx * dx * ny, dy * dy * nx
    c = (lhs > rhs) - (lhs < rhs)
    return c if sx >= 0 else -c


def _cos_pair(dot: int, nn: int) -> Tuple[int, int]:
    """sigma_q as (dot, |x|^2 |y|^2); a zero norm gives sigma_q = 0."""
    return (int(dot), int(nn) or 1)   # nn = 0 implies dot = 0


def sigma_q_cmp(qx: np.ndarray, qy: np.ndarray, theta_q: int) -> int:
    """Sign of sigma_q(qx, qy) - theta_q / FIXED_ONE, exact in Python integers:
    sigma_q >= p / r iff <qx,qy> >= 0 and r^2 <qx,qy>^2 >= p^2 |qx|^2 |qy|^2 (for p >= 0)."""
    _int8(qx, qy)
    _int(theta_q)
    x, y = qx.tolist(), qy.tolist()
    dot = sum(a * b for a, b in zip(x, y))
    nn = sum(a * a for a in x) * sum(b * b for b in y)
    return _cos_cmp(_cos_pair(dot, nn), (theta_q, FIXED_ONE * FIXED_ONE))


# Float64 cosines of exact integers carry relative error ~1e-15; any two candidates whose float
# cosines differ by more than this band are ordered correctly by the float. Within the band (one
# fixed-point step) the order is decided in Python integers.
_BAND = 1.0 / FIXED_ONE


def _keep(a: float, pair: Tuple[int, int], sigma_lo: int) -> bool:
    """Floor test sigma_q >= sigma_lo / FIXED_ONE: the float cosine ``a`` decides outside the band,
    the exact ``pair`` inside it."""
    lo = sigma_lo / FIXED_ONE
    if abs(a - lo) >= _BAND:
        return a > lo
    return _cos_cmp(pair, (sigma_lo, FIXED_ONE * FIXED_ONE)) >= 0


def _approx(dots: np.ndarray, nn: np.ndarray) -> np.ndarray:
    """Float64 cosines of the exact integer (dot, |x|^2 |y|^2) pairs: the pre-sort only."""
    return dots / np.sqrt(nn)


def topm_exact(qb: np.ndarray, Q: np.ndarray, m: int, sigma_lo: int, *,
               Qf: np.ndarray = None, norms: np.ndarray = None) -> List[int]:
    """Row indices of the top-m rows of Q by sigma_q(qb, Q[i]) among those with
    sigma_q >= sigma_lo / FIXED_ONE, descending, exact ties by row index. Callers pass Q's rows in
    canonical order, so index ties are ties by that order (r3 3.3).

    Float32 GEMM of the cast vectors (exact integers), float64 cosine for ranking, then every cluster
    of candidates chained within ``_BAND`` of each other (which covers the m-th cut and every tie) is
    re-sorted with the integer comparator; the floor is decided in integers inside the band. Exact
    for any band above the float error. A caller folding many rows may pass ``Qf`` = Q as float32 and
    ``norms`` = the int64 squared row norms, kept incrementally (the fold does); both are checked
    for shape only, so they must be exactly those values."""
    _int(m, sigma_lo)
    _int8(qb, Q)
    if m < 0:
        raise ValueError("m must be >= 0")
    if m == 0 or len(Q) == 0:
        return []
    if Q.shape[-1] != qb.shape[-1] or Q.shape[-1] > 1024:
        raise ValueError("dimensions must match and d <= 1024 (2^24 exactness bound)")
    if Qf is None:
        Qf = Q.astype(np.float32)
    if norms is None:
        norms = (Q.astype(np.int64) ** 2).sum(axis=1)       # exact: int64 cast before the product
    if Qf.shape != Q.shape or norms.shape != (len(Q),):
        raise ValueError("Qf / norms do not match Q")
    dots = (Qf @ qb.astype(np.float32)).astype(np.int64)     # exact: partial sums below 2^24
    nb = int((qb.astype(np.int64) ** 2).sum())
    nn = norms * nb                                          # <= (1024 * 127^2)^2 < 2^63
    nn = np.where(nn == 0, 1, nn)                            # zero norm: dot = 0, sigma_q = 0
    approx = _approx(dots.astype(np.float64), nn.astype(np.float64))
    lo = sigma_lo / FIXED_ONE
    near = np.abs(approx - lo) < _BAND
    keep_mask = (approx > lo) & ~near
    for i in np.flatnonzero(near).tolist():                  # the floor in integers, in the band
        keep_mask[i] = _keep(float(approx[i]), (int(dots[i]), int(nn[i])), sigma_lo)
    keep = np.flatnonzero(keep_mask)
    keep = keep[np.argsort(-approx[keep], kind="stable")].tolist()
    ap = approx.tolist()
    exact = cmp_to_key(lambda i, j: -_cos_cmp((int(dots[i]), int(nn[i])), (int(dots[j]), int(nn[j])))
                       or (i > j) - (i < j))
    out: List[int] = []
    k = 0
    while k < len(keep) and len(out) < m:
        e = k + 1
        while e < len(keep) and ap[keep[e - 1]] - ap[keep[e]] <= _BAND:
            e += 1
        out.extend(sorted(keep[k:e], key=exact))
        k = e
    return out[:m]


# ---------------------------------------------------------------------------
# chi: three-way NLI
# ---------------------------------------------------------------------------

# Minimal antonym/negation lexicon for the deterministic offline backend. This
# is a heuristic for tests + local smoke; the cross-encoder backend is the
# stronger rung. Pairs are symmetric.
# Known antonym/incompatible pairs (lemma-form). Used in addition to the
# general "single differing slot on otherwise-overlapping text" heuristic.
_ANTONYM_PAIRS = [
    {"true", "false"}, {"on", "off"}, {"active", "revoked"},
    {"increase", "decrease"}, {"up", "down"}, {"hot", "cold"},
    {"open", "closed"}, {"increased", "decreased"}, {"rose", "fell"},
]

_STOP = {"is", "are", "in", "the", "a", "an", "of", "to", "make", "makes",
         "all", "some", "do", "does", "lies", "this", "named", "called",
         "has", "have", "owns", "own", "scheduled", "for", "went", "daily"}

_NEG = {"not", "no", "never", "nt"}


def _stem(tok: str) -> str:
    for suf in ("ed", "es", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def _content(tokens) -> set:
    return {_stem(t) for t in tokens if t not in _STOP and t not in _NEG}


def _lexical_nli(a: str, b: str) -> str:
    """Deterministic offline NLI heuristic (for tests + local smoke; the
    cross-encoder is the stronger rung for real data).

    Contradiction is detected by either (i) an explicit negation flip on
    otherwise-overlapping content, (ii) a known antonym swap, or (iii) the
    GENERAL pattern: two texts share most content but differ in exactly one
    content slot (a swapped entity/value) -- "X is in France" vs "X is in
    Italy", "owns a dog" vs "owns a cat", "budget increased" vs "budget
    decreased". This generalises beyond an enumerated antonym list.
    """
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta or not tb:
        return "neutral"

    # (i) explicit negation flip on otherwise-overlapping content
    if (ta & _NEG) ^ (tb & _NEG):
        if len(_content(ta) & _content(tb)) >= 2:
            return "contradicts"

    ca, cb = _content(ta), _content(tb)
    shared = ca & cb
    only_a, only_b = ca - cb, cb - ca

    # (ii) known antonym swap on otherwise-overlapping content
    for pair in _ANTONYM_PAIRS:
        sa, sb = pair & (only_a | shared), pair & (only_b | shared)
        if (pair & only_a) and (pair & only_b) and (pair & only_a) != (pair & only_b):
            if len(shared) >= 1:
                return "contradicts"

    # entailment FIRST: pure containment (one side adds no new content token)
    # is sound lexical entailment and must not be read as a value swap.
    if shared and (not only_a or not only_b):
        return "entails"

    # (iii) single differing content slot on shared content -> an incompatible
    # value swap (contradiction): "X in France" vs "X in Italy", "owns a dog"
    # vs "owns a cat", "budget increased" vs "budget decreased".
    if len(shared) >= 1 and len(only_a) == 1 and len(only_b) == 1:
        return "contradicts"

    # weaker containment -> entailment
    if cb and len(shared) / len(cb) >= 0.6:
        return "entails"
    if ca and len(shared) / len(ca) >= 0.6:
        return "entails"
    return "neutral"


class _CrossEncoderNLI:
    """Stronger NLI rung via a cross-encoder (downloads weights once). The model id is
    a constructor argument so a sweep can swap NLI models (R5)."""

    _LABELS = ("contradicts", "entails", "neutral")

    def __init__(self, model_id: str = "cross-encoder/nli-deberta-v3-small"):
        self._model_id = model_id
        self._model = None

    def probs(self, a: str, b: str) -> Tuple[float, float, float]:
        """Softmax triple (p_c, p_e, p_n), premise a. The slots are read through the model's own
        config.id2label (bart/distilbart-mnli order differs from the deberta cross-encoders); a model
        whose three labels do not map one-to-one onto contradiction/entailment/neutral is refused."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_id)
        names = {str(v).lower(): int(k) for k, v in self._model.config.id2label.items()}
        if sorted(names) != ["contradiction", "entailment", "neutral"]:
            raise ValueError(f"cannot map NLI labels {self._model.config.id2label!r}")
        p = self._model.predict([(a, b)], apply_softmax=True)[0]
        return tuple(float(p[names[k]]) for k in ("contradiction", "entailment", "neutral"))

    def __call__(self, a: str, b: str) -> str:
        return self._LABELS[int(np.argmax(self.probs(a, b)))]


# Alias -> HuggingFace cross-encoder NLI model id. Adding an NLI is data, not a branch.
_NLI_MODEL_IDS = {
    "mnli-cross-encoder": "cross-encoder/nli-deberta-v3-small",
    "deberta-base-nli": "cross-encoder/nli-deberta-v3-base",
    "roberta-nli": "cross-encoder/nli-roberta-base",
}
_CROSS_ENCODERS: "dict[str, _CrossEncoderNLI]" = {}


def new_cross_encoder(model: str) -> _CrossEncoderNLI:
    """A fresh cross-encoder NLI instance for an alias of ``_NLI_MODEL_IDS`` (not cached)."""
    if model not in _NLI_MODEL_IDS:
        raise ValueError(f"unknown NLI model {model!r}")
    return _CrossEncoderNLI(_NLI_MODEL_IDS[model])


def _cross_encoder(model: str) -> _CrossEncoderNLI:
    if model not in _CROSS_ENCODERS:
        _CROSS_ENCODERS[model] = new_cross_encoder(model)
    return _CROSS_ENCODERS[model]


def _lexical_probs(a: str, b: str) -> Tuple[float, float, float]:
    """Probability form of the lexical rung: the label of ``_lexical_nli`` gets 0.5 + 0.5 J, the other
    two split the rest, J = Jaccard of the content-token sets (0 if both empty). Deterministic; argmax
    is the lexical label because 0.5 + 0.5 J > (0.5 - 0.5 J) / 2 for every J >= 0. The lexical rung
    is symmetric in (a, b) and cannot express direction; directional evidence needs a cross-encoder."""
    ca, cb = _content(set(_tokenize(a))), _content(set(_tokenize(b)))
    j = len(ca & cb) / len(ca | cb) if ca | cb else 0.0
    top, rest = 0.5 + 0.5 * j, (0.5 - 0.5 * j) / 2
    k = _CrossEncoderNLI._LABELS.index(_lexical_nli(a, b))
    return tuple(top if i == k else rest for i in range(3))  # type: ignore[return-value]


def nli_probs(a: str, b: str, *, model: str) -> Tuple[float, float, float]:
    """(p_c, p_e, p_n) with premise ``a``, hypothesis ``b`` (r3 2.1). ``model`` as in ``chi``. Callers
    pass canon(text) of each claim key. Fixed-point form for a Sig payload: ``to_fixed16``."""
    if model == "lexical":
        return _lexical_probs(a, b)
    return _cross_encoder(model).probs(a, b)


def to_fixed16(probs: Sequence[float]) -> Tuple[int, ...]:
    """Each p in [0, 1] -> round(p * FIXED_ONE), Python round (half to even) of the float64 product,
    the rule of protocol.objects. Out-of-range, non-finite or bool input is refused. The triple is
    converted per component, so it need not sum to FIXED_ONE (Sig joins are not probability vectors
    either)."""
    out = []
    for p in probs:
        if isinstance(p, bool):
            raise TypeError("bool is not a probability")
        if not 0.0 <= p <= 1.0:  # also false for nan
            raise ValueError(f"probability out of [0, 1]: {p!r}")
        out.append(round(float(p) * FIXED_ONE))
    return tuple(out)


def chi(a: str, b: str, *, model: str) -> str:
    """Three-way NLI: 'entails' | 'neutral' | 'contradicts'. ``model`` required
    (no silent default): 'lexical' (offline, deterministic) or a cross-encoder alias
    from ``_NLI_MODEL_IDS`` (stronger; weights download once and the instance is cached)."""
    if model == "lexical":
        return _lexical_nli(a, b)
    return _cross_encoder(model)(a, b)


def evidence_set(endpoints: Iterable) -> Set[str]:
    """Helper: build a grounding-endpoint key set from GROUNDS-link endpoints."""
    return {e if isinstance(e, str) else getattr(e, "node_id", b"").hex() for e in endpoints}


__all__ = ["kappa", "sigma", "epsilon", "chi", "EmbeddingSimilarity", "evidence_set",
           "quantize_int8", "int8_gram", "sigma_q_cmp", "topm_exact", "nli_probs", "to_fixed16",
           "new_cross_encoder"]
