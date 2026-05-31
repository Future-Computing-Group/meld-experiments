"""The four merge signals: kappa / sigma / epsilon / chi.

REUSE posture
-------------
- sigma REUSES the experiment-template embedding plumbing: embed once, cache per
  (encoder, text-hash) to disk, return a real cosine. The encoder ladder mirrors
  the h-residual template: a weak deterministic 'hashing' rung (offline, fast,
  for tests + local smoke) and a 'minilm' rung (sentence-transformers, for
  stronger sweeps). NO silent default: the encoder must be named.
- kappa is a one-line content-addressed claim-key identity over the vendored
  proto Claim.
- epsilon is a set Jaccard over shared grounding (GROUNDS) endpoints.
- chi is a three-way NLI; the model is config-driven (no silent default). A
  deterministic 'lexical' backend keeps tests/local-smoke offline and fast; a
  cross-encoder backend ('mnli-cross-encoder') is available for stronger runs.

The numbers this module emits are real (cosine over cached embeddings, lexical
NLI heuristics); none are fabricated. Stronger encoder/NLI rungs only sharpen
the same signals.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable, Set

import numpy as np

from smp.proto import Claim

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

    _MODEL_IDS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2"}

    def __init__(self, rung: str):
        self.rung = rung
        self._model = None

    def encode_one(self, text: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._MODEL_IDS[self.rung])
        emb = self._model.encode([text], normalize_embeddings=True,
                                 show_progress_bar=False)
        return np.asarray(emb[0], dtype=np.float64)


def _make_encoder(name: str):
    if name == "hashing":
        return _HashingEncoder()
    if name == "minilm":
        return _SentenceTransformerEncoder("minilm")
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
# chi: three-way NLI
# ---------------------------------------------------------------------------

# Minimal antonym/negation lexicon for the deterministic offline backend. This
# is a heuristic for tests + local smoke; the cross-encoder backend is the
# stronger rung. Pairs are symmetric.
_ANTONYM_PAIRS = [
    {"france", "italy"}, {"france", "germany"}, {"italy", "germany"},
    {"true", "false"}, {"on", "off"}, {"active", "revoked"},
    {"increase", "decrease"}, {"up", "down"}, {"hot", "cold"},
    {"monday", "tuesday"}, {"open", "closed"},
]


def _lexical_nli(a: str, b: str) -> str:
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta or not tb:
        return "neutral"
    # contradiction: shared subject tokens but a known antonym swap, OR explicit
    # negation flip on otherwise-overlapping text.
    for pair in _ANTONYM_PAIRS:
        if pair & ta and pair & tb and (pair & ta) != (pair & tb):
            # both texts mention opposite members of an antonym pair
            if len(pair & ta) == 1 and len(pair & tb) == 1 and (pair & ta) != (pair & tb):
                return "contradicts"
    neg = {"not", "no", "never", "n't"}
    if (ta & neg) ^ (tb & neg) and len((ta - neg) & (tb - neg)) >= 2:
        return "contradicts"
    # entailment: hypothesis tokens largely covered by / overlap premise (after
    # dropping stopwords), high lexical containment.
    stop = {"is", "are", "in", "the", "a", "an", "of", "to", "make", "makes",
            "all", "some", "do", "does", "lies"}
    ca, cb = ta - stop, tb - stop
    if cb and len(ca & cb) / len(cb) >= 0.5:
        return "entails"
    if ca and len(ca & cb) / len(ca) >= 0.5:
        return "entails"
    return "neutral"


class _CrossEncoderNLI:
    """Stronger NLI rung via a cross-encoder (downloads weights once)."""

    _MODEL_ID = "cross-encoder/nli-deberta-v3-small"
    _LABELS = ("contradicts", "entails", "neutral")

    def __init__(self):
        self._model = None

    def __call__(self, a: str, b: str) -> str:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._MODEL_ID)
        scores = self._model.predict([(a, b)])[0]
        # cross-encoder NLI label order is [contradiction, entailment, neutral]
        idx = int(np.argmax(scores))
        return ("contradicts", "entails", "neutral")[idx]


_CROSS_ENCODER: "_CrossEncoderNLI | None" = None


def chi(a: str, b: str, *, model: str) -> str:
    """Three-way NLI: 'entails' | 'neutral' | 'contradicts'. ``model`` required
    (no silent default): 'lexical' (offline, deterministic) or
    'mnli-cross-encoder' (stronger)."""
    if model == "lexical":
        return _lexical_nli(a, b)
    if model == "mnli-cross-encoder":
        global _CROSS_ENCODER
        if _CROSS_ENCODER is None:
            _CROSS_ENCODER = _CrossEncoderNLI()
        return _CROSS_ENCODER(a, b)
    raise ValueError(f"unknown NLI model {model!r}")


def evidence_set(endpoints: Iterable) -> Set[str]:
    """Helper: build a grounding-endpoint key set from GROUNDS-link endpoints."""
    return {e if isinstance(e, str) else getattr(e, "node_id", b"").hex() for e in endpoints}


__all__ = ["kappa", "sigma", "epsilon", "chi", "EmbeddingSimilarity", "evidence_set"]
