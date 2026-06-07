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

    def __call__(self, a: str, b: str) -> str:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_id)
        scores = self._model.predict([(a, b)])[0]
        # cross-encoder NLI label order is [contradiction, entailment, neutral]
        idx = int(np.argmax(scores))
        return ("contradicts", "entails", "neutral")[idx]


# Alias -> HuggingFace cross-encoder NLI model id. Adding an NLI is data, not a branch.
_NLI_MODEL_IDS = {
    "mnli-cross-encoder": "cross-encoder/nli-deberta-v3-small",
    "deberta-base-nli": "cross-encoder/nli-deberta-v3-base",
    "roberta-nli": "cross-encoder/nli-roberta-base",
}
_CROSS_ENCODERS: "dict[str, _CrossEncoderNLI]" = {}


def chi(a: str, b: str, *, model: str) -> str:
    """Three-way NLI: 'entails' | 'neutral' | 'contradicts'. ``model`` required
    (no silent default): 'lexical' (offline, deterministic) or a cross-encoder alias
    from ``_NLI_MODEL_IDS`` (stronger; weights download once and the instance is cached)."""
    if model == "lexical":
        return _lexical_nli(a, b)
    if model in _NLI_MODEL_IDS:
        ce = _CROSS_ENCODERS.get(model)
        if ce is None:
            ce = _CrossEncoderNLI(_NLI_MODEL_IDS[model])
            _CROSS_ENCODERS[model] = ce
        return ce(a, b)
    raise ValueError(f"unknown NLI model {model!r}")


def evidence_set(endpoints: Iterable) -> Set[str]:
    """Helper: build a grounding-endpoint key set from GROUNDS-link endpoints."""
    return {e if isinstance(e, str) else getattr(e, "node_id", b"").hex() for e in endpoints}


__all__ = ["kappa", "sigma", "epsilon", "chi", "EmbeddingSimilarity", "evidence_set"]
