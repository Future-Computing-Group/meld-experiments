"""Task 1.1 — the four merge signals kappa / sigma / epsilon / chi.

- kappa: content-addressed claim-key identity (exact, cheap, no model call).
- sigma: dense cosine similarity of claim texts (REUSES cached embeddings).
- epsilon: Jaccard overlap of grounding-evidence endpoint sets.
- chi: three-way NLI (entails / neutral / contradicts).

Load-bearing test (encoder-strength): sigma is a real symmetric cosine and the
per-text-hash cache is reused, so the sweep never re-embeds. NLI/encoder models
are config-driven with NO silent default.
"""
from __future__ import annotations

import pytest

from meld.merge.signals import chi, epsilon, kappa, sigma
from meld.proto import Claim


# ---- kappa ----
def test_kappa_exact_claim_key_match():
    assert kappa(Claim(claim_key=b"k1"), Claim(claim_key=b"k1")) == 1
    assert kappa(Claim(claim_key=b"k1"), Claim(claim_key=b"k2")) == 0


# ---- sigma (encoder-strength: symmetric + cached) ----
def test_sigma_symmetric_and_cached(tmp_path):
    from meld.merge.signals import EmbeddingSimilarity

    es = EmbeddingSimilarity(encoder="hashing", cache_dir=str(tmp_path))
    s1 = es.sigma("Paris is in France", "Paris lies in France")
    s2 = es.sigma("Paris lies in France", "Paris is in France")
    assert abs(s1 - s2) < 1e-6            # symmetry
    assert es.cache_hits >= 1             # per-text-hash cache reused
    assert -1.0 - 1e-9 <= s1 <= 1.0 + 1e-9


def test_sigma_identical_text_is_one(tmp_path):
    from meld.merge.signals import EmbeddingSimilarity

    es = EmbeddingSimilarity(encoder="hashing", cache_dir=str(tmp_path))
    assert es.sigma("the sky is blue", "the sky is blue") == pytest.approx(1.0, abs=1e-6)


def test_sigma_encoder_strength_orders_paraphrase_above_unrelated(tmp_path):
    # a real similarity signal: a paraphrase scores higher than an unrelated text.
    from meld.merge.signals import EmbeddingSimilarity

    es = EmbeddingSimilarity(encoder="hashing", cache_dir=str(tmp_path))
    para = es.sigma("the cat sat on the mat", "the cat sat on the mat")
    unrel = es.sigma("the cat sat on the mat", "quantum chromodynamics is hard")
    assert para > unrel


def test_sigma_requires_explicit_encoder():
    # no silent default: constructing without an encoder is an error.
    from meld.merge.signals import EmbeddingSimilarity

    with pytest.raises(TypeError):
        EmbeddingSimilarity()  # type: ignore[call-arg]


# ---- epsilon (Jaccard of grounding evidence) ----
def test_epsilon_jaccard_of_evidence():
    assert epsilon({"e1", "e2"}, {"e2", "e3"}) == pytest.approx(1 / 3)


def test_epsilon_empty_sets_is_zero():
    assert epsilon(set(), set()) == 0.0  # documented edge case


# ---- chi (three-way NLI) ----
# The 'lexical' backend is a deterministic offline heuristic: it soundly detects
# CONTAINMENT entailment and single-slot value-swap contradiction. Hyponym
# entailment ("bark" => "make sounds") is beyond lexical reach and needs the
# cross-encoder rung -- not asserted here.
def test_chi_entails():
    assert chi("Dogs make loud sounds", "Dogs make sounds", model="lexical") == "entails"


def test_chi_contradicts():
    assert chi("Paris is in France", "Paris is in Italy", model="lexical") == "contradicts"


def test_chi_neutral():
    assert chi("The meeting is on Monday", "I like coffee", model="lexical") == "neutral"


def test_chi_requires_explicit_model():
    with pytest.raises(TypeError):
        chi("a", "b")  # type: ignore[call-arg]
