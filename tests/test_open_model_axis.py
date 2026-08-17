"""G3 — the OPEN-MODEL AXIS: each registry baseline runs on >= 2 open models via
OpenWeightLLM (Qwen2.5-7B + Mistral-7B-Instruct; Gemma-2-9B-it fallback).

These tests pin the axis declaration + the import-light contract: constructing an
OpenWeightLLM for any axis model must NOT import transformers/vllm (the offline
path stays clean; heavy deps load only at generate-time on the gated host). They
do NOT run a real model (that is the real-model CSC run).
"""
from __future__ import annotations

import sys

import pytest

from meld.agent.llm import (OPEN_MODEL_AXIS, OpenWeightLLM, open_model,
                           DEFAULT_OPEN_WEIGHT_MODEL)


def test_axis_has_qwen_and_mistral_at_least():
    models = {v["model"] for v in OPEN_MODEL_AXIS.values()}
    assert any("Qwen2.5-7B" in m for m in models)
    assert any("Mistral-7B" in m for m in models)
    # default is Qwen (primary)
    assert OPEN_MODEL_AXIS["qwen2.5-7b"]["model"] == DEFAULT_OPEN_WEIGHT_MODEL


def test_axis_excludes_llama_3_1_8b():
    """Open-weight-quirks lesson: Llama-3.1-8B is deliberately NOT on the axis."""
    models = " ".join(v["model"].lower() for v in OPEN_MODEL_AXIS.values())
    assert "llama-3.1-8b" not in models and "llama-3-1-8b" not in models


@pytest.mark.parametrize("key", ["qwen2.5-7b", "mistral-7b", "gemma-2-9b"])
def test_open_model_constructs_without_heavy_deps(key):
    """Constructing the client must be import-light: no transformers/vllm loaded
    (they import lazily only when .complete actually generates on the host)."""
    # Robust to test ordering: another test may have already loaded the heavy
    # deps into sys.modules (e.g. an encoder/NLI eval). The contract is that
    # CONSTRUCTING the client does not itself TRIGGER the import -- so assert the
    # deps are not NEWLY added by construction, not that they are globally absent.
    pre_transformers = "transformers" in sys.modules
    pre_vllm = "vllm" in sys.modules
    llm = open_model(key)
    assert isinstance(llm, OpenWeightLLM)
    assert llm.model == OPEN_MODEL_AXIS[key]["model"]
    if not pre_transformers:
        assert "transformers" not in sys.modules
    if not pre_vllm:
        assert "vllm" not in sys.modules


def test_open_model_unknown_key_raises():
    with pytest.raises(KeyError):
        open_model("llama-3.1-8b")


# -- M-1 model-SIZE axis: two Qwen2.5 entries bracketing the 7B -------------


@pytest.mark.parametrize("key,hf_id,role", [
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "size-small"),
    ("qwen2.5-32b", "Qwen/Qwen2.5-32B-Instruct", "size-large"),
])
def test_size_axis_entries_resolve_to_exact_hf_ids(key, hf_id, role):
    """The size-axis brackets the existing 7B (same Qwen2.5 family, Apache-2.0)
    so model SIZE is the only varied factor. Each must resolve to its EXACT HF
    repo and carry the size role tag + Apache-2.0 license."""
    assert key in OPEN_MODEL_AXIS
    entry = OPEN_MODEL_AXIS[key]
    assert entry["model"] == hf_id
    assert entry["license"] == "Apache-2.0"
    assert entry["role"] == role


@pytest.mark.parametrize("key,hf_id", [
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen2.5-32b", "Qwen/Qwen2.5-32B-Instruct"),
])
def test_size_axis_round_trips_through_open_model(key, hf_id):
    """Round-trip exactly as the existing 7B entry does: open_model(key) builds
    an OpenWeightLLM whose .model is the exact HF id, import-light."""
    pre_transformers = "transformers" in sys.modules
    pre_vllm = "vllm" in sys.modules
    llm = open_model(key)
    assert isinstance(llm, OpenWeightLLM)
    assert llm.model == hf_id
    if not pre_transformers:
        assert "transformers" not in sys.modules
    if not pre_vllm:
        assert "vllm" not in sys.modules


def test_size_axis_entries_mirror_7b_field_schema():
    """The new size entries carry EXACTLY the fields the 7B entry carries (no
    invented fields) so the axis schema stays uniform."""
    ref_fields = set(OPEN_MODEL_AXIS["qwen2.5-7b"])
    for key in ("qwen2.5-1.5b", "qwen2.5-32b"):
        assert set(OPEN_MODEL_AXIS[key]) == ref_fields
