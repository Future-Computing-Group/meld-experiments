"""G3 — the OPEN-MODEL AXIS: each registry baseline runs on >= 2 open models via
OpenWeightLLM (Qwen2.5-7B + Mistral-7B-Instruct; Gemma-2-9B-it fallback).

These tests pin the axis declaration + the import-light contract: constructing an
OpenWeightLLM for any axis model must NOT import transformers/vllm (the offline
path stays clean; heavy deps load only at generate-time on the gated host). They
do NOT run a real model (that is the PI-gated CSC run).
"""
from __future__ import annotations

import sys

import pytest

from smp.agent.llm import (OPEN_MODEL_AXIS, OpenWeightLLM, open_model,
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
