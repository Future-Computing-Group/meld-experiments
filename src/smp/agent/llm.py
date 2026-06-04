"""LLM clients for the competence-gate agent.

Two backends behind one ``complete(prompt) -> str`` contract:

- ``MockLLM`` -- a deterministic, offline, scripted client used for the LOCAL
  WIRING smoke and the tests. It returns plain ReAct text turns ("Thought: ...
  Action: tool[arg]"); it produces NO GAIA score.
- ``OpenWeightLLM`` -- the GATED CSC backend. It serves a single open-weight
  instruct model via transformers (or a vLLM OpenAI-compatible endpoint). The
  default model is **Qwen2.5-7B-Instruct** (Apache-2.0). Honoring the
  open-weight-quirks lesson, the agent contract is a PLAIN ReAct text turn, NOT
  a JSON-output prompt (the JSON-matching prompt is what triggers Llama-3.1-8B's
  Python-code-emission quirk); and we prefer the Qwen/Mistral/Gemma families
  over Llama-3.1-8B. This class is import-light: heavy deps (transformers/vllm)
  are imported lazily ONLY when an open-weight backend is actually constructed,
  so the offline test/smoke path never needs them.
"""
from __future__ import annotations

from typing import List, Optional

# The default open-weight model for the gated CSC run. Apache-2.0 license.
DEFAULT_OPEN_WEIGHT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OPEN_WEIGHT_LICENSE = "Apache-2.0"

# The OPEN-MODEL AXIS for the gated run: each registry baseline runs on >= 2 of
# these via OpenWeightLLM. Qwen2.5-7B (default) + Mistral-7B-Instruct are the two
# cross-family models; Gemma-2-9B-it is the fallback. Llama-3.1-8B is DELIBERATELY
# EXCLUDED (open-weight-quirks lesson: its JSON-output-prompt code-emission quirk
# is avoided structurally by the plain ReAct text contract, but we still prefer
# the Qwen/Mistral/Gemma families for cross-family validation).
#
# The MODEL-SIZE sub-axis (M-1 F1 corroboration) brackets the 7B primary with a
# smaller and a larger member of the SAME Qwen2.5 family (Apache-2.0), so model
# SIZE is the only varied factor in the cross-size answer-F1 comparison; recall
# /merge/summarize are model-independent and unchanged.
# Each entry carries the HF repo (for the transformers transport) AND the Ollama
# tag (for the ollama transport, which serves quantized open weights over an
# OpenAI-compatible endpoint -- used for the larger members of the size ladder).
# recall/merge/summarize are model-independent; this axis only varies the answer-F1
# corroboration: a size ladder 1.5B->72B across three families (Qwen/Mistral/Gemma).
OPEN_MODEL_AXIS = {
    "qwen2.5-1.5b": {"model": "Qwen/Qwen2.5-1.5B-Instruct", "ollama": "qwen2.5:1.5b",
                     "license": "Apache-2.0", "role": "size-small"},
    "qwen2.5-7b": {"model": "Qwen/Qwen2.5-7B-Instruct", "ollama": "qwen2.5:7b",
                   "license": "Apache-2.0", "role": "primary"},
    "qwen2.5-32b": {"model": "Qwen/Qwen2.5-32B-Instruct", "ollama": "qwen2.5:32b",
                    "license": "Apache-2.0", "role": "size-large"},
    "qwen2.5-72b": {"model": "Qwen/Qwen2.5-72B-Instruct", "ollama": "qwen2.5:72b",
                    "license": "Apache-2.0", "role": "size-xlarge"},
    "mistral-7b": {"model": "mistralai/Mistral-7B-Instruct-v0.3", "ollama": "mistral:7b",
                   "license": "Apache-2.0", "role": "cross-family"},
    "gemma-4-12b": {"model": "google/gemma-4-12B-it", "ollama": "gemma4:12b",
                    "license": "Gemma (open-weights, gated accept)",
                    "role": "cross-family-midsize"},
    "gemma-3-12b": {"model": "google/gemma-3-12b-it", "ollama": "gemma3:12b",
                    "license": "Gemma (open-weights, gated accept)",
                    "role": "cross-family-midsize"},
    "gemma-2-9b": {"model": "google/gemma-2-9b-it", "ollama": "gemma2:9b",
                   "license": "Gemma (open-weights, gated accept)",
                   "role": "fallback"},
}


def open_model(key: str = "qwen2.5-7b", *, transport: str = "transformers",
               **kwargs) -> "OpenWeightLLM":
    """Construct an ``OpenWeightLLM`` for a named model on the open-model axis.
    For the ``ollama`` transport the request uses the Ollama tag; otherwise the HF
    repo. Import-light: no heavy dep is loaded until the model actually generates."""
    if key not in OPEN_MODEL_AXIS:
        raise KeyError(f"unknown open model {key!r}; axis: {list(OPEN_MODEL_AXIS)}")
    entry = OPEN_MODEL_AXIS[key]
    if transport == "ollama":
        model = entry.get("ollama")
        if not model:
            raise KeyError(f"no ollama tag for {key!r}")
    else:
        model = entry["model"]
    return OpenWeightLLM(model=model, transport=transport, **kwargs)


class MockLLM:
    """Deterministic scripted client (offline). Cycles through ``script`` turns;
    repeats the last turn if the agent runs longer than the script."""

    def __init__(self, script: List[str]):
        if not script:
            raise ValueError("MockLLM needs at least one scripted turn")
        self.script = list(script)
        self._i = 0
        self.calls: List[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        turn = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        return turn


class OpenWeightLLM:
    """GATED open-weight backend (Qwen2.5-7B-Instruct by default).

    Not exercised in the offline path. Two transports:
      - ``transport="transformers"``: local generate() via a HF pipeline.
      - ``transport="vllm-openai"``: POST to a vLLM OpenAI-compatible server.
    Heavy deps are imported lazily here so importing this module is cheap.
    """

    def __init__(self, model: str = DEFAULT_OPEN_WEIGHT_MODEL,
                 transport: str = "transformers",
                 endpoint: Optional[str] = None,
                 max_new_tokens: int = 512, temperature: float = 0.0):
        self.model = model
        self.transport = transport
        self.endpoint = endpoint
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipe = None

    def complete(self, prompt: str) -> str:
        if self.transport == "transformers":
            return self._complete_transformers(prompt)
        # ollama exposes the same OpenAI /v1/completions contract as vLLM, so the
        # same POST path serves both (self.model is the ollama tag for ollama).
        if self.transport in ("vllm-openai", "ollama"):
            return self._complete_vllm(prompt)
        raise ValueError(f"unknown transport {self.transport!r}")

    def _complete_transformers(self, prompt: str) -> str:
        if self._pipe is None:
            import os
            from transformers import (AutoModelForCausalLM,  # lazy, gated path
                                      AutoTokenizer, pipeline)

            # Force local-only loading when offline is set: pipeline(model=name)
            # otherwise hits the HF API to RESOLVE the repo and dies under
            # HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE even with the weights cached.
            # Loading model+tokenizer explicitly with local_files_only resolves
            # straight from $HF_HOME (the pattern that loads cleanly offline).
            local_only = (os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
                          or os.environ.get("HF_HUB_OFFLINE", "0") == "1")
            # Resolve to the cached snapshot DIRECTORY so from_pretrained loads a
            # local path with no name resolution / API call at all (passing the
            # repo NAME still triggers an offline-mode API probe).
            model_ref = self.model
            if local_only:
                from huggingface_hub import snapshot_download
                model_ref = snapshot_download(self.model, local_files_only=True)
            tok = AutoTokenizer.from_pretrained(
                model_ref, local_files_only=local_only)
            mdl = AutoModelForCausalLM.from_pretrained(
                model_ref, local_files_only=local_only,
                device_map="auto", torch_dtype="auto")
            self._pipe = pipeline("text-generation", model=mdl, tokenizer=tok)
        out = self._pipe(prompt, max_new_tokens=self.max_new_tokens,
                         do_sample=self.temperature > 0,
                         temperature=max(self.temperature, 1e-3),
                         return_full_text=False)
        return out[0]["generated_text"]

    def _complete_vllm(self, prompt: str) -> str:
        import json
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            (self.endpoint or "http://localhost:8000/v1/completions"),
            data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:  # noqa: S310 (gated, trusted)
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["text"]


__all__ = ["MockLLM", "OpenWeightLLM", "open_model", "OPEN_MODEL_AXIS",
           "DEFAULT_OPEN_WEIGHT_MODEL", "DEFAULT_OPEN_WEIGHT_LICENSE"]
