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
OPEN_MODEL_AXIS = {
    "qwen2.5-7b": {"model": "Qwen/Qwen2.5-7B-Instruct", "license": "Apache-2.0",
                   "role": "primary"},
    "qwen2.5-1.5b": {"model": "Qwen/Qwen2.5-1.5B-Instruct",
                     "license": "Apache-2.0", "role": "size-small"},
    "qwen2.5-32b": {"model": "Qwen/Qwen2.5-32B-Instruct",
                    "license": "Apache-2.0", "role": "size-large"},
    "mistral-7b": {"model": "mistralai/Mistral-7B-Instruct-v0.3",
                   "license": "Apache-2.0", "role": "cross-family"},
    "gemma-2-9b": {"model": "google/gemma-2-9b-it",
                   "license": "Gemma (open-weights, gated accept)",
                   "role": "fallback"},
}


def open_model(key: str = "qwen2.5-7b", **kwargs) -> "OpenWeightLLM":
    """Construct an ``OpenWeightLLM`` for a named model on the open-model axis.
    Import-light: no heavy dep is loaded until the model actually generates."""
    if key not in OPEN_MODEL_AXIS:
        raise KeyError(f"unknown open model {key!r}; axis: {list(OPEN_MODEL_AXIS)}")
    return OpenWeightLLM(model=OPEN_MODEL_AXIS[key]["model"], **kwargs)


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
        if self.transport == "vllm-openai":
            return self._complete_vllm(prompt)
        raise ValueError(f"unknown transport {self.transport!r}")

    def _complete_transformers(self, prompt: str) -> str:
        if self._pipe is None:
            from transformers import pipeline  # lazy, gated path only

            self._pipe = pipeline("text-generation", model=self.model,
                                  device_map="auto")
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
