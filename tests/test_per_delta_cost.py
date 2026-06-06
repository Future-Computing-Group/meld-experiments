"""M2-edge: per-delta cost decomposition harness (TDD).

Verifies the timing decomposition with fake embed/NLI steps of known duration, independent
of the real model cost; the real CPU measurement is a smoke-ladder step. Run:
  PYTHONPATH=src python3 -m pytest tests/test_per_delta_cost.py -q
"""
import time

from smp.eval.per_delta_cost import measure_per_delta_cost


def test_decomposes_embed_and_nli_timing():
    res = measure_per_delta_cost(embed_fn=lambda: time.sleep(0.005),
                                 nli_fn=lambda: time.sleep(0.02), n=10, warmup=2)
    assert 3.0 <= res["embed_ms_mean"] <= 10.0, res     # ~5 ms
    assert 15.0 <= res["nli_ms_mean"] <= 30.0, res      # ~20 ms
    assert res["n"] == 10
    # the per-delta total is at least the two components
    assert res["total_ms_mean"] >= res["embed_ms_mean"] + res["nli_ms_mean"] - 2.0, res
    # std fields are present and non-negative
    assert res["embed_ms_std"] >= 0.0 and res["nli_ms_std"] >= 0.0
