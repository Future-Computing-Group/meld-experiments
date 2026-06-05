"""A-1 gate ablation: drop each admission gate in turn and quantify its contribution to
the false-merge and stale-promotion rates. Ablating a gate = forcing its input to the
permissive value (Context -> ctx_ok=True; inference -> chi non-contradicting; staleness
-> ACTIVE/not-stale), then re-classifying. The full classifier is the safe baseline; each
ablation must make the corresponding error WORSE. Pure simulation over smp.merge.classify
(no LLM, no testbed) -- independent of the multi-host runs.
"""
from __future__ import annotations


def _cfg():
    from smp.config import load_thresholds
    return load_thresholds()


def _cases(cfg):
    hi = min(cfg.theta_merge + 0.05, 1.0)        # above the merge threshold
    return [
        # Context gate: high-sigma, non-contradicting, ctx_ok=False -> RELATE (NOT merge).
        {"gold": "RELATE", "kappa": 0, "sigma": hi, "chi": "neutral",
         "ctx_ok": False, "retention_below_min": False, "status": "ACTIVE", "stale": False},
        # inference gate: distinct-key (kappa=0) high-sigma contradiction -> CONFLICT
        # (NOT merge); dropping the gate flips it via the dangerous SEMANTIC path.
        {"gold": "CONFLICT", "kappa": 0, "sigma": hi, "chi": "contradicts",
         "ctx_ok": True, "retention_below_min": False, "status": "ACTIVE", "stale": False},
        # staleness gate: a distinct-key non-ACTIVE local -> REJECT (must not propagate).
        {"gold": "REJECT", "kappa": 0, "sigma": hi, "chi": "neutral",
         "ctx_ok": True, "retention_below_min": False, "status": "DEPRECATED", "stale": True},
        # clean MERGE control: high-sigma, compatible, fresh -> MERGE (semantic path).
        {"gold": "MERGE", "kappa": 0, "sigma": hi, "chi": "neutral",
         "ctx_ok": True, "retention_below_min": False, "status": "ACTIVE", "stale": False},
    ]


def test_full_classifier_is_the_safe_baseline():
    from smp.merge.gate_ablation import gate_ablation_metrics
    cfg = _cfg()
    m = gate_ablation_metrics(_cases(cfg), cfg)
    assert m[None]["false_merge_rate"] == 0.0       # nothing non-MERGE silently merged
    assert m[None]["stale_promotion_rate"] == 0.0   # the stale local was rejected


def test_ablating_context_gate_raises_false_merge():
    from smp.merge.gate_ablation import gate_ablation_metrics
    cfg = _cfg()
    m = gate_ablation_metrics(_cases(cfg), cfg)
    assert m["context"]["false_merge_rate"] > m[None]["false_merge_rate"]


def test_ablating_inference_gate_raises_false_merge():
    from smp.merge.gate_ablation import gate_ablation_metrics
    cfg = _cfg()
    m = gate_ablation_metrics(_cases(cfg), cfg)
    assert m["inference"]["false_merge_rate"] > m[None]["false_merge_rate"]


def test_ablating_staleness_gate_raises_stale_promotion():
    from smp.merge.gate_ablation import gate_ablation_metrics
    cfg = _cfg()
    m = gate_ablation_metrics(_cases(cfg), cfg)
    assert m["staleness"]["stale_promotion_rate"] > m[None]["stale_promotion_rate"]
