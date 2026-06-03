"""eval-c multi-host (in-process dispatch): concurrent conflicting status updates under a
partition, then heal. With the status-CRDT every brain converges to the SAME per-claim
status after heal (order-independent lattice join); last-writer-wins stays diverged because
delivery order picks the surviving status. This is the in-process logic the local-docker
and testbed runs replay over the wire. Pure simulation (no testbed).
"""
from __future__ import annotations


def _cfg():
    from smp.agent.run_hotpot_gate import _calibrated_cfg
    return _calibrated_cfg("hashing")


def test_partition_heal_crdt_reconverges_lww_does_not():
    from smp.sync.eval_c import eval_c_partition_heal
    r = eval_c_partition_heal(n_nodes=4, claim="the policy is X", encoder="hashing",
                              nli="lexical", cfg=_cfg(),
                              effect_a="DEPRECATED", effect_b="OVERRULED")
    assert r["crdt_during"] < 1.0          # diverged while partitioned (two sides, two effects)
    assert r["crdt_healed"] == 1.0         # status-CRDT reconverges after heal+resync
    assert r["lww_healed"] < 1.0           # last-writer-wins stays diverged (order-dependent)


def test_identical_effects_never_diverge():
    # a sanity floor: if both sides apply the SAME effect, there is no divergence to heal.
    from smp.sync.eval_c import eval_c_partition_heal
    r = eval_c_partition_heal(n_nodes=4, claim="claim Y", encoder="hashing", nli="lexical",
                              cfg=_cfg(), effect_a="OVERRULED", effect_b="OVERRULED")
    assert r["crdt_during"] == 1.0 and r["crdt_healed"] == 1.0
