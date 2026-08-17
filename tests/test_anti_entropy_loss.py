"""Loss-injection convergence experiment for the anti-entropy digest.

N brains publish a fixed contested-lifecycle status-link set; each delivery is
dropped independently at the loss rate (the protocol's lossy semantic routing).
WITHOUT the digest, lost links are permanent -> per-claim status diverges. WITH
periodic anti-entropy rounds, the digest detects + reconciles the lost links ->
agreement returns to 1.000. Deterministic in the seed.
"""
from __future__ import annotations


# --- CHANGE 2 RED->GREEN -----------------------------------------------------

def test_zero_loss_agrees_without_digest():
    # No loss -> every brain receives every link -> full agreement even without
    # the digest (the digest is only needed to repair LOSS).
    from meld.eval.anti_entropy_eval import loss_convergence_cell
    cell = loss_convergence_cell(loss=0.0, n_brains=5, seed=0, digest=False)
    assert cell["agreement"] == 1.0
    assert cell["n_distinct_maps"] == 1


def test_loss_without_digest_diverges():
    # Under loss, without reconciliation, some brains miss links -> the per-claim
    # status map set fragments (agreement < 1, more than one distinct map).
    from meld.eval.anti_entropy_eval import loss_convergence_cell
    cell = loss_convergence_cell(loss=0.10, n_brains=8, seed=1, digest=False)
    assert cell["agreement"] < 1.0
    assert cell["n_distinct_maps"] >= 2


def test_loss_with_digest_reconverges_to_full_agreement():
    # The SAME lossy delivery, but with periodic anti-entropy rounds, reconciles
    # every brain to the complete link set -> agreement 1.000, one distinct map.
    from meld.eval.anti_entropy_eval import loss_convergence_cell
    cell = loss_convergence_cell(loss=0.10, n_brains=8, seed=1, digest=True)
    assert cell["agreement"] == 1.0
    assert cell["n_distinct_maps"] == 1
    # the converged map equals the canonical lattice-join target
    assert cell["status_map"] == cell["expected_status_map"]
    # the digest carried a measurable cost and did real repair work
    assert cell["digest_bytes"] > 0
    assert cell["links_repaired"] > 0
    assert cell["rounds_to_converge"] >= 1


def test_with_digest_converges_at_the_reviewer_cited_rate():
    # The 0.3% loss the reviewers cited: without the digest a few links are lost;
    # with it, agreement is restored to 1.000.
    from meld.eval.anti_entropy_eval import loss_convergence_cell
    with_d = loss_convergence_cell(loss=0.003, n_brains=12, seed=2, digest=True)
    assert with_d["agreement"] == 1.0
    assert with_d["status_map"] == with_d["expected_status_map"]


def test_sweep_is_deterministic_in_seed():
    from meld.eval.anti_entropy_eval import loss_convergence_sweep
    a = loss_convergence_sweep(losses=[0.0, 0.01], n_brains=6, seeds=[0, 1])
    b = loss_convergence_sweep(losses=[0.0, 0.01], n_brains=6, seeds=[0, 1])
    assert a == b


def test_sweep_with_digest_is_one_everywhere_without_drops_with_loss():
    from meld.eval.anti_entropy_eval import loss_convergence_sweep
    rep = loss_convergence_sweep(losses=[0.0, 0.003, 0.01, 0.05, 0.10],
                                 n_brains=10, seeds=[0, 1, 2])
    for row in rep["rows"]:
        # WITH digest: full agreement at every loss rate
        assert row["with_digest_agreement"] == 1.0
        if row["loss"] > 0.0:
            # WITHOUT digest: loss degrades agreement (mean < 1) at positive loss
            assert row["without_digest_agreement"] < 1.0
        else:
            assert row["without_digest_agreement"] == 1.0
