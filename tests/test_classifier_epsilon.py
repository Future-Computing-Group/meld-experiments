"""TDD for the epsilon (evidence-overlap) rule in classify().

epsilon promotes a BORDERLINE-similarity pair (sigma in [sigma_lo, theta_merge))
to MERGE when the two claims share grounding evidence (eps >= eps_hi), are not
contradicting, and share a compatible frame. It NEVER overrides the contradiction
guard, NEVER merges below sigma_lo, and is INERT when eps_hi=0 / eps=0 (so
ungrounded datasets are unaffected -> M-1/M-2 headline numbers preserved).
"""
from smp.config import Thresholds
from smp.merge.classifier import classify


def _cfg(eps_hi=0.5):
    # theta_merge/sigma_lo as in M-2 (0.90/0.12); eps_hi opt-in.
    return Thresholds(theta_merge=0.90, sigma_lo=0.12, r_min=0.2, tau=0.5,
                      eps_hi=eps_hi)


# --- epsilon PROMOTES a borderline-sigma pair to MERGE, but ONLY on positive
# entailment (chi == entails). A mere non-contradiction (neutral) is NOT enough:
# that is what stops an NLI-missed contradiction (labelled neutral) from being
# silently epsilon-merged. ---
def test_epsilon_promotes_borderline_to_merge_on_entailment():
    cfg = _cfg(eps_hi=0.5)
    # sigma in the RELATE band, ENTAILS, ctx_ok, strong evidence overlap.
    assert classify(cfg=cfg, kappa=0, sigma=0.5, epsilon=0.9, chi="entails",
                    ctx_ok=True) == "MERGE"


def test_epsilon_neutral_does_not_promote():
    cfg = _cfg(eps_hi=0.5)
    # high overlap + borderline sigma but only NEUTRAL -> stays RELATE (no
    # positive entailment, so a possible missed contradiction is not merged).
    assert classify(cfg=cfg, kappa=0, sigma=0.5, epsilon=0.9, chi="neutral",
                    ctx_ok=True) == "RELATE"


def test_same_pair_without_overlap_stays_relate():
    cfg = _cfg(eps_hi=0.5)
    # entails, but eps below eps_hi -> the promotion must NOT fire.
    assert classify(cfg=cfg, kappa=0, sigma=0.5, epsilon=0.1, chi="entails",
                    ctx_ok=True) == "RELATE"


# --- safety: epsilon never overrides the contradiction guard ---
def test_epsilon_never_merges_a_contradiction():
    cfg = _cfg(eps_hi=0.5)
    assert classify(cfg=cfg, kappa=0, sigma=0.5, epsilon=1.0, chi="contradicts",
                    ctx_ok=True) == "CONFLICT"


# --- safety: epsilon never merges below the actionable floor sigma_lo ---
def test_epsilon_does_not_merge_below_sigma_lo():
    cfg = _cfg(eps_hi=0.5)
    # sigma below sigma_lo: no actionable relation; high eps + entails must not rescue it.
    assert classify(cfg=cfg, kappa=0, sigma=0.05, epsilon=1.0, chi="entails",
                    ctx_ok=True) == "NO_RELATION"


# --- safety: epsilon needs a compatible frame (ctx) to promote ---
def test_epsilon_respects_context_gate():
    cfg = _cfg(eps_hi=0.5)
    assert classify(cfg=cfg, kappa=0, sigma=0.5, epsilon=0.9, chi="entails",
                    ctx_ok=False) == "RELATE"


# --- inertness: eps_hi=0 (default / ungrounded) reproduces the old behaviour ---
def test_epsilon_inert_when_eps_hi_zero():
    cfg = _cfg(eps_hi=0.0)
    # borderline pair must stay RELATE (no promotion) when promotion disabled,
    # regardless of eps/chi -- this is the HotpotQA / M-2 ungrounded path.
    assert classify(cfg=cfg, kappa=0, sigma=0.5, epsilon=1.0, chi="entails",
                    ctx_ok=True) == "RELATE"


def test_epsilon_default_param_is_zero_backward_compatible():
    cfg = _cfg(eps_hi=0.5)
    # call WITHOUT epsilon (old call sites) -> eps defaults 0 -> no promotion.
    assert classify(cfg=cfg, kappa=0, sigma=0.5, chi="neutral",
                    ctx_ok=True) == "RELATE"


# --- high-sigma MERGE still fires (epsilon path doesn't shadow it) ---
def test_high_sigma_merge_unchanged():
    cfg = _cfg(eps_hi=0.5)
    assert classify(cfg=cfg, kappa=0, sigma=0.95, epsilon=0.0, chi="neutral",
                    ctx_ok=True) == "MERGE"
