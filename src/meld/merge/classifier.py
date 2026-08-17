"""The pair-adjudication classifier (four verdicts) plus the admission gate
(architecture A.2).

The genuinely NEW contribution: an executable classifier over three core signals
(kappa/sigma/chi) + the Context gate -- plus an optional, off-by-default
evidence-overlap signal (epsilon) -- emitting exactly one of
MERGE / RELATE / CONFLICT / NO_RELATION. Thresholds (theta_merge, sigma_lo) come
from config (CFG), NEVER hardcoded; they are calibrated in M-2, not derived.
This module is the pair-adjudication core of the five-outcome decision procedure
(manuscript Sec. 4.2); the admission gate (REJECT) and the novelty test (INSERT)
are the two outcomes that live one level up, in
``meld.protocol.merge_on_receive``.

Decision procedure (Sec. 4.2), a first-match cascade, five outcomes total:

    REJECT    (R1, admission gate) if  stale/overruled gate fails -- evaluated
              here first because it takes precedence over everything (Cor. 2.1),
              but it is a PRE-VERDICT, not a pair-adjudication verdict.
    INSERT    (R2, novel) if no local candidate has kappa==1 or sigma>=sigma_lo
              -- decided by the caller (merge_on_receive), not this function.
    CONFLICT  (R3) if  chi == contradicts  AND  (kappa == 1 OR sigma >= sigma_lo)
    MERGE     (R4, exact key) if  kappa == 1 -- exempt from Context/authority:
              the claim key hashes content together with discrete scope
              (Sec. 4.1), so kappa==1 already means same assertion, same scope.
    MERGE     (R5, similarity) if  sigma >= theta_merge AND chi != contradicts
              AND ctx_ok AND authority_ok
    RELATE    (R6, otherwise) if  sigma_lo <= sigma < theta_merge
              OR  (sigma >= theta_merge AND (not ctx_ok OR not authority_ok))
              with chi != contradicts
    NO_RELATION (otherwise: no actionable relation: sigma below sigma_lo, no
              key); the protocol layer maps this to INSERT.
"""
from __future__ import annotations

from meld.config import Thresholds


def classify(
    *,
    cfg: Thresholds,
    kappa: int = 0,
    sigma: float = 0.0,
    epsilon: float = 0.0,
    chi: str = "neutral",
    ctx_ok: bool = True,
    authority_ok: bool = True,
    retention_below_min: bool = False,
    status: str = "ACTIVE",
) -> str:
    """Return the merge decision for an (incoming, local) claim pair.

    ``kappa`` exact claim-key identity (0/1); ``sigma`` dense cosine;
    ``epsilon`` grounding-evidence Jaccard overlap; ``chi`` NLI label;
    ``ctx_ok`` Context-gate result; ``retention_below_min`` / ``status`` drive
    the wire-admission REJECT gate. ``epsilon`` defaults to 0 so existing call
    sites (and ungrounded datasets, where eps_hi=0) are unchanged.

    ``authority_ok`` (§5.3) admits a SIMILARITY merge only when the incoming
    source is at least as authoritative as the more-canonical local target. When
    False, a similarity-merge (kappa=0) is downgraded from MERGE to a candidate
    RELATE, so a canonical brain is never silently rewritten by a low-authority
    peer. An EXACT-KEY (kappa=1) match is the SAME claim (no rewrite risk) and is
    unaffected. Defaults True ⇒ existing callers and M-2 (equal default authority)
    are byte-for-byte unchanged.
    """
    # --- Gate precedence: stale or non-ACTIVE never propagates (Cor. 2.1) ---
    if retention_below_min or status != "ACTIVE":
        return "REJECT"

    # --- CONFLICT: a contradiction between claims that are at all RELATED
    # (sigma >= sigma_lo) or share a key. Gated on sigma_lo, the actionable-
    # relation floor, NOT theta_merge: a contradiction need not be near-identical
    # in surface form to be a contradiction. Gating it on theta_merge instead
    # would route lexically divergent contradictions (sigma in
    # [sigma_lo, theta_merge)) silently to NO_RELATION -- the conflict-recall blind
    # spot. No silent winner; the contradiction is surfaced. ---
    conflict_actionable = (kappa == 1) or (sigma >= cfg.sigma_lo)
    if chi == "contradicts" and conflict_actionable:
        return "CONFLICT"

    # --- MERGE: exact key, OR high-sigma non-contradicting in a shared frame ---
    # An EXACT-KEY match is the SAME claim (no silent-rewrite risk), so it merges
    # regardless of authority. This exemption from BOTH ctx_ok and authority_ok is
    # licensed by the scoped claim key (§4.1): the key hashes canonical
    # content TOGETHER WITH the claim's discrete scope, so kappa==1 already means
    # the two claims are the same assertion in the same scope and there is no
    # cross-boundary rewrite to guard against. The key does NOT carry validity or
    # freshness, so the FRAME must still be compatible: an exact-key pair in an
    # incompatible Context (e.g. disjoint validity intervals) RELATEs instead of
    # merging. A SIMILARITY
    # merge carries no such guarantee, so it requires authority_ok: a
    # low-authority source into a more-canonical local is downgraded to a candidate
    # RELATE below (§5.3), not auto-merged. authority_ok defaults True ⇒ unchanged.
    if kappa == 1:
        return "MERGE" if ctx_ok else "RELATE"
    if (sigma >= cfg.theta_merge and chi != "contradicts" and ctx_ok
            and authority_ok):
        return "MERGE"

    # --- MERGE (epsilon-corroborated): a BORDERLINE-similarity pair
    # (sigma in [sigma_lo, theta_merge)) whose claims share grounding evidence
    # (epsilon >= eps_hi) AND positively entail (chi == entails) is promoted from
    # RELATE to MERGE -- shared provenance is independent evidence of sameness,
    # complementary to surface similarity. The entailment requirement (not the
    # weaker chi != contradicts) is deliberate: it stops an NLI-MISSED
    # contradiction (labelled neutral) from being silently epsilon-merged, so
    # epsilon's recall gain does not come at a false-merge cost. Gated so it NEVER
    # (a) merges a non-entailing pair, (b) merges below the actionable floor
    # sigma_lo, or (c) crosses an incompatible frame (ctx_ok). With eps_hi=0
    # (ungrounded default) this path is dead, so M-1/M-2 are unaffected. ---
    if (cfg.eps_hi > 0.0 and chi == "entails" and ctx_ok
            and cfg.sigma_lo <= sigma < cfg.theta_merge
            and epsilon >= cfg.eps_hi):
        return "MERGE"

    # --- RELATE: same meaning in a different frame, or mid-band similarity, or a
    # high-sigma similarity merge withheld because the source is less authoritative
    # than the canonical local (§5.3) -- admitted as a candidate, not a rewrite. ---
    if chi != "contradicts":
        if sigma >= cfg.theta_merge and (not ctx_ok or not authority_ok):
            return "RELATE"
        if cfg.sigma_lo <= sigma < cfg.theta_merge:
            return "RELATE"

    # --- otherwise: no actionable relation (sigma below sigma_lo, no key) ---
    # NOT a REJECT: REJECT is the admission gate (:56 above) alone. This is a
    # pair-adjudication verdict the caller (merge_on_receive) maps to INSERT.
    return "NO_RELATION"


__all__ = ["classify"]
