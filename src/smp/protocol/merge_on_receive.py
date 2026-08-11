"""Merge-on-receive orchestration (architecture A.1/A.2, manuscript Sec. 4.2).

A receiving brain does NOT blindly apply an incoming delta. For each incoming
Claim it finds the best local candidate, computes the three core signals (plus an
optional, off-by-default evidence-overlap signal) + Context gate, runs the
four-way pair-adjudication classifier, and emits exactly ONE Patch carrying
which gates fired and the resulting deltas -- one of five outcomes:

    INSERT    -> a new node admitted ACTIVE (no local candidate at/above the
                 relatedness floor -- the claim is novel, not rejected)
    MERGE     -> a claim-status UPDATE (the claim is admitted/merged)
    RELATE    -> a Mapping (ASSOCIATES) link, the two claims stay distinct
    CONFLICT  -> a first-class CONTRADICTS link, NO silent winner
    REJECT    -> a recorded no-op (the admission gate: stale or overruled)

The Patch -- not the raw delta -- is what the brain later applies to mutate state.
"""
from __future__ import annotations

from typing import List, Optional

from smp.config import Thresholds
from smp.merge.classifier import classify
from smp.merge.context_gate import Context, context_compatible
from smp.merge.signals import EmbeddingSimilarity, chi, epsilon, kappa
from smp.proto import AUTHORITY_LEVELS, Claim, Endpoint, MemoryLink, MemoryUnit
from smp.protocol.messages import Patch


def _claims(unit: MemoryUnit) -> List[Claim]:
    return list(unit.claims)


def _authority_ok(incoming: MemoryUnit, local: MemoryUnit) -> bool:
    """§5.3 admission: True unless the incoming SOURCE is strictly LESS
    authoritative than the more-canonical local TARGET. In the proto authority
    ranking PRECEDENT is most authoritative (smallest enum number); OBSERVATION,
    CLAIM, SPECULATION descend from there. So a SMALLER rank number == MORE
    authoritative, and "incoming less authoritative" == incoming rank number
    GREATER than local's. Equal authority (both default OBSERVATION) -> True ->
    no downgrade, so M-2/headline are unchanged."""
    inc_rank = AUTHORITY_LEVELS[incoming.authority]
    loc_rank = AUTHORITY_LEVELS[local.authority]
    return not (inc_rank > loc_rank)


def _context_of(unit: MemoryUnit, claim: Claim) -> Context:
    # frame derived from the unit's own cell/scope (no longer a hardcoded
    # constant), so two units in different cells are Context-incompatible and a
    # high-similarity cross-cell pair RELATEs on the LIVE path. Default cell
    # ("agent.public") keeps single-frame runs (M-2/headline) unchanged. Freshness
    # is the status field.
    return Context(cell=unit.cell, valid_from=0, valid_until=None,
                   is_fresh=(unit.status == "ACTIVE"))


def merge_on_receive(
    incoming: MemoryUnit,
    local_units: List[MemoryUnit],
    *,
    cfg: Thresholds,
    es: EmbeddingSimilarity,
    nli: str,
    sign: bytes = b"sig",
    version: int = 1,
) -> List[Patch]:
    """Return one Patch per incoming claim."""
    patches: List[Patch] = []
    # §5.3: a local unit/claim that is no longer ACTIVE is not an admission
    # target. Filtered HERE (not inside _best_candidate) so the exclusion covers
    # BOTH the exact-key fast path and the similarity scan from one site: a
    # DEPRECATED/OVERRULED local cannot be revived by an exact-key match from a
    # peer, and the status lifecycle stays monotone.
    local_claims: List[tuple] = [
        (lu, lc) for lu in local_units if lu.status == "ACTIVE"
        for lc in _claims(lu) if lc.claim_status == "ACTIVE"
    ]

    for ic in _claims(incoming):
        best, best_elig = _best_candidate(
            ic, local_claims, es, sigma_lo=cfg.sigma_lo,
            incoming=incoming, theta_merge=cfg.theta_merge)
        if best is None:
            # no real local candidate (best similarity below sigma_lo, or no
            # local claims at all) -> the incoming concept is NOVEL; admit it as
            # its own first-class outcome, INSERT (R2, manuscript Sec. 4.2). The
            # four-way classifier only governs the case where an actual
            # candidate match exists.
            patches.append(_make_patch(
                "INSERT", incoming, ic, gates=["novel"], sign=sign, version=version,
            ))
            continue

        lu, lc, sigma_val = best
        k = kappa(ic, lc)
        chi_val = chi(ic.text, lc.text, model=nli)
        # epsilon: grounding-evidence overlap of the two claims. Empty grounding
        # (ungrounded corpora) -> 0.0 -> the epsilon path is inert (and disabled
        # outright when cfg.eps_hi=0), so HotpotQA / M-2 are unaffected.
        eps_val = epsilon(set(ic.grounding), set(lc.grounding))
        ctx_ok = context_compatible(_context_of(incoming, ic), _context_of(lu, lc))
        # §5.3: a similarity PUBLISH from a low-authority brain into a more-
        # canonical local brain is admitted as a candidate (RELATE), not a silent
        # rewrite. Equal/default authority -> True -> no downgrade (M-2 unchanged).
        auth_ok = _authority_ok(incoming, lu)
        retention_below = (incoming.relevance_magnitude < cfg.r_min
                           and incoming.relevance_magnitude > 0)

        decision = classify(
            cfg=cfg, kappa=k, sigma=sigma_val, epsilon=eps_val, chi=chi_val,
            ctx_ok=ctx_ok, authority_ok=auth_ok, retention_below_min=retention_below,
            status=incoming.status,
        )
        if decision == "RELATE" and best_elig is not None and chi_val != "contradicts":
            # Gate-aware fallback (reviewer round 4): the max-similarity primary is
            # merge-ineligible (Context or authority) and does not contradict; a
            # gate-admissible candidate exists, so adjudicate that one instead.
            # Conflict priority is preserved: a contradicting primary never reaches
            # this branch, and the fallback pair runs its own NLI check below.
            lu2, lc2, sigma2 = best_elig
            chi2 = chi(ic.text, lc2.text, model=nli)
            if chi2 == "contradicts":
                decision, lu, lc, sigma_val, chi_val = "CONFLICT", lu2, lc2, sigma2, chi2
                ctx_ok = True
            else:
                decision, lu, lc, sigma_val, chi_val = "MERGE", lu2, lc2, sigma2, chi2
                ctx_ok = True
            k = 0
        if decision == "NO_RELATION":
            # unreachable today (a candidate implies kappa=1 or sigma>=sigma_lo, so
            # the classifier's below-floor verdict cannot fire here); mapped rather
            # than asserted so a future candidate-selection change degrades to
            # INSERT rather than to an invalid Patch decision.
            decision = "INSERT"
        gates = _gates_for(decision, k, sigma_val, chi_val, ctx_ok)
        patches.append(_make_patch(
            decision, incoming, ic, peer_claim=lc, sigma=sigma_val,
            gates=gates, sign=sign, version=version,
        ))
    return patches


def _best_candidate(ic: Claim, local_claims, es: EmbeddingSimilarity, sigma_lo: float,
                    *, incoming=None, theta_merge=None):
    """Return (primary, eligible): the max-similarity candidate (or exact-key hit)
    and, when it differs, the best candidate that also satisfies the merge-
    admission gates (sigma >= theta_merge, compatible Context, admissible
    authority). Gate-aware selection stops a Context/authority-ineligible
    high-similarity candidate from shadowing an admissible merge target
    (reviewer round 4). Both are None below the relatedness floor (novel).
    Exact-key hits are primary immediately; the eligible track is only consulted
    when the primary is merge-ineligible and does not contradict."""
    best = None
    best_sigma = -1.0
    best_elig = None
    best_elig_sigma = -1.0
    primary_exact = None
    for lu, lc in local_claims:
        if primary_exact is None and kappa(ic, lc) == 1:
            # exact-key hit: the provisionally selected candidate (first in
            # admission order). Do NOT return yet: keep scanning so the
            # gate-admissible track fills, or an exact hit whose frame fails
            # the Context gate would shadow an admissible merge (round 5).
            primary_exact = (lu, lc, 1.0)
            continue
        s = es.sigma(ic.text, lc.text)
        if s > best_sigma:
            best_sigma, best = s, (lu, lc, s)
        if (incoming is not None and theta_merge is not None and s >= theta_merge
                and s > best_elig_sigma
                and context_compatible(_context_of(incoming, ic), _context_of(lu, lc))
                and _authority_ok(incoming, lu)):
            best_elig_sigma, best_elig = s, (lu, lc, s)
    if primary_exact is not None:
        if best_elig is not None and best_elig[1] is primary_exact[1]:
            best_elig = None
        return primary_exact, best_elig
    if best is None or best_sigma < sigma_lo:
        return None, None
    if best_elig is not None and best_elig[1] is best[1]:
        best_elig = None  # same claim; no separate eligible track needed
    return best, best_elig


def _gates_for(decision, k, sigma_val, chi_val, ctx_ok) -> List[str]:
    g: List[str] = []
    if k == 1:
        g.append("kappa")
    else:
        g.append("sigma")
    if decision == "CONFLICT":
        g.append("chi")
    if decision in ("MERGE", "RELATE"):
        g.append("context")
    return g


def _make_patch(decision, incoming, claim, peer_claim: Optional[Claim] = None,
                sigma: float = 0.0, gates=None, sign=b"sig", version=1) -> Patch:
    target = Endpoint(node_id=incoming.id, claim_key=claim.claim_key)
    emitted: list = []
    if decision in ("INSERT", "MERGE"):
        emitted.append(incoming)
        # A same-enough merge via SIMILARITY (kappa=0, sigma>=theta_merge) means the
        # incoming claim is a redundant paraphrase of an existing local canonical.
        # Keep it for provenance/audit, but mark ITS status DEPRECATED (superseded)
        # and record a SUPERSEDES consolidation link, so a status-aware recall
        # surfaces one representative -- this is what makes merge differ from naive
        # union, and it converges STATUS, not truth. An exact-key (kappa=1) or novel
        # insert is canonical and stays ACTIVE.
        paraphrase = peer_claim is not None and "sigma" in (gates or [])
        if paraphrase:
            emitted.append(("STATUS", claim.claim_key,
                            (incoming.id.hex() or "v", "DEPRECATED")))
            emitted.append(MemoryLink(
                type="SUPERSEDES", weight=float(sigma), version=version,
                signature=sign, provenance="CONSOLIDATION",
                src=Endpoint(node_id=incoming.id, claim_key=claim.claim_key),
                dst=Endpoint(claim_key=peer_claim.claim_key),
            ))
        else:
            emitted.append(("STATUS", claim.claim_key,
                            (incoming.id.hex() or "v", "ACTIVE")))
    elif decision == "RELATE":
        # "overlaps": the two claims STAY DISTINCT (Section 4) -- so the incoming
        # claim is RETAINED as a retrievable unit AND an ASSOCIATES link records the
        # relation. Emitting only the link (the prior behavior) dropped the claim
        # from the store, collapsing any store fed related content (centralized
        # RELATEd ~37/40 claims away -> recall@5 0.056).
        emitted.append(incoming)
        emitted.append(MemoryLink(
            type="ASSOCIATES", weight=float(sigma), version=version,
            signature=sign, provenance="INFERRED",
            src=Endpoint(node_id=incoming.id, claim_key=claim.claim_key),
            dst=Endpoint(claim_key=peer_claim.claim_key if peer_claim else b""),
        ))
    elif decision == "CONFLICT":
        emitted.append(MemoryLink(
            type="CONTRADICTS", weight=1.0, version=version, signature=sign,
            provenance="INFERRED",
            src=Endpoint(node_id=incoming.id, claim_key=claim.claim_key),
            dst=Endpoint(claim_key=peer_claim.claim_key if peer_claim else b""),
        ))
    # REJECT: emitted stays empty (recorded no-op)
    return Patch(decision=decision, target=target, emitted=emitted,
                 gates_that_fired=gates or [], version=version, signature=sign)


__all__ = ["merge_on_receive"]
