"""Merge-on-receive orchestration (architecture A.1/A.2).

A receiving brain does NOT blindly apply an incoming delta. For each incoming
Claim it finds the best local candidate, computes the three core signals (plus an
optional, off-by-default evidence-overlap signal) + Context gate, runs the
four-way classifier, and emits exactly ONE Patch carrying which gates fired and
the resulting deltas:

    MERGE     -> a claim-status UPDATE (the claim is admitted/merged)
    RELATE    -> a Mapping (ASSOCIATES) link, the two claims stay distinct
    CONFLICT  -> a first-class CONTRADICTS link, NO silent winner
    REJECT    -> a recorded no-op

A novel incoming concept with no local candidate is admitted as a MERGE (insert).
The Patch -- not the raw delta -- is what the brain later applies to mutate state.
"""
from __future__ import annotations

from typing import List, Optional

from smp.config import Thresholds
from smp.merge.classifier import classify
from smp.merge.context_gate import Context, context_compatible
from smp.merge.signals import EmbeddingSimilarity, chi, epsilon, kappa
from smp.proto import Claim, Endpoint, MemoryLink, MemoryUnit
from smp.protocol.messages import Patch


def _claims(unit: MemoryUnit) -> List[Claim]:
    return list(unit.claims)


def _context_of(unit: MemoryUnit, claim: Claim) -> Context:
    # frame derived from the unit's scope (here a single default cell for sim);
    # real wiring reads Endpoint cell/ACL; freshness is the status field.
    return Context(cell="agent.public", valid_from=0, valid_until=None,
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
    local_claims: List[tuple] = [
        (lu, lc) for lu in local_units for lc in _claims(lu)
    ]

    for ic in _claims(incoming):
        best = _best_candidate(ic, local_claims, es, sigma_lo=cfg.sigma_lo)
        if best is None:
            # no real local candidate (best similarity below sigma_lo, or no
            # local claims at all) -> the incoming concept is NOVEL; admit it as
            # an insert (MERGE). The four-way classifier only governs the case
            # where an actual candidate match exists.
            patches.append(_make_patch(
                "MERGE", incoming, ic, gates=["novel"], sign=sign, version=version,
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
        retention_below = (incoming.relevance_magnitude < cfg.r_min
                           and incoming.relevance_magnitude > 0)

        decision = classify(
            cfg=cfg, kappa=k, sigma=sigma_val, epsilon=eps_val, chi=chi_val,
            ctx_ok=ctx_ok, retention_below_min=retention_below,
            status=incoming.status,
        )
        gates = _gates_for(decision, k, sigma_val, chi_val, ctx_ok)
        patches.append(_make_patch(
            decision, incoming, ic, peer_claim=lc, sigma=sigma_val,
            gates=gates, sign=sign, version=version,
        ))
    return patches


def _best_candidate(ic: Claim, local_claims, es: EmbeddingSimilarity, sigma_lo: float):
    """Return the best local candidate (lu, lc, sigma) only if it is a REAL
    match: an exact claim-key hit, or a similarity at/above sigma_lo. Below the
    floor there is no candidate (the incoming claim is novel)."""
    best = None
    best_sigma = -1.0
    for lu, lc in local_claims:
        if kappa(ic, lc) == 1:
            return (lu, lc, 1.0)
        s = es.sigma(ic.text, lc.text)
        if s > best_sigma:
            best_sigma, best = s, (lu, lc, s)
    if best is None or best_sigma < sigma_lo:
        return None
    return best


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
    if decision == "MERGE":
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
