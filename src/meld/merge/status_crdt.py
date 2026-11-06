"""Authored status (design §2.1, §3.4, Prop. 4).

auth(k) = join of effect(e) over the authorized, valid effects e in A with c1(e) = k, joined
with ACTIVE, on the chain ACTIVE <= DEPRECATED <= OVERRULED <= REVOKED. Effects map
SUPERSEDE -> DEPRECATED, OVERRULE -> OVERRULED, REVOKE -> REVOKED, DISTINGUISHES -> none.

``authorized(e, policy)`` reads only e's carried fields (author_cell, author_auth, scope,
kind) and the M_pol dict: the author's cell lies in e's scope and author_auth is at most the
policy's maximum rank for e.kind (``policy["auth_policy"][kind]``; a kind the policy does not
name is never authorized). Authorization is scope-granular: with validity (target cell in
e's scope) it places author cell and target cell in one scope, not in one cell. Threat model:
author_cell and author_auth are self-asserted by the author and covered by the group MAC
(benign model, design §2.3); per-author keys (R1-21) would bind them to a principal.
Validity (``objects.effect_valid``) reads e, M_pol and the held c1
only, so it is monotone on nodes and auth never decreases as objects arrive (Prop. 4).
An unauthorized or invalid effect stays in A (auditable) and never materializes.

``StatusState`` is the grow-only set of effects keyed by eid; merge is set union, so
replicas holding the same effects compute the same auth whatever the delivery order.

LEGACY (retired by T6): ``LinkState`` and ``effective_status``/``effective_status_gated``/
``merge`` keep the pre-r3 per-claim link OR-set for the old merge path and its harnesses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional

from meld.protocol.objects import (
    AuthoredEffect, canonical_bytes, effect_valid, object_id, scope_partition,
)

# Monotone lattice rank: higher == more restricted (top = REVOKED).
_RANK = {"ACTIVE": 0, "DEPRECATED": 1, "OVERRULED": 2, "REVOKED": 3}
_BY_RANK = {v: k for k, v in _RANK.items()}
EFFECT_STATUS = {"SUPERSEDE": "DEPRECATED", "OVERRULE": "OVERRULED", "REVOKE": "REVOKED",
                 "DISTINGUISHES": None}


def join(a: str, b: str) -> str:
    """Lattice-join (least upper bound) of two statuses: the more-restricted."""
    return _BY_RANK[max(_RANK[a], _RANK[b])]


def authorized(e: AuthoredEffect, policy: dict,
               scope_of: Optional[Mapping[str, str]] = None) -> bool:
    """A function of e's carried fields and M_pol alone (see module docstring). ``scope_of``
    is ``scope_partition(policy)``, precomputed by callers that test many effects."""
    max_rank = policy["auth_policy"].get(e.kind)
    if scope_of is None:
        scope_of = scope_partition(policy)
    return (max_rank is not None and e.author_auth <= max_rank
            and scope_of.get(e.author_cell) == e.scope)


def epoch_authorized(E, policy: dict) -> bool:
    """An epoch record counts iff its author is the scope's calibration authority, named in
    M_pol as ``policy["auth_policy"]["calibration_authority"][scope]`` (fail closed)."""
    return E.author == policy["auth_policy"].get("calibration_authority", {}).get(E.scope, None)


def auth_map(effects: Iterable[AuthoredEffect], policy: dict,
             cell_of: Mapping[bytes, str]) -> Dict[bytes, str]:
    """auth(k) for every k some counted effect targets; absent keys are ACTIVE.
    ``cell_of`` maps each held claim key to its cell (the validity check)."""
    scope_of = scope_partition(policy)
    out: Dict[bytes, str] = {}
    for e in effects:
        st = EFFECT_STATUS[e.kind]
        if st and authorized(e, policy, scope_of) and effect_valid(e, scope_of, cell_of):
            out[e.c1] = join(out.get(e.c1, "ACTIVE"), st)
    return out


@dataclass(frozen=True)
class StatusState:
    """Grow-only set of authored effects keyed by eid (content address of the canonical
    bytes). Immutable; ``add``/``merge`` return new states."""

    effects: Mapping[bytes, AuthoredEffect] = field(default_factory=dict)

    def add(self, e: AuthoredEffect) -> "StatusState":
        return StatusState({**self.effects, object_id(canonical_bytes(e)): e})

    def merge(self, other: "StatusState") -> "StatusState":
        return StatusState({**self.effects, **other.effects})

    def auth(self, k: bytes, policy: dict, cell_of: Mapping[bytes, str]) -> str:
        return auth_map(self.effects.values(), policy, cell_of).get(k, "ACTIVE")


# ------------------------------------------------------------------------------- LEGACY
@dataclass(frozen=True)
class LinkState:
    """LEGACY per-claim OR-set of status links (link id -> effect status), pre-r3."""

    links: Dict[str, str] = field(default_factory=dict)

    def add(self, link_id: str, effect: str) -> "LinkState":
        new = dict(self.links)
        new[link_id] = join(new[link_id], effect) if link_id in new else effect
        return LinkState(links=new)


def merge(a: LinkState, b: LinkState) -> LinkState:
    """LEGACY: set-union of two link OR-sets, lattice-joining effects on id collision."""
    merged = dict(a.links)
    for lid, eff in b.links.items():
        merged[lid] = join(merged[lid], eff) if lid in merged else eff
    return LinkState(links=merged)


def effective_status(state: LinkState) -> str:
    """LEGACY: lattice-join of all link effects. Empty -> ACTIVE."""
    return effective_status_gated(state)


def effective_status_gated(state: LinkState, *, stale=frozenset()) -> str:
    """LEGACY: the join over links not in ``stale`` (the old read-time staleness gate of
    sync/node.py). There is no caller-supplied unauthorized set: authorization is
    ``authorized(e, policy)`` over carried fields."""
    status = "ACTIVE"
    for lid, eff in state.links.items():
        if lid not in stale:
            status = join(status, eff)
    return status


__all__ = ["StatusState", "EFFECT_STATUS", "authorized", "epoch_authorized", "auth_map", "join",
           "LinkState", "merge", "effective_status", "effective_status_gated"]
