"""Replica-comparable views of one replica's state for a scope (design §7.1 harnesses, Props 1, 3).

The five components the multi-replica harnesses count divergences on, restricted to one scope
(two replicas compare derived state only for scopes both subscribe to, §2.0): the authored sets
(Merkle roots of D, A, Cal, Ep), Sig (its root), the representatives with vis (from
Store.fingerprint_inputs), the raw links (``derived.links``, sorted, as the certificate reads
them), and the decisions (the certificate's decision digest). The joint post-pass fingerprint
is the certificate's (fold.readjudicate), so nothing here is a second fingerprint.
``WINDOW`` names the derived components a pre-pass window counts; Sig differs by record
batching and arrival, and is reported apart.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

from meld.proto import h16
from meld.protocol.fold import canon_json, decisions_json
from meld.sync.merkle import merkle_root

COMPONENTS = ("authored", "sig", "reps", "links", "decisions")
WINDOW = ("reps", "links", "decisions")


def components(store, scope: str, decisions) -> Dict[str, bytes]:
    """component -> digest; ``decisions`` are the admission cache's (pre-pass) or a pass's."""
    scope_of, cell_of = store.objects.scope_of, store.cell_of()
    mine = {k for k in store.nodes() if scope_of.get(cell_of[k]) == scope}
    vis, groups, _ = store.fingerprint_inputs()
    vis = [v for v in vis if v[0] in mine]
    groups = [g for g in groups if g[0] in mine]
    links = sorted(lk for lk in store.derived.links if lk[1] in mine)
    return {"authored": h16(b"".join(merkle_root(store.objects.ids(k, scope))
                                     for k in ("D", "A", "Cal", "Ep"))),
            "sig": merkle_root(store.objects.ids("Sig", scope)),
            "reps": h16(canon_json((vis, groups))),
            "links": h16(canon_json(links)),
            "decisions": h16(decisions_json(decisions))}


def divergence(views: List[Dict[str, bytes]]) -> Dict[str, int]:
    """Per component, the number of replicas whose value differs from the modal one (0 iff all
    agree)."""
    return {c: len(views) - Counter(v[c] for v in views).most_common(1)[0][1] if views else 0
            for c in COMPONENTS}


__all__ = ["COMPONENTS", "WINDOW", "components", "divergence"]
