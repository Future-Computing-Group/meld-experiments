"""Loss-injection convergence experiment: the anti-entropy digest closes the SEC
gap under lossy semantic routing.

N brains each receive a fixed contested-lifecycle status-link set, but every
delivery is dropped independently with probability ``loss`` (the protocol's lossy
semantic routing: ~0.3% of deltas dropped at recall 0.997). We measure the
fraction of brains agreeing on the FULL per-claim status map in two conditions:

  (i)  WITHOUT the digest -- lost links are permanent, so agreement falls and the
       brains fragment into several distinct status maps as loss grows; and
  (ii) WITH periodic anti-entropy rounds -- the digest detects the per-claim
       symmetric difference of link ids and pulls the missing links, so every
       brain reconverges to the complete set -> agreement 1.000.

This is sync-layer only: it reuses the canonical status-link scenario and the
status CRDT, and never touches the merge classifier (the M-2 headline). Pure
simulation, deterministic in the seed.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Tuple

from meld.merge.status_crdt import join
from meld.sync.anti_entropy import LinkBook, anti_entropy_round

# Effects in lattice-join order (DEPRECATED < OVERRULED < REVOKED).
_EFFECTS = ("DEPRECATED", "OVERRULED", "REVOKED")


def _canonical_link_set(n_claims: int = 24, seed: int = 7) -> List[Tuple[str, str, str]]:
    """A fixed contested-lifecycle status-link set: each of ``n_claims`` claims is
    overruled/superseded by several peers issuing DIFFERENT effects (the exact
    situation the status CRDT exists to resolve), plus one idempotent re-delivery.

    Larger than the 4-claim in-process scenario so that even a small loss rate has
    a measurable chance of dropping a link somewhere across the brains. Fixed +
    deterministic in its own seed (independent of the per-cell delivery seed)."""
    rng = random.Random(seed)
    links: List[Tuple[str, str, str]] = []
    for c in range(n_claims):
        ck = f"c{c}"
        # 2-3 competing effect links per claim, drawn from the join lattice.
        k = 2 + (c % 2)                       # alternate 2 and 3 links per claim
        chosen = [_EFFECTS[rng.randrange(len(_EFFECTS))] for _ in range(k)]
        # ensure at least two DISTINCT effects so the join is non-trivial
        if len(set(chosen)) == 1:
            chosen[0] = _EFFECTS[(_EFFECTS.index(chosen[0]) + 1) % len(_EFFECTS)]
        for j, eff in enumerate(chosen):
            links.append((ck, f"{ck}.l{j}", eff))
        # one idempotent re-delivery (OR-set add) on the first link of each claim
        links.append((ck, f"{ck}.l0", chosen[0]))
    return links


def _expected_status_map(links: List[Tuple[str, str, str]]) -> Dict[str, str]:
    """The converged per-claim status: the lattice-join of every claim's effects."""
    out: Dict[str, str] = {}
    for ck, _lid, eff in links:
        out[ck] = join(out.get(ck, "ACTIVE"), eff)
    return out


def _status_map(book: LinkBook) -> Dict[str, str]:
    return book.effective_statuses()


def _map_key(m: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(m.items()))


def _agreement(maps: List[Dict[str, str]]) -> Tuple[float, int]:
    """(fraction of brains sharing the modal status map, number of distinct maps)."""
    keys = [_map_key(m) for m in maps]
    clusters = Counter(keys)
    modal = clusters.most_common(1)[0][1]
    return modal / len(maps), len(clusters)


def _deliver_with_loss(links, n_brains: int, loss: float,
                       rng: random.Random) -> List[LinkBook]:
    """Deliver every link to every brain, dropping each delivery independently with
    probability ``loss`` (the lossy semantic routing). Returns one LinkBook per
    brain holding only the links it received."""
    books = [LinkBook() for _ in range(n_brains)]
    for ck, lid, eff in links:
        for b in range(n_brains):
            if rng.random() >= loss:               # delivered (1 - loss)
                books[b].add(ck, lid, eff)
    return books


def _reconcile(books: List[LinkBook], *, max_rounds: int = 20) -> Tuple[int, int, int]:
    """Run periodic anti-entropy rounds (ring-gossip: brain i with brain i+1) until
    all brains agree, or ``max_rounds``. Returns
    (rounds_to_converge, total_links_repaired, total_digest_bytes)."""
    n = len(books)
    total_repaired = 0
    total_bytes = 0
    rounds = 0
    for r in range(max_rounds):
        rounds = r + 1
        round_repaired = 0
        for i in range(n):
            cost = anti_entropy_round(books[i], books[(i + 1) % n])
            round_repaired += cost.links_pulled_into_a + cost.links_pulled_into_b
            total_bytes += cost.bytes_exchanged
        total_repaired += round_repaired
        maps = [_status_map(b) for b in books]
        if _agreement(maps)[1] == 1:               # one distinct map -> converged
            break
    return rounds, total_repaired, total_bytes


def loss_convergence_cell(*, loss: float, n_brains: int, seed: int,
                          digest: bool, n_claims: int = 24) -> Dict:
    """One experiment cell: deliver the canonical link set to ``n_brains`` under
    independent per-delivery ``loss``; report per-claim status agreement WITHOUT
    the digest, or WITH periodic anti-entropy reconciliation. Deterministic in
    ``seed``."""
    links = _canonical_link_set(n_claims=n_claims)
    expected = _expected_status_map(links)
    rng = random.Random((seed << 16) ^ int(loss * 1e6) ^ n_brains)
    books = _deliver_with_loss(links, n_brains, loss, rng)

    rounds = links_repaired = digest_bytes = 0
    if digest:
        rounds, links_repaired, digest_bytes = _reconcile(books)

    maps = [_status_map(b) for b in books]
    agreement, n_distinct = _agreement(maps)
    modal_map = dict(Counter(_map_key(m) for m in maps).most_common(1)[0][0])
    return {
        "loss": loss, "n_brains": n_brains, "seed": seed, "digest": digest,
        "n_claims": n_claims, "n_links": len(links),
        "agreement": agreement, "n_distinct_maps": n_distinct,
        "status_map": modal_map, "expected_status_map": expected,
        "rounds_to_converge": rounds, "links_repaired": links_repaired,
        "digest_bytes": digest_bytes,
    }


def loss_convergence_sweep(*, losses: List[float], n_brains: int,
                           seeds: List[int], n_claims: int = 24) -> Dict:
    """Sweep loss rates x seeds, both WITHOUT and WITH the digest. Per loss rate,
    report the mean without-digest agreement (and max distinct maps observed), the
    with-digest agreement (expected 1.0), and the digest cost (mean bytes/round and
    rounds to convergence). Deterministic in the seed list."""
    rows: List[Dict] = []
    for loss in losses:
        wo_agree: List[float] = []
        wo_distinct: List[int] = []
        w_agree: List[float] = []
        w_bytes: List[int] = []
        w_rounds: List[int] = []
        w_repaired: List[int] = []
        for seed in seeds:
            c0 = loss_convergence_cell(loss=loss, n_brains=n_brains, seed=seed,
                                       digest=False, n_claims=n_claims)
            c1 = loss_convergence_cell(loss=loss, n_brains=n_brains, seed=seed,
                                       digest=True, n_claims=n_claims)
            wo_agree.append(c0["agreement"])
            wo_distinct.append(c0["n_distinct_maps"])
            w_agree.append(c1["agreement"])
            w_bytes.append(c1["digest_bytes"])
            w_rounds.append(c1["rounds_to_converge"])
            w_repaired.append(c1["links_repaired"])
        n = len(seeds)
        rows.append({
            "loss": loss, "n_brains": n_brains, "n_seeds": n,
            "without_digest_agreement": sum(wo_agree) / n,
            "without_digest_max_distinct_maps": max(wo_distinct),
            "with_digest_agreement": sum(w_agree) / n,
            "digest_bytes_total_mean": sum(w_bytes) / n,
            "rounds_to_converge_mean": sum(w_rounds) / n,
            "links_repaired_mean": sum(w_repaired) / n,
        })
    return {"n_brains": n_brains, "seeds": seeds, "n_claims": n_claims, "rows": rows}


__all__ = ["loss_convergence_cell", "loss_convergence_sweep",
           "_canonical_link_set", "_expected_status_map"]
