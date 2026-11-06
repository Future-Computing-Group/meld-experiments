"""§7-r: semantic Patch routing vs broadcast.

The routing ablation: a Patch about topic X should be delivered only to brains
whose content is relevant to X (semantic routing), not broadcast to every brain.
Ground-truth "needs it" = the brain would MERGE/RELATE the claim (relevant), not
REJECT it. Metrics: routing precision (delivered that were needed / delivered),
recall (needed that were delivered / needed), wasted-delivery rate (delivered but
not needed / delivered). The win over broadcast is equal recall at far lower waste.
Built on the MELD sigma signal (meld.merge.signals).
"""
from __future__ import annotations


def _brains():
    return {
        "b_paris": ["Paris is the capital of France", "France is a country in Europe"],
        "b_everest": ["Mount Everest is the tallest mountain on Earth"],
        "b_orwell": ["George Orwell wrote the novel 1984"],
    }


def test_semantic_route_delivers_to_relevant_brain_only():
    from meld.sync.router import semantic_route
    delivered = semantic_route("Paris is the capital city of France", _brains(),
                               encoder="hashing", sigma_lo=0.3)
    assert delivered == {"b_paris"}


def test_semantic_route_reuses_shared_es_to_avoid_reencoding():
    # perf: scoring many claims against peer content must NOT re-encode per claim. A shared
    # EmbeddingSimilarity caches peer-content (and the claim) so the 2nd route hits the cache.
    from meld.merge.signals import EmbeddingSimilarity
    from meld.sync.router import semantic_route
    es = EmbeddingSimilarity(encoder="hashing")
    semantic_route("Paris is the capital of France", _brains(),
                   encoder="hashing", sigma_lo=0.0, es=es)
    hits = es.cache_hits
    semantic_route("Paris is the capital of France", _brains(),
                   encoder="hashing", sigma_lo=0.0, es=es)
    assert es.cache_hits > hits          # second route hits the cache (no re-encode)


def test_score_brains_exposes_ranked_relevance():
    # the scoring seam NR never built: per-brain relevance score, not a bare set.
    from meld.sync.router import score_brains
    s = score_brains("Paris is the capital city of France", _brains(), encoder="hashing")
    assert set(s) == {"b_paris", "b_everest", "b_orwell"}
    assert s["b_paris"] > s["b_everest"]
    assert s["b_paris"] > s["b_orwell"]


def test_semantic_route_top_k_keeps_highest_scoring():
    # two brains clear a low floor; top_k=1 keeps only the most relevant (score-ranked,
    # NOT a positional slice -- the NR router.py:606 bug we avoid).
    from meld.sync.router import semantic_route
    brains = {"b_france": ["France is a country in Europe"],
              "b_paris": ["Paris is the capital of France"]}
    d = semantic_route("Paris is the capital city of France", brains,
                       encoder="hashing", sigma_lo=0.05, top_k=1)
    assert d == {"b_paris"}


def test_broadcast_route_delivers_to_all():
    from meld.sync.router import broadcast_route
    assert broadcast_route(_brains()) == {"b_paris", "b_everest", "b_orwell"}


def test_routing_metrics_precision_recall_waste():
    from meld.sync.router import routing_metrics
    # one patch: needed by {a}, delivered to {a,b} -> precision 1/2, recall 1/1, waste 1/2
    m = routing_metrics(delivered=[{"a", "b"}], needed=[{"a"}])
    assert abs(m["precision"] - 0.5) < 1e-9
    assert abs(m["recall"] - 1.0) < 1e-9
    assert abs(m["wasted_delivery_rate"] - 0.5) < 1e-9


def test_semantic_beats_broadcast_on_waste_at_equal_recall():
    # across a few topic-specific patches, semantic routing recalls the needed
    # brain every time while wasting far fewer deliveries than broadcast.
    from meld.sync.router import broadcast_route, routing_metrics, semantic_route
    brains = _brains()
    patches = [("Paris is the capital city of France", {"b_paris"}),
               ("Everest is the highest mountain on Earth", {"b_everest"}),
               ("Orwell authored the book 1984", {"b_orwell"})]
    sem_deliv = [semantic_route(p, brains, encoder="hashing", sigma_lo=0.3) for p, _ in patches]
    bc_deliv = [broadcast_route(brains) for _ in patches]
    needed = [n for _, n in patches]
    sem = routing_metrics(delivered=sem_deliv, needed=needed)
    bc = routing_metrics(delivered=bc_deliv, needed=needed)
    assert sem["recall"] == bc["recall"] == 1.0          # both deliver to the needed brain
    assert sem["wasted_delivery_rate"] < bc["wasted_delivery_rate"]   # semantic wastes less


# --- budget-free selection rules (wp6h): R1 floor = sigma_lo alone, R2 margin, R3 largest gap ---

def test_select_peers_floor_only_is_r1():
    from meld.sync.router import select_peers
    s = {"a": 0.95, "b": 0.40, "c": 0.10}
    assert select_peers(s, sigma_lo=0.5) == {"a"}
    assert select_peers(s, sigma_lo=0.12) == {"a", "b"}


def test_select_peers_margin_keeps_scores_within_delta_of_max():
    from meld.sync.router import select_peers
    s = {"a": 0.95, "b": 0.90, "c": 0.60}
    assert select_peers(s, sigma_lo=0.0, margin=0.1) == {"a", "b"}
    assert select_peers(s, sigma_lo=0.0, margin=0.01) == {"a"}
    assert select_peers(s, sigma_lo=0.92, margin=0.5) == {"a"}      # the floor still applies


def test_select_peers_gap_cuts_at_largest_gap_with_floor_as_bottom():
    from meld.sync.router import select_peers
    s = {"a": 1.0, "b": 0.98, "c": 0.45, "d": 0.40}
    assert select_peers(s, sigma_lo=0.1, gap=True) == {"a", "b"}     # gap 0.53 between b and c
    # the floor is the virtual bottom score: when every eligible peer sits far above it,
    # the largest gap is to the floor and every eligible peer is kept
    s2 = {"a": 0.99, "b": 0.97, "c": 0.95}
    assert select_peers(s2, sigma_lo=0.3, gap=True) == {"a", "b", "c"}
    assert select_peers({"a": 0.2}, sigma_lo=0.3, gap=True) == set()  # nothing eligible
    assert select_peers({"a": 0.5}, sigma_lo=0.3, gap=True) == {"a"}


def test_select_peers_default_matches_semantic_route_contract():
    from meld.sync.router import select_peers
    s = {"a": 0.9, "b": 0.8, "c": 0.05}
    assert select_peers(s, sigma_lo=0.1) == {"a", "b"}
    assert select_peers(s, sigma_lo=0.1, top_k=1) == {"a"}


def test_semantic_route_forwards_margin_and_gap():
    from meld.sync.router import semantic_route
    brains = {"b_france": ["France is a country in Europe"],
              "b_paris": ["Paris is the capital of France"],
              "b_orwell": ["George Orwell wrote the novel 1984"]}
    claim = "Paris is the capital of France"
    assert semantic_route(claim, brains, encoder="hashing", sigma_lo=0.0, margin=1e-6) == {"b_paris"}
    assert "b_paris" in semantic_route(claim, brains, encoder="hashing", sigma_lo=0.0, gap=True)
