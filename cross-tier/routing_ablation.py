#!/usr/bin/env python3
"""§7-r routing ablation, four strategies, NPS-aligned.

Compares the delivery cost of four routing strategies over the MELD relevance
signal, mirroring the federated-pub/sub baselines (broadcast / random / round-robin
/ semantic). Each Patch needs exactly the brains whose interest it matches (ground
truth by construction); a strategy delivers a set of brains per Patch; we report
recall (needed brains reached) and wasted-delivery rate (deliveries to brains that
did not need it). Semantic routing should reach the needed brains at near-zero waste;
random/round-robin spend the same budget blindly; broadcast reaches everyone at high
waste. Built on the tested meld.sync.router.

Run:  PYTHONPATH=src python3 cross-tier/routing_ablation.py
"""
import json, os, random, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from meld.sync.router import (broadcast_route, semantic_route, random_route,  # noqa: E402
                             round_robin_route, routing_metrics)
from meld.merge.signals import EmbeddingSimilarity                            # noqa: E402

# Three well-separated interest domains -> one brain each; each Patch belongs to one
# domain and needs exactly that brain (so broadcast wastes 2/3 = 0.667 by construction).
DOMAINS = {
    "b-weather": ["the weather forecast and rainfall", "temperature and climate trends",
                  "storms, wind, and humidity"],
    "b-finance": ["stock prices and equity markets", "interest rates and bond yields",
                  "currency exchange and inflation"],
    "b-medicine": ["human anatomy and the heart", "diseases, symptoms, and diagnosis",
                   "drugs, dosage, and treatment"],
}
PATCHES = [  # (text, needed-brain) -- the ground-truth target by construction
    ("rain is expected with high humidity tomorrow", "b-weather"),
    ("the storm brings strong winds and cold air", "b-weather"),
    ("equity markets fell as bond yields rose", "b-finance"),
    ("the central bank raised interest rates again", "b-finance"),
    ("the drug lowers blood pressure in the heart", "b-medicine"),
    ("symptoms include fever and the diagnosis is clear", "b-medicine"),
]


def run_ablation(*, sigma_lo, encoder="minilm", seed=0):
    brains = DOMAINS
    es = EmbeddingSimilarity(encoder=encoder)
    rng = random.Random(seed)
    needed = [{nb} for _, nb in PATCHES]
    out = {}
    for strat in ("broadcast", "random", "round_robin", "semantic"):
        delivered = []
        for i, ((ptext, _), need) in enumerate(zip(PATCHES, needed)):
            if strat == "broadcast":
                d = broadcast_route(brains)
            elif strat == "semantic":
                # rank-and-select the same budget as random/round-robin, so the
                # comparison isolates *which* brains each strategy picks, not how many.
                d = semantic_route(ptext, brains, encoder=encoder, sigma_lo=sigma_lo,
                                   top_k=len(need), es=es)
            elif strat == "random":
                d = random_route(brains, k=len(need), rng=rng)
            else:
                d = round_robin_route(brains, k=len(need), offset=i)
            delivered.append(d)
        m = routing_metrics(delivered=delivered, needed=needed)
        out[strat] = {"recall": round(m["recall"], 3),
                      "wasted_delivery_rate": round(m["wasted_delivery_rate"], 3),
                      "n_delivered": m["n_delivered"]}
    return out


if __name__ == "__main__":
    res = run_ablation(sigma_lo=0.12)
    print(json.dumps(res, indent=2))
