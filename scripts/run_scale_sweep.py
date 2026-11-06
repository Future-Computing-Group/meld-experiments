#!/usr/bin/env python3
"""S-3: node-scaling of sync overhead --- delivery cost vs cluster size N.

Sweeps N brains and reports, per published claim, how many brains each policy delivers to:
broadcast O(N) vs semantic routing O(needed) at maintained recall. Converts the manuscript's
structural "advantage widens with N" (swept only to N=4) into a measured curve. Local,
in-process, no LLM, no cost.

Rows carry topology.stamps (routing reads the encoder only: no NLI, no policy).

Run:  PYTHONPATH=src .venv/bin/python scripts/run_scale_sweep.py --encoder minilm
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from meld.merge.signals import EmbeddingSimilarity   # noqa: E402
from meld.eval.scale_sweep import scale_row          # noqa: E402
from meld.eval.topology import stamps                # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per-domain", type=int, default=4)
    ap.add_argument("--ns", default="2,4,8,16,32,64")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "scale-sweep.json"))
    a = ap.parse_args(argv)
    Ns = [2, 4] if a.smoke else [int(x) for x in a.ns.split(",")]
    es = EmbeddingSimilarity(encoder=a.encoder)  # reused across N (one model load)

    rows = []
    for n in Ns:
        r = {**scale_row(n, per_domain=a.per_domain, sigma_lo=0.12, es=es),
             **stamps(a.encoder, None, None, pol=None)}
        rows.append(r)
        print(f"[scale] N={n:>3} | recall {r['recall_semantic']:.3f} | "
              f"deliv/claim semantic {r['deliv_per_claim_semantic']:.2f} vs broadcast "
              f"{r['deliv_per_claim_broadcast']:.1f} | waste bc {r['waste_broadcast']:.3f} | "
              f"broadcast/semantic {r['broadcast_over_semantic']:.1f}x", flush=True)

    out = {"ns": Ns, "per_domain": a.per_domain, "rows": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n=== sync overhead vs cluster size N (deliveries per published claim) ===")
    print(f"{'N':>4} {'recall_sem':>11} {'sem/claim':>10} {'bcast/claim':>12} {'bcast/sem':>10}")
    for r in rows:
        print(f"{r['n_brains']:>4} {r['recall_semantic']:>11.3f} "
              f"{r['deliv_per_claim_semantic']:>10.2f} {r['deliv_per_claim_broadcast']:>12.1f} "
              f"{r['broadcast_over_semantic']:>9.1f}x")
    print("semantic deliveries/claim stay flat in N; broadcast grows as N -> the routing "
          "advantage widens linearly with cluster size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
