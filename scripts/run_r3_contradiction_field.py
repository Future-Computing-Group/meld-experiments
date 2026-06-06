#!/usr/bin/env python3
"""R3: measure the NLI's field contradiction-miss (false-negative) rate.

Builds a naturalistic-form labelled sample from real HotpotQA gold sentences (controlled
value-swap contradictions + related-but-non-contradicting neighbours), runs the real NLI
(cross-encoder/nli-deberta-v3-small), and reports contradiction recall/precision and the
missed-contradiction rate -- the deployment safety determinant (the rate at which a real
contradiction is NOT flagged and would silently merge).

Run (smoke):  PYTHONPATH=src .venv/bin/python scripts/run_r3_contradiction_field.py --smoke
Run (full):   PYTHONPATH=src .venv/bin/python scripts/run_r3_contradiction_field.py --n 40
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from smp.merge.signals import chi                                  # noqa: E402
from smp.agent.hotpot_loader import load_hotpot_fixture, load_hotpot_real  # noqa: E402
from smp.eval.contradiction_field import build_field_sample, contradiction_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "r3-contradiction-field.json"))
    a = ap.parse_args()

    if a.smoke:
        tasks, nli = load_hotpot_fixture(n=4), "lexical"
    else:
        tasks, nli = load_hotpot_real(split="validation", config="distractor", n=a.n), "mnli-cross-encoder"

    sample = build_field_sample(tasks, max_pairs=80)
    gold = [lab for lab, _, _ in sample]
    print(f"[run] {len(sample)} labelled real-sentence pairs "
          f"({gold.count('contradicts')} contradictions), nli={nli} ...", flush=True)
    pred = [chi(x, y, model=nli) for _, x, y in sample]
    m = contradiction_metrics(gold, pred)
    m["nli"] = nli
    m["n_pairs"] = len(sample)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2)
    print(json.dumps(m, indent=2))
    print(f"=== NLI field contradiction recall {m['contradiction_recall']}, "
          f"MISSED-contradiction rate {m['missed_contradiction_rate']} "
          f"({m['fn']}/{m['n_contradictions']}), precision {m['contradiction_precision']} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
