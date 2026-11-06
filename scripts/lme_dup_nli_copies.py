"""Exploratory check for outputs/wp6-results/lme-dup-diagnostic-2026-10.md: how often the deployed NLI
cross-encoder calls a duplicated text and its light_paraphrase copies contradictory, on 60 LongMemEval
user turns and 60 HotpotQA gold sentences (seed 0). Pairs (original, copy) and (copy, copy), either
direction. Run: PYTHONPATH=<freeze-r1.2 export>/src python scripts/lme_dup_nli_copies.py
         data/bench/lme-cleaned/longmemeval_s_cleaned.json"""
import json, random, sys, itertools, collections
from meld.agent.mem_conditions import light_paraphrase
from meld.protocol.fold import nli_provider
from meld.protocol.objects import contradicts
from meld.proto import canon
p = nli_provider("mnli-cross-encoder")
items = json.load(open(sys.argv[1]))
rng = random.Random(0)
turns = [t["content"].strip() for it in items[:300] for s in it["haystack_sessions"] for t in s
         if t["role"] == "user" and t["content"].strip()]
turns = rng.sample(turns, 60)
def rate(texts, tag):
    c = collections.Counter(); n = collections.Counter()
    for t in texts:
        vs = [t] + [light_paraphrase(t, v) for v in range(7)]
        for i, j in itertools.combinations(range(8), 2):
            key = "orig" if i == 0 else "copy"
            n[key] += 1; c[key] += contradicts(p(canon(vs[j]), canon(vs[i]))) or contradicts(p(canon(vs[i]), canon(vs[j])))
    print(tag, {k: f"{c[k]}/{n[k]} = {c[k]/n[k]:.3f}" for k in n}, flush=True)
rate(turns, "lme")
from meld.agent.hotpot_loader import load_hotpot_real
from meld.agent.recall_metrics import gold_supporting_sentences
hs = [s for t in load_hotpot_real()[:200] for s in gold_supporting_sentences(t)]
rate(rng.sample(hs, 60), "hotpot")
