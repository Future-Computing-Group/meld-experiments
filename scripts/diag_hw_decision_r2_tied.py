"""Tied deltas (equal created_at) of the R2 overlap 0.5/0.75 distributed_merge builds whose int8 q has an element
within argv[2] (scaled units) of its rounding boundary. argv: EMB_CACHE_DIR THRESHOLD OUT.json"""
import json, sys, numpy as np
from collections import defaultdict
from meld.agent.hotpot_loader import load_hotpot_real
from meld.agent.mem_conditions import split_across_brains
from meld.eval.substrates import eval_pool
from meld.merge.signals import EmbeddingSimilarity
from meld.proto import canon
es = EmbeddingSimilarity('minilm', cache_dir=sys.argv[1]); TH = float(sys.argv[2])
tasks = eval_pool(load_hotpot_real(split="validation", config="distractor"))[:200]
out = []
for ov in (0.5, 0.75):
    for i, t in enumerate(tasks):
        sp = split_across_brains(t, n_brains=3, overlap_fraction=ov, dup_degree=2)
        by = defaultdict(set)
        for b in sp.per_brain:
            for x in b:
                x = (x or "").strip()
                if x: by[sp.when[x]].add(x)
        for w, xs in by.items():
            if len(xs) < 2: continue
            for x in xs:
                v = es._embed(canon(x)); s = 127 * v / np.abs(v).max()
                d = np.abs(np.abs(s - np.floor(s)) - 0.5); j = int(np.argmin(d))
                if d[j] < TH:
                    out.append({"overlap": ov, "task": i, "when": w, "text": x, "elem": j, "dist": float(d[j]), "emb_margin": float(d[j] * np.abs(v).max() / 127)})
json.dump(out, open(sys.argv[3], "w"), indent=0)
print(len(out))
