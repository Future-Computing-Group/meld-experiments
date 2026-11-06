"""For each tied delta with an int8 element near its rounding boundary, mirror that element across the
boundary (the int8 flip another platform's float jitter would cause) and rebuild the task."""
import importlib.util, json, sys
import numpy as np
spec = importlib.util.spec_from_file_location("diag", sys.argv[1]); D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
from meld.merge import signals, classifier
from meld.config import read_epoch
from meld.eval.runner_args import read_key_file
from meld.agent.hotpot_loader import load_hotpot_real
from meld.agent.mem_conditions import build_condition
from meld.agent.recall_metrics import recall_at_k_and_mrr
from meld.eval.substrates import eval_pool
from meld.proto import canon
epoch = read_epoch(sys.argv[2], read_key_file(sys.argv[3])); cands = json.load(open(sys.argv[4])); out = sys.argv[5]
tasks = eval_pool(load_hotpot_real(split="validation", config="distractor"))[:200]
FLIP = {}
orig = signals.EmbeddingSimilarity._embed
def embed(self, text):
    v = orig(self, text)
    if text in FLIP:
        j = FLIP[text]; v = v.copy(); M = np.abs(v).max(); s = 127 * v[j] / M; b = np.floor(s) + 0.5
        v[j] = (2 * b - s) * M / 127
    return v
signals.EmbeddingSimilarity._embed = embed
signals._CrossEncoderNLI.probs = D._probs; classifier.step = D._step
for c in cands:
    FLIP.clear(); FLIP[canon(c["text"])] = c["elem"]; D.EVENTS.clear()
    t = tasks[c["task"]]
    b = build_condition(t, "distributed_merge", encoder="minilm", nli="mnli-cross-encoder", epoch=epoch,
                        overlap_fraction=c["overlap"], cache_dir=sys.argv[6])
    rec = {str(k): recall_at_k_and_mrr(b.brain, t, k=k, source=b.split.source)["recall_at_k"] for k in (1, 3, 5, 10, 20)}
    with open(out, "a") as fh:
        fh.write(json.dumps({**c, "live": b.live_units, "recall": rec, "n_merge": b.n_merge}) + "\n")
