"""Signal-model tournament, dev only. Usage: score.py <model_id> <kind: nli|para>
Writes probs-<model>.json: per set, per pair, both orders (NLI: [pc,pe,pn] per direction; para: p_para per order).
Sets: paws_dev (manifest S1/S2 dev, from PAWS train), paws_val (PAWS official validation, 300+300 sample),
sick_dev (S8 dev from the sickdev worktree manifest). Test and cal rows are filtered out at load."""
import sys, json, os, time, numpy as np, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, "src")
from meld.bench.loaders import detok_paws
torch.set_num_threads(8)
OUT = "outputs/tournament"
WT = ".worktrees/sickdev/data/bench/manifest.json"


def sets():
    m = json.load(open(WT))
    dev = [p for p in m["pairs"] if p["split"] == "dev"]          # never touch other splits
    S = {"paws_dev": [(p["text_a"], p["text_b"], p["stratum"], bool(p["mergeable"])) for p in dev if p["source"] == "paws"],
         "sick_dev": [(p["text_a"], p["text_b"], p["stratum"], bool(p["mergeable"])) for p in dev if p["source"] == "sick"]}
    v = pd.read_parquet(f"{OUT}/pawsval/validation.parquet")
    rng = np.random.RandomState(0); idx = []
    for lab in (1, 0):
        ii = np.flatnonzero(v.label.values == lab); idx += list(rng.choice(ii, 300, replace=False))
    S["paws_val"] = [(detok_paws(v.sentence1.iloc[i]), detok_paws(v.sentence2.iloc[i]),
                      "para" if v.label.iloc[i] == 1 else "nonpara", bool(v.label.iloc[i] == 1)) for i in sorted(idx)]
    return S


class Model:
    def __init__(self, mid, kind):
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.mod = AutoModelForSequenceClassification.from_pretrained(mid).eval()
        self.kind = kind
        lab = {int(k): str(v).lower() for k, v in self.mod.config.id2label.items()}
        if kind == "nli":
            names = {v: k for k, v in lab.items()}
            assert sorted(names) == ["contradiction", "entailment", "neutral"], lab
            self.idx = [names["contradiction"], names["entailment"], names["neutral"]]
        else:
            n = self.mod.config.num_labels
            # single logit -> sigmoid; two labels -> softmax slot of the positive class
            self.pos = None if n == 1 else next((k for k, v in lab.items() if v in ("equivalent", "label_1", "paraphrase", "duplicate")), None)
            assert n == 1 or self.pos is not None, lab

    @torch.no_grad()
    def __call__(self, pairs, bs=16):
        o = []
        for i in range(0, len(pairs), bs):
            b = pairs[i:i + bs]
            x = self.tok([p[0] for p in b], [p[1] for p in b], return_tensors="pt", padding=True, truncation=True, max_length=512)
            lg = self.mod(**x).logits
            if self.kind == "nli":
                o += torch.softmax(lg, -1)[:, self.idx].tolist()
            elif self.pos is None:
                o += torch.sigmoid(lg[:, 0]).tolist()
            else:
                o += torch.softmax(lg, -1)[:, self.pos].tolist()
        return o


def timing(model, pairs):
    """Seconds per pair for both calls (a->b and b->a), batch 1 and batch 16."""
    r = {}
    for bs, n in ((1, 48), (16, 192)):
        P = pairs[:n]; model(P[:bs])                               # warm-up
        t0 = time.perf_counter(); model(P, bs); model([(b, a) for a, b, *_ in P], bs)
        r[f"s_per_pair_bs{bs}"] = (time.perf_counter() - t0) / len(P)
    return r


if __name__ == "__main__":
    mid, kind = sys.argv[1], sys.argv[2]
    M = Model(mid, kind)
    sanity = M([("The cat sat on the mat.", "On the mat sat the cat."), ("The dog bit the man.", "The man bit the dog."),
                ("A man is playing a guitar.", "A man is not playing a guitar.")], 1)
    print(mid, M.mod.config.id2label, "sanity", [np.round(s, 3).tolist() for s in sanity], flush=True)
    S = sets(); res = {"model": mid, "kind": kind, "sanity": sanity}
    for k, L in S.items():
        res[k] = {"fwd": M([(a, b) for a, b, *_ in L]), "bwd": M([(b, a) for a, b, *_ in L]),
                  "stratum": [x[2] for x in L], "y": [x[3] for x in L]}
        print(k, len(L), flush=True)
    res["timing"] = timing(M, S["sick_dev"] + S["paws_dev"])
    print(res["timing"], flush=True)
    json.dump(res, open(f"{OUT}/probs-{mid.replace('/', '_')}.json", "w"))
