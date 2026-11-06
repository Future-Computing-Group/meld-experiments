"""Tournament metrics and the selection rule, from probs-*.json. Usage: analyse.py [paws_set]"""
import sys, json, glob, numpy as np
from sklearn.metrics import roc_auc_score
OUT = "outputs/tournament"
PAWS = sys.argv[1] if len(sys.argv) > 1 else "paws_val"
BASE = "cross-encoder/nli-deberta-v3-small"
PAWS_TRAINED = {"semindan/paws_x_xlm_r_only_en", "domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector"}
R = {json.load(open(f))["model"]: json.load(open(f)) for f in glob.glob(f"{OUT}/probs-*.json")}


def s_of(m, st):
    d = R[m][st]
    if R[m]["kind"] == "nli":
        return np.minimum(np.array(d["fwd"])[:, 1], np.array(d["bwd"])[:, 1])
    return (np.array(d["fwd"]) + np.array(d["bwd"])) / 2


def contra(nli):
    d = R[nli]["sick_dev"]; f, b = np.array(d["fwd"]), np.array(d["bwd"])
    fire = lambda p: p[:, 0] > np.maximum(p[:, 1], p[:, 2])          # objects.contradicts on one direction
    c = np.array(d["stratum"]) == "S8_sick_contra"
    return float((fire(f) | fire(b))[c].mean())


def sep(s, y):
    y = np.array(y); q = float(np.quantile(s[~y], .95, method="higher"))
    return dict(auc=float(roc_auc_score(y, s)), q95=q, recall=float((s[y] > q).mean()))


def cost(*ms):
    return {k: sum(R[m]["timing"][k] for m in ms) for k in ("s_per_pair_bs1", "s_per_pair_bs16")}


def row(name, s_fn, gate, cost_models, kind):
    r = dict(name=name, kind=kind, gate=gate, contra=contra(gate), cost=cost(*cost_models))
    for st in ("paws_dev", "paws_val", "sick_dev"):
        r[st] = sep(s_fn(st), R[BASE][st]["y"])
    r["primary"] = (r[PAWS]["recall"] + r["sick_dev"]["recall"]) / 2
    return r


rows = []
for m, d in R.items():
    if d["kind"] == "nli":
        rows.append(row(m, lambda st, m=m: s_of(m, st), m, [m], "a"))
    else:
        rows.append(row(m, lambda st, m=m: s_of(m, st), BASE, [m, BASE], "b"))
base = next(r for r in rows if r["name"] == BASE)
for r in rows:
    r["cost_ratio"] = {k: r["cost"][k] / base["cost"][k] for k in r["cost"]}
    r["eligible"] = (base["contra"] - r["contra"] <= 0.05) and max(r["cost_ratio"].values()) <= 4
best = {k: max((r for r in rows if r["kind"] == k), key=lambda r: r["primary"]) for k in ("a", "b")}
bestel = {k: max((r for r in rows if r["kind"] == k and r["eligible"]), key=lambda r: r["primary"]) for k in ("a", "b")}
for tag, B in (("best", best), ("best-eligible", bestel)):
    a, b = B["a"]["name"], B["b"]["name"]
    name = f"min[{a} | {b}]"
    if any(r["name"] == name for r in rows):
        continue
    r = row(name, lambda st, a=a, b=b: np.minimum(s_of(a, st), s_of(b, st)), a, [a, b], "c")
    r["cost_ratio"] = {k: r["cost"][k] / base["cost"][k] for k in r["cost"]}
    r["eligible"] = (base["contra"] - r["contra"] <= 0.05) and max(r["cost_ratio"].values()) <= 4
    r["combo_of"] = tag
    rows.append(r)

print(f"PAWS set for the primary score: {PAWS}\n")
h = f"{'candidate':72s} {'k':1s} {'AUCpd':>5s} {'Rpd':>5s} {'AUCpv':>5s} {'Rpv':>5s} {'AUCs':>5s} {'Rs':>5s} {'contra':>6s} {'ms1':>6s} {'ms16':>6s} {'x1':>5s} {'x16':>5s} {'prim':>5s} el"
print(h)
for r in sorted(rows, key=lambda r: -r["primary"]):
    print(f"{r['name'][:72]:72s} {r['kind']} {r['paws_dev']['auc']:.3f} {r['paws_dev']['recall']:.3f} {r['paws_val']['auc']:.3f} "
          f"{r['paws_val']['recall']:.3f} {r['sick_dev']['auc']:.3f} {r['sick_dev']['recall']:.3f} {r['contra']:.3f} "
          f"{1e3*r['cost']['s_per_pair_bs1']:6.1f} {1e3*r['cost']['s_per_pair_bs16']:6.1f} {r['cost_ratio']['s_per_pair_bs1']:5.2f} "
          f"{r['cost_ratio']['s_per_pair_bs16']:5.2f} {r['primary']:.3f} {'Y' if r['eligible'] else 'n'}{' PAWS-TRAINED' if r['name'] in PAWS_TRAINED or any(p in r['name'] for p in PAWS_TRAINED) else ''}")
bp = max(rows, key=lambda r: r[PAWS]["recall"]); bs = max(rows, key=lambda r: r["sick_dev"]["recall"])
print(f"\ncross-domain: best on PAWS = {bp['name']} (PAWS R {bp[PAWS]['recall']:.3f}, SICK R {bp['sick_dev']['recall']:.3f}); "
      f"best on SICK = {bs['name']} (SICK R {bs['sick_dev']['recall']:.3f}, PAWS R {bs[PAWS]['recall']:.3f})")
win = max((r for r in rows if r["eligible"]), key=lambda r: r["primary"])
margin = win["primary"] - base["primary"]
keep = win["name"] == BASE or margin < 0.05
print(f"rule: best eligible = {win['name']} primary {win['primary']:.3f}; base {base['primary']:.3f}; margin {100*margin:.1f} pts -> "
      f"{'KEEP ' + BASE if keep else 'SWITCH to ' + win['name']}")
json.dump(rows, open(f"{OUT}/metrics-{PAWS}.json", "w"), indent=1)
