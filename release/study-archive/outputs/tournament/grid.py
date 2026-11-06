"""All NLI x paraphrase min-combinations (ablation beyond the rule's best-a x best-b), plus a bootstrap of the winner's margin."""
import sys, numpy as np
sys.argv = ["x", "paws_val"]
exec(open("outputs/tournament/analyse.py").read().split("print(f\"PAWS set")[0])
A = [m for m in R if R[m]["kind"] == "nli"]; B = [m for m in R if R[m]["kind"] == "para"]
out = []
for a in A:
    for b in B:
        r = row(f"min[{a.split('/')[-1]} | {b.split('/')[-1]}]", lambda st, a=a, b=b: np.minimum(s_of(a, st), s_of(b, st)), a, [a, b], "c")
        r["x"] = max(r["cost"][k] / base["cost"][k] for k in r["cost"])
        out.append(r)
for r in sorted(out, key=lambda r: -r["primary"])[:12]:
    print(f"{r['name'][:75]:75s} Rpv {r['paws_val']['recall']:.3f} Rpd {r['paws_dev']['recall']:.3f} Rs {r['sick_dev']['recall']:.3f} contra {r['contra']:.2f} cost x{r['x']:.2f} prim {r['primary']:.3f} {'Y' if r['x']<=4 else 'n'}")
# bootstrap margin: winner vs base, resample pairs within each set (stratified by label)
rng = np.random.RandomState(0)
W = lambda st: np.minimum(s_of("cross-encoder/nli-deberta-v3-base", st), s_of("semindan/paws_x_xlm_r_only_en", st))
Bs = lambda st: s_of(BASE, st)
def rec(s, y):
    q = np.quantile(s[~y], .95, method="higher"); return (s[y] > q).mean()
d = []
for _ in range(2000):
    v = 0
    for st in ("paws_val", "sick_dev"):
        y = np.array(R[BASE][st]["y"]); i = np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum()) for c in (True, False)])
        v += (rec(W(st)[i], y[i]) - rec(Bs(st)[i], y[i])) / 2
    d.append(v)
print("margin bootstrap 95% CI (pts):", np.round(100 * np.quantile(d, [.025, .5, .975]), 1))
