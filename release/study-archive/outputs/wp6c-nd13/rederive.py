"""Independent re-derivation of the WP6c headline numbers (PH-1 to PH-6) from the raw archive files.

Stdlib only, no import from the repo. Recomputes point estimates, n and counts from the rows (not from any
summary block), writes rederived.json, then compares them with the analysis output
outputs/wp6-results/wp6c-nd13-tables.md (3-decimal rounding) and exits 1 on any mismatch.

  python3 -I rederive.py [path/to/wp6c-tables.md]
"""
import glob, json, os, re, statistics, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "outputs", "wp6-results", "wp6c-nd13-tables.md")


def load(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, pattern))):
        with open(f) as fh:
            out += [json.loads(l)["result"] for l in fh if l.strip()]
    return out


def r3(x):
    return None if x is None else round(x + 0.0, 3)


def n1(pattern):
    cell = defaultdict(dict)
    for r in load(pattern):
        cell[(r["n_brains"], r["dup"], r["condition"])][r["world"]] = r
    c = cell.pop((1, 1, "centralized"))
    res = {}
    for (N, d, cond) in sorted(k for k in cell if k[2] == "meld"):
        m, u = cell[(N, d, "meld")], cell[(N, d, "naive_union")]
        out = {"worlds": len(m)}
        for k in ("recall@5", "recall@10"):
            diffs = [m[w][k] - c[w][k] for w in m if w in c and None not in (m[w][k], c[w][k])]
            out[k] = {"n": len(diffs), "meld_minus_centr": r3(statistics.fmean(diffs)),
                      "dz": r3(statistics.fmean(diffs) / statistics.stdev(diffs)) if statistics.stdev(diffs) > 0 else None}
        du = [m[w]["recall@5"] - u[w]["recall@5"] for w in m if w in u and None not in (m[w]["recall@5"], u[w]["recall@5"])]
        out["meld_minus_union_r5"] = r3(statistics.fmean(du))
        out["live_meld"] = round(statistics.fmean(r["live"] for r in m.values()), 1)
        out["live_union"] = round(statistics.fmean(r["live"] for r in u.values()), 1)
        res[f"{N},{d}"] = out
    return res


def a2a(apat, spat):
    star = {(r["n_brains"], r["dup"], r["world"]): r for r in load(spat) if r["condition"] == "meld"}
    rows = load(apat)
    eq = sum(star[(r["n_brains"], r["dup"], r["world"])]["recall@5"] == r["recall@5"]
             and star[(r["n_brains"], r["dup"], r["world"])]["live"] == r["live"] for r in rows)
    conv = sum(len(set(r["convergence"]["fingerprints"])) == 1 and not any(r["convergence"]["post_pass"].values()) for r in rows)
    return {"runs": len(rows), "worlds": len({r["world"] for r in rows}), "equal_to_star": eq, "converged": conv}


def e2t2(fn):
    d = json.load(open(os.path.join(HERE, fn)))
    out = {}
    for k in map(str, d["ks"]):
        rec = [w["recall"][k] for w in d["rows"] if w["recall"].get(k)]
        mean = {m: statistics.fmean(x[m] for x in rec) for m in d["modes"]}
        out[k] = {"n": len(rec), **{f"typed-{m}": r3(mean["typed"] - mean[m]) for m in ("node_only", "rewired", "untyped")}}
    return out


def baselines():
    meld = {(r["n_brains"], r["world"]): r["recall@5"] for r in load("n1-hotpot-star/shard-*.jsonl") if r["condition"] == "meld" and r["dup"] == 2}
    out = {}
    for pol in ("mem0-update", "mem0-full", "graphiti", "caura"):
        rows = [r for r in load(f"bl/{pol}/nb-hotpot/shard-*.jsonl") if r["condition"] == pol]
        for N in (3, 8):
            ds = [r["recall@5"] - meld[(N, r["world"])] for r in rows if r["n_brains"] == N and (N, r["world"]) in meld]
            out[f"{pol},{N}"] = {"n": len(ds), "baseline_minus_meld_r5": r3(statistics.fmean(ds))}
    return out


def s1(fn):
    d = json.load(open(os.path.join(HERE, fn)))
    return {f"{k}": {c: d["cells"][f"0.5|{c}|{k}"]["mean_recall"] for c in d["conditions"]} for k in d["ks"]}


def num(s):
    return float(s)


def compare(R):
    T = open(TABLES).read()
    bad, n = [], 0
    for wl, key in (("HotpotQA", "hotpot"), ("LongMemEval", "lme")):
        sec = T.split(f"### PH table {wl}")[1].split("###")[0]
        for line in sec.splitlines():
            m = re.match(r"\| (\d+) \| (\d+) \| (\d+) \| (5|10) \| (-?[\d.]+) \|.*?\| ([-\d.n/a]+) \| (equivalent|not shown)(.*)", line)
            if not m:
                continue
            N, d, nn, k, diff, dzv = m.group(1), m.group(2), int(m.group(3)), m.group(4), num(m.group(5)), m.group(6)
            x = R[key][f"{N},{d}"][f"recall@{k}"]
            n += 1
            if x["n"] != nn or abs(x["meld_minus_centr"] - diff) > 0.0011 or (dzv != "n/a" and x["dz"] is not None and abs(x["dz"] - num(dzv)) > 0.0011):
                bad.append((wl, N, d, k, x, line[:80]))
            if k == "5":
                mu = re.search(r"\| (-?[\d.]+) \[", m.group(8))
                lv = re.search(r"\| [\d.]+ / ([\d.]+) / ([\d.]+) \|", m.group(8))
                y = R[key][f"{N},{d}"]
                n += 1
                if abs(num(mu.group(1)) - y["meld_minus_union_r5"]) > 0.0011 or abs(num(lv.group(1)) - y["live_meld"]) > 0.051 or abs(num(lv.group(2)) - y["live_union"]) > 0.051:
                    bad.append((wl, N, d, "union/live", y, line[:80]))
    sec = T.split("### E2-T2 typed-link value on HotpotQA")[1].split("###")[0]
    for line in sec.splitlines():
        m = re.match(r"\| (\d+) \|.*?\| (-?[\d.]+) \[.*?\| (-?[\d.]+) \[.*?\| (-?[\d.]+) \[", line)
        if m:
            x = R["e2t2_hotpot"][m.group(1)]
            for j, c in ((2, "typed-node_only"), (3, "typed-rewired"), (4, "typed-untyped")):
                n += 1
                if abs(num(m.group(j)) - x[c]) > 0.0011:
                    bad.append(("e2t2", m.group(1), c, x[c], m.group(j)))
    for line in T.split("### Baselines on the recall")[1].splitlines():
        m = re.match(r"\| hotpot \| (\d) \| ([\w-]+) \| (\d+) \|.*?\| (-?[\d.]+) \[", line)
        if m:
            x = R["baselines_hotpot"][f"{m.group(2)},{m.group(1)}"]
            n += 1
            if x["n"] != int(m.group(3)) or abs(x["baseline_minus_meld_r5"] - num(m.group(4))) > 0.0011:
                bad.append(("bl", m.group(2), m.group(1), x, m.group(4)))
    return n, bad


if __name__ == "__main__":
    R = {"hotpot": n1("n1-hotpot-star/shard-*.jsonl"), "lme": n1("n1-lme-star/shard-*.jsonl"),
         "a2a_hotpot": a2a("n1-hotpot-a2a/shard-*.jsonl", "n1-hotpot-star/shard-*.jsonl"),
         "a2a_lme": a2a("n1-lme-a2a/shard-*.jsonl", "n1-lme-star/shard-*.jsonl"),
         "s1_overlap0.5": s1("s1/r2-overlap-k-n200.json"), "e2t2_hotpot": e2t2("e2t2/hotpot-n200.json"),
         "baselines_hotpot": baselines()}
    with open(os.path.join(HERE, "rederived.json"), "w") as fh:
        json.dump(R, fh, indent=1, sort_keys=True)
    n, bad = compare(R)
    print(f"compared {n} values against {os.path.relpath(TABLES, HERE)}; mismatches {len(bad)}")
    for b in bad:
        print("MISMATCH", b)
    print("a2a", R["a2a_hotpot"], R["a2a_lme"])
    sys.exit(1 if bad or n == 0 else 0)
