"""Exploratory diagnostic (not a result run): why MELD loses recall@k against centralized on
LongMemEval as the duplication degree d rises. Builds centralized (one-brain plan) and MELD star
(N, d) for the given worlds exactly as scripts/run_nbrains.py does (topology.build), then traces
per gold turn and classifies each centralized-hit / MELD-miss at k:

  a  every MELD node of the gold turn is DEPRECATED into a representative of another turn;
  b  a live node of the gold turn exists and would be in the top k if the read listed each
     source once (copies of already-listed turns took the slots: crowding);
  c  no node of the gold turn is held (REJECT or not admitted);
  d  a live node exists but other turns outrank it even after per-source dedup (score shift).

It also re-reads each built store under two read variants, applied identically to both arms:
  nd   near-duplicate suppression: walk the fused ranking, skip a unit whose dense cosine with an
       already listed unit is >= --tau (0.95, fixed before looking at any result, not tuned);
  mem  member indexing: a live representative scores the max fused score over its group's
       member texts (the deprecated members are indexed too).
Exploratory only: same LongMemEval evaluation worlds as the reported run, never used to tune.

Run: PYTHONPATH=<freeze-r1.2 export>/src OMP_NUM_THREADS=2 python scripts/lme_dup_diagnostic.py \
       --epoch outputs/calibration-labelled/substrate-r4-a0.05-lme/lme-seed-a0.05.epoch.json \
       --key-file outputs/calibration-labelled/key --worlds 0f05491a ... --out trace.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

from meld.agent.recall_metrics import recalled_sources
from meld.brain.store import _fuse
from meld.config import read_epoch
from meld.eval import topology
from meld.eval.lme_supersession import _answer_turns, session_split
from meld.eval.runner_args import runner_key
from meld.proto import canon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ranking(mem, query, members=False):
    """Full fused ranking [(text, score)] over live units (the store's graph_walk_read with
    k = all); ``members``: a representative scores the max over its group's member texts."""
    st, enc = mem.store, mem.encoder
    es = st._es.setdefault(enc, mem._es)
    nodes, live = st.nodes(), set(st.live_nodes())
    rep = st.derived.rep
    idx = list(nodes) if members else list(live)
    sc = _fuse(query, {n: canon(nodes[n][1].text) for n in idx},
               ((s, d, w) for _, s, d, w in st.resolved_links()), es, 0.3, 0.5, 0.2)
    unit = {}
    for n, s in sc.items():
        r = rep.get(n, n)
        if r not in live:
            continue
        u = nodes[r][1].unit_id
        unit[u] = max(unit.get(u, s), s)
    texts = mem.texts()
    return [(texts[u], s) for u, s in sorted(unit.items(), key=lambda x: (-x[1], x[0]))]


def suppress(mem, ranked, k, tau):
    es = mem._es
    out = []
    for t, _ in ranked:
        if all(es.sigma(canon(t), canon(o)) < tau for o in out):
            out.append(t)
        if len(out) == k:
            break
    return out


def srcs(texts, source):
    return list(dict.fromkeys(source.get(t, t) for t in texts))


def trace_world(it, n, d, ks, kw, tau):
    gold = sorted({t for *_, t in _answer_turns(it)})
    q = it["question"]
    out = {"world": it["question_id"], "type": it["question_type"], "n_gold": len(gold), "arms": {}}
    for arm, plan, cond in (("centralized", session_split(it, n_brains=1), "centralized"),
                            ("meld", session_split(it, n_brains=n, dup_degree=d), "meld")):
        b = topology.build(plan, cond, "star", **kw)
        mem, src = b["memory"], plan.source
        full = ranking(mem, q)
        assert [t for t, _ in full[:10]] == mem.recall(q, k=10), "ranking != store read"
        memb = ranking(mem, q, members=True)
        st, nodes = mem.store, mem.store.nodes()
        live, rep = set(st.live_nodes()), st.derived.rep
        dec = {x.b: x.decision for x in mem.admission.decisions("agent")}
        per_gold = {}
        for g in gold:
            ks_ = [kk for kk, (_, dd) in nodes.items() if src.get(dd.text, dd.text) == g]
            per_gold[g[:80]] = {
                "nodes": len(ks_), "live": sum(kk in live for kk in ks_),
                "rep_other": sum(kk not in live and src.get(nodes[rep.get(kk, kk)][1].text,
                                                             nodes[rep.get(kk, kk)][1].text) != g
                                 for kk in ks_),
                "decisions": sorted(dec.get(kk, "none") for kk in ks_),
                "src_rank": next((i + 1 for i, s in enumerate(srcs([t for t, _ in full], src))
                                  if s == g), None)}
        arm_out = {"live": b["live"], "decisions": b["decisions"], "gold": per_gold, "top": {}}
        for k in ks:
            top = [t for t, _ in full[:k]]
            arm_out["top"][k] = {
                "base": srcs(top, src),
                "nd": srcs(suppress(mem, full, k, tau), src),
                "mem": srcs([t for t, _ in memb[:k]], src),
                "slots_distinct": len(srcs(top, src))}
        arm_out["top10_text"] = [[t[:100], round(s, 4), src.get(t, t)[:60]] for t, s in full[:10]]
        out["arms"][arm] = arm_out
        del b, mem
    out["gold"] = [g[:80] for g in gold]
    return out


def hotpot_world(pooled, members, n, d, k, kw, tau, overlap):
    """HotpotQA contrast (recall only, no miss classification): recall@k per read variant and
    the mean number of distinct sources in the top k, per arm, averaged over member questions."""
    from meld.agent.mem_conditions import split_across_brains
    from meld.agent.recall_metrics import _matches, gold_supporting_sentences
    out = {"world": pooled.task_id[:40], "arms": {}}
    for arm, nn, dd, cond in (("centralized", 1, 1, "centralized"), ("meld", n, d, "meld")):
        plan = split_across_brains(pooled, n_brains=nn, dup_degree=dd, overlap_fraction=overlap)
        b = topology.build(plan, cond, "star", **kw)
        mem, src = b["memory"], plan.source
        acc = {"base": [], "nd": [], "mem": [], "distinct": []}
        for t in members:
            gold = gold_supporting_sentences(t)
            if not gold:
                continue
            full = ranking(mem, t.question)
            memb = ranking(mem, t.question, members=True)
            for v, lst in (("base", [x for x, _ in full[:k]]), ("nd", suppress(mem, full, k, tau)),
                           ("mem", [x for x, _ in memb[:k]])):
                got = srcs(lst, src)
                acc[v].append(sum(any(_matches(g, x) for x in got) for g in gold) / len(gold))
            acc["distinct"].append(len(srcs([x for x, _ in full[:k]], src)))
        out["arms"][arm] = {"live": b["live"], **{v: sum(x) / len(x) for v, x in acc.items()}}
        del b, mem
    return out


def classify(row, k):
    """Per gold turn: centralized hit and MELD miss at k -> a/b/c/d; the reverse -> 'gain'."""
    c, m = row["arms"]["centralized"], row["arms"]["meld"]
    out = []
    for g in row["gold"]:
        hc = any(s[:80] == g for s in c["top"][str(k)]["base"])
        hm = any(s[:80] == g for s in m["top"][str(k)]["base"])
        if hc and not hm:
            gi = m["gold"][g]
            out.append("c" if gi["nodes"] == 0 else "a" if gi["live"] == 0 and gi["rep_other"]
                       else "b" if gi["src_rank"] is not None and gi["src_rank"] <= k else "d")
        elif hm and not hc:
            out.append("gain")
    return out


def summarize(trace, archive, n, d, k=5):
    """Miss classes, sanity against the archived rows, and the stratified MELD - centralized
    recall@k per read variant (strata = archived sign of the recall@5 difference at (n, d), the
    variable the sample was drawn on, at every k)."""
    import collections
    import glob
    arch = {}
    for f in glob.glob(os.path.join(archive, "shard-*.jsonl")):
        for line in open(f):
            r = json.loads(line)["result"]
            arch[(r["world"], r["n_brains"], r["dup"], r["condition"])] = r[f"recall@{k}"]
            arch[(r["world"], r["n_brains"], r["dup"], r["condition"], 5)] = r["recall@5"]
    stratum, size = {}, collections.Counter()
    for w in {key[0] for key in arch}:
        c, m = arch.get((w, 1, 1, "centralized", 5)), arch.get((w, n, d, "meld", 5))
        if c is None or m is None:
            continue
        stratum[w] = "loss" if m < c else "gain" if m > c else "tie"
        size[stratum[w]] += 1
    rows = [json.loads(line) for line in open(trace)]
    cls, agree, per = collections.Counter(), 0, collections.defaultdict(lambda: collections.defaultdict(list))
    distinct, mismatch = collections.defaultdict(list), []
    for r in rows:
        cls.update(classify(r, k))
        rec = {}
        for arm in ("centralized", "meld"):
            top = r["arms"][arm]["top"][str(k)]
            distinct[arm].append(top["slots_distinct"])
            for v in ("base", "nd", "mem"):
                rec[arm, v] = sum(any(s[:80] == g for s in top[v]) for g in r["gold"]) / len(r["gold"])
        same = (abs(rec["centralized", "base"] - arch[(r["world"], 1, 1, "centralized")]) < 1e-9
                and abs(rec["meld", "base"] - arch[(r["world"], n, d, "meld")]) < 1e-9)
        agree += same
        if not same:
            mismatch.append(r["world"])
        for v in ("base", "nd", "mem"):
            per[stratum[r["world"]]][v].append(rec["meld", v] - rec["centralized", v])
            per[stratum[r["world"]]]["c_" + v].append(rec["centralized", v] - rec["centralized", "base"])
    tot = sum(size.values())
    est = {v: sum(size[s] / tot * sum(per[s][v]) / len(per[s][v]) for s in size if per[s][v])
           for v in ("base", "nd", "mem", "c_nd", "c_mem")}
    return {"worlds": len(rows), "agree_with_archive": agree, "mismatch": mismatch, "classes": dict(cls),
            "strata_sizes": dict(size), "strata_sampled": {s: len(per[s]["base"]) for s in per},
            "per_stratum_mean": {s: {v: round(sum(x) / len(x), 4) for v, x in per[s].items()} for s in per},
            "stratified_estimate": {v: round(x, 4) for v, x in est.items()},
            "mean_distinct_top": {a: round(sum(x) / len(x), 2) for a, x in distinct.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "data", "bench", "lme-cleaned",
                                                   "longmemeval_s_cleaned.json"))
    ap.add_argument("--epoch", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--workload", choices=("lme", "hotpot"), default="lme")
    ap.add_argument("--worlds", nargs="+", required=True,
                    help="LME question ids, or HotpotQA pooled-world indices (eval-pool order)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summarize", metavar="ARCHIVE_DIR",
                    help="summarize --out against the archived rows in ARCHIVE_DIR and exit")
    a = ap.parse_args(argv)
    if a.summarize:
        print(json.dumps({k: summarize(a.out, a.summarize, a.n, a.d, k) for k in a.ks}, indent=1))
        return
    epoch = read_epoch(a.epoch, runner_key(a.key_file))
    kw = dict(encoder="minilm", nli="mnli-cross-encoder", epoch=epoch, cache_dir=a.cache_dir)
    if a.workload == "hotpot":
        from meld.agent.hotpot_loader import load_hotpot_real
        from meld.agent.mem_conditions import pool_tasks
        from meld.eval.substrates import eval_pool
        pools = pool_tasks(eval_pool(load_hotpot_real()), 20)
        for w in a.worlds:
            row = hotpot_world(*pools[int(w)], a.n, a.d, a.ks[0], kw, a.tau, 0.5)
            row["index"] = int(w)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(w, row["arms"], flush=True)
        return
    items = {x["question_id"]: x for x in json.load(open(a.data, encoding="utf-8"))}
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)["world"] for line in open(a.out)}
    for w in a.worlds:
        if w in done:
            continue
        row = trace_world(items[w], a.n, a.d, a.ks, kw, a.tau)
        row["class@5"], row["class@10"] = classify(json.loads(json.dumps(row)), 5), \
            classify(json.loads(json.dumps(row)), 10)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(w, row["class@5"], flush=True)


if __name__ == "__main__":
    main()
