"""PH-18: per-delta admission cost (ms) on the local CPU (edge-CPU proxy), release code, deployed models, batch size 1.

One SMPMemory (all-MiniLM-L6-v2 with the int8 wire embedding, cross-encoder/nli-deberta-v3-small, s = mutual
entailment min(p_e(x->y), p_e(y->x)) as score_s computes it) holds the HotpotQA round-4 substrate epoch (dev-signed
copy; content id checked against the deployed id). The texts of one evaluation-pool HotpotQA world as the N-brains
sink receives them (per_store 20, N 8, d 4, in world-time order) are written one at a time (``SMPMemory.write``:
encode, sign, insert, admission step). Per delta the wall time is split by timers wrapped around the encoder call
(SMPMemory._es._embed), the exact top-m (classifier.topm_exact) and the NLI provider; the rest (signing, insertion,
Sig publication, bookkeeping) is ``other``. Exact copies (no admission) and the first --warmup deltas are not timed.

PH-18b: --device (cpu, cuda, mps) places BOTH models on that device explicitly (the protocol code leaves the encoder
on SMP_ST_DEVICE, default cpu, and the cross-encoder on sentence-transformers' automatic choice, which is mps on an
Apple laptop and cuda on a GPU node: the laptop run at 9572922 had its NLI on mps). On cuda every timer
read is preceded by torch.cuda.synchronize(), and the GPU name and torch/CUDA versions are stamped. --reference-rows
compares this run's decision per delta (by write index) with another run's rows file; the count of differing
decisions is reported, never assumed zero. --max-texts stops after that many writes (tests). --epoch/--key-file
default to the dev-signed copy and dev key; on the cluster pass the group-signed epoch and the group key (the
content id is checked either way).

Run (repo root):  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/python -I scripts/run_wp6e_admission_cost.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from meld.agent.hotpot_loader import load_hotpot_real  # noqa: E402
from meld.agent.memory import DEFAULT_POL, SMPMemory  # noqa: E402
from meld.config import read_epoch  # noqa: E402
from meld.eval import substrates as S  # noqa: E402
from meld.eval import wp6e as W  # noqa: E402
from meld.eval.topology import stamps  # noqa: E402
from meld.merge import classifier  # noqa: E402
from meld.protocol.fold import nli_provider  # noqa: E402
from meld.protocol.objects import canonical_bytes, object_id  # noqa: E402

ENC, NLI = "minilm", "mnli-cross-encoder"
EPOCH = ("outputs/calibration-labelled/substrate-r4-a0.05-hotpot/hotpot-seed-a0.05.epoch.json",
         "59b847dab1ffa1000f665c6425e6d66a")
PJ = lambda *p: os.path.join(ROOT, *p)  # noqa: E731


def compare_decisions(rows, ref_rows) -> dict:
    """Decision per write index of ``rows`` against ``ref_rows``: compared, differing, unmatched, changes."""
    a, b = {r["i"]: r["decision"] for r in rows}, {r["i"]: r["decision"] for r in ref_rows}
    both = sorted(a.keys() & b.keys())
    changes: dict = {}
    for i in both:
        if a[i] != b[i]:
            changes[f"{b[i]}->{a[i]}"] = changes.get(f"{b[i]}->{a[i]}", 0) + 1
    return {"compared": len(both), "differing": sum(changes.values()), "only_this_run": len(a.keys() - b.keys()),
            "only_reference": len(b.keys() - a.keys()), "changes": changes}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-brains", type=int, default=8)
    ap.add_argument("--dup", type=int, default=4)
    ap.add_argument("--world", type=int, default=0, help="index of the eval-pool world")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out-dir", default=PJ("outputs/wp6e/ph18"))
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    ap.add_argument("--epoch", default=PJ(EPOCH[0]))
    ap.add_argument("--key-file", default=PJ("outputs/calibration-labelled/key"))
    ap.add_argument("--max-texts", type=int, help="stop after this many writes (default all)")
    ap.add_argument("--reference-rows", help="rows .jsonl of another run: decisions compared per write index")
    a = ap.parse_args(argv)
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("FATAL: --device cuda but torch.cuda.is_available() is False")
    sync = torch.cuda.synchronize if a.device == "cuda" else (lambda: None)

    def tick():
        sync()
        return time.perf_counter()
    key = open(a.key_file, "rb").read().strip()
    ep = read_epoch(a.epoch, key)
    assert object_id(canonical_bytes(ep)).hex() == EPOCH[1], "not the deployed HotpotQA r4 epoch"
    worlds = S.hotpot_worlds(S.eval_pool(load_hotpot_real()), n_brains=a.n_brains, dup_degree=a.dup, per_store=20)
    for _ in range(a.world):
        next(worlds)
    wid, plan, _, _ = next(worlds)
    placed = [(plan.when.get(t, 0), i, j, t) for i, b in enumerate(plan.per_brain) for j, t in enumerate(b)]
    texts = [t for *_, t in sorted(placed)][:a.max_texts]

    acc = {"encode": 0.0, "topm": 0.0, "nli": 0.0, "calls": 0}
    base = nli_provider(NLI, own=True)

    # the lazy loads of signals._CrossEncoderNLI / _SentenceTransformerEncoder, with the device made explicit
    base.model._model = CrossEncoder(base.model._model_id, device=a.device)

    def prov(p, h):
        t0 = tick()
        r = base(p, h)
        acc["nli"] += tick() - t0
        acc["calls"] += 1
        return r
    mem = SMPMemory(encoder=ENC, nli=NLI, epoch=ep, pol=DEFAULT_POL, provider=prov)
    enc = mem._es.encoder
    enc._model = SentenceTransformer(enc._MODEL_IDS[enc.rung], device=a.device)
    emb, topm = mem._es._embed, classifier.topm_exact

    def t_emb(x):
        t0 = tick()
        r = emb(x)
        acc["encode"] += tick() - t0
        return r

    def t_topm(*x, **k):
        t0 = tick()
        r = topm(*x, **k)
        acc["topm"] += tick() - t0
        return r
    mem._es._embed, classifier.topm_exact = t_emb, t_topm     # timers only; the functions are called unchanged
    rows, refolds0 = [], None
    for i, t in enumerate(texts):
        for k in ("encode", "topm", "nli", "calls"):
            acc[k] = 0
        t0 = tick()
        decs = mem.write(t)
        total = tick() - t0
        if i == a.warmup:
            refolds0 = mem.admission.counters["refolds"]
        if i < a.warmup or not decs:
            continue
        d = decs[0]
        ms = {k: acc[k] * 1000 for k in ("encode", "topm", "nli")}
        rows.append({"i": i, "n_prior": d.n_prior, "decision": d.decision, "nli_calls": acc["calls"],
                     "total_ms": total * 1000, **{f"{k}_ms": v for k, v in ms.items()},
                     "other_ms": total * 1000 - sum(ms.values())})
    refolds = mem.admission.counters["refolds"] - (refolds0 or 0)
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() \
        if sys.platform == "darwin" else platform.processor()
    summ = {k: W.summary_ms([r[k] for r in rows]) for k in ("total_ms", "encode_ms", "topm_ms", "nli_ms", "other_ms")}
    big = [r for r in rows if r["n_prior"] >= 500]
    out = {**stamps(ENC, NLI, ep, DEFAULT_POL), "epoch_note": "dev-signed copy, content id = deployed " + EPOCH[1],
           "cpu": cpu, "machine": platform.machine(), "torch_threads": torch.get_num_threads(),
           "omp": os.environ.get("OMP_NUM_THREADS"), "torch": torch.__version__, "device": a.device, "batch_size": 1,
           "gpu": torch.cuda.get_device_name(0) if a.device == "cuda" else None, "cuda": torch.version.cuda,
           "cuda_synchronized": a.device == "cuda", "encoder_device": str(enc._model.device),
           "nli_device": str(base.model._model.model.device), "max_texts": a.max_texts, "predicate": "mutual entailment, s = min(p_e(x->y), p_e(y->x))",
           "world": wid, "n_brains": a.n_brains, "dup_degree": a.dup, "texts": len(texts), "timed": len(rows),
           "warmup": a.warmup, "refolds_in_timed_window": refolds, "m": ep.m,
           "summary": summ, "summary_n_prior_ge_500": {k: W.summary_ms([r[k] for r in big]) for k in summ} if big else None,
           "nli_calls_per_delta": W.summary_ms([r["nli_calls"] for r in rows]),
           "share_of_mean_total": {k: summ[k]["mean"] / summ["total_ms"]["mean"] for k in ("encode_ms", "topm_ms", "nli_ms", "other_ms")},
           "decisions": {x: sum(r["decision"] == x for r in rows) for x in sorted({r["decision"] for r in rows})},
           "max_n_prior": max(r["n_prior"] for r in rows)}
    if a.reference_rows:
        with open(a.reference_rows) as fh:
            out["decisions_vs_reference"] = {"path": a.reference_rows,
                                             **compare_decisions(rows, [json.loads(x) for x in fh])}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "admission-cost.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(a.out_dir, "admission-cost-rows.jsonl"), "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps({k: out[k] for k in ("cpu", "timed", "refolds_in_timed_window", "summary", "nli_calls_per_delta",
                                          "share_of_mean_total", "decisions", "device", "gpu", "encoder_device",
                                          "nli_device") + (("decisions_vs_reference",) if a.reference_rows else ())},
                     indent=1))
    return out


if __name__ == "__main__":
    main()
