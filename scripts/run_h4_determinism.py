"""H4 replica determinism (design r3 §7.2 H4, §7.4 freeze gate; planning 07 §6): one fold workload
on several hardware cells (device x batch size x torch threads), compared on the raw float signals,
their quantised forms, and the post-pass certificate fingerprints after a Sig exchange.

Stages (each a subcommand; files are JSON plus one .npy of float embeddings per cell and workload):

  deltas  the reference deltas (the production issuer path, ``order_shuffle.claims``: batch-1 MiniLM,
          shipped int8 q) and the epoch object of each workload, as signed raw objects. Workloads:
          ``gold`` (the 26-seed gold workload, scope "agent", a development epoch at --dev-theta),
          ``paws`` and ``sick`` (the cal-split streams of eval/calibration with their SEED epochs).
  cell    on this hardware: float embeddings of every node text at --batch (and their int8 q), then
          pass 1 = ``fold.readjudicate`` over the reference deltas with this cell's NLI as provider
          (float softmax recorded per direction). With --table (batch > 1) the directions the
          reference cell read are evaluated up front in batches and served from that table; a
          direction outside it falls back to one batch-1 call (counted).
  pass2   the Sig exchange: a fresh replica per cell holds the reference deltas, the epoch and the
          union of every cell's Sig records, and folds again (provider = this cell's models, loaded
          only on a miss). Fingerprint and decisions per cell.
  report  across cells against the reference cell: raw float disagreement (embeddings, NLI) before
          quantisation, int8 and uint16 flips, selection flips if each cell had issued its own q
          (``fold.tested_pairs``), theta flips of the tested score, contradicts-predicate flips,
          pass-1 and pass-2 fingerprint agreement (target 1.000 after the exchange), decision changes
          pass 1 -> pass 2 and group-gate releases (Lemma 7b).

Own-issuance arm (``cell --own``, ``report --own``; no pass 2): each cell issues its OWN deltas from
the same texts, i.e. the reference deltas with q replaced by the cell's int8 of its own embedding and
re-signed (a delta's other fields do not depend on the hardware), and folds them with its own NLI.
The report counts, per workload against the reference cell and before any exchange: int8 q rows and
elements that differ, delta ids that differ, canonical-order swaps among deltas with equal created_at,
admission decisions and tested pairs that differ (by claim key), live units and fingerprints. Workload
``hotpot`` = the R2 overlap x k distributed_merge input (``mem_conditions.split_across_brains``, 3
sovereign brains, dup degree 2, the first --hotpot-n eval-pool validation tasks at each of
--hotpot-overlaps), one scope per (overlap, task) named hotpot-o<overlap>-t<task>, under --hotpot-epoch
(read with --hotpot-epoch-key, default --key-file, re-signed under --key-file like every epoch here).

Every per-workload record of every stage carries topology.stamps (the workload's epoch, policy,
--encoder and --nli, both required); each decision row ends with its epoch bin.

Run (cluster wrappers: csc/roihu/meld-h4-*.sbatch):
  PYTHONPATH=src python scripts/run_h4_determinism.py deltas --key-file K --out R/ref.json
  PYTHONPATH=src python scripts/run_h4_determinism.py cell --ref R/ref.json --key-file K --name x86-b1-t1 \
      --device cpu --batch 1 --threads 1 --out R/cells
  PYTHONPATH=src python scripts/run_h4_determinism.py pass2 --ref R/ref.json --key-file K --cells R/cells \
      --names x86-b1-t1 ... --device cpu --out R/pass2
  PYTHONPATH=src python scripts/run_h4_determinism.py report --ref R/ref.json --key-file K --cells R/cells \
      --pass2 R/pass2 --reference x86-b1-t1 --out R/h4-report.json
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from meld.agent.memory import DEFAULT_POL, DEFAULT_SCOPE
from meld.brain.store import Store
from meld.config import read_epoch
from meld.eval.runner_args import runner_key
from meld.eval import calibration as C
from meld.eval.order_shuffle import gold_workload
from meld.eval.topology import bin_for, stamps
from meld.merge.signals import _NLI_MODEL_IDS, _SentenceTransformerEncoder, quantize_int8, to_fixed16
from meld.proto import CANON_VERSION
from meld.protocol.fold import readjudicate, tested_pairs
from meld.protocol.objects import EpochRecord, canonical_bytes, from_bytes, m_nli, signed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUP_GATES = ("group_contradiction", "group_cannot_link")
Embed = Callable[[List[str], int], np.ndarray]                       # (texts, batch) -> float rows
NLI = Callable[[List[Tuple[str, str]], int], List[Tuple[float, float, float]]]  # (p_c, p_e, p_n)


# -- workloads and replicas ------------------------------------------------------------------------

def pol_of(workload: str) -> Tuple[dict, str]:
    if workload == "gold" or workload.startswith("hotpot"):
        return DEFAULT_POL, "agent"
    return C.policy(workload, DEFAULT_POL["bootstrap_selector"]), workload


def hotpot_deltas(task, overlap: float, encoder: str, nli: str) -> List:
    """The deltas the 3 sovereign brains of run_r2_overlap_k's distributed_merge publish for ``task``
    (the production issuer: SMPMemory.delta, created_at = world time + 1, origin agent<i>)."""
    from meld.agent.mem_conditions import _fed, split_across_brains
    split = split_across_brains(task, overlap_fraction=overlap)
    mems = [_fed(obs, split.when, "publish", encoder, nli, None, None, origin=f"agent{i}")
            for i, obs in enumerate(split.per_brain)]
    return [m.store.objects.get("D", did) for m in mems for did in m.store.objects.ids("D", DEFAULT_SCOPE)]


def build_ref(workloads: Sequence[str], *, encoder: str, nli: str, key: bytes, manifest: Optional[dict],
              epoch_dir: Optional[str], dev_theta: float, hotpot_tasks: Sequence = (),
              hotpot_epoch: Optional[EpochRecord] = None) -> Dict:
    """workload -> {"scope", "objects": [[hex, mac, scope]]}: deltas then the epoch, MAC'd under key.
    ``hotpot`` expands to one workload per (overlap, task index, task) of ``hotpot_tasks``."""
    from run_nbrains import dev_epoch
    jobs = []
    for wl in workloads:
        if wl == "hotpot":
            jobs += [(f"hotpot-o{ov}-t{i:03d}", (ov, t)) for ov, i, t in hotpot_tasks]
        else:
            jobs.append((wl, None))
    out = {}
    for wl, hp in jobs:
        pol, scope = pol_of(wl)
        if hp is not None:
            deltas, epoch = hotpot_deltas(hp[1], hp[0], encoder, nli), hotpot_epoch
        elif wl == "gold":
            deltas, _ = gold_workload(encoder)
            epoch = dev_epoch(encoder, nli, dev_theta)
        else:
            store, *_ = C.stream(C.split_rows(manifest, wl, "cal"), scope=scope, encoder=encoder,
                                 selector=DEFAULT_POL["bootstrap_selector"], key=key)
            deltas = list(store.objects.iter("D", scope))
            epoch = read_epoch(os.path.join(epoch_dir, f"{wl}-seed-a0.05.epoch.json"), key)
        objs = [signed(d, key) for d in deltas] + [signed(epoch, key)]
        out[wl] = {"scope": scope, "objects": [[b.hex(), mac, scope] for b, mac in objs],
                   **stamps(encoder, nli, epoch, pol)}
    return out


def epoch_of(ref: Dict, wl: str) -> EpochRecord:
    """The workload's epoch: the last object of its reference set."""
    h, mac, _ = ref[wl]["objects"][-1]
    return from_bytes(EpochRecord, bytes.fromhex(h), mac)


def _stamps(ref: Dict, wl: str, encoder: str, nli_name: str) -> Dict:
    return stamps(encoder, nli_name, epoch_of(ref, wl), pol_of(wl)[0])


def family(wl: str) -> str:
    return wl.split("-")[0]


def replica(wl: str, ref: Dict, key: bytes, encoder: str, extra: Sequence = ()) -> Store:
    pol, _ = pol_of(wl)
    s = Store(pol, key, encoder=encoder)
    for h, mac, scope in list(ref[wl]["objects"]) + list(extra):
        if s.apply(bytes.fromhex(h), mac, scope) is None:
            raise ValueError(f"{wl}: object refused (wrong key or encoder?)")
    return s


def node_texts(store: Store, scope: str) -> Dict[str, str]:
    """claim key hex -> text, in canonical node order."""
    return {d.claim_key.hex(): d.text for _, d in store.nodes().values()
            if store.objects.scope_of.get(d.cell) == scope}


# -- one cell --------------------------------------------------------------------------------------

def table_provider(nli: NLI, batch: int, table_dirs: Sequence[Tuple[str, str]], record: Dict):
    """A fold provider over a table evaluated up front in batches; misses fall back to batch 1.
    Every float triple it serves lands in ``record`` (premise, hyp) -> triple."""
    if table_dirs:
        record.update(zip(map(tuple, table_dirs), nli(list(map(tuple, table_dirs)), batch)))
    stats = {"batch1_calls": 0}

    def provider(a: str, b: str):
        if (a, b) not in record:
            record[a, b] = nli([(a, b)], 1)[0]
            stats["batch1_calls"] += 1
        return to_fixed16(record[a, b])
    return provider, stats


def decisions_of(res, epoch) -> List[List]:
    """[b, decision, gates, tested, s_q, theta_q, bin] per decision (bin last: older records lack it)."""
    return [[d.b.hex(), d.decision, list(d.gates), [k.hex() for k in d.tested] if d.tested else None,
             d.s_q, d.theta_q, bin_for(d.n_prior, epoch)] for d in res.decisions]


def run_cell(ref: Dict, *, name: str, embed: Embed, nli: NLI, batch: int, key: bytes, encoder: str,
             nli_name: str, table: Optional[Dict] = None, own: bool = False) -> Dict[str, Dict]:
    """workload -> pass-1 record (and "emb": float rows in node order, saved separately). ``own``:
    pass 1 folds the cell's own issuance (``own_replica``) and the record adds the fold's canonical
    order [[claim key, delta id]] and its live units."""
    mn = m_nli(encoder, nli_name, "int8", CANON_VERSION)
    out = {}
    for wl in ref:
        scope = ref[wl]["scope"]
        store = replica(wl, ref, key, encoder)
        texts = node_texts(store, scope)
        t0 = time.time()
        emb = np.asarray(embed(list(texts.values()), batch), dtype=np.float32)
        t_emb = time.time() - t0
        q = [quantize_int8(v.astype(np.float64)).tobytes().hex() for v in emb]
        if own:
            store = own_replica(wl, ref, key, encoder, dict(zip(texts, q)))
        record: Dict = {}
        dirs = (table or {}).get(wl, {}).get("nli_dirs", []) if batch > 1 else []
        provider, stats = table_provider(nli, batch, dirs, record)
        t0 = time.time()
        res = readjudicate(store, scope, m_nli=mn, provider=provider, issuer=name, hw=name)
        sig = [[canonical_bytes(r).hex(), r.mac, r.scope] for r in res.new_sig]
        out[wl] = {"cell": name, "keys": list(texts), "emb": emb, "q": q,
                   "nli_dirs": [list(k) for k in record],
                   "nli_float": [[float(p).hex() for p in v] for v in record.values()],
                   "fingerprint1": res.certificate.fingerprint.hex(),
                   "decisions1": decisions_of(res, epoch_of(ref, wl)),
                   "sig": sig, "nli_calls": res.counters["nli_calls"], **stats,
                   "seconds": {"embed": round(t_emb, 3), "fold": round(time.time() - t0, 3)},
                   **({"own": True, "live": len(store.live_nodes()),
                       "order": [[k.hex(), did.hex()] for k, (did, _) in store.nodes().items()]} if own else {}),
                   **_stamps(ref, wl, encoder, nli_name)}
    return out


def run_pass2(ref: Dict, cells: Dict[str, Dict[str, Dict]], names: Sequence[str], *, key: bytes,
              encoder: str, nli_name: str, provider_for: Callable[[str], Callable]) -> Dict:
    """name -> workload -> {fingerprint2, decisions2, misses}: fresh replica, union of all Sig."""
    mn = m_nli(encoder, nli_name, "int8", CANON_VERSION)
    out = {}
    for name in names:
        out[name] = {}
        for wl in ref:
            union = [s for c in cells.values() for s in c[wl]["sig"]]
            store = replica(wl, ref, key, encoder, union)
            res = readjudicate(store, ref[wl]["scope"], m_nli=mn, provider=provider_for(name),
                               issuer=name, hw=name)
            out[name][wl] = {"fingerprint2": res.certificate.fingerprint.hex(),
                             "decisions2": decisions_of(res, epoch_of(ref, wl)),
                             "misses": res.counters["pass_time_misses"],
                             **_stamps(ref, wl, encoder, nli_name)}
    return out


# -- comparison ------------------------------------------------------------------------------------

def _contradicts(p) -> bool:
    return p[0] > max(p[1], p[2])


def compare_cell(r: Dict, c: Dict, *, sigma_q_sel: Optional[Tuple[List, List]] = None) -> Dict:
    """Cell ``c`` against reference ``r`` (one workload, pass-1 records)."""
    assert r["keys"] == c["keys"], "cells hold different node sets"
    E, F = np.asarray(r["emb"], np.float32), np.asarray(c["emb"], np.float32)
    qr, qc = [np.frombuffer(bytes.fromhex(h), np.int8) for h in r["q"]], \
             [np.frombuffer(bytes.fromhex(h), np.int8) for h in c["q"]]
    q_elem = sum(int((a != b).sum()) for a, b in zip(qr, qc))
    rn = {tuple(k): [float.fromhex(x) for x in v] for k, v in zip(r["nli_dirs"], r["nli_float"])}
    cn = {tuple(k): [float.fromhex(x) for x in v] for k, v in zip(c["nli_dirs"], c["nli_float"])}
    common = sorted(rn.keys() & cn.keys())
    fr = {k: to_fixed16(rn[k]) for k in common}
    fc = {k: to_fixed16(cn[k]) for k in common}
    # theta flips of the tested score: s = min(p_e fwd, p_e back) > theta, both from the same cell
    texts = dict(zip(r["keys"], r["emb_texts"])) if "emb_texts" in r else None
    theta_flips = theta_total = 0
    if texts:
        for b, dec, _g, tested, s_q, theta, *_bin in r["decisions1"]:
            if not tested or theta is None:
                continue
            a_t, b_t = texts[tested[0]], texts[tested[1]]
            pr = [fr.get((b_t, a_t)), fr.get((a_t, b_t))]
            pc = [fc.get((b_t, a_t)), fc.get((a_t, b_t))]
            if None in pr or None in pc:
                continue
            theta_total += 1
            theta_flips += (min(pr[0][1], pr[1][1]) > theta) != (min(pc[0][1], pc[1][1]) > theta)
    out = {"emb_rows": len(E), "emb_rows_bit_identical": int((E == F).all(axis=1).sum()),
           "emb_max_abs_diff": float(np.abs(E - F).max()) if len(E) else 0.0,
           "q_rows_differing": sum(bool((a != b).any()) for a, b in zip(qr, qc)),
           "q_elements_differing": q_elem,
           "nli_dirs_common": len(common),
           "nli_dirs_only_ref": len(rn.keys() - cn.keys()), "nli_dirs_only_cell": len(cn.keys() - rn.keys()),
           "nli_float_bit_identical": sum(rn[k] == cn[k] for k in common),
           "nli_max_abs_diff": max((abs(a - b) for k in common for a, b in zip(rn[k], cn[k])), default=0.0),
           "fixed16_differing": sum(fr[k] != fc[k] for k in common),
           "contradicts_flips": sum(_contradicts(fr[k]) != _contradicts(fc[k]) for k in common),
           "argmax_flips": sum(int(np.argmax(rn[k])) != int(np.argmax(cn[k])) for k in common),
           "theta_tested": theta_total, "theta_flips": theta_flips,
           "fingerprint1_equal": r["fingerprint1"] == c["fingerprint1"],
           "decisions1_differing": sum(a[1] != b[1] for a, b in zip(r["decisions1"], c["decisions1"]))}
    if sigma_q_sel is not None:
        a, b = sigma_q_sel
        out["selection_flips_own_q"] = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
    return out


def gate_releases(d1: List[List], d2: List[List]) -> Dict:
    """Pass 1 -> pass 2 (more Sig): decision changes, merges created, and group-gate releases
    (a group-level gate fired at pass 1 and not at pass 2: the only way Lemma 7b admits a new merge)."""
    by1 = {d[0]: d for d in d1}
    changes = Counter()
    released = new_merges = unexplained = 0
    for d in d2:
        p = by1[d[0]]
        if p[1] != d[1]:
            changes[f"{p[1]}->{d[1]}"] += 1
        rel = bool(set(p[2]) & set(GROUP_GATES)) and not set(d[2]) & set(GROUP_GATES)
        released += rel
        if d[1] == "MERGE" and p[1] != "MERGE":
            new_merges += 1
            unexplained += not rel
    return {"decision_changes": dict(changes), "group_gate_releases": released,
            "new_merges": new_merges, "new_merges_without_release": unexplained}


def own_replica(wl: str, ref: Dict, key: bytes, encoder: str, q: Dict[str, str]) -> Store:
    """The workload's set as a cell that issued its own deltas holds it: every reference delta with
    q = the cell's int8 (claim key hex -> q hex), re-signed; the epoch unchanged."""
    store = replica(wl, ref, key, encoder)
    scope = ref[wl]["scope"]
    s2 = Store(store.pol, key, encoder=encoder)
    for d in store.objects.iter("D", scope):
        s2.apply(*signed(dataclasses.replace(d, q=bytes.fromhex(q[d.claim_key.hex()])), key), scope)
    h, mac, sc = ref[wl]["objects"][-1]
    if s2.apply(bytes.fromhex(h), mac, sc) is None:
        raise ValueError(f"{wl}: epoch refused")
    return s2


def selection_with_own_q(ref: Dict, wl: str, rec: Dict, key: bytes, encoder: str) -> List:
    """Tested pairs (a_e, b) of the fold's selector had the cell issued its own q (same deltas
    otherwise, same epoch): the sigma_q quantisation-boundary effect on selection."""
    s2 = own_replica(wl, ref, key, encoder, dict(zip(rec["keys"], rec["q"])))
    lo, _, m = s2.pol["bootstrap_selector"]
    return [(a.hex(), b.hex()) for a, b, _ in tested_pairs(s2, ref[wl]["scope"], sigma_lo=lo, m=m)]


def report(ref: Dict, cells: Dict[str, Dict[str, Dict]], p2: Dict[str, Dict], reference: str, *,
           key: bytes, encoder: str, nli_name: str) -> Dict:
    out = {}
    for wl in ref:
        r = cells[reference][wl]
        own = {n: selection_with_own_q(ref, wl, c[wl], key, encoder) for n, c in cells.items()}
        texts = node_texts(replica(wl, ref, key, encoder), ref[wl]["scope"])
        per = {}
        for n, c in cells.items():
            rr = {**r, "emb_texts": [texts[k] for k in r["keys"]]}
            per[n] = {**compare_cell(rr, c[wl], sigma_q_sel=(own[reference], own[n])),
                      **({} if n not in p2 else gate_releases(c[wl]["decisions1"], p2[n][wl]["decisions2"])),
                      "pass2_misses": p2.get(n, {}).get(wl, {}).get("misses")}
        f1 = Counter(c[wl]["fingerprint1"] for c in cells.values())
        f2 = Counter(x[wl]["fingerprint2"] for x in p2.values())
        out[wl] = {"cells": per, "n_cells": len(cells),
                   "fingerprint1_distinct": len(f1),
                   "fingerprint1_agreement": f1.most_common(1)[0][1] / len(cells),
                   "fingerprint2_distinct": len(f2),
                   "fingerprint2_agreement": f2.most_common(1)[0][1] / len(p2) if p2 else None,
                   **_stamps(ref, wl, encoder, nli_name)}
    return out


def tie_swaps(order_r: Sequence[str], order_c: Sequence[str], created_at: Dict[str, int]) -> Tuple[int, int]:
    """(pairs of nodes with equal created_at, those whose relative canonical order differs)."""
    pr, pc = {k: i for i, k in enumerate(order_r)}, {k: i for i, k in enumerate(order_c)}
    by: Dict[int, List[str]] = {}
    for k in order_r:
        by.setdefault(created_at[k], []).append(k)
    pairs = swaps = 0
    for g in by.values():
        for i, x in enumerate(g):
            for y in g[i + 1:]:
                pairs += 1
                swaps += (pr[x] < pr[y]) != (pc[x] < pc[y])
    return pairs, swaps


def compare_own(r: Dict, c: Dict, created_at: Dict[str, int]) -> Dict:
    """Own-issuance cell ``c`` against the reference cell ``r`` (one workload, pass 1, no exchange)."""
    assert r["keys"] == c["keys"], "cells hold different node sets"
    qr, qc = [np.frombuffer(bytes.fromhex(h), np.int8) for h in r["q"]], \
             [np.frombuffer(bytes.fromhex(h), np.int8) for h in c["q"]]
    ir, ic = dict(map(tuple, r["order"])), dict(map(tuple, c["order"]))
    pairs, swaps = tie_swaps([k for k, _ in r["order"]], [k for k, _ in c["order"]], created_at)
    dr, dc = {d[0]: d for d in r["decisions1"]}, {d[0]: d for d in c["decisions1"]}
    changes = Counter(f"{dr[k][1]}->{dc[k][1]}" for k in dr if k in dc and dr[k][1] != dc[k][1])
    return {"q_rows_differing": sum(bool((a != b).any()) for a, b in zip(qr, qc)),
            "q_elements_differing": sum(int((a != b).sum()) for a, b in zip(qr, qc)),
            "delta_ids_differing": sum(ir[k] != ic.get(k) for k in ir),
            "tie_pairs": pairs, "tie_swaps": swaps,
            "decisions": len(dr),
            "decisions_differing": sum(dc.get(k, [None, None])[1] != d[1] for k, d in dr.items())
            + len(dc.keys() - dr.keys()),
            "decision_changes": dict(changes),
            "tested_differing": sum(dc.get(k, [None] * 4)[3] != d[3] for k, d in dr.items()),
            "live": c["live"], "live_diff": c["live"] - r["live"],
            "fingerprint_equal": r["fingerprint1"] == c["fingerprint1"]}


OWN_SUMS = ("q_rows_differing", "q_elements_differing", "delta_ids_differing", "tie_pairs", "tie_swaps",
            "decisions", "decisions_differing", "tested_differing", "live_diff")


def report_own(ref: Dict, cells: Dict[str, Dict[str, Dict]], reference: str, *, key: bytes, encoder: str,
               nli_name: str) -> Dict:
    """workload -> {cells: name -> compare_own, fingerprint agreement}; "_summary": family -> cell ->
    sums over the family's workloads and the number of workloads with any difference."""
    out, summ = {}, {}
    for wl in ref:
        created_at = {k.hex(): d.created_at for k, (_, d) in replica(wl, ref, key, encoder).nodes().items()}
        per = {n: compare_own(cells[reference][wl], c[wl], created_at) for n, c in cells.items()}
        f1 = Counter(c[wl]["fingerprint1"] for c in cells.values())
        out[wl] = {"cells": per, "n_cells": len(cells), "fingerprint1_distinct": len(f1),
                   "fingerprint1_agreement": f1.most_common(1)[0][1] / len(cells),
                   **_stamps(ref, wl, encoder, nli_name)}
        for n, x in per.items():
            s = summ.setdefault(family(wl), {}).setdefault(n, {"workloads": 0, **{k: 0 for k in OWN_SUMS},
                                                                "workloads_tie_swapped": 0,
                                                                "workloads_decisions_differing": 0,
                                                                "workloads_live_differing": 0,
                                                                "workloads_fingerprint_differing": 0})
            s["workloads"] += 1
            for k in OWN_SUMS:
                s[k] += x[k]
            s["workloads_tie_swapped"] += x["tie_swaps"] > 0
            s["workloads_decisions_differing"] += x["decisions_differing"] > 0
            s["workloads_live_differing"] += x["live_diff"] != 0
            s["workloads_fingerprint_differing"] += not x["fingerprint_equal"]
    out["_summary"] = {"reference": reference, **summ}
    return out


# -- real models -----------------------------------------------------------------------------------

def real_models(encoder: str, nli_name: str, device: str) -> Tuple[Embed, NLI]:
    """The production calls with an explicit device and batch size: SentenceTransformer.encode with
    normalised embeddings, CrossEncoder.predict with softmax, slots by config.id2label."""
    from sentence_transformers import CrossEncoder, SentenceTransformer
    st = SentenceTransformer(_SentenceTransformerEncoder._MODEL_IDS[encoder], device=device)
    ce = CrossEncoder(_NLI_MODEL_IDS[nli_name], device=device)
    names = {str(v).lower(): int(k) for k, v in ce.config.id2label.items()}
    slots = [names[k] for k in ("contradiction", "entailment", "neutral")]

    def embed(texts, batch):
        return st.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False)

    def nli(pairs, batch):
        P = ce.predict(pairs, batch_size=batch, apply_softmax=True, show_progress_bar=False)
        return [tuple(float(p[i]) for i in slots) for p in np.atleast_2d(P)]
    return embed, nli


def _env(device: str) -> Dict:
    import torch
    env = {"host": platform.node(), "machine": platform.machine(), "torch": torch.__version__,
           "threads": torch.get_num_threads(), "device": device}
    if device == "cuda":
        if not torch.cuda.is_available():
            sys.exit("FATAL: --device cuda but no CUDA device")
        env["gpu"] = torch.cuda.get_device_name(0)
    env["git_sha"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                    text=True).stdout.strip() or os.environ.get("MELD_GIT_SHA", "")
    return env


def _load_cells(d: str, names: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Dict]]:
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        rec = json.load(open(f))
        n = rec["cell"]
        if names and n not in names:
            continue
        for wl, x in rec["workloads"].items():
            x["emb"] = np.load(os.path.join(d, f"{n}.{wl}.npy"))
        out[n] = rec["workloads"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("deltas", "cell", "pass2", "report"))
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--workloads", nargs="+", default=["gold", "paws", "sick"],
                    help="gold paws sick hotpot (hotpot = one workload per overlap and task)")
    ap.add_argument("--manifest", default=os.path.join(REPO, "data", "bench", "manifest.json"))
    ap.add_argument("--epoch-dir", default=os.path.join(REPO, "outputs", "calibration-labelled", "a0.05"))
    ap.add_argument("--dev-theta", type=float, default=0.9)
    ap.add_argument("--ref")
    ap.add_argument("--name")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="torch threads; 0 = all cores")
    ap.add_argument("--table", help="reference cell dir + name as DIR:NAME, for batch > 1")
    ap.add_argument("--cells")
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--pass2", nargs="+", default=[])
    ap.add_argument("--reference")
    ap.add_argument("--own", action="store_true", help="own-issuance arm (cell and report stages)")
    ap.add_argument("--hotpot-epoch", help="HotpotQA r4 a0.05 epoch file (workload hotpot)")
    ap.add_argument("--hotpot-epoch-key", help="key the epoch file is MAC'd under (default --key-file)")
    ap.add_argument("--hotpot-n", type=int, default=200)
    ap.add_argument("--hotpot-overlaps", type=float, nargs="+", default=[0.5, 0.75])
    ap.add_argument("--hotpot-tasks", type=int, nargs="+", help="0-based task indices (default all of --hotpot-n)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    key = runner_key(a.key_file)
    if a.stage == "deltas":
        man = json.load(open(a.manifest)) if set(a.workloads) - {"gold", "hotpot"} else None
        hp, hp_epoch = [], None
        if "hotpot" in a.workloads:
            from meld.agent.hotpot_loader import load_hotpot_real
            from meld.eval.substrates import eval_pool
            if not a.hotpot_epoch:
                sys.exit("workload hotpot needs --hotpot-epoch")
            hp_epoch = read_epoch(a.hotpot_epoch, runner_key(a.hotpot_epoch_key or a.key_file))
            tasks = eval_pool(load_hotpot_real(split="validation", config="distractor"))[:a.hotpot_n]
            hp = [(ov, i, tasks[i]) for ov in a.hotpot_overlaps
                  for i in (a.hotpot_tasks if a.hotpot_tasks is not None else range(len(tasks)))]
        ref = build_ref(a.workloads, encoder=a.encoder, nli=a.nli, key=key, manifest=man,
                        epoch_dir=a.epoch_dir, dev_theta=a.dev_theta, hotpot_tasks=hp, hotpot_epoch=hp_epoch)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(ref, open(a.out, "w"))
        print({wl: len(v["objects"]) for wl, v in ref.items()})
        return ref
    ref = json.load(open(a.ref))
    ref = {wl: v for wl, v in ref.items() if family(wl) in a.workloads}
    if a.stage == "report" and a.own:
        rep = report_own(ref, _load_cells(a.cells), a.reference, key=key, encoder=a.encoder, nli_name=a.nli)
        json.dump(rep, open(a.out, "w"), indent=1)
        print(json.dumps(rep["_summary"], indent=1))
        return rep
    if a.stage == "report":
        cells = _load_cells(a.cells)
        p2 = {}
        for f in a.pass2:
            p2.update(json.load(open(f)))
        rep = report(ref, cells, p2, a.reference, key=key, encoder=a.encoder, nli_name=a.nli)
        json.dump(rep, open(a.out, "w"), indent=1)
        for wl, x in rep.items():
            print(wl, {k: v for k, v in x.items() if k != "cells"})
        return rep
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    env = _env(a.device)
    if a.stage == "cell":
        table = None
        if a.batch > 1:
            if not a.table:
                sys.exit("--batch > 1 needs --table DIR:NAME (the reference cell's read set)")
            d, n = a.table.rsplit(":", 1)
            table = json.load(open(os.path.join(d, f"{n}.json")))["workloads"]
        embed, nli = real_models(a.encoder, a.nli, a.device)
        t0 = time.time()
        res = run_cell(ref, name=a.name, embed=embed, nli=nli, batch=a.batch, key=key, encoder=a.encoder,
                       nli_name=a.nli, table=table, own=a.own)
        os.makedirs(a.out, exist_ok=True)
        for wl, x in res.items():
            np.save(os.path.join(a.out, f"{a.name}.{wl}.npy"), x.pop("emb"))
        json.dump({"cell": a.name, "env": env, "batch": a.batch, "own": a.own,
                   "seconds": round(time.time() - t0, 1), "workloads": res}, open(os.path.join(a.out, f"{a.name}.json"), "w"))
        for wl, x in res.items():
            print(a.name, wl, "fp1", x["fingerprint1"], "nli_calls", x["nli_calls"],
                  "batch1_calls", x["batch1_calls"], x["seconds"])
        return res
    # pass2
    cells = _load_cells(a.cells)
    models: Dict = {}

    def provider_for(name):
        def p(x, y):
            if "m" not in models:
                models["m"] = real_models(a.encoder, a.nli, a.device)[1]
            return to_fixed16(models["m"]([(x, y)], 1)[0])
        return p
    res = run_pass2(ref, cells, a.names, key=key, encoder=a.encoder, nli_name=a.nli,
                    provider_for=provider_for)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    for n, x in res.items():
        print(n, {wl: (v["fingerprint2"], v["misses"]) for wl, v in x.items()})
    return res


if __name__ == "__main__":
    main()
