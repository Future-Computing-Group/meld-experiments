"""Service throughput knee of one deployed brain (merges/s and publish-to-merge latency vs offered rate).

Two steps, so the driver host needs no models:
  sign   (where the encoder is installed, e.g. the laptop): a donor replica with the run's key and
         the brain's encoder signs one delta per HotpotQA claim; writes one /patch body per line.
  run    (any host that reaches the brain): per offered rate, POST /reset (no peers), pre-fill the
         store with the first --prefill envelopes (default 200, the eval-s/r store), then offer the
         next --n envelopes one per POST /patch at that rate through
         eval/sink_throughput.run_saturation. The store size at the start of every rate is the
         same, so the rate is the only variable. Without --rates the rates are multiples of the
         capacity a sequential probe measures. Rows carry topology.stamps from the brain's own
         /metrics (encoder, NLI, M_nli, epoch ids), the brain host's CPU model and the rate.

Run:  PYTHONPATH=src python scripts/run_service_knee.py sign --encoder minilm --nli mnli-cross-encoder \
          --key-file run.key --out knee-envelopes.jsonl
      PYTHONPATH=src python scripts/run_service_knee.py run --url http://BRAIN --envelopes knee-envelopes.jsonl \
          [--prefill 200] [--n 60] [--rates 1 2 4] --out knee.json
A brain reached through its host's watchdog takes --url http://HOST:8090/brain.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, List, Optional

from meld.eval.sink_throughput import run_saturation, saturation_knee
from meld.eval.topology import stamps

MULTIPLES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def sign_envelopes(texts: List[str], *, encoder: str, nli: str, key: bytes) -> List[bytes]:
    """One packed, signed delta per text from a donor replica under ``key`` (the brain's M_nli)."""
    from meld.agent.memory import DEFAULT_CELL
    from meld.protocol.objects import signed
    from meld.sync.node import BrainNode
    from meld.sync.wire_delta import pack
    donor = BrainNode("knee-donor", encoder=encoder, nli=nli, key=key)
    scope = donor.scope_of[DEFAULT_CELL]
    return [pack([(*signed(donor.mem.delta(t), key), scope)]) for t in texts]


def _check(path: str, status_body) -> None:
    status, body = status_body
    if status != 200:
        raise RuntimeError(f"{path} -> {status} {body[:200]!r}")


def knee(post: Callable, get: Callable, prefill: List[bytes], measure: List[bytes], *,
         rates: Optional[List[float]] = None, n_per_rate: int = 60) -> List[dict]:
    """``post(path, body) -> (status, body)``, ``get(path) -> dict``. Returns one row per rate."""
    def fresh():
        _check("/reset", post("/reset", json.dumps({"peers": []}).encode()))
        for e in prefill:
            _check("/patch", post("/patch", e))
        return get("/metrics")["live_count"]

    if rates is None:                         # sequential capacity probe on the same store size
        fresh()
        k = min(10, len(measure))
        t = time.perf_counter()
        for e in measure[:k]:
            _check("/patch", post("/patch", e))
        cap = k / (time.perf_counter() - t)
        rates = [round(m * cap, 3) for m in MULTIPLES]
    if n_per_rate > len(measure):
        raise ValueError(f"{len(measure)} envelopes for {n_per_rate} per rate")
    m = get("/metrics")
    stamp = {**stamps(m["encoder"], m["nli"], None, None), "m_nli": m["m_nli"],
             "epoch_ids": m["epoch_ids"], "host_cpu": m.get("host_cpu")}
    rows = []
    for r in rates:
        live = fresh()
        res = run_saturation(lambda e: _check("/patch", post("/patch", e)), [r], n_per_rate,
                             claim_source=lambda i: measure[i])
        rows.append({**res[r], "prefill": len(prefill), "live_before": live, **stamp})
    return rows


def _http(url: str):
    import requests
    s = requests.Session()
    post = lambda path, body=b"": (lambda r: (r.status_code, r.content))(      # noqa: E731
        s.post(url.rstrip("/") + path, data=body, timeout=120))
    get = lambda path: s.get(url.rstrip("/") + path, timeout=60).json()        # noqa: E731
    return post, get


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sign")
    s.add_argument("--encoder", required=True)
    s.add_argument("--nli", required=True)
    s.add_argument("--key-file", required=True)
    s.add_argument("--n-claims", type=int, default=400)
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--envelopes", required=True)
    r.add_argument("--prefill", type=int, default=200)
    r.add_argument("--n", type=int, default=60)
    r.add_argument("--rates", type=float, nargs="+")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "sign":
        from meld.eval.runner_args import read_key_file
        from meld.sync.workload import load_hotpot_claims
        texts = list(dict.fromkeys(t for t, _ in load_hotpot_claims(a.n_claims)))
        envs = sign_envelopes(texts, encoder=a.encoder, nli=a.nli, key=read_key_file(a.key_file))
        with open(a.out, "w", encoding="ascii") as fh:
            fh.writelines(e.decode("ascii") + "\n" for e in envs)
        print(f"{len(envs)} envelopes -> {a.out}")
        return envs
    envs = [ln.strip().encode("ascii") for ln in open(a.envelopes, encoding="ascii") if ln.strip()]
    post, get = _http(a.url)
    rows = knee(post, get, envs[:a.prefill], envs[a.prefill:], rates=a.rates, n_per_rate=a.n)
    for row in rows:
        print(f"rate {row['offered_rate']:>8}: {row['merged_per_s']:>7} merges/s  p50 {row['p50_ms']} ms  "
              f"p95 {row['p95_ms']} ms  n {row['n']}", flush=True)
    out = {"rows": rows, "knee_merges_per_s": saturation_knee({r["offered_rate"]: r for r in rows})}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"knee {out['knee_merges_per_s']} merges/s -> {a.out}")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
