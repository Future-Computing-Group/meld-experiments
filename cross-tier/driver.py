#!/usr/bin/env python3
"""Cross-tier experiment driver (runs on the laptop hub).

Publishes claims to the brokered `claims` topic and collects per-brain `acks`,
measuring with ONE clock (the laptop's) so there is no cross-tier skew:

  run_latency()          -> per-brain publish->merge->ack RTT (p50/p95) over K probes
  run_running_example()  -> per-brain MERGE/RELATE/CONFLICT decisions + the c2
                            status after a deprecate+overrule (expect OVERRULED on
                            every tier: the order-independent CRDT join)

Signals in the running-example beats are the values verified against the encoder/
NLI earlier (M-2 validates the dense rung); the driver is therefore torch-free.
A separate partition-heal phase lives in the orchestrator (drop a tier's tunnel).
"""
import argparse, json, math, socket, sys, threading, time

# Verified running-example signals (reproduced from src/smp; see manuscript Fig. 1).
BEATS = [
    {"key": "merge",     "op": "publish",   "claim_id": "c1", "kappa": 0, "sigma": 0.93, "chi": "entails",     "ctx_ok": True},
    {"key": "relate",    "op": "publish",   "claim_id": "c1", "kappa": 0, "sigma": 0.93, "chi": "entails",     "ctx_ok": False},
    {"key": "conflict",  "op": "publish",   "claim_id": "c2", "kappa": 0, "sigma": 0.86, "chi": "contradicts", "ctx_ok": True},
    {"key": "deprecate", "op": "deprecate", "claim_id": "c2", "link_id": "L_dep"},
    {"key": "overrule",  "op": "overrule",  "claim_id": "c2", "link_id": "L_ovr"},
]


def _connect(host, port):
    return socket.create_connection((host, int(port)), timeout=30)


def _publish(host, port, topic, payload):
    s = _connect(host, port)
    f = s.makefile("rw")
    f.write("PUB %s %s\n" % (topic, payload)); f.flush()
    f.readline()
    s.close()


class _AckCollector(threading.Thread):
    def __init__(self, host, port):
        super().__init__(daemon=True)
        self.host, self.port = host, port
        self.acks, self.lock, self._ready = [], threading.Lock(), threading.Event()

    def run(self):
        s = _connect(self.host, self.port)
        f = s.makefile("rw")
        f.write("SUB acks\n"); f.flush()
        f.readline()
        self._ready.set()
        for line in f:
            if line.startswith("MSG acks "):
                t = time.time()
                try:
                    rec = json.loads(line[len("MSG acks "):])
                except json.JSONDecodeError:
                    continue
                rec["t_ack"] = t
                with self.lock:
                    self.acks.append(rec)

    def wait_ready(self, t=5):
        self._ready.wait(t)

    def snapshot(self):
        with self.lock:
            return list(self.acks)


def _collect_until(coll, want, timeout):
    end = time.time() + timeout
    while time.time() < end and len(coll.snapshot()) < want:
        time.sleep(0.05)
    return coll.snapshot()


def _pct(vals, p):
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = max(0, min(len(vals) - 1, int(math.ceil(p / 100.0 * len(vals)) - 1)))
    return round(vals[idx], 2)


def run_latency(host, port, brains, k=50, timeout=20):
    coll = _AckCollector(host, port); coll.start(); coll.wait_ready()
    t_pub = {}
    for i in range(k):
        nonce = "lat-%d" % i
        t_pub[nonce] = time.time()
        _publish(host, port, "claims", json.dumps(
            {"nonce": nonce, "op": "publish", "claim_id": "probe",
             "kappa": 0, "sigma": 0.0, "chi": "neutral", "ctx_ok": True}))
        time.sleep(0.02)
    acks = _collect_until(coll, k * len(brains), timeout)
    per = {b: {"acked": 0, "rtts": []} for b in brains}
    for a in acks:
        nonce, b = str(a.get("nonce", "")), a.get("brain")
        if nonce.startswith("lat-") and b in per:
            per[b]["acked"] += 1
            per[b]["rtts"].append((a["t_ack"] - t_pub.get(nonce, a["t_ack"])) * 1000.0)
    out = {"phase": "latency", "k": k, "per_brain": {}}
    for b in brains:
        r = per[b]["rtts"]
        out["per_brain"][b] = {"acked": per[b]["acked"], "p50_ms": _pct(r, 50), "p95_ms": _pct(r, 95)}
    return out


def run_running_example(host, port, brains, timeout=15):
    coll = _AckCollector(host, port); coll.start(); coll.wait_ready()
    for beat in BEATS:
        _publish(host, port, "claims", json.dumps({"nonce": beat["key"], **beat}))
        time.sleep(0.2)
    acks = _collect_until(coll, len(BEATS) * len(brains), timeout)
    per = {b: {} for b in brains}
    for a in acks:
        b = a.get("brain")
        if b not in per:
            continue
        per[b][a.get("nonce")] = a.get("decision") if a.get("op") == "publish" else a.get("eff_status")
    out = {"phase": "running-example", "per_brain": {}, "all_converged": True}
    for b in brains:
        d = per[b]
        rec = {"merge": d.get("merge"), "relate": d.get("relate"),
               "conflict": d.get("conflict"), "c2_status": d.get("overrule")}
        out["per_brain"][b] = rec
        if rec["c2_status"] != "OVERRULED":
            out["all_converged"] = False
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker-host", required=True)
    ap.add_argument("--broker-port", required=True)
    ap.add_argument("--brains", required=True, help="comma-separated brain ids")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    brains = [b.strip() for b in args.brains.split(",") if b.strip()]
    results = {
        "latency": run_latency(args.broker_host, args.broker_port, brains, k=args.k),
        "running_example": run_running_example(args.broker_host, args.broker_port, brains),
    }
    blob = json.dumps(results, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(blob)
    print(blob)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
