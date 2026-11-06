#!/usr/bin/env python3
"""Cross-tier experiment driver (runs on the laptop hub).

Publishes signed objects to the brokered `claims` topic and collects per-brain `acks`,
measuring with ONE clock (the laptop's) so there is no cross-tier skew:

  run_latency()          -> per-brain publish->admit->ack RTT (p50/p95) over K probe deltas
  run_running_example()  -> per-brain INSERT/MERGE/RELATE/CONFLICT decisions on real claim
                            deltas + the c2 status after an authored deprecate (SUPERSEDE)
                            and overrule (expect OVERRULED on every tier: the order-
                            independent join of authored effects)

Every brain admits each delta with the protocol's own step (brain_core). The MERGE beat
merges only under an epoch: pass the calibration authority's signed epoch file
(``--epoch``, config.load_epoch); the driver publishes it first. Without one the paraphrase
RELATEs, by design (no default operating point). The brains and the driver must agree on
the encoder and NLI model (M_nli); the defaults are the torch-free hashing + lexical rung.
"""
import argparse, json, math, os, socket, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from brain_core import OTHER_CELL, replica, signed_json   # noqa: E402  (brains drop unsigned objects)
from meld.agent.memory import DEFAULT_CELL                 # noqa: E402
from meld.proto import claim_key                           # noqa: E402
from meld.protocol.objects import AuthoredEffect           # noqa: E402

ENCODER, NLI = "hashing", "lexical"
SEED = "the valve on line two is open"
PARA = "the valve on line two is open now"
NEG = "the valve on line two is not open"
C2_ID = claim_key(NEG, DEFAULT_CELL).hex()     # the ack claim_id of the contradiction "c2"


def _effect(kind, text=NEG):
    c2 = claim_key(text, DEFAULT_CELL)
    return AuthoredEffect(scope="agent", kind=kind, c1=c2,
                          c2=claim_key(text, DEFAULT_CELL, 1) if kind == "SUPERSEDE" else b"",
                          target_cell=DEFAULT_CELL, author="driver", author_cell=DEFAULT_CELL,
                          author_auth=0, created_at=0)


def running_example(encoder=ENCODER, nli=NLI):
    """[(nonce, signed message)]: seed, its paraphrase (merge), the paraphrase in another
    cell (relate), a contradiction (conflict), then deprecate and overrule the contradiction."""
    f = replica("driver", encoder=encoder, nli=nli)
    beats = [("seed", f.delta(SEED)), ("merge", f.delta(PARA)),
             ("relate", f.delta(PARA, OTHER_CELL)), ("conflict", f.delta(NEG)),
             ("deprecate", _effect("SUPERSEDE")), ("overrule", _effect("OVERRULE"))]
    return [(n, signed_json(o, n)) for n, o in beats]


def effect_message(nonce, kind):
    """An authored effect on the contradiction c2. A brain reports its status once it holds
    c2 as a node (``c2_message``): auth is read for nodes only."""
    return signed_json(_effect(kind), nonce)


def c2_message(nonce="c2", encoder=ENCODER, nli=NLI):
    return signed_json(replica("driver", encoder=encoder, nli=nli).delta(NEG), nonce)


def trial_messages(i, encoder=ENCODER, nli=NLI):
    """One partition-heal trial on a fresh claim: its signed delta, then a signed deprecate
    (SUPERSEDE) and overrule on it; ``claim_id`` is the ack claim_id the brains report."""
    text = "the pump on line %d is running" % i
    return {"claim_id": claim_key(text, DEFAULT_CELL).hex(),
            "delta": signed_json(replica("driver", encoder=encoder, nli=nli).delta(text), "t%d" % i),
            "deprecate": signed_json(_effect("SUPERSEDE", text), "d%d" % i),
            "overrule": signed_json(_effect("OVERRULE", text), "o%d" % i)}


def probe_messages(k, encoder=ENCODER, nli=NLI):
    """[(nonce, signed message)] of k distinct probe deltas."""
    f = replica("driver", encoder=encoder, nli=nli)
    return [("lat-%d" % i, signed_json(f.delta("latency probe %d" % i), "lat-%d" % i))
            for i in range(k)]


def running_messages(epoch=None, encoder=ENCODER, nli=NLI):
    """The running example as published: the signed epoch first when one is given."""
    return ([("epoch", epoch_message(epoch))] if epoch else []) + running_example(encoder, nli)


def epoch_message(epoch):
    """The calibration authority's signed epoch record (config.load_epoch) as a message."""
    received, mac = epoch
    return json.dumps({"nonce": "epoch", "o": received.decode("ascii"), "m": mac})


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
    for nonce, msg in probe_messages(k):
        t_pub[nonce] = time.time()
        _publish(host, port, "claims", msg)
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


def run_running_example(host, port, brains, epoch=None, timeout=15):
    coll = _AckCollector(host, port); coll.start(); coll.wait_ready()
    msgs = running_messages(epoch)
    for _, msg in msgs:
        _publish(host, port, "claims", msg)
        time.sleep(0.2)
    acks = _collect_until(coll, len(msgs) * len(brains), timeout)
    per = {b: {} for b in brains}
    for a in acks:
        b = a.get("brain")
        if b not in per:
            continue
        per[b][a.get("nonce")] = a.get("decision") if a.get("op") == "publish" else a.get("eff_status")
    out = {"phase": "running-example", "per_brain": {}, "all_converged": True}
    for b in brains:
        d = per[b]
        rec = {"seed": d.get("seed"), "merge": d.get("merge"), "relate": d.get("relate"),
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
    ap.add_argument("--epoch", default=None, help="signed epoch file (config.load_epoch)")
    args = ap.parse_args(argv)
    brains = [b.strip() for b in args.brains.split(",") if b.strip()]
    epoch = None
    if args.epoch:
        from meld.config import load_epoch
        epoch = load_epoch(args.epoch)
    results = {
        "latency": run_latency(args.broker_host, args.broker_port, brains, k=args.k),
        "running_example": run_running_example(args.broker_host, args.broker_port, brains,
                                               epoch=epoch),
    }
    blob = json.dumps(results, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(blob)
    print(blob)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
