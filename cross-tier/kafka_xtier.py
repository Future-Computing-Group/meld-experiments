#!/usr/bin/env python3
"""Local end-to-end over REAL Apache Kafka (the standard transport).

The driver publishes the running-example objects + latency probe deltas to a claims
topic; N brains (kafka_brain) admit each and publish acks; the driver measures. Same assertions as the relay end-to-end (test_xtier), now over Kafka ---
proving the MELD merge protocol runs over the standard fabric, not a stand-in relay.

Run (needs a broker at localhost:9092):
    PYTHONPATH=cross-tier:src python3 cross-tier/kafka_xtier.py
"""
import json, math, os, subprocess, sys, time, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
from kafka_transport import KafkaTransport     # noqa: E402
from driver import ENCODER, NLI, probe_messages, running_messages  # noqa: E402
from meld.config import load_epoch                # noqa: E402

BOOTSTRAP = os.environ.get("SMP_KAFKA_BOOTSTRAP", "localhost:9092")
KBRAIN = os.path.join(HERE, "kafka_brain.py")


def broker_up():
    try:
        from kafka import KafkaProducer
        KafkaProducer(bootstrap_servers=BOOTSTRAP, request_timeout_ms=2000,
                      api_version_auto_timeout_ms=2000).close(timeout=2)
        return True
    except Exception:
        return False


def _pct(vals, p):
    if not vals:
        return 0.0
    vals = sorted(vals)
    return round(vals[max(0, min(len(vals) - 1, int(math.ceil(p / 100.0 * len(vals)) - 1)))], 2)


def run_running_example(brains, claims_topic, acks_topic, epoch=None, timeout=25,
                        encoder=ENCODER, nli=NLI):
    t = KafkaTransport(BOOTSTRAP)
    coll = t.consumer(acks_topic, from_beginning=True, timeout_ms=500)
    msgs = running_messages(epoch, encoder, nli)
    for _, msg in msgs:
        t.publish(claims_topic, msg)
        time.sleep(0.1)
    per = {b: {} for b in brains}
    want, got, end = len(msgs) * len(brains), 0, time.time() + timeout
    while time.time() < end and got < want:
        for tp, msgs in coll.poll(timeout_ms=500).items():
            for m in msgs:
                a = json.loads(m.value); b = a.get("brain")
                if b in per:
                    per[b][a.get("nonce")] = a.get("decision") if a.get("op") == "publish" else a.get("eff_status")
                    got += 1
    coll.close(); t.close()
    out = {"per_brain": {}, "all_converged": True}
    for b in brains:
        d = per[b]
        rec = {"seed": d.get("seed"), "merge": d.get("merge"), "relate": d.get("relate"),
               "conflict": d.get("conflict"), "c2_status": d.get("overrule")}
        out["per_brain"][b] = rec
        if rec["c2_status"] != "OVERRULED":
            out["all_converged"] = False
    return out


def run_latency(brains, claims_topic, acks_topic, k=20, timeout=25, encoder=ENCODER, nli=NLI):
    t = KafkaTransport(BOOTSTRAP)
    coll = t.consumer(acks_topic, from_beginning=True, timeout_ms=500)
    t_pub = {}
    for nonce, msg in probe_messages(k, encoder, nli):
        t_pub[nonce] = time.time()
        t.publish(claims_topic, msg)
        time.sleep(0.01)
    per = {b: {"acked": 0, "rtts": []} for b in brains}
    want, got, end = k * len(brains), 0, time.time() + timeout
    while time.time() < end and got < want:
        for tp, msgs in coll.poll(timeout_ms=500).items():
            for m in msgs:
                a = json.loads(m.value); b = a.get("brain"); nonce = str(a.get("nonce", ""))
                if b in per and nonce.startswith("lat-"):
                    per[b]["acked"] += 1
                    per[b]["rtts"].append((time.time() - t_pub.get(nonce, time.time())) * 1000.0)
                    got += 1
    coll.close(); t.close()
    return {"per_brain": {b: {"acked": per[b]["acked"], "p50_ms": _pct(per[b]["rtts"], 50),
                              "p95_ms": _pct(per[b]["rtts"], 95)} for b in brains}}


def _local_end_to_end(epoch_path=None):
    """Three local brains and the driver over the broker. The MERGE beat needs an epoch:
    ``epoch_path`` or SMP_EPOCH (a signed epoch file for the brains' M_nli under the group key)."""
    if not broker_up():
        print("SKIP: no Kafka broker at %s" % BOOTSTRAP); return 0
    run = uuid.uuid4().hex[:8]
    claims, acks = "meld-claims-%s" % run, "meld-acks-%s" % run
    brains = ["local", "hpc", "edge"]
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src") + ":" + HERE)
    procs = []
    import tempfile
    logs = {}
    for b in brains:
        lg = tempfile.NamedTemporaryFile(prefix="kbrain-%s-" % b, suffix=".jsonl", delete=False); lg.close()
        logs[b] = lg.name
        procs.append(subprocess.Popen(
            [sys.executable, KBRAIN, "--bootstrap", BOOTSTRAP, "--id", b,
             "--claims-topic", claims, "--acks-topic", acks, "--log", lg.name,
             "--encoder", ENCODER, "--nli", NLI, "--run-seconds", "30"], env=env))
    try:
        time.sleep(4)  # let the brain consumers join + start polling
        lat = run_latency(brains, claims, acks, k=10)
        path = epoch_path or os.environ.get("SMP_EPOCH")
        if not path:
            print("FAIL: no epoch (pass one or set SMP_EPOCH); without it the MERGE beat RELATEs")
            return 1
        rex = run_running_example(brains, claims, acks, epoch=load_epoch(path))
        print(json.dumps({"transport": "kafka", "latency": lat, "running_example": rex}, indent=2))
        ok = all(lat["per_brain"][b]["acked"] == 10 for b in brains) \
            and all(rex["per_brain"][b] == {"seed": "INSERT", "merge": "MERGE", "relate": "RELATE",
                                            "conflict": "CONFLICT", "c2_status": "OVERRULED"}
                    for b in brains) and rex["all_converged"]
        print("=== %s ===" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


def _drive(argv):
    """Run the driver against already-running brains (used by the cross-tier
    orchestrator). Topics + brain ids are passed in; brains live on remote tiers."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--brains", required=True)
    ap.add_argument("--claims-topic", required=True)
    ap.add_argument("--acks-topic", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epoch", default=None, help="signed epoch file (config.load_epoch)")
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--nli", default=NLI)
    a = ap.parse_args(argv)
    brains = [b.strip() for b in a.brains.split(",") if b.strip()]
    results = {"transport": "kafka", "encoder": a.encoder, "nli": a.nli,
               "latency": run_latency(brains, a.claims_topic, a.acks_topic, k=a.k,
                                      encoder=a.encoder, nli=a.nli),
               "running_example": run_running_example(
                   brains, a.claims_topic, a.acks_topic,
                   epoch=load_epoch(a.epoch) if a.epoch else None, encoder=a.encoder, nli=a.nli)}
    blob = json.dumps(results, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(blob)
    print(blob)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "drive":
        sys.exit(_drive(sys.argv[2:]))
    sys.exit(_local_end_to_end())
