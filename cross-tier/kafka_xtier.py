#!/usr/bin/env python3
"""Local end-to-end over REAL Apache Kafka (the standard transport).

The driver publishes the verified running-example beats + latency probes to a
claims topic; N brains (kafka_brain) merge-on-receive and publish acks; the driver
measures. Same assertions as the relay end-to-end (test_xtier), now over Kafka ---
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
from driver import BEATS                        # noqa: E402  (reuse the verified beats)

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


def run_running_example(brains, claims_topic, acks_topic, timeout=25):
    t = KafkaTransport(BOOTSTRAP)
    coll = t.consumer(acks_topic, from_beginning=True, timeout_ms=500)
    for beat in BEATS:
        t.publish(claims_topic, json.dumps({"nonce": beat["key"], **beat}))
        time.sleep(0.1)
    per = {b: {} for b in brains}
    want, got, end = len(BEATS) * len(brains), 0, time.time() + timeout
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
        rec = {"merge": d.get("merge"), "relate": d.get("relate"),
               "conflict": d.get("conflict"), "c2_status": d.get("overrule")}
        out["per_brain"][b] = rec
        if rec["c2_status"] != "OVERRULED":
            out["all_converged"] = False
    return out


def run_latency(brains, claims_topic, acks_topic, k=20, timeout=25):
    t = KafkaTransport(BOOTSTRAP)
    coll = t.consumer(acks_topic, from_beginning=True, timeout_ms=500)
    t_pub = {}
    for i in range(k):
        nonce = "lat-%d" % i
        t_pub[nonce] = time.time()
        t.publish(claims_topic, json.dumps(
            {"nonce": nonce, "op": "publish", "claim_id": "probe",
             "kappa": 0, "sigma": 0.0, "chi": "neutral", "ctx_ok": True}))
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


def _local_end_to_end():
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
             "--run-seconds", "30"], env=env))
    try:
        time.sleep(4)  # let the brain consumers join + start polling
        lat = run_latency(brains, claims, acks, k=10)
        rex = run_running_example(brains, claims, acks)
        print(json.dumps({"transport": "kafka", "latency": lat, "running_example": rex}, indent=2))
        ok = all(lat["per_brain"][b]["acked"] == 10 for b in brains) \
            and all(rex["per_brain"][b] == {"merge": "MERGE", "relate": "RELATE",
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
    a = ap.parse_args(argv)
    brains = [b.strip() for b in a.brains.split(",") if b.strip()]
    results = {"transport": "kafka",
               "latency": run_latency(brains, a.claims_topic, a.acks_topic, k=a.k),
               "running_example": run_running_example(brains, a.claims_topic, a.acks_topic)}
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
