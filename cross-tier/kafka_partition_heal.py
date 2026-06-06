#!/usr/bin/env python3
"""M3a: seeded multi-trial cross-tier partition-heal over REAL Apache Kafka.

Upgrades the single-trial existence proof (kafka_xtier) to a quantitative distributed
result: a reconvergence-success fraction and a p50/p95 heal-latency distribution over N
seeded trials. Each trial drives three real kafka_brain subprocesses and models the
partition the way the main paper does --- as the absence of delivery to one brain ---
by freezing the edge brain's consumer with SIGSTOP, publishing the overrule (the peers
diverge), then SIGCONT: the frozen brain re-consumes the missed offset and the grow-only
status set converges, order-independently and without a coordinator.

  * aggregate_trials(trials): pure aggregation of per-trial records (unit-tested).
  * run_one_trial(seed): one partition-heal trial -> {converged, heal_ms, diverged_first}.
  * run_partition_heal(n_trials, seed0): loop + aggregate.

Run (needs a broker at $SMP_KAFKA_BOOTSTRAP, default localhost:9092):
    PYTHONPATH=src:cross-tier python3 cross-tier/kafka_partition_heal.py --trials 10
"""
import json, math, os, random, signal, subprocess, sys, tempfile, time, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

BOOTSTRAP = os.environ.get("SMP_KAFKA_BOOTSTRAP", "localhost:9092")
KBRAIN = os.path.join(HERE, "kafka_brain.py")
EDGE = "edge"
PEERS = ("local", "hpc")


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


def aggregate_trials(trials):
    """Pure: reconvergence-success fraction + heal-latency p50/p95/mean over the
    CONVERGED trials. all_diverged_first guards that the partition was actually
    exercised on every trial (else the run is not a real partition-heal measurement)."""
    n = len(trials)
    converged = [t for t in trials if t.get("converged")]
    heals = [float(t["heal_ms"]) for t in converged]
    return {
        "n_trials": n,
        "n_converged": len(converged),
        "reconvergence_success_fraction": round(len(converged) / n, 4) if n else 0.0,
        "heal_ms_p50": _pct(heals, 50),
        "heal_ms_p95": _pct(heals, 95),
        "heal_ms_mean": round(sum(heals) / len(heals), 2) if heals else 0.0,
        "all_diverged_first": all(t.get("diverged_first") for t in trials) if n else False,
    }


def _last_c2_status(log_path):
    """The brain's latest effective status for claim c2 (parsed from its merge log)."""
    eff = None
    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("ev") == "merge" and r.get("claim_id") == "c2":
                    eff = r.get("eff_status")
    except FileNotFoundError:
        pass
    return eff


def _wait_status(log_path, want, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if _last_c2_status(log_path) == want:
            return True
        time.sleep(0.1)
    return False


def run_one_trial(seed, bootstrap=None, join_s=4.0, timeout=12.0):
    """One seeded partition-heal trial over real Kafka. Returns a per-trial record."""
    from kafka_transport import KafkaTransport
    bootstrap = bootstrap or BOOTSTRAP
    rng = random.Random(seed)
    run = uuid.uuid4().hex[:8]
    claims, acks = "smp-ph-claims-%s" % run, "smp-ph-acks-%s" % run
    brains = [EDGE] + list(PEERS)
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src") + ":" + HERE)
    logs, procs = {}, {}
    for b in brains:
        lg = tempfile.NamedTemporaryFile(prefix="ph-%s-" % b, suffix=".jsonl", delete=False)
        lg.close()
        logs[b] = lg.name
        procs[b] = subprocess.Popen(
            [sys.executable, KBRAIN, "--bootstrap", bootstrap, "--id", b,
             "--claims-topic", claims, "--acks-topic", acks, "--log", lg.name,
             "--run-seconds", "60"], env=env)

    def _resume_all():
        for p in procs.values():
            try:
                p.send_signal(signal.SIGCONT)
            except Exception:
                pass

    t = None
    try:
        time.sleep(join_s)  # let every consumer join the group + start polling
        t = KafkaTransport(bootstrap)
        # [1] baseline: deprecate c2; all three converge to DEPRECATED
        t.publish(claims, json.dumps({"nonce": "d0", "op": "deprecate",
                                      "claim_id": "c2", "link_id": "L_dep"}))
        for b in brains:
            _wait_status(logs[b], "DEPRECATED", timeout)

        # [2] partition: freeze the edge brain's consumer (no delivery to edge)
        procs[EDGE].send_signal(signal.SIGSTOP)
        time.sleep(0.2 + rng.uniform(0.0, 0.3))  # seeded jitter, reproducible

        # [3] overrule during the partition: peers apply it, edge is cut off
        t.publish(claims, json.dumps({"nonce": "o1", "op": "overrule",
                                      "claim_id": "c2", "link_id": "L_ovr"}))
        peers_ovr = all(_wait_status(logs[b], "OVERRULED", timeout) for b in PEERS)
        edge_status = _last_c2_status(logs[EDGE])
        diverged_first = bool(peers_ovr and edge_status == "DEPRECATED")

        # [4] heal: SIGCONT -> the frozen consumer re-reads the missed offset
        t_heal = time.time()
        procs[EDGE].send_signal(signal.SIGCONT)
        converged = _wait_status(logs[EDGE], "OVERRULED", timeout)
        heal_ms = round((time.time() - t_heal) * 1000.0, 2) if converged else 0.0
        return {"seed": seed, "converged": converged, "heal_ms": heal_ms,
                "diverged_first": diverged_first}
    finally:
        _resume_all()  # never leave a stopped process behind
        if t is not None:
            t.close()
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        for path in logs.values():
            try:
                os.unlink(path)
            except OSError:
                pass


def run_partition_heal(n_trials, seed0=1, bootstrap=None, progress=None):
    trials = []
    for i in range(n_trials):
        rec = run_one_trial(seed0 + i, bootstrap=bootstrap)
        trials.append(rec)
        if progress:
            progress("trial %d/%d seed=%d converged=%s diverged_first=%s heal_ms=%.1f"
                     % (i + 1, n_trials, rec["seed"], rec["converged"],
                        rec["diverged_first"], rec["heal_ms"]))
    agg = aggregate_trials(trials)
    agg["trials"] = trials
    agg["bootstrap"] = bootstrap or BOOTSTRAP
    return agg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--bootstrap", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not broker_up():
        print("SKIP: no Kafka broker at %s" % (a.bootstrap or BOOTSTRAP)); sys.exit(0)
    res = run_partition_heal(a.trials, seed0=a.seed0, bootstrap=a.bootstrap,
                             progress=lambda m: print("[m3a]", m, file=sys.stderr, flush=True))
    blob = json.dumps(res, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(blob)
    print(blob)
