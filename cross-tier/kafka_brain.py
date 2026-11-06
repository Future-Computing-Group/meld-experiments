#!/usr/bin/env python3
"""MELD brain over Apache Kafka (the standard transport).

Consumes signed objects from a claims topic, runs the shared receive step (brain_core:
the replica's MAC-verified sets and the protocol's admission step), and publishes acks to
an acks topic. Each brain uses a unique consumer group => broadcast (it sees every delta);
routing strategies that restrict delivery sit above this, choosing the topic.

Torch-free with the hashing encoder and the lexical NLI rung; the brain evaluates its own
Sig payloads.
"""
import argparse, json, os, sys, time, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from kafka_transport import KafkaTransport       # noqa: E402
from brain_core import process_delta, replica    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:9092")
    ap.add_argument("--id", required=True)
    ap.add_argument("--claims-topic", required=True)
    ap.add_argument("--acks-topic", required=True)
    ap.add_argument("--encoder", required=True, help="M_nli encoder (deltas under another are dropped)")
    ap.add_argument("--nli", required=True, help="M_nli NLI model the brain evaluates Sig with")
    ap.add_argument("--log", default=None)
    ap.add_argument("--run-seconds", type=float, default=0.0, help="0 = run until killed")
    args = ap.parse_args()

    rep = replica(args.id, encoder=args.encoder, nli=args.nli)
    logf = open(args.log, "a", encoding="utf-8") if args.log else None

    def log(rec):
        if logf:
            logf.write(json.dumps(rec) + "\n"); logf.flush()

    t = KafkaTransport(args.bootstrap)
    consumer = t.consumer(args.claims_topic,
                          group="meld-brain-%s-%s" % (args.id, uuid.uuid4().hex[:6]),
                          from_beginning=True, timeout_ms=500)
    log({"ev": "ready", "brain": args.id, "t": time.time()})
    deadline = (time.time() + args.run_seconds) if args.run_seconds > 0 else None
    try:
        while deadline is None or time.time() < deadline:
            for tp, msgs in consumer.poll(timeout_ms=500).items():
                for m in msgs:
                    t_recv = time.time()
                    try:
                        d = json.loads(m.value)
                    except (ValueError, RecursionError):   # incl. non-UTF-8 bytes
                        d = None
                    ack = process_delta(d, rep, args.id, t_recv)
                    if ack is None:   # unverifiable, malformed or unknown op: dropped
                        log({"ev": "drop", "brain": args.id, "t": t_recv})
                        continue
                    log({"ev": "merge", **ack})
                    t.publish(args.acks_topic, json.dumps(ack))
    finally:
        consumer.close(); t.close()
        if logf:
            logf.close()


if __name__ == "__main__":
    main()
