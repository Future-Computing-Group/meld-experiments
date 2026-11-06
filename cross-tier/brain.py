#!/usr/bin/env python3
"""MELD brain for the cross-tier continuum run: one replica over the relay.

Subscribes to `claims`, runs the shared MAC-verified receive step (`brain_core.process_delta`:
insertion into the five replicated sets and the protocol's admission step) on each delivered
object, and publishes an `acks` record back through the broker so the laptop driver can
measure everything with one clock (no cross-tier skew). Torch-free with the hashing encoder
and the lexical NLI rung; the brain evaluates its own Sig payloads.
"""
import argparse, json, os, socket, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from brain_core import process_delta, replica                 # noqa: E402


def _connect(host, port):
    return socket.create_connection((host, int(port)), timeout=30)


def _publish(host, port, topic, payload):
    s = _connect(host, port)
    f = s.makefile("rw")
    f.write("PUB %s %s\n" % (topic, payload)); f.flush()
    f.readline()
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker-host", required=True)
    ap.add_argument("--broker-port", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--encoder", required=True, help="M_nli encoder (deltas under another are dropped)")
    ap.add_argument("--nli", required=True, help="M_nli NLI model the brain evaluates Sig with")
    ap.add_argument("--log", default=None)
    ap.add_argument("--once", action="store_true",
                    help="exit after one disconnect instead of reconnecting (tests)")
    args = ap.parse_args()

    rep = replica(args.id, encoder=args.encoder, nli=args.nli)
    logf = open(args.log, "a", encoding="utf-8") if args.log else None

    def log(rec):
        if logf:
            logf.write(json.dumps(rec) + "\n"); logf.flush()

    def handle(line):
        line = line.rstrip("\n")
        if not line.startswith("MSG claims "):
            return
        t_recv = time.time()
        try:
            d = json.loads(line[len("MSG claims "):])
        except (ValueError, RecursionError):
            d = None
        ack = process_delta(d, rep, args.id, t_recv)
        if ack is None:   # unverifiable, malformed or unknown op: dropped, never acked
            log({"ev": "drop", "brain": args.id, "t": t_recv})
            return
        log({"ev": "merge", **ack})
        _publish(args.broker_host, args.broker_port, "acks", json.dumps(ack))

    # Reconnect loop: a dropped SUB connection (a partition: the tunnel goes down)
    # must NOT lose the brain's replicated state. `rep` lives outside the loop, so on
    # reconnect (heal) the brain re-SUBs, receives the re-delivered links, and the
    # grow-only join converges it -- the real-tier analogue of C-1. Killed externally.
    while True:
        try:
            sub = _connect(args.broker_host, args.broker_port)
            f = sub.makefile("rw")
            f.write("SUB claims\n"); f.flush()
            f.readline()  # OK SUB
            log({"ev": "ready", "brain": args.id, "t": time.time()})
            for line in f:
                handle(line)
            log({"ev": "disconnect", "brain": args.id, "t": time.time()})  # EOF
        except OSError as e:
            log({"ev": "disconnect", "brain": args.id, "err": str(e), "t": time.time()})
        if args.once:
            break
        time.sleep(1.0)  # partition: retry until the tunnel heals


if __name__ == "__main__":
    main()
