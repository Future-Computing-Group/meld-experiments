#!/usr/bin/env python3
"""MELD brain for the cross-tier continuum run: merge-on-receive over the relay.

Subscribes to `claims`, runs the *tested* four-way decision procedure
(`meld.merge.classifier.classify`) + the status CRDT (`meld.merge.status_crdt`) on
each delivered delta, and publishes an `acks` record back through the broker so
the laptop driver can measure everything with one clock (no cross-tier skew).

Torch-free by design: the dense signals (sigma, chi) are computed ONCE by the
driver (validated separately by M-2) and travel in the delta; the brain applies
the decision rule + CRDT, which are pure Python. The only third-party dep is
PyYAML, for the calibrated thresholds (sourced from config, never hardcoded).
"""
import argparse, json, os, socket, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from meld.config import load_thresholds                       # noqa: E402
from meld.merge.classifier import classify                    # noqa: E402
from meld.merge.status_crdt import StatusState, effective_status  # noqa: E402

MINILM = os.path.join(os.path.dirname(HERE), "src", "meld", "config", "thresholds-minilm.yaml")


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
    ap.add_argument("--thresholds", default=MINILM)
    ap.add_argument("--log", default=None)
    ap.add_argument("--once", action="store_true",
                    help="exit after one disconnect instead of reconnecting (tests)")
    args = ap.parse_args()

    cfg = load_thresholds(args.thresholds)
    status = {}  # claim_id -> StatusState
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
        except json.JSONDecodeError:
            return
        op = d.get("op", "publish")
        cid = d.get("claim_id", "")
        if op == "publish":
            decision = classify(
                cfg=cfg, kappa=int(d.get("kappa", 0)), sigma=float(d.get("sigma", 0.0)),
                chi=d.get("chi", "neutral"), ctx_ok=bool(d.get("ctx_ok", True)))
            eff = effective_status(status.get(cid, StatusState()))
            ack = {"nonce": d.get("nonce"), "brain": args.id, "op": op, "claim_id": cid,
                   "decision": decision, "eff_status": eff, "t_recv": t_recv}
        elif op in ("overrule", "deprecate"):
            effect = "OVERRULED" if op == "overrule" else "DEPRECATED"
            link_id = d.get("link_id", "%s-%s" % (op, d.get("nonce")))
            st = status.get(cid, StatusState()).add(link_id, effect)
            status[cid] = st  # grow-only set: retained across reconnects -> heal converges
            ack = {"nonce": d.get("nonce"), "brain": args.id, "op": op, "claim_id": cid,
                   "decision": None, "eff_status": effective_status(st), "t_recv": t_recv}
        else:
            return
        log({"ev": "merge", **ack})
        _publish(args.broker_host, args.broker_port, "acks", json.dumps(ack))

    # Reconnect loop: a dropped SUB connection (a partition: the tunnel goes down)
    # must NOT lose the brain's CRDT state. `status` lives outside the loop, so on
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
