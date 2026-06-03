#!/usr/bin/env python3
"""eval-c local-docker smoke: status-CRDT partition-heal reconvergence over the wire.

Drives the docker-compose brain stack: ingest one claim into every brain, then apply
CONFLICTING overrule links to different brains (a partition: A sees link La, C sees Lb,
B sees neither) so the per-claim status DIVERGES; then "heal" by delivering BOTH links to
every brain and confirm the status-CRDT RECONVERGES to one status everywhere (agreement
1.0), order-independently. Exits non-zero on any failed check (run after compose up).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from collections import Counter

URL = {"A": "http://127.0.0.1:8801", "B": "http://127.0.0.1:8802", "C": "http://127.0.0.1:8803"}
CLAIM = "the canonical policy is X"
LINKS = [("A", "La", "DEPRECATED"), ("C", "Lb", "OVERRULED")]   # conflicting, per-side


def _post(url, data=b""):
    with urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST"),
                                timeout=8) as r:
        return json.loads(r.read().decode())


def _status(node):
    with urllib.request.urlopen(URL[node] + "/metrics", timeout=8) as r:
        return json.loads(r.read().decode())["status"].get(CLAIM, "ACTIVE")


def _agreement():
    s = [_status(n) for n in URL]
    return max(Counter(s).values()) / len(s)


def main():
    for d in range(40):
        try:
            if all(urllib.request.urlopen(URL[n] + "/healthz", timeout=3).status == 200 for n in URL):
                break
        except Exception:
            time.sleep(2)
    for n in URL:
        _post(URL[n] + "/ingest", CLAIM.encode())

    # PARTITION: conflicting overrules reach different brains only.
    for node, lid, eff in LINKS:
        _post(URL[node] + "/status", json.dumps({"claim": CLAIM, "link_id": lid, "effect": eff}).encode())
    during = _agreement()
    print(f"  during partition: agreement={during:.2f} ({ {n: _status(n) for n in URL} })")
    if during >= 1.0:
        print("FAIL: expected divergence during the partition", file=sys.stderr); sys.exit(1)

    # HEAL: deliver BOTH links to every brain -> the status-CRDT joins them.
    for n in URL:
        for _node, lid, eff in LINKS:
            _post(URL[n] + "/status", json.dumps({"claim": CLAIM, "link_id": lid, "effect": eff}).encode())
    healed = _agreement()
    print(f"  after heal:       agreement={healed:.2f} ({ {n: _status(n) for n in URL} })")
    if healed != 1.0:
        print("FAIL: status-CRDT did not reconverge after heal", file=sys.stderr); sys.exit(2)
    print("EVAL-C SMOKE OK (CRDT reconverged over the wire)")


if __name__ == "__main__":
    main()
