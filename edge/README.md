# Edge testbed orchestrator — sync latency / routing / partition-heal

Generic, site-agnostic orchestration for the distributed protocol experiments
(eval-s sync latency, eval-r routing, eval-c partition-heal convergence) on a
multi-host edge testbed. All host aliases, IPs, SSH targets, ports, image tags,
and the netem RTT / partition grid live in a gitignored `testbed.yaml`; this tree
ships only the generic logic and a sanitized `testbed.example.yaml`.

## Run

```bash
cd edge
cp testbed.example.yaml testbed.yaml     # testbed.yaml is gitignored
$EDITOR testbed.yaml                      # fill in your nodes, ports, grid

# build + deploy the brain image to every node, then run the campaigns
./deploy.sh
python run_campaign.py                    # eval-s latency + eval-r routing grid
python run_eval_c.py                      # partition -> heal -> CRDT agreement
```

`config.py` reads the topology and grid from `testbed.yaml` (override with
`SMP_TESTBED=/path/to/your.yaml`); the orchestrators run locally on the node
marked `orchestrator` and reach every other node by SSH. Containers are built
from `edge/docker/` (a hashing-encoder `:smoke` image and a MiniLM `:minilm`
campaign image).

## Prerequisites

- A multi-host testbed (bare-metal, VMs, or containers) reachable by SSH, each
  with Docker and `tc`/`netem` (for the emulated RTT and the healable partition).
- `pip install pyyaml` on the orchestrator host; the brain images carry their own
  Python deps (see `edge/docker/`).
