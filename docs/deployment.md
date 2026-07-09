# Deployment Guide

This guide describes how to deploy `scpn-control` in various environments,
from local Docker containers to HPC clusters and real-time control systems.

## 1. Docker (Python-only, CPU)

For standard simulations and development without specialized hardware.

**Dockerfile Example:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir .
RUN pip install --no-cache-dir matplotlib scipy

CMD ["python", "examples/digital_twin_demo.py"]
```

## 2. Docker GPU (CUDA + JAX)

For high-performance ensemble runs and gradient-based optimization.

**Dockerfile Example:**
```dockerfile
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install --no-cache-dir .

ENV JAX_PLATFORM_NAME=gpu
CMD ["python", "validation/benchmark_transport.py"]
```

## 3. HPC / SLURM

Template for running large-scale parameter scans on a cluster.

**batch_job.slurm:**
```bash
#!/bin/bash
#SBATCH --job-name=scpn_scan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

module load python/3.11
source venv/bin/activate

# Parallel scan using jax.vmap or multiprocessing
python tools/run_parameter_scan.py --config configs/iter_scan.json
```

## 4. Native execution mode (Rust + PyO3)

Use the native bridge when timing or formal-runtime ownership matters. Python
loads configuration and campaign parameters; the fused Rust/PyO3 data plane owns
the hot loop, transport publisher, formal-mode telemetry, and optional AOT
certificate monitor.

Build the editable extension from the PyO3 crate:

```bash
python -m pip install maturin
cd scpn-control-rs/crates/control-python
maturin develop --release
```

Then compare Python and native execution at the same campaign boundary:

```bash
cd ../../..
PYTHONPATH=src python scripts/benchmark_native_handoff.py \
  --steps 5000 \
  --tick-interval-s 0.0001 \
  --transport-backend std \
  --json-out validation/reports/native_handoff_comparison.json \
  --markdown-out validation/reports/native_handoff_comparison.md
```

For production timing claims, run on isolated cores with recorded host load, CPU
governor/frequency context, runtime versions, and concurrent-job status.
Workstation reports without that metadata are local-regression evidence only.

## 5. WebSocket Monitoring

Deploy the real-time monitoring dashboard for live telemetry.

1.  **Start Stream Server**:
    ```bash
    python -m scpn_control.phase.ws_phase_stream --host 127.0.0.1 --port 8765 --api-key "$SCPN_PHASE_WS_API_KEY"
    ```
2.  **Connect Dashboard**:
    Open `docs/dashboard.html` in a browser and point to `ws://localhost:8765`.
3.  **Remote Exposure Requirements**:
    The stream requires `SCPN_PHASE_WS_API_KEY` or `--api-key` unless an
    operator deliberately starts a local development server with
    `--allow-unauthenticated-clients`.  Clients must send that key with
    `Authorization: Bearer <key>` or `X-SCPN-API-Key`; query-string token
    authentication is disabled unless `--allow-query-token-auth` is explicitly
    supplied.  Use `--tls-cert`, `--tls-key`, and `--require-tls` for `wss://`
    transport when the stream crosses a host boundary.  Plaintext non-loopback
    binds are rejected unless an isolated lab deployment explicitly supplies
    `--allow-insecure-remote`.  Command frames are capped by
    `--max-payload-bytes` (`65536` bytes by default) and accepted commands are
    rate-limited per connection.  Browser clients with an `Origin` header are
    rejected unless that origin is explicitly listed with `--allowed-origin`.
    Deployments can reduce command authority with repeated `--allowed-action`
    options, for example allowing `set_psi` while denying `reset` and `stop`.

## 6. Integration with IMAS

For ITER data exchange using the Integrated Modelling and Analysis Suite.

```python
from scpn_control.core.imas_adapter import ImasClient

# Load IDS (Interface Data Structure)
client = ImasClient(shot=12345, run=1)
equilibrium = client.get_ids("equilibrium")

# Map to scpn-control config
cfg = equilibrium.to_scpn_config()
```
Requires `imas` Python bindings to be installed in the environment.

## Deployment boundary and approval route

This guide is a technical deployment map, not a facility certification manual.

Use this sequence for a safe rollout:

1. Start with containerized or local simulations to validate command wiring.
2. Move to native execution only for timing-sensitive loops.
3. Record host/core isolation metadata for any hard real-time claim.
4. Keep proof, benchmark, and runtime evidence in matching validators before
   proposing production timing claims.

The same configuration can be technically correct but claim-insufficient for
facility integration if the environment metadata and admission artifacts are
absent.

## Deployment by stakeholder role

This section helps teams decide which deployment path to run first for their
decision context.

### Research teams and model iteration

Use containerized or local Python-only runs for:

- fast algorithm comparison,
- configuration debugging,
- dataset-driven sensitivity work,
- pre-merger evidence generation inside `validation/`.

For this path, the goal is reproducibility and bounded local evidence, not
plant timing.

### Timing-sensitive control experiments

Use the native Rust/PyO3 path when the team needs:

- consistent microsecond-level compute ownership,
- explicit hot-path boundaries (`PyO3` compiled extension),
- reproducible formal-verification mode switching (`async_drop`, `sync_stride`,
  `aot_certificate`),
- deterministic packet publication behavior.

Use the native benchmark scripts from this repository and keep host metadata in
the report (`core` usage, governor mode, concurrent load tags).

### Integration and review

Use container + Rust + validation artifacts before external sharing with reviewers
or safety teams. For this context, each report should include:

- command transcript,
- report and manifest hashes,
- admission mode,
- declared claim boundary,
- and the exact deployment gate that admits the result.

Treat these artifacts as part of the release evidence bundle, not as optional
appendices.

## Practical use and scope

Use this page to align deployment mode with runtime assumptions.

- Read the environment matrix before selecting transport, scheduling, and hardware assumptions.
- Do not use a Python-only path for hard real-time PCS claims.
- Confirm security, admission, and benchmark evidence after deployment choices are set.
