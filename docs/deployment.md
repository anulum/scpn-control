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

## 4. Real-Time Mode (Rust + PyO3)

To achieve the 11.9 µs P50 latency target, ensure the Rust backend is 
compiled and available.

1.  **Build Rust Extension**:
    ```bash
    cd scpn-control-rs
    cargo build --release
    cp target/release/libscpn_control_rs.so ../src/scpn_control/
    ```
2.  **Verify Dispatch**:
    Check the logs for `[scpn_control] Backend: RUST` during initialization.

## 5. WebSocket Monitoring

Deploy the real-time monitoring dashboard for live telemetry.

1.  **Start Stream Server**:
    ```bash
    python src/scpn_control/phase/ws_phase_stream.py --port 8765
    ```
2.  **Connect Dashboard**:
    Open `docs/dashboard.html` in a browser and point to `ws://localhost:8765`.

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
