# SCPN Control — The Fastest Open-Source Fusion Controller

<p align="center">
  <img src="scpn_control_header.png" alt="SCPN Control" width="100%">
</p>

---

## The Problem

Fusion energy is within reach, but **real-time plasma control** remains the
bottleneck. Today's tokamak control systems are:

- **Slow** — physics loops at 4--10 kHz, limited by Fortran/C legacy code
- **Fragile** — no formal verification, no disruption prediction
- **Monolithic** — tightly coupled to specific machines (DIII-D, ITER)
- **GPU-locked** — P-EFIT needs CUDA for sub-ms reconstruction

**scpn-control solves all four.**

---

## The Solution

A standalone neuro-symbolic control engine that compiles Stochastic Petri Nets
into spiking neural network controllers — with formal verification, sub-ms
latency, and zero GPU dependency.

### 11.9 microsecond control loop

Faster than any open-source fusion code. Faster than the DIII-D PCS physics
loops. On commodity hardware. No FPGA. No InfiniBand.

| Metric | scpn-control | DIII-D PCS | TORAX | ITER PCS |
|--------|-------------|-----------|-------|----------|
| Control frequency | **10--30 kHz** | 4--10 kHz | Offline | ~100 Hz |
| Step latency (P50) | **11.9 us** | 100--250 us | ~ms | 5--10 ms |
| Language | Rust + Python | C/Fortran | JAX | TBD |
| GPU required | **No** | No | Yes | TBD |

### 0.39 ms neural equilibrium — without GPU

P-EFIT achieves <1 ms on GPU hardware. scpn-control matches this on **CPU
only** using a PCA + MLP surrogate trained on SPARC geometries.

### Full-stack control in one package

| Capability | Status |
|-----------|--------|
| Grad-Shafranov equilibrium solver | Production |
| 1.5D coupled transport | Production |
| PID / MPC / H-infinity controllers | Production |
| Spiking neural network controller | Production |
| ML disruption prediction | Production |
| SPI ablation mitigation | Production |
| Real-time digital twin | Production |
| Phase dynamics (Kuramoto/UPDE) | Production |
| WebSocket live telemetry | Production |
| Formal contract verification | Production |

---

## Why It Matters

### For Fusion Startups

You need real-time control **now**, not after a 3-year bespoke development
cycle. scpn-control gives you:

- Drop-in controller with 675 tests and CI-gated RMSE validation
- Runs on edge hardware (no data center required)
- AGPL-licensed with commercial licensing available

### For National Labs

Your PCS is decades old. scpn-control offers:

- Modern Rust + Python stack replacing legacy Fortran
- Formal verification via contract-based pre/post-condition checking
- Digital twin for offline commissioning and operator training

### For ITER / DEMO

The 100 Hz diagnostic cycle won't cut it for disruption mitigation.
scpn-control's 30 kHz loop gives you the headroom for:

- Real-time disruption prediction (ML-based, <1 ms inference)
- SPI pellet injection with halo current / runaway electron physics
- Closed-loop SNN feedback replacing hardwired interlock logic

---

## Architecture

```
48 Python modules | 5 Rust crates | 675 tests | 14 CI jobs
```

```
src/scpn_control/
+-- scpn/       Petri Net -> SNN compiler (formal contracts)
+-- core/       GS solver, transport, scaling laws
+-- control/    PID, MPC, H-inf, SNN, digital twin
+-- phase/      Paper 27 Kuramoto/UPDE engine (7 modules)

scpn-control-rs/
+-- control-types/    PlasmaState, EquilibriumConfig
+-- control-math/     LIF neurons, Boris pusher, Kuramoto
+-- control-core/     GS solver, transport, scaling
+-- control-control/  PID, MPC, H-inf, SNN
+-- control-python/   PyO3 bindings
```

---

## Live Demo

**Streamlit Dashboard:** [scpn-control.streamlit.app](https://scpn-control.streamlit.app)

Real-time 16-layer Kuramoto-Sakaguchi phase sync with global field driver.
Interactive controls for coupling strength, oscillator count, and Psi driver.

**Phase sync convergence (500 ticks, 16 layers x 50 oscillators):**

<p align="center">
  <img src="phase_sync_live.gif" alt="Phase Sync Convergence" width="100%">
</p>

---

## Validation

Every claim is CI-verified. Every benchmark is reproducible.

| Validation | Method | Result |
|-----------|--------|--------|
| DIII-D shot replay | 16 reference shots, RMSE gated | < 15% Te RMSE |
| SPARC equilibrium | 8 EFIT reference equilibria | < 5% flux error |
| IPB98(y,2) scaling | ITPA multi-machine database | 26.6% RMSE |
| Kuramoto convergence | R -> 0.92, V -> 0, lambda < 0 | 500-tick verified |
| Control latency | Criterion benchmark (P50/P99) | 11.9 / 23.9 us |
| Neural equilibrium | PCA + MLP vs Picard ground truth | 0.39 ms mean |

---

## Getting Started

```bash
pip install scpn-control          # From PyPI (v0.2.0)
scpn-control demo --steps 1000    # Closed-loop control demo
scpn-control benchmark            # PID vs SNN timing
scpn-control live --zeta 0.5      # Real-time WS phase sync
```

```bash
# Rust acceleration (optional)
cd scpn-control-rs
cargo test --workspace
cd crates/control-python && maturin develop --release
```

---

## Publications

- **Paper 27:** "The Knm Matrix" — 16-layer Kuramoto-Sakaguchi phase dynamics
  with exogenous global field driver. [arXiv:2004.06344](https://arxiv.org/abs/2004.06344)
- **Competitive Analysis:** [Full benchmark comparison](competitive_analysis.md)
  against DIII-D PCS, TORAX, FUSE, GENE, JINTRAC, P-EFIT

---

## Licensing

| | |
|---|---|
| **Open Source** | GNU AGPL v3 — free for research and open-source projects |
| **Commercial** | Dual-license available for proprietary integration |
| **Contact** | [protoscience@anulum.li](mailto:protoscience@anulum.li) |
| **Organization** | ANULUM CH & LI |
| **Authors** | Miroslav Sotek ([ORCID](https://orcid.org/0009-0009-3560-0851)), Michal Reiprich |

---

## Next Steps

1. **Try it:** `pip install scpn-control`
2. **See benchmarks:** [Competitive Analysis](competitive_analysis.md)
3. **Live demo:** [scpn-control.streamlit.app](https://scpn-control.streamlit.app)
4. **Talk to us:** [protoscience@anulum.li](mailto:protoscience@anulum.li)
