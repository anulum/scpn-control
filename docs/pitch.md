# SCPN Control — Neuro-Symbolic Fusion Controller

<p align="center">
  <img src="../scpn_control_header.png" alt="SCPN Control" width="100%">
</p>

---

## The Problem

Fusion energy is within reach, but **real-time plasma control** remains a
bottleneck. Current tokamak control systems are:

- **Slow** — physics loops at 4--10 kHz, limited by Fortran/C legacy code
- **Coupled** — tightly bound to specific machines (DIII-D, ITER)
- **GPU-locked** — P-EFIT needs CUDA for sub-ms reconstruction
- **No SNN path** — no open-source Petri Net → SNN compilation for fusion

scpn-control addresses these with a different architecture.

---

## The Solution

A standalone neuro-symbolic control engine that compiles Stochastic Petri Nets
into spiking neural network controllers — with contract-based pre/post-condition
checking, sub-ms kernel latency, and zero GPU dependency.

### 11.9 microsecond kernel step

The Rust control kernel benchmarks at 11.9 µs P50 (Criterion-verified on
GitHub Actions ubuntu-latest). This is a **single kernel step**, not a
complete control cycle including I/O, diagnostics, and actuator commands.

| Metric | scpn-control | DIII-D PCS | TORAX | ITER PCS |
|--------|-------------|-----------|-------|----------|
| Kernel step (P50) | **11.9 us** | 100--250 us (physics cycle) | ~ms (sim step) | 5--10 ms |
| Language | Rust + Python | C/Fortran | JAX | TBD |
| GPU required | **No** | No | Yes | TBD |
| Deployment | Research / Alpha | Production | Offline sim | Spec |

> **Caveat:** DIII-D PCS timings include I/O, diagnostics, and actuator
> commands within a full physics cycle. scpn-control's 11.9 µs is a bare
> kernel call. A fair comparison would require equivalent end-to-end
> measurement on comparable hardware.

### 0.39 ms neural equilibrium — without GPU

P-EFIT achieves <1 ms on GPU hardware. scpn-control's PCA + MLP surrogate
achieves 0.39 ms on **CPU only**. Not validated against P-EFIT on identical
equilibria — the MLP was trained on SPARC geometries, not the DIII-D shapes
P-EFIT typically reconstructs.

### Full-stack control in one package

| Capability | Status |
|-----------|--------|
| Grad-Shafranov equilibrium solver | Tested (CI-gated RMSE) |
| 1.5D coupled transport | Tested |
| PID controller | Tested |
| H-infinity controller (Riccati) | Tested |
| MPC (gradient-based, surrogate dynamics) | Tested |
| Spiking neural network controller (pure LIF+NEF engine) | Tested (mocked CI; Loihi untested) |
| Phase dynamics (Kuramoto/UPDE) | Tested |
| WebSocket live telemetry | Tested |
| Contract-based pre/post-condition checking | Tested |
| **Native linear GK eigenvalue solver** | **Tested (Cyclone Base Case)** |
| **External GK coupling (TGLF/GENE/GS2/CGYRO/QuaLiKiz)** | **Tested (mock subprocess)** |
| **Hybrid surrogate+GK validation** | **Tested (OOD + correction + online learning)** |
| **GK → UPDE phase bridge** | **Tested** |
| ML disruption prediction (Transformer) | Experimental (synthetic data only) |
| SPI ablation mitigation | Experimental |
| Real-time digital twin | Experimental |
| Neuro-cybernetic controller | Experimental |
| JAX autodiff (transport + GS solver) | Tested |
| QLKNN-10D neural transport | Tested |
| PPO reinforcement learning agent | Tested (beats MPC + PID) |
| GPU dispatch (JAX) | Tested |

---

## Why It Matters

### For Fusion Startups

You need real-time control prototyping **now**, not after a 3-year bespoke
development cycle. scpn-control gives you:

- A tested controller with 3,300+ tests (100% coverage) and CI-gated RMSE validation
- Five-tier gyrokinetic transport (native solver + 5 external codes + hybrid validation)
- Runs on commodity hardware (no GPU or data center required)
- AGPL-3.0 open source; commercial licensing available

**Caveat:** This is Alpha-stage research software, not a production PCS.
Integration with real hardware requires significant additional work.

### For National Labs

scpn-control offers a modern alternative for offline analysis and
rapid prototyping:

- Modern Rust + Python stack alongside legacy Fortran
- Contract-based pre/post-condition checking on control boundaries
- Digital twin for offline commissioning and algorithm development

### For ITER / DEMO (Speculative)

The architecture *could* support future integration, but:

- Not currently hardened for ITER CODAC or EPICS
- Disruption prediction is trained on synthetic data only
- SPI mitigation physics is experimental, not validated against
  real disruption databases

---

## Architecture

```
125 Python modules | 5 Rust crates | 3,300+ tests (100% coverage) | 20 CI jobs
```

```
src/scpn_control/
+-- scpn/       Petri Net -> SNN compiler (formal contracts)
+-- core/       GS solver, transport, scaling laws, gyrokinetic (16 GK modules)
+-- control/    PID, MPC, H-inf, SNN, digital twin
+-- phase/      Paper 27 Kuramoto/UPDE engine (9 modules)

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
  <img src="../phase_sync_live.gif" alt="Phase Sync Convergence" width="100%">
</p>

---

## Validation

All benchmarks are CI-reproducible. See [VALIDATION.md](https://github.com/anulum/scpn-control/blob/main/VALIDATION.md)
for scope and limitations.

| Validation | Method | Result | Data Source |
|-----------|--------|--------|-------------|
| DIII-D shot replay | 16 reference shots, RMSE gated | < 15% Te RMSE | **Synthetic** (mock_diiid.py) |
| SPARC equilibrium | 8 EFIT reference equilibria | < 5% flux error | Public GEQDSK files |
| IPB98(y,2) scaling | ITPA multi-machine database | 26.6% RMSE | Published coefficients |
| Kuramoto convergence | R -> 0.92, V -> 0, lambda < 0 | 500-tick verified | Simulation |
| Control latency | Criterion benchmark (P50/P99) | 11.9 / 23.9 us | CI ubuntu-latest |
| Neural equilibrium | PCA + MLP vs Picard ground truth | 0.39 ms mean | Simulation |

> **Important:** "DIII-D shot replay" uses **synthetic mock shots**, not real
> MDSplus experimental data. No real tokamak data has been ingested or validated
> against. The SPARC GEQDSK files are publicly available design equilibria.

---

## Getting Started

```bash
pip install scpn-control          # From PyPI
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
| **Open Source** | AGPL-3.0-or-later (strong copyleft); commercial licensing available |
| **Contact** | [protoscience@anulum.li](mailto:protoscience@anulum.li) |
| **Organization** | ANULUM CH & LI |
| **Authors** | Miroslav Sotek ([ORCID](https://orcid.org/0009-0009-3560-0851)) |

---

## Next Steps

1. **Try it:** `pip install scpn-control`
2. **See benchmarks:** [Competitive Analysis](competitive_analysis.md)
3. **Live demo:** [scpn-control.streamlit.app](https://scpn-control.streamlit.app)
4. **Talk to us:** [protoscience@anulum.li](mailto:protoscience@anulum.li)
