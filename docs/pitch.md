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
checking, a runtime-selectable multi-controller stack, and zero GPU dependency.

### What makes it different

The differentiator is the architecture, not raw speed:

- **Petri Net → SNN compilation.** A control specification written as a
  Stochastic Petri Net is compiled into leaky integrate-and-fire neuron pools —
  a documented open-source neuro-symbolic control path for fusion-control
  experiments.
- **Formal safety contracts on every action.** Pre/post-condition contracts
  (Z3-backed, certificate-bundled) are checked on every control observation and
  command, fail-closed.
- **One interface, five controllers.** PID, nonlinear MPC, H∞, SNN, and a
  neuro-cybernetic controller share a single contract interface and are
  runtime-selectable, so the same safety case covers the whole stack.
- **CPU-only, Rust-accelerated.** No GPU dependency anywhere in the control path.

### Real-time budget

Control compute is not the bottleneck. The native integrated control cycle runs
in ~5 µs P50 on CI (~3 µs local) — comfortably under the 1–10 kHz real-time
band (100 µs–1 ms period). In a fielded loop the dominant latency is diagnostics
acquisition, equilibrium reconstruction, and actuation, not the controller, so
this is reported as *meeting the budget with margin*, not as a competitive speed
claim. Per-controller and per-backend tables, with the local/CI side-by-side and
the `acados`-gated MPC caveat, are in [benchmarks](benchmarks.md).

### Neural equilibrium research path — without GPU

P-EFIT achieves <1 ms on GPU hardware. scpn-control's PCA + MLP surrogate
currently has bounded synthetic pretraining evidence, not an admitted latency
claim against P-EFIT on identical equilibria. The tracked pretraining report is
`validation/reports/neural_equilibrium_pretraining.json`; real EFIT/P-EFIT
latency or accuracy claims remain blocked until matched reference artefacts are
admitted.

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
| PPO reinforcement learning agent | Research benchmark evidence |
| GPU dispatch (JAX) | Tested |

---

## Why It Matters

### For Fusion Startups

You need real-time control prototyping **now**, not after a 3-year bespoke
development cycle. scpn-control gives you:

- A tested controller stack with module-specific tests, a 100% configured coverage gate, and
  CI-gated bounded validation
- Five-tier gyrokinetic transport research surfaces with strict boundaries for
  external-code agreement
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
161 Python control/physics modules | 5 Rust crates / 64 Rust source files
483 Python test files | 10 GitHub Actions workflows
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
| DIII-D shot replay | Reference GEQDSK + disruption-shot archive, checksum gated | Manifest + replay gates pass | Immutable repository reference artefacts |
| SPARC equilibrium | 8 EFIT reference equilibria | < 5% flux error | Public GEQDSK files |
| IPB98(y,2) scaling | ITPA multi-machine database | 26.6% RMSE | Published coefficients |
| Kuramoto convergence | R -> 0.92, V -> 0, lambda < 0 | 500-tick verified | Simulation |
| Control latency | `benchmark_native_handoff.py` (P50/P99) | ~5 / ~6 us native cycle | CI (EPYC 7763) + local |
| Neural equilibrium | PCA + MLP synthetic pretraining | Claim boundary report tracked | Simulation |

> **Important:** "DIII-D shot replay" is validated against immutable repository
> reference artefacts with manifest checksums. It is not a live MDSplus
> acquisition or facility-control claim. Synthetic fixtures remain only for CI
> plumbing tests and are not evidence for public physics claims.

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
  with exogenous global field driver.
  [Paper 27 manuscript](https://www.academia.edu/143833534/27_SCPN_The_Knm_Matrix).
- **Related Kuramoto-Sakaguchi finite-size reference:**
  [arXiv:2004.06344](https://arxiv.org/abs/2004.06344).
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

## Decision-context guide for this pitch

This document presents the technical value of the stack and the market position. For external communication, use this map:

- **What the platform proves today:** bounded controller execution, contract-admitted validation workflow, and benchmark evidence in its declared context.
- **What it does not prove yet:** facility commissioning equivalence, measured-shot deployment claims, and CODAC/EPICS production acceptance.
- **What to state when pitching:** separate local/controlled benchmarks from hardware deployment claims.

A practical investor-facing summary should pair this page with:

- `docs/production_readiness.md` (admission gates and limitations),
- `docs/validation.md` (validator outputs and claim boundaries),
- `docs/benchmarks.md` (timing context and evidence classes).

If a claim depends on external-code or facility validation, indicate that this is
an explicit next-work item, not a completed fact.

## Practical use and scope

Use this page as the investor-facing positioning artifact.

- Align each claim with corresponding evidence artifacts before external circulation.
- Keep this pitch synchronized with funding and readiness checkpoints.
- Route any measurable claim changes through benchmark and validation updates.
