<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# SCPN Control: Architectural Capabilities & Use-Case Scenarios

## 1. Core Architecture: Neuro-Symbolic Actuation

`scpn-control` is a real-time control research layer for tokamak plasma scenarios. It ingests high-frequency diagnostics, computes stability and transport quantities, and issues actuator commands through fail-closed admission gates.

### Technical Specifications

*   **Strict Fallback Degradation:** Compute dispatch attempts `scpn-control-rs` (Rust) first, falls back to `sc-neurocore` (JAX/GPU acceleration) where available, and finally to strictly typed pure NumPy/Python. A missing native backend degrades the compute path; it does not crash the loop.
*   **Control-Loop Latency:** Measured latency artefacts (per-kernel and end-to-end percentiles, with recorded host, affinity, and isolation context) live in `validation/reports/` and are regression-gated in CI. All committed figures carry an evidence class and are local-regression evidence, not facility or production timing claims; see `VALIDATION.md` for the current claim boundary.
*   **Stochastic Petri Nets (SPNs):** Supervisory logic is expressed as SPNs executed via populations of Leaky Integrate-and-Fire (LIF) neurons, providing graceful behaviour under diagnostic noise and sensor dropouts.

## 2. Dynamic Domain Actuation

The architecture is domain-configurable: the controller updates its $K_{nm}$ coupling matrix online via the `AdaptiveKnmEngine`, and layer adapters map domain telemetry onto the phase-coupling model.

*   **Plasma Control Loop:** Ingests $\beta_N$, Mirnov RMS, and Greenwald limits. Executes Singular Value Decomposition (SVD) and $H_\infty$ control synthesis to adjust magnetic coil voltages ($\Delta V_{coil}$) for Neoclassical Tearing Mode (NTM) suppression studies.

## 3. Gyrokinetic Transport (v0.17.0)

Three gyrokinetic transport paths are implemented:

*   **Path A — External GK Coupling:** Interfaces to five first-principles codes (TGLF, GENE, GS2, CGYRO, QuaLiKiz) via subprocess with automatic input deck generation and output parsing; fail-closed when the external executable is absent. Transport solver mode: `transport_model="external_gk"`.
*   **Path B — Native Linear GK Solver:** Self-contained ballooning-space eigenvalue solver (Miller geometry, Sugama collision operator, Gauss-Legendre velocity grid), benchmarked against Cyclone Base Case reference ranges (Dimits et al. 2000). Quantitative cross-code agreement with external GK codes is an open validation target; see `VALIDATION.md`.
*   **Path C — Hybrid Surrogate+GK Validation:** A QLKNN-style surrogate runs at microsecond speed; Mahalanobis + ensemble OOD detection triggers GK spot-checks, with a correction layer (multiplicative/additive/replace with EMA smoothing) and online retraining.
*   **SCPN Phase Bridge:** An opt-in bridge maps GK growth rates and diffusivities onto the 8-layer UPDE Kuramoto coupling matrix (P0↔P1 microturbulence↔zonal flows, P1↔P4 zonal↔transport barrier, P3↔P4 sawtooth↔barrier). The bridge currently records coupling metadata; closed-loop actuation through the phase layer is a research lane, not a shipped control path.

## 4. Use-Case Scenarios

*   **Nuclear Fusion (Digital Twin & Flight Sim):** A full `tokamak_flight_sim.py` and `tokamak_digital_twin.py` allow continuous integration testing of real-time control algorithms against a synthetic plasma model before any hardware exposure.
*   **Industrial Control Research:** The same SPN/phase-coupling architecture is applicable to nonlinear multi-actuator control problems (manufacturing, chemical reactors, grid balancing) as a research substrate; no production deployment in these domains is claimed.
