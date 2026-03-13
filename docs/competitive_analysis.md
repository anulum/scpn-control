# Competitive Analysis — scpn-control

> **Last updated:** 2026-03-12 (v0.15.0).
> Community code timings are from published literature (references at end).
> SCPN timings are CI-verified on GitHub Actions ubuntu-latest unless noted.

## 1. Real-Time Control Loop

| Code | Control Freq | Step Latency | Language | Source |
|------|-------------|-------------|----------|--------|
| **scpn-control (Rust)** | **10--30 kHz** | **11.9 us P50 / 23.9 us P99** | Rust + Python | CI Criterion |
| DIII-D PCS (production) | 4--10 kHz (physics loops) | 100--250 us per physics cycle | C / Fortran | Penaflor 2024; Barr 2024 |
| P-EFIT (GPU) | N/A (reconstruction) | 300--375 us per iter (129x129) | Fortran + CUDA | Sabbagh 2023 |
| RT-GSFit | ~5 kHz | ~200 us | C++ | Tokamak Energy 2025 |
| TORAX | N/A (offline sim) | ~ms per timestep | Python / JAX | Citrin 2024 |
| Gym-TORAX | 10--100 Hz | ~10 ms (RL env step) | Python / JAX | DeepMind 2025 |
| ITER PCS (spec) | ~100 Hz diagnostics | 5--10 ms processing | TBD | ITER RTF docs |
| FUSE | N/A (design code) | N/A | Julia | Meneghini 2024 |

> **Note on DIII-D:** The raw data-acquisition cycle runs at ~16.7 kHz (60 us),
> but the physics-level control algorithms (rtEFIT, shape control, NTM
> feedback) execute at 4--10 kHz depending on the algorithm. scpn-control's
> 11.9 us P50 is still faster than any published DIII-D physics control loop
> and operates without dedicated FPGA or InfiniBand hardware.

## 2. Transport Simulation Speed

| Code | Type | Runtime | Physics | Source |
|------|------|---------|---------|--------|
| GENE / CGYRO | Gyrokinetic | 10^5--10^6 CPU-hours | Nonlinear 5D Vlasov | Jenko 2000; Belli 2008 |
| JINTRAC + QuaLiKiz | Full integrated | ~217 hours (16 cores) | First-principles turbulence | TU/e 2021 |
| JINTRAC + QLKNN | NN surrogate | ~2 hours (1 core) | ML surrogate | van de Plassche 2020 |
| TORAX | 1D JAX | Faster than real-time (~seconds) | QLKNN10D | Citrin 2024 |
| FUSE | 1D Julia | ~25 ms per step (TJLF) | TJLF surrogate | Meneghini 2024 |
| **scpn-control (Rust)** | 1.5D step | **1.5--5.5 us per step** | Crit-gradient + neoclassical | CI Criterion |
| **scpn-control (MLP)** | Neural surrogate | **24 ns single-point** | Trained surrogate | CI Criterion |
| QLKNN (TensorFlow) | NN inference | ~100 us (25 outputs) | Surrogate | van de Plassche 2020 |

> **Fidelity caveat:** scpn-control uses a critical-gradient transport model,
> not QLKNN or TGLF trained on gyrokinetic data. The speed advantage is partly
> because the physics is simpler. This is an intentional trade-off: reactor-
> grade control latency in exchange for reduced turbulence fidelity.

## 3. Equilibrium Reconstruction

| Code | Grid | Method | Runtime | Source |
|------|------|--------|---------|--------|
| EFIT (Fortran) | 65x65 | Current-filament Picard | ~2 s full recon | Lao 1985 |
| P-EFIT (GPU) | 65x65 | GPU-accelerated Picard | <1 ms per iter | Sabbagh 2023 |
| RT-GSFit | 65x65 | Real-time reconstruction | ~200 us | Tokamak Energy 2025 |
| CHEASE (Fortran) | 257x257 | Fixed-boundary cubic Hermite | ~5 s | Lutjens 1996 |
| HELENA | 201 flux | Isoparametric | ~10 s | Huysmans 1991 |
| FreeGS | Variable | Picard + multigrid | ~seconds | FreeGS GitHub |
| FreeGSNKE | Variable | Newton-Krylov | Faster than FreeGS | FreeGSNKE 2024 |
| **scpn-control (Rust)** | 65x65 | Picard + SOR | **~100 ms** | Measured |
| **scpn-control (Neural)** | 129x129 | PCA + MLP surrogate | **0.39 ms mean** | CI verified |
| **scpn-control (Multigrid)** | 65x65 | V-cycle | **~12 ms** | Measured v0.15.0 |

> The Neural Equilibrium Kernel achieves P-EFIT-class speed (0.39 ms) on
> **CPU only**, without requiring CUDA or GPU hardware.

## 4. Feature Breadth

| Feature | scpn-control | TORAX | PROCESS | FREEGS | FUSE | DREAM |
|---------|-------------|-------|---------|--------|------|-------|
| GS Equilibrium | Yes (multigrid) | Yes (spectral) | No | Yes (Picard) | Yes | No |
| Free-boundary solve | **Yes (v0.15.0)** | Partial | No | Yes | Yes | No |
| Transport solver | 1.5D coupled | 1D flux-driven | 0D | No | 1D | 0--1D |
| **Neuro-symbolic SNN** | **Yes** | No | No | No | No | No |
| **Disruption prediction (ML)** | **Yes** | No | No | No | No | N/A |
| **SPI mitigation** | **Yes** | No | No | No | No | Yes |
| Neutronics / TBR | Yes (1-D slab) | No | Yes | No | Yes | No |
| **Digital twin (real-time)** | **Yes** | No | No | No | No | No |
| **Rust native backend** | **Yes (5 crates)** | No | No | No | No | No |
| IMAS Integration | **Yes (Dec 2025)** | Yes | No | No | No | No |
| GPU acceleration | **Yes (JAX)** | Yes (JAX) | No | No | JAX | No |
| Autodifferentiation | **Yes (JAX, full GS)** | **Yes (JAX)** | No | No | **Yes (Julia)** | No |

## 5. Where Competitors Lead

| Weakness | Detail | Who Does It Better |
|----------|--------|-------------------|
| ~~Equilibrium autodiff depth~~ | **RESOLVED** v0.13.0: JAX Picard GS solver with `jax.grad` through full solve | — |
| No peer-reviewed publication | JOSS paper drafted but not yet submitted | TORAX (NF 2024), FUSE (FED 2024) |
| Smaller community | Single-team vs DeepMind / General Atomics resources | TORAX, FUSE |
| ~~RL agent maturity~~ | **RESOLVED** v0.15.0: PPO 500K beats MPC (143.7 vs 58.1), 0% disruption | — |

### Resolved since v0.10.0
- GPU equilibrium: JAX neural eq with GPU dispatch (v0.11.0)
- Transport autodiff: JAX-traced Thomas + CN + neural eq (v0.10.0–v0.11.0)
- Trained transport model: QLKNN-10D MLP with auto-discovery (v0.12.0)
- RL agent: PPO on TokamakEnv with PID/MPC benchmark (v0.12.0)
- Equilibrium autodiff depth: JAX Picard GS solver with `jax.grad` through full solve (v0.13.0)
- RL agent maturity: PPO 500K on JarvisLabs, beats MPC and PID, 3-seed reproducible (v0.14.0)

## 6. Codebase Metrics (v0.15.0)

| Metric | Value |
|--------|-------|
| Python source modules | 57 |
| Python source LOC | ~22,900 |
| Rust crates | 5 |
| Rust LOC (all .rs) | ~61,900 |
| Test files | 136 |
| Tests collected | 2,683 |
| Test coverage | 100.0% (10,142 stmts, gate=99%) |
| CI jobs | 26 |
| Real DIII-D shots | 17 disruption + 1 safe baseline |
| SPARC GEQDSK files | 3 |
| Pretrained weight files | 5 (MLP, FNO, neural eq, QLKNN, PPO) |

## 7. scpn-control Unique Position

1. **Fastest open-source kernel step** — 11.9 µs P50 (Criterion-verified).
   This is a bare kernel call, not a complete control cycle. No head-to-head
   end-to-end comparison has been published.

2. **Neuro-symbolic SNN + contract checking + digital twin** — the Petri
   Net to SNN compiler with runtime contract assertions is architecturally
   unique in the fusion simulation space.

3. **Neural equilibrium at 0.39 ms without GPU** — not cross-validated
   against P-EFIT on identical equilibria, but demonstrates CPU-only
   sub-ms reconstruction is achievable.

4. **Full-stack control breadth** — equilibrium, transport, control,
   disruption mitigation, digital twin in one focused 57-module package.
   Trade-off: breadth over depth in any single area.

## 8. Gap Resolution Status

| Gap | Resolution | Status |
|-----|-----------|--------|
| Transport-only autodiff | JAX neural equilibrium MLP (v0.11.0) | **RESOLVED** |
| No GPU equilibrium | JAX neural eq with GPU dispatch (v0.11.0) | **RESOLVED** |
| Simpler turbulence | QLKNN-10D trained MLP (v0.12.0) | **RESOLVED** |
| No RL validation | PPO + PID + MPC benchmark (v0.12.0) | **RESOLVED** |
| Equilibrium autodiff depth | JAX Picard GS solver, `jax.grad` through full solve (v0.13.0) | **RESOLVED** |
| No peer-reviewed pub | JOSS paper fact-checked, ready for submission | v0.13.0 |
| Smaller community | External action: talks, workshops, issue triage | Ongoing |

## References

- Lao, L.L. et al. (1985). *Nucl. Fusion* 25, 1611 (EFIT).
- Sabbagh, S.A. et al. (2023). GPU-accelerated EFIT (P-EFIT). ACM SC23.
- Lutjens, H. et al. (1996). *Comput. Phys. Commun.* 97, 219 (CHEASE).
- Huysmans, G.T.A. et al. (1991). *Proc. CP90* (HELENA).
- Romanelli, M. et al. (2014). *Plasma Fusion Res.* 9, 3403023 (JINTRAC).
- Citrin, J. et al. (2024). *arXiv:2406.06718* (TORAX).
- Meneghini, O. et al. (2024). *arXiv:2409.05894* (FUSE).
- Jenko, F. et al. (2000). *Phys. Plasmas* 7, 1904 (GENE).
- Belli, E.A. & Candy, J. (2008). *Phys. Plasmas* 15, 092510 (CGYRO).
- Hoppe, M. et al. (2021). *Comput. Phys. Commun.* 268, 108098 (DREAM).
- van de Plassche, K.L. et al. (2020). *Phys. Plasmas* 27, 022310 (QLKNN).
- Penaflor, B.G. et al. (2024). DIII-D PCS. *Fus. Eng. Des.*
- Barr, J.L. et al. (2024). *arXiv:2511.11964* (Parallelised RT physics on DIII-D).
- FreeGS: https://github.com/freegs-plasma/freegs
- FreeGSNKE: https://docs.freegsnke.com/
- Gym-TORAX: *arXiv:2510.11283*
