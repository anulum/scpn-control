<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Benchmarks -->

# Benchmarks

This project has two benchmark tracks:

1. Python CLI micro-benchmark (`scpn-control benchmark`)
2. Rust Criterion benches (`cargo bench --workspace`)

## Python CLI benchmark

Run:

```bash
python -m pip install -e .
scpn-control benchmark --n-bench 5000
```

JSON output:

```bash
scpn-control benchmark --n-bench 5000 --json-out
```

Current outputs include:

- `pid_us_per_step`
- `snn_us_per_step`
- `speedup_ratio`

## Kuramoto Phase Sync — Python vs Rust Speedup

Single `kuramoto_sakaguchi_step()` with ζ=0.5, Ψ=0.3.
Python: NumPy vectorised (AMD Ryzen, single-thread).
Rust: Rayon `par_chunks_mut(64)` + criterion harness.

| N | Python (ms) | Rust (ms) | Speedup |
|------:|------------:|----------:|--------:|
| 64 | 0.050 | 0.003 | 17.3× |
| 256 | 0.029 | 0.033 | 0.9× |
| 1 000 | 0.087 | 0.062 | 1.4× |
| 4 096 | 0.328 | 0.180 | 1.8× |
| 16 384 | 1.240 | 0.544 | 2.3× |
| 65 536 | 5.010 | 1.980 | 2.5× |

N=64: Rust wins on per-element throughput (no NumPy dispatch overhead).
N=256: parity — NumPy SIMD matches rayon at this size.
N≥1000: Rust rayon parallelism scales; **sub-ms for N=16k** (0.544 ms).

The Rust Criterion harness also includes the phase-lagged Sakaguchi case
`sakaguchi_alpha/alpha_0.37_zeta_0.5` for N=1000, 4096, 16 384, and 65 536.
This keeps the `alpha != 0` production path under the same regression benchmark
surface as the baseline and global-driver kernels.

### Knm 16-Layer UPDE PAC Benchmark

Full 16-layer outer loop (16 × 256 oscillators, Paper 27 Knm, ζ=0.5).
Criterion harness, AMD Ryzen.

| Config | Median (µs) | 95% CI |
|--------|------------:|-------:|
| PAC γ=1.0 | 909 | [860, 921] |
| No PAC γ=0 | 811 | [807, 827] |

PAC gate overhead: ~12% (98 µs per step).
See `docs/bench_pac_vs_nopac.vl.json` for Vega-Lite breakdown.

### Lyapunov Exponent vs ζ Strength

N=1000, 200 steps @ dt=1ms, Ψ=0.3 (exogenous driver).

| ζ | λ (K=0) | λ (K=2) |
|------:|--------:|--------:|
| 0.0 | +0.01 | +0.04 |
| 0.1 | −0.03 | −0.02 |
| 0.5 | −0.23 | −0.24 |
| 1.0 | −0.49 | −0.53 |
| 3.0 | −1.65 | −1.83 |
| 5.0 | −3.01 | −3.35 |

λ < 0 ⟹ stable convergence toward Ψ.
See `docs/bench_lyapunov_vs_zeta.vl.json` for Vega-Lite plot.

Benchmark source: `benches/bench_fusion_snn_hook.py` (Python, pytest-benchmark).

### Interactive Visualization

All three benchmark datasets (speedup, λ-vs-ζ, PAC latency) in a single
interactive Vega-Lite chart with legend-click filtering:

`docs/bench_interactive.vl.json`

Open in the [Vega Editor](https://vega.github.io/editor/) or embed via
`<vega-embed>` / `vegaEmbed()`.  Click legend entries to isolate series.

## Gyrokinetic Linear Benchmark (v0.17.0)

The native linear GK eigenvalue solver is benchmarked via
`validation/benchmark_gk_linear.py`:

| Case | Parameters | gamma_max | Dominant | Runtime |
|------|-----------|-----------|----------|---------|
| Cyclone Base Case | R/a=2.78, q=1.4, s_hat=0.78, R/L_Ti=6.9 | >0 | ITG | ~2s (12 k_y, n_theta=32) |
| SPARC mid-radius | R0=1.85, B0=12.2, q=1.8 | finite | — | ~1s (6 k_y) |
| ITER mid-radius | R0=6.2, B0=5.3, q=1.5 | finite | — | ~1s (6 k_y) |

Multi-code comparison (`benchmark_gk_linear.run_multi_code_comparison()`):

| Model | gamma_max | chi_i | chi_e |
|-------|-----------|-------|-------|
| Native GK eigenvalue | from solver | from quasilinear | from quasilinear |
| Quasilinear dispersion | from analytic | from mixing-length | from mixing-length |

Hybrid accuracy (`validation/benchmark_hybrid_accuracy.py`) measures the
correction layer convergence over 20 transport steps with periodic GK
spot-checks.

## RZIP Calibration Benchmark

`validation/benchmark_rzip_calibration.py` publishes bounded local regression
evidence for the RZIP rigid-plasma vertical-stability plant. The generated
report records the declared vertical inertia, wall time constant, growth rate,
growth time, and explicit facility-claim boundary.

Report artefacts:

- `validation/reports/rzip_calibration.json`
- `validation/reports/rzip_calibration.md`

Facility vertical-control claims still require documented public, external-code,
or measured-discharge RZIP reference evidence that passes the strict admission
gate.

## RWM Claim-Admission Benchmark

`validation/benchmark_rwm_claims.py` publishes bounded local regression evidence
for the resistive-wall-mode feedback model. The generated report records beta
limits, wall-gap correction, rotation, sensor/coil topology, controller latency,
coil coupling, open-loop growth, closed-loop growth, and the explicit
facility-claim boundary.

Report artefacts:

- `validation/reports/rwm_claims.json`
- `validation/reports/rwm_claims.md`

Facility RWM-control claims still require documented public, external MHD, or
measured-shot evidence that passes the strict admission gate.

## EFIT-lite Claim-Admission Benchmark

`validation/benchmark_efit_lite_claims.py` publishes bounded synthetic
regression evidence for the fixed-boundary EFIT-lite reconstruction path. The
generated report records diagnostic provenance, grid shape, flux-loop and
B-probe counts, Rogowski radius, reconstructed current, q95, beta_pol, li, and
the explicit facility-claim boundary.

Report artefacts:

- `validation/reports/efit_lite_claims.json`
- `validation/reports/efit_lite_claims.md`

Facility equilibrium claims still require matched EFIT/P-EFIT, documented
public, or measured-discharge evidence for psi, Ip, q95, beta_pol, and li that
passes the strict admission gate.

## Kinetic EFIT Claim-Admission Benchmark

`validation/benchmark_kinetic_efit_claims.py` publishes bounded synthetic
regression evidence for kinetic pressure, q-profile, anisotropy, diagnostic
provenance, profile provenance, fast-ion provenance, MSE calibration, and
normalised elliptic-rho interpolation geometry.

Report artefacts:

- `validation/reports/kinetic_efit_claims.json`
- `validation/reports/kinetic_efit_claims.md`

Facility kinetic-EFIT claims still require matched EFIT/P-EFIT, documented
public, or measured-discharge references for pressure, q-profile, and
anisotropy that pass the strict admission gate.

## Differentiable Transport Gradient-Latency Benchmark

The controller-tuning facade measures the audited admission path for JAX
transport gradients via `validation/benchmark_differentiable_transport_latency.py`.
The timed path includes gradients for transport coefficients and source
schedules plus the sampled independent finite-difference audit used before
controller-tuning admission.

Report artefacts:

- `validation/reports/differentiable_transport_latency.json`
- `validation/reports/differentiable_transport_latency.md`

The report is local latency evidence for the audited gradient-admission path.
It is not a real-time control-loop guarantee and does not replace external
transport validation.

## VMEC-lite Claim-Admission Benchmark

`validation/benchmark_vmec_lite_claims.py` publishes bounded synthetic
regression evidence for the fixed-boundary VMEC-lite spectral facade. The
generated report records Fourier truncation, field periods, pressure and
rotational-transform profile provenance, current-assumption provenance,
positive sampled major-radius bounds, force residual, and q-domain.

Report artefacts:

- `validation/reports/vmec_lite_claims.json`
- `validation/reports/vmec_lite_claims.md`

Full VMEC or 3D MHD equilibrium claims still require matched VMEC, documented
public, external-MHD, or measured-stellarator references for `R_mn`, `Z_mn`,
rotational transform, convergence, and residual tolerance.

## Neural-equilibrium Claim-Admission Benchmark

`validation/benchmark_neural_equilibrium_pretraining.py` publishes bounded
synthetic pretraining evidence for the neural-equilibrium surrogate and records
claim-admission evidence around the generated weights. The generated report
captures sample count, grid shape, PCA component count, explained variance,
synthetic MSE, Grad-Shafranov residual, weight checksum, and the explicit
predictive-claim boundary.

Generated artefacts:

- `validation/reports/neural_equilibrium_pretraining.json`
- `validation/reports/neural_equilibrium_pretraining.md`
- `validation/reports/neural_equilibrium_synthetic_pretrain.npz`

Facility predictive claims remain blocked until a strict P-EFIT or documented
public reference artefact validates the same weight checksum and declares
psi, pressure, q-profile, boundary, and magnetic-axis errors inside stated
tolerances.

## Neural-transport Claim-Admission Benchmark

`validation/benchmark_neural_transport_claims.py` publishes bounded local
regression evidence for the neural-transport claim boundary. The generated
report records the deterministic analytic-fallback benchmark cases, local
channel agreement, local diffusivity errors, feature-schema contract, and the
explicit quantitative-claim admission status.

Generated artefacts:

- `validation/reports/neural_transport_claims.json`
- `validation/reports/neural_transport_claims.md`

Quantitative QuaLiKiz, QLKNN, or documented-reference neural-transport claims
remain blocked until a strict reference artefact validates the same neural
weight checksum and declares chi_i, chi_e, D_e, and unstable-branch metrics
inside stated tolerances.

## Orbit-following Claim-Admission Benchmark

`validation/benchmark_orbit_following_claims.py` publishes bounded synthetic
regression evidence for guiding-centre orbit-following claim admission. The
generated report records geometry provenance, particle provenance,
collision-model provenance, loss-boundary provenance, banana width,
first-orbit loss, and ensemble classification counts.

Report artefacts:

- `validation/reports/orbit_following_claims.json`
- `validation/reports/orbit_following_claims.md`

External orbit-following claims still require matched external-code,
documented-public, published-benchmark, or measured fast-ion diagnostic
references for banana width and loss fraction.

## UQ Claim-Admission Benchmark

`validation/benchmark_uq_claims.py` publishes bounded synthetic regression
evidence for full-chain uncertainty quantification claim admission. The
generated report records scenario provenance, prior provenance, propagation
chain, seed, sample count, ordered percentile checks, finite outputs, D-T fuel
dilution, and density/temperature sensitivity provenance.

Report artefacts:

- `validation/reports/uq_claims.json`
- `validation/reports/uq_claims.md`

Calibrated predictive-UQ claims still require matched measured scenario,
documented-public, external-UQ, or facility validation references for central
values and sigma statistics.

## Density-control Claim-Admission Benchmark

`validation/benchmark_density_control_claims.py` publishes bounded synthetic
regression evidence for density-control claim admission. The generated report
records geometry provenance, transport provenance, actuator provenance,
diagnostic provenance, CFL limiting, Greenwald fraction, source integral,
particle inventory change, and actuator command bounds.

Report artefacts:

- `validation/reports/density_control_claims.json`
- `validation/reports/density_control_claims.md`

Facility-calibrated density-control claims still require matched measured
discharge, documented-public, external particle-balance, or facility replay
references for Greenwald fraction and particle inventory change.

## Rust Criterion benchmarks

Run from the Rust workspace root:

```bash
cd scpn-control-rs
cargo bench --workspace
```

Current benchmark targets:

- `benches/bench_boris.rs`
- `benches/bench_lif.rs`
- `benches/bench_transport.rs`
- `benches/bench_kuramoto.rs`

Criterion artifacts are generated under:

- `scpn-control-rs/target/criterion/`

## CI benchmark jobs

### Rust Criterion (Job 8)

- `cargo bench --workspace`
- Uploads `bench-results` from `scpn-control-rs/target/criterion/`

### Python phase-sync benchmark — DIII-D scale (Job 9)

Runs `kuramoto_sakaguchi_step` at N=1000 and N=4096 (DIII-D PCS scale),
plus a `RealtimeMonitor.tick()` (16 layers × 50 oscillators).

Gates:
- Single-step P50 < 5 ms (N=4096)
- RealtimeMonitor tick P50 < 50 ms

## Reproducibility notes

- Run benchmarks on an idle machine.
- Keep `--n-bench` fixed for comparable CLI timing runs.
- Compare same Python/Rust versions and CPU class when evaluating trends.
