# Validation

## Scope & Limitations

> This section states what is and is not validated. Read this first.

| Claim | Validated Against | Data Source | Limitation |
|-------|-------------------|-------------|------------|
| "DIII-D shot replay" | Synthetic mock shots | `tests/mock_diiid.py` | **Not real MDSplus data** |
| SPARC equilibrium RMSE | Published GEQDSK design files | CFS public data | Design equilibria, not experimental |
| IPB98(y,2) scaling | Published ITPA coefficients | Wesson, ITER Physics Basis | Coefficient comparison only |
| Control latency 11.9 µs | Rust Criterion benchmark | CI ubuntu-latest | Bare kernel step, not E2E cycle |
| Neural equilibrium 0.39 ms | PCA+MLP vs Picard solver | Internal simulation | Not cross-validated against P-EFIT |
| "Formal verification" | Runtime contract assertions | `scpn/contracts.py` | Not theorem-proved (no Coq/Lean) |
| Disruption prediction | Synthetic training data | Internal generator | Not validated on real disruption DBs |
| SPI mitigation physics | Physics equations only | Literature constants | Not validated against JET/ITER data |
| SNN controller | Mocked Nengo CI tests | Simulated neurons | Nengo Loihi hardware untested |

**What does NOT exist:**
- No real MDSplus shot data ingestion
- No experimental tokamak validation (DIII-D, JET, KSTAR, EAST)
- No peer-reviewed fusion journal publication
- No real hardware deployment (PCS, EPICS, CODAC)
- No cross-validation against P-EFIT, TORAX, or FUSE on identical scenarios
- No radiation-hardened or ARM deployment testing

## Test Matrix

| Suite | Count | Scope |
|-------|------:|-------|
| Python unit/integration | 3,300+ | `pytest tests/` across 235 files |
| Rust engine | 140+ | `cargo test --workspace` in `scpn-control-rs/` |
| Rust-Python interop | 3 files | PyO3 parity tests via maturin |
| Notebooks | 5 | Executed in CI via `nbconvert` |
| E2E (DIII-D mock) | 1 file | Full shot-driven control loop (**synthetic data**) |
| RMSE gate | 1 file | Regression against SPARC GEQDSK + synthetic DIII-D |
| GK eigenvalue | 54 | Native linear GK solver (geometry, species, eigenvalue, quasilinear) |
| GK external codes | 34 | TGLF, GENE, GS2, CGYRO, QuaLiKiz (mock subprocess, input/output parsing) |
| GK hybrid layer | 55 | OOD detection, scheduling, correction, online learning, verification |
| GK validation | 20 | Cyclone Base Case, SPARC/ITER scans, multi-code comparison, hybrid accuracy |

CI runs tests on Python 3.10-3.14 (Ubuntu), Python 3.12 (Windows + macOS), Rust on Ubuntu.

## Gyrokinetic Transport Validation

| Claim | Validated Against | Data Source | Limitation |
|-------|-------------------|-------------|------------|
| Cyclone Base Case ITG | GENE/GS2/GYRO published γ_max | Dimits et al. 2000 | Linear solver within 21% of GENE for CBC |
| Dimits shift | Zero transport below critical gradient at n_kx=256 | Dimits et al. 2000 | Proven with adiabatic electrons; kinetic electrons yield chi_i=1.3 χ_gB |
| Turbulent saturation | chi_i = 2.0 χ_gB (adiabatic), 1.3 χ_gB (kinetic) | GENE CBC range 1-5 χ_gB | Within expected range |
| SPARC/ITER GK parameters | Internal eigenvalue solver | Synthetic equilibrium | Not cross-validated against TGLF runs |
| Hybrid surrogate correction | Internal GK vs critical-gradient | Synthetic spot-checks | No experimental validation |
| 5 external code interfaces | Mock subprocess tests | Input deck generation + output parsing | No actual GK binaries in CI |

**What does NOT exist for GK:**
- No cross-code nonlinear GK validation (GENE/CGYRO flux comparison on identical equilibria)
- No experimental turbulence profile comparison (DIII-D BES, KSTAR ECE)
- No TGLF binary in CI (mock subprocess only)

## CI Validation Gates

All gates must pass before merge to `main`.

| Gate | Workflow | Enforces |
|------|----------|----------|
| ruff check | `ci.yml` | Import hygiene, code quality |
| ruff format | `ci.yml` | Consistent formatting |
| bandit | `ci.yml` | Security static analysis (SAST) |
| test + coverage | `ci.yml` | `pytest --cov-fail-under=99` on Python 3.12 |
| mypy | `ci.yml` | Type checking (scoped files) |
| notebook smoke | `ci.yml` | All tutorial notebooks execute |
| package quality | `ci.yml` | `twine check` on built sdist/wheel |
| pip-audit | `ci.yml` | No known vulnerabilities in deps |
| RMSE gate | `ci.yml` | Regression bounds vs. reference data |
| E2E DIII-D | `ci.yml` | End-to-end mock shot test |
| real DIII-D | `ci.yml` | 17 real disruption shots (synthetic) validation |
| JAX parity | `ci.yml` | JAX transport, neural eq, GS solver parity |
| LIF+NEF SNN | `ci.yml` | SNN wrapper emulator tests |
| E2E benchmark | `ci.yml` | Control latency regression |
| rust-tests | `ci.yml` | `cargo test --workspace` + clippy + fmt |
| rust-python-interop | `ci.yml` | Maturin build + parity tests |
| rust-benchmarks | `ci.yml` | Criterion benchmarks (no regression gate) |
| cargo-deny | `ci.yml` | License + advisory supply-chain policy |
| cargo-audit | `ci.yml` | Rust dependency vulnerability scan |
| pre-commit | `pre-commit.yml` | Trailing whitespace, YAML, formatting |
| CodeQL | `codeql.yml` | GitHub CodeQL security analysis |
| Scorecard | `scorecard.yml` | OpenSSF Scorecard supply-chain audit |

## Coverage Policy

- Threshold: 99% (enforced by `pytest --cov-fail-under=99`). Current: 100% (10,142 statements, 0 missed).
- Excluded lines: `pragma: no cover`, `if __name__`, `raise NotImplementedError`,
  conditional imports (`HAS_TORCH`, `HAS_NENGO`, `HAS_SC_NEUROCORE`, `except ImportError`)

## RMSE Validation

The `validation/rmse_dashboard.py` script computes pointwise RMSE against
reference GEQDSK equilibria (SPARC design files, synthetic DIII-D) and
enforces bounds in CI via `tools/ci_rmse_gate.py`.

> The DIII-D reference files are **synthetically generated** by
> `tests/mock_diiid.py`, not downloaded from MDSplus or D3D archives.

### RMSE Regression Thresholds

Thresholds are set ~30% above current best values. They catch regressions,
not noise. Source: `tools/ci_rmse_gate.py`.

| Metric | Threshold | Current Best | Source |
|--------|-----------|-------------|--------|
| `confinement_itpa_tau_rmse_s` | 0.20 s | ~0.129 s | 20 ITPA H-mode points |
| `sparc_axis_rmse_m` | 2.50 m | ~1.60 m | SPARC GEQDSKs (synthetic lmode dominates) |
| `beta_iter_sparc_beta_n_rmse` | 0.10 | ~0.042 | ITER/SPARC design points |
| `disruption_fpr` | 0.15 | — | Hard gate since v3.1 |
| `tbr_min` / `tbr_max` | 1.00 / 1.40 | — | Tritium self-sufficiency range |
| `q_max` | 15.0 | — | 0-D model artifact ceiling |

## Benchmark Gates

| Benchmark | Budget | Scope |
|-----------|--------|-------|
| Kuramoto step (N=1000) | < 5 ms P50 | Single-step latency |
| Kuramoto step (N=4096) | < 5 ms P50 | DIII-D scale |
| RealtimeMonitor tick | < 50 ms P50 | 16 layers x 50 oscillators |

> Benchmark budgets are regression gates (vs. previous CI runs), not
> comparisons against external codes. No head-to-head benchmarks against
> TORAX, FUSE, or DIII-D PCS have been published.
