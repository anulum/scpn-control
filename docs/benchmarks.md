<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Benchmarks -->

# Benchmarks

## Current native runtime evidence package

The v0.20.4 release candidate includes the following repository reports as
local-regression evidence for the native execution and formal-runtime lane:

| Report family | Purpose | Claim boundary |
| --- | --- | --- |
| `native_handoff_comparison.*` | Compare Python orchestration with fused Rust/PyO3 execution at one campaign boundary | Local execution-ownership evidence; not target PCS timing |
| `native_formal_modes_*.*` | Compare disabled, `async_drop`, `sync_stride`, and `aot_certificate` formal modes | Shows coverage/drop/timing semantics; non-isolated timings remain local regression |
| `native_formal_aot_certificate_admission_*.*` | Persist digest-bound AOT certificate admission evidence | Hot-path certificate evidence; production use requires production benchmark context |
| `native_formal_spin_pacing_*.*` and `native_control_spin_pacing_*.*` | Exercise opt-in spin pacing under workstation constraints | Short timing experiments only; do not cite as deployment timing |

The reports intentionally keep workstation limitations visible: workspace state,
proof-sampling drops where applicable, certificate digests, and evidence-class
metadata. Do not promote their timing numbers into market, release, or facility
claims unless the matching JSON report records production benchmark context and
the validator admits it.

This project has three benchmark tracks:

1. Python CLI micro-benchmark (`scpn-control benchmark`)
2. Rust Criterion benches (`cargo bench --workspace`)
3. Native handoff comparison (`scripts/benchmark_native_handoff.py`)

## Native handoff comparison

Use this track after changes to the control-loop execution boundary:

- `src/scpn_control/core/rust_engine.py`
- `src/scpn_control/cli.py`
- `scpn-control-rs/crates/control-python`

The benchmark forces both execution modes at the same campaign boundary:

- `python`: Python orchestration with Rust-compatible controller and transport
  primitives.
- `native`: fused PyO3 Rust loop with cumulative native cycle telemetry.

The native loop also owns runtime formal verification. Python supplies bounded
Petri-net checking policy through
`NeuroCyberneticEngine.configure_native_formal_verification(...)`; the PyO3
crate either spawns the Z3 worker inside Rust or evaluates a compiled
certificate monitor in the native loop. Worker-backed modes pin to `core_z3`
when host affinity is available and pass only fixed numeric snapshots over a
bounded `crossbeam-channel`. No Z3 ASTs, solver contexts, or proof objects
cross the Python boundary during the fused campaign loop.

Three formal-verification execution modes are benchmarkable:

- `async_drop`: non-blocking proof sampling. The control loop never waits for
  Z3; saturated snapshots are counted as drops.
- `sync_stride`: deterministic stride verification. The control loop blocks on
  each configured stride step until the Rust Z3 worker returns a proof result.
- `aot_certificate`: deterministic compiled-certificate monitoring. The control
  loop evaluates the admitted Petri invariant directly and does not construct
  Z3 contexts or enqueue proof work in the hot path. The current certificate is
  a sound sufficient condition for the configured bounded contract and fails
  closed when the state needs full Z3 search to admit. Admission is bound to a
  canonical certificate-assumption payload covering schema version, certificate
  identifier, Petri topology, maximum marking, maximum depth, and contract
  semantics. Runtime telemetry exposes `certificate_admitted`,
  `certificate_schema_version`, `certificate_id`, `certificate_contract`, and
  `certificate_assumption_sha256` so benchmark artifacts identify the exact
  admitted monitor used in the hot path.

Use `scripts/benchmark_native_formal_modes.py` to quantify the difference:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_native_formal_modes.py \
  --steps 5000 \
  --repeats 3 \
  --tick-interval-s 0 \
  --pacing-modes sleep \
  --strides 1,5,20,30 \
  --transports std,io-uring
```

The report includes generated, submitted, checked, dropped, failure counts,
certificate-admission fields, and sync wait timing. A strict certification
argument must use `sync_stride` as the ground-truth proof engine or
`aot_certificate` with `certificate_admitted=true` and one stable
`certificate_assumption_sha256` across the relevant comparison cases.
`async_drop` must be described as asynchronous proof sampling.

The native fused loop also exposes pacing modes:

- `sleep`: default scheduler-yield pacing. This is safe for normal developer
  runs but measures host wake-up latency as part of wall time.
- `spin`: opt-in busy-wait pacing. The Rust loop uses `std::hint::spin_loop`
  instead of `sleep`, holds the native execution thread on-core, and is intended
  only for short deterministic timing experiments on isolated cores. Spin pacing
  rejects tick intervals above `0.01 s` to prevent accidental long-duration core
  burn.

Compare sleep and spin pacing on the AOT hot path with:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_native_formal_modes.py \
  --steps 5000 \
  --repeats 3 \
  --tick-interval-s 0.0001 \
  --formal-modes disabled,aot_certificate \
  --pacing-modes sleep,spin \
  --strides 1 \
  --transports std \
  --evidence-class local_regression
```

Admit the persisted AOT certificate evidence before using it in a release or
safety-case argument:

```bash
python validation/validate_native_formal_certificate_evidence.py \
  validation/reports/native_formal_aot_certificate_admission_20260604T103219Z.json \
  --max-aot-p99-cycle-us 10.0
```

The validator rejects reports with malformed JSON, the wrong benchmark schema,
missing benchmark context, invalid evidence-class metadata, missing AOT cases,
unstable certificate digests, missing certificate admission, nonzero drops,
nonzero formal failures, incomplete generated/submitted/checked coverage, or
AOT p99 cycle latency above the configured threshold. Reports generated on a
loaded workstation or without explicit CPU/core isolation must use
`evidence_class=local_regression` and `production_claim_allowed=false`.
`evidence_class=production_benchmark` requires explicit isolation metadata,
a clean workspace, and a declared yes/no value for concurrent heavy jobs.

`scpn-control validate` runs the same native formal certificate gate by default
and emits the result under `native_formal_certificate`. The release-evidence
admission step requires this section to pass, requires at least one admitted AOT
certificate case, and binds the report to the certificate-assumption digest,
benchmark-report digest, benchmark evidence class, and production-claim
boundary. Local regression reports must keep `production_claim_allowed=false`;
production benchmark reports must set it explicitly and must carry no
validator errors. Use
`--no-native-formal-certificate` only for local diagnostics; release evidence
and preflight admission must not skip it.

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_native_handoff.py \
  --steps 5000 \
  --tick-interval-s 0.0001 \
  --transport-backend std \
  --json-out validation/reports/native_handoff_comparison.json \
  --markdown-out validation/reports/native_handoff_comparison.md
```

The JSON output is the machine-readable evidence artifact. The Markdown output
is the review table. A valid native run must report zero drops and zero publish
failures. For formal-runtime evidence, inspect `native.formal_verification` in
the returned campaign summary. The expected backend is `rust-z3`, and any
nonzero `failures` count means the fused loop tripped the fail-closed formal
contract instead of continuing under Python control-plane intervention.
For AOT certificate runs, the expected backend is `compiled-certificate`; strict
release evidence must include the schema, certificate identifier, contract label,
and full SHA-256 assumption digest.

This benchmark isolates execution ownership. Use the transport-specific Rust
benchmark and UDP fault-tolerance benchmark for `std` versus `io-uring`
transport measurements.

Current local evidence in `validation/reports/native_handoff_comparison.json`
records 5000 delivered UDP sink packets for each execution path, zero drops,
zero publish failures, Python-orchestrated active-cycle average `11.9141358 us`,
native active-cycle average `5.7648218 us`, and native wall-time speedup
`1.052610246860105x` under a `100 us` campaign tick.

## Capacitor-bank energy ledger

Use this track after changes to the CONTROL-owned capacitor-bank RLC admission
surface:

- `src/scpn_control/control/capacitor_bank_state.py`
- `scpn-control-rs/crates/control-control/src/capacitor_bank.rs`
- `scpn-control-rs/crates/control-python/src/lib.rs`
- `benchmarks/bench_capacitor_bank_energy.py`
- `scpn-control-rs/crates/control-control/examples/bench_capacitor_bank_energy.rs`

The benchmark measures one discharge report per sample. Each report includes
the total RLC energy ledger, residual, relative residual, and pass/fail flag.
Python and Rust commands use the same capacitance, inductance, resistance,
initial voltage, initial current, waveform, step size, and discharge length.

```bash
PYTHONPATH=src python benchmarks/bench_capacitor_bank_energy.py \
  --steps 500 \
  --warmup 50 \
  --discharge-steps 200 \
  --dt-s 1.0e-7 \
  --json-out validation/reports/capacitor_bank_energy_python.json \
  --markdown-out validation/reports/capacitor_bank_energy_python.md

cargo run --release --manifest-path scpn-control-rs/Cargo.toml \
  -p control-control --example bench_capacitor_bank_energy -- \
  --steps 500 \
  --warmup 50 \
  --discharge-steps 200 \
  --dt-s 1.0e-7 \
  --json-out validation/reports/capacitor_bank_energy_rust.json \
  --markdown-out validation/reports/capacitor_bank_energy_rust.md
```

The JSON artifacts are the machine-readable evidence. Markdown reports are for
review. Runs without hard CPU isolation must retain
`evidence_class=local_regression` and `production_claim_allowed=false`.

## JAX GK parity evidence

`validation/benchmark_jax_gk_parity.py` persists schema-versioned parity
artifacts for the JAX linear gyrokinetic backend against the repository native
local-dispersion solver. Each artifact records backend, device kind, platform,
JAX/JAXLIB versions, dtype, X64 state, solver kwargs, growth-rate and
real-frequency tolerances, case-parameter metadata, mode-spectrum agreement,
and canonical SHA-256 digests for solver kwargs, case parameters, and the
complete payload. The default benchmark emits the built-in CBC, kinetic-electron
TEM, and low-drive stable-mode parity cases.

Run:

```bash
python validation/benchmark_jax_gk_parity.py --json-out
JAX_PLATFORM_NAME=cpu python validation/benchmark_jax_gk_parity.py --json-out
```

Strict admission:

```bash
python validation/validate_jax_gk_parity.py \
  --artifact-root validation/reports/jax_gk_parity \
  --require-parity-artifacts \
  --require-cases cyclone_base_case,tem_kinetic_electron,stable_mode \
  --require-backends cpu,gpu
```

The benchmark command also writes aggregate timing evidence outside the artifact
directory so strict admission does not accidentally ingest benchmark summaries
as parity artifacts:

```text
validation/reports/jax_gk_parity_benchmark.json
validation/reports/jax_gk_parity_benchmark.md
```

Current local CPU run, generated with `JAX_PLATFORM_NAME=cpu`, regenerated the
three CPU artifacts in `2.963800` seconds total. Per-case timings were:

| Case | Backend | Device | Elapsed s |
|---|---|---|---:|
| `cyclone_base_case` | `cpu` | `cpu` | 2.731885 |
| `tem_kinetic_electron` | `cpu` | `cpu` | 0.106864 |
| `stable_mode` | `cpu` | `cpu` | 0.096412 |

The persisted campaign currently contains three CPU and three GPU parity
artefacts over CBC, kinetic-electron TEM, and low-drive stable-mode cases. The
strict CPU/GPU admission gate reports complete required case/backend coverage,
maximum gamma relative error `1.5386142994101046e-06`, maximum omega absolute
error `2.9658060068937786e-07`, and entries payload SHA-256
`7c7d3c7eefd5d2577579d1fd89d1fdaa056eebc13aa9d7f06f14cb1e8e755dfb`. The claim
boundary is backend parity only. These artifacts do not replace external TGLF,
GENE, GS2, CGYRO, or QuaLiKiz validation for quantitative gyrokinetic claims.

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

## Runtime admission benchmark

Run this benchmark after changes to `scpn_control.core.runtime_admission`,
`NeuroCyberneticEngine.execute_hardware_loop(...)`, `run-hardware-campaign`, or
the PyO3 `runtime_admission_snapshot()` counterpart:

```bash
taskset -c 4,5,6,7 env PYTHONPATH=src python benchmarks/bench_runtime_admission.py \
  --iterations 500 \
  --warmup 50 \
  --core-snn 4 \
  --core-z3 5 \
  --core-net 6 \
  --core-hb 7 \
  --json-out validation/reports/runtime_admission_release_20260605T000000Z.json \
  --md-out validation/reports/runtime_admission_release_20260605T000000Z.md
```

This measures launch-time admission overhead only. It is not a control-loop
hot-path benchmark and does not qualify hard real-time PCS timing by itself. A
production timing claim still requires `--runtime-admission-policy require`,
PREEMPT_RT or realtime sysfs evidence, SCHED_FIFO/SCHED_RR execution, requested
cores inside the process affinity mask, performance CPU governors, adequate
memory-lock limits, heartbeat configuration, and hard-isolated benchmark
context.

Current local regression evidence:

| Evidence | Samples | Warmup | Median | p95 | p99 | Admission result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `validation/reports/runtime_admission_release_20260605T000000Z.md` | 500 | 50 | 140.395 us | 182.384 us | 216.253 us | failed strict production admission: no PREEMPT_RT, no SCHED_FIFO/SCHED_RR, non-performance governors |

## Pulsed-shot MPC adapter regression

Use this benchmark after changes to the pulsed MPC admission boundary:

- `src/scpn_control/control/fusion_sota_mpc.py`
- `src/scpn_control/control/pulsed_scenario_scheduler_v2.py`
- `src/scpn_control/control/capacitor_bank_state.py`
- `scpn-control-rs/crates/control-control/src/mpc.rs`
- `scpn-control-rs/crates/control-python/src/lib.rs`

Run the local regression harness with explicit output paths:

```bash
PYTHONPATH=src python benchmarks/bench_pulsed_mpc_adapter.py \
  --steps 2000 \
  --warmup 200 \
  --json-out validation/reports/pulsed_mpc_adapter_local_regression.json \
  --md-out validation/reports/pulsed_mpc_adapter_local_regression.md
```

If the optional PyO3 extension was rebuilt for the current Rust source, the
Python report includes `pyo3_non_burn_mask` and
`pyo3_burn_infeasible_safe` rows. On this workstation, build the editable PyO3
extension with a target directory on `/tmp`; the repository checkout is on a
`fuseblk` volume, and maturin's rpath patching path can fail against generated
shared objects in the repository target directory.

```bash
cd scpn-control-rs/crates/control-python
../../../.venv/bin/python -m maturin develop \
  --release \
  --features io-uring \
  --target-dir /tmp/scpn_control_rs_maturin_target
```

If `patchelf` is available on `PATH` and maturin reports an ELF parse error for
`libscpn_control_rs.so`, remove that optional Python package from the virtual
environment and rerun the command above. The extension does not require
committing generated shared objects.

For soft core affinity on a developer workstation:

```bash
taskset -c 4,5 env PYTHONPATH=src python benchmarks/bench_pulsed_mpc_adapter.py \
  --steps 2000 \
  --warmup 200 \
  --evidence-class local_regression \
  --json-out validation/reports/pulsed_mpc_adapter_soft_isolated.json \
  --md-out validation/reports/pulsed_mpc_adapter_soft_isolated.md
```

The v1.1 report records Python adapter timing for non-burn masking, feasible
burn admission, and infeasible-bank safe-action replacement. Each case also
preserves the latest `scpn-control.pulsed-mpc-decision-evidence.v1`
admission digest, action digest, safe-action digest, and burn-mask digest. If
the optional PyO3 extension is installed and rebuilt with
`PyMpcController.plan_pulsed()`, the same report records Rust/PyO3 adapter
timing and evidence fields. Reports generated on a loaded workstation or with
soft affinity only must keep
`production_claim_allowed=false`; they are local regression evidence, not
target-hardware timing evidence.

Run the native Rust adapter benchmark when Rust control-surface timing changed
or when the PyO3 extension is unavailable:

```bash
cargo run --manifest-path scpn-control-rs/Cargo.toml \
  -p control-control \
  --example bench_pulsed_mpc_adapter \
  --release \
  -- \
  --steps 2000 \
  --warmup 200 \
  --json-out validation/reports/pulsed_mpc_adapter_rust_local_regression.json \
  --md-out validation/reports/pulsed_mpc_adapter_rust_local_regression.md
```

This example times the Rust `MPController.plan_pulsed()` surface directly and
writes a separate digest-bound JSON/Markdown report whose case payloads include
the same pulsed-MPC decision evidence fields. Use the Python and Rust reports
together as polyglot regression evidence.

Current PyO3-inclusive local regression evidence:

| Evidence | Cases | Median range | p99 range | Claim boundary |
| --- | --- | ---: | ---: | --- |
| `validation/reports/pulsed_mpc_adapter_pyo3_decision_evidence_python_20260604T171015Z.md` | Python + PyO3 | 40.112-873.7225 us | 57.746-1264.718 us | local regression only |
| `validation/reports/pulsed_mpc_adapter_pyo3_decision_evidence_rust_20260604T171015Z.md` | native Rust | 34.826-36.843 us | 46.76-49.0 us | local regression only |

## Multi-shot campaign regression

Use this benchmark after changes to the multi-shot orchestration boundary:

- `src/scpn_control/control/multi_shot_campaign.py`
- `src/scpn_control/control/pulsed_scenario_scheduler_v2.py`
- `src/scpn_control/control/capacitor_bank_state.py`
- `scpn-control-rs/crates/control-control/src/multi_shot_campaign.rs`
- `scpn-control-rs/crates/control-python/src/lib.rs`

The current harnesses exercise two complete shots with per-shot
`pulsed_mpc_admission_digest` evidence so Python, Rust, and PyO3 surfaces are
compared against the same digest-bound replay contract.

Python:

```bash
taskset -c 4,5 env PYTHONPATH=src python benchmarks/bench_multi_shot_campaign.py \
  --steps 2000 \
  --warmup 200 \
  --evidence-class local_regression \
  --json-out validation/reports/multi_shot_campaign_soft_isolated.json \
  --md-out validation/reports/multi_shot_campaign_soft_isolated.md
```

Rust:

```bash
taskset -c 4,5 cargo run --manifest-path scpn-control-rs/Cargo.toml \
  -p control-control \
  --example bench_multi_shot_campaign \
  --release \
  -- \
  --steps 2000 \
  --warmup 200 \
  --json-out validation/reports/multi_shot_campaign_rust_soft_isolated.json \
  --md-out validation/reports/multi_shot_campaign_rust_soft_isolated.md
```

These reports compare the Python campaign adapter, native Rust campaign kernel,
and PyO3 table bridge evidence contract. Loaded workstation reports and
soft-affinity reports are local regression evidence only.

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

## Nonlinear Cyclone Base Case Evidence

`validation/gk_nonlinear_cyclone.py` publishes schema-versioned nonlinear CBC
diagnostic and saturation-admission evidence. The report separates quick
diagnostic checks from saturated `chi_i` admission, binds the payload with a
canonical SHA-256 digest, and writes both JSON and Markdown summaries:

- `validation/reports/gk_nonlinear_cyclone.json`
- `validation/reports/gk_nonlinear_cyclone.md`

The current local benchmark passed the linear recovery, energy-conservation,
and zonal-flow diagnostics. The saturated nonlinear CBC claim remains blocked:
the V4 run used `200` steps, produced `chi_i_gB=1.6568813509166032e-09`, failed
the `1.0..5.0` CBC reference band, and had tail relative drift
`0.30041712853638713` above the configured `0.10` threshold. Use
`--require-saturation` for publication or release gates that must fail unless a
long enough saturated campaign is admitted.

## RZIP Calibration Benchmark

`validation/benchmark_rzip_calibration.py` publishes bounded local regression
evidence for the RZIP rigid-plasma vertical-stability plant. The generated
report records the declared vertical inertia, wall time constant, growth rate,
growth time, tamper-evident evidence payload SHA-256 digest, and explicit
facility-claim boundary.

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

## Free-boundary Tracking Claim-Admission Benchmark

`validation/benchmark_free_boundary_tracking_claims.py` publishes bounded
repository-regression evidence for the direct free-boundary tracking claim
boundary. The generated report records true objective residuals, response-rank
health, actuator bounds, latency-compensation status, supervisor actions, and
the explicit facility-claim boundary.

Report artefacts:

- `validation/reports/free_boundary_tracking_claims.json`
- `validation/reports/free_boundary_tracking_claims.md`

Facility free-boundary tracking claims still require documented public,
measured-replay, or external equilibrium benchmark evidence that passes the
strict admission gate.

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
The same benchmark script also writes a separate multi-step source-rollout
latency report. That path measures the JAX rollout source-gradient plus sampled
NumPy finite-difference audit used before NMPC source-rollout admission.

Report artefacts:

- `validation/reports/differentiable_transport_latency.json`
- `validation/reports/differentiable_transport_latency.md`
- `validation/reports/differentiable_transport_rollout_latency.json`
- `validation/reports/differentiable_transport_rollout_latency.md`
- `validation/reports/differentiable_transport_full_fidelity_readiness.json`
- `validation/reports/differentiable_transport_full_fidelity_readiness.md`

Admission:

```bash
python validation/validate_differentiable_transport_latency.py --require-admitted --json-out
```

The report is local latency evidence for the audited gradient-admission path.
It is not a real-time control-loop guarantee and does not replace external
transport validation. Full-fidelity differentiable-transport promotion must
also pass `transport_full_fidelity_readiness_evidence()` with bound one-step and
rollout reports, controller proof digest, equilibrium-coupled campaign
metadata, and an admitted external reference artefact.

## TORAX Code-to-Code External-Reference Evidence

`validation/code_to_code_benchmark.py` runs the local transport stack on a
declared ITER-like scenario and can optionally execute TORAX on the same
scenario. The script now emits schema-versioned JSON and Markdown evidence with
a canonical payload digest, scenario digest, external-reference status, blocked
reasons, and finite comparison metrics when TORAX is available.

Report artefacts:

- `validation/reports/code_to_code_benchmark.json`
- `validation/reports/code_to_code_benchmark.md`

Admission commands:

```bash
python validation/code_to_code_benchmark.py --with-torax
python validation/code_to_code_benchmark.py --with-torax --require-external
```

`--require-external` exits non-zero unless TORAX actually runs and the report
contains finite scpn-control and TORAX profile/comparison payloads. Reports
without TORAX remain explicit blocked evidence and do not satisfy full-fidelity
external-reference requirements. The current local evidence run executed the
scpn-control scenario path with average `Te=8.142 keV`, average `Ti=8.109 keV`,
energy-balance error `1.3548e-02`, particle-balance error `7.4357e-03`, and
blocked TORAX admission because TORAX is not installed in this environment.

## End-to-End Control Latency Evidence

`benchmarks/e2e_control_latency.py` records the full sensor, equilibrium,
transport, controller, and actuator-clamp path.  Use `--output-json` when
publishing evidence, and always supply `--target-hardware-id`,
`--target-hardware-class`, and `--rt-kernel` for Raspberry Pi, Jetson,
industrial PC, or other qualified target-hardware runs.  Reports without those
operator-qualified fields remain local latency evidence only and do not support
hardware-in-the-loop or sub-millisecond real-time claims.

Persisted reports use the `scpn-control.e2e-latency.v1` schema and include a
canonical `payload_sha256` over the latency payload. The admission validator
rejects digest tampering, non-positive iteration counts, unordered percentiles,
non-finite timing values, mismatched E2E/kernel overhead factors, unqualified
hardware metadata, and reports that alter the local-evidence claim boundary.

Before a report is cited as target-hardware evidence, run:

```bash
python validation/validate_e2e_latency_evidence.py validation/reports/e2e_control_latency.json \
  --max-e2e-p95-us 1000 --json-out
```

The validator rejects unqualified local-host metadata, missing RT-kernel
evidence, non-finite percentile data, missing claim-boundary text, and optional
P95 latency threshold regressions.

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

MAST EFM full-output baseline training is prepared through
`validation/train_mast_efm_neural_equilibrium.py`. The checked-in dry-run launch
report records the expected supervised-dataset SHA-256, current workstation
payload visibility, ML350 storage-only execution policy, and fail-closed pre-run
admission status. The companion result-template report binds the launch digest
and declares the holdout, latency, GPU-cost, and admission-certificate outputs
that a future workstation or cloud execution must publish before strict
predictive admission is requested.

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

## Neural-turbulence Claim-Admission Benchmark

`validation/benchmark_neural_turbulence_claims.py` publishes bounded local
regression evidence for the neural-turbulence claim boundary. The generated
report records the deterministic analytic-target sample count, gyro-Bohm
Q_i/Q_e/Gamma_e errors, critical-gradient activity agreement, feature-schema
contract, and explicit quantitative-claim admission status.

Generated artefacts:

- `validation/reports/neural_turbulence_claims.json`
- `validation/reports/neural_turbulence_claims.md`

Quantitative gyrokinetic, QuaLiKiz, or documented-reference turbulence claims
remain blocked until a strict reference artefact validates the same neural
weight checksum and declares Q_i, Q_e, Gamma_e, flux-relative error, and
critical-gradient metrics inside stated tolerances.

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

## Burn-control Claim-Admission Benchmark

`validation/benchmark_burn_control_claims.py` publishes bounded repository
regression evidence for the DT burn-control and alpha-heating claim boundary.
The generated report records alpha power, auxiliary power, Q, Lawson margin,
burn fraction, reactivity exponent, thermal stability, controller limits, and
the explicit reactor-claim boundary.

Report artefacts:

- `validation/reports/burn_control_claims.json`
- `validation/reports/burn_control_claims.md`

Reactor burn-control claims still require documented public, integrated
transport benchmark, or measured burn replay references for alpha power, Q,
Lawson margin, burn fraction, and reactivity-exponent agreement.

## Volt-second Claim-Admission Benchmark

`validation/benchmark_volt_second_claims.py` publishes bounded repository
regression evidence for the scenario volt-second accounting claim boundary. The
generated report records ramp, flat-top, and ramp-down flux consumption, Ejima
startup flux, bootstrap-current correction, remaining flat-top time, budget
margin, and the explicit facility-claim boundary.

Report artefacts:

- `validation/reports/volt_second_claims.json`
- `validation/reports/volt_second_claims.md`

Pulse-duration or central-solenoid commissioning claims still require documented
public, measured loop-voltage replay, or external scenario benchmark references
for total flux, flat-top duration, Ejima flux, bootstrap current, and budget
margin agreement.

## Current-drive Claim-Admission Benchmark

`validation/benchmark_current_drive_claims.py` publishes bounded repository
regression evidence for the ECCD, LHCD, and NBI current-drive claim boundary.
The generated report records grid-normalised absorbed power, total driven
current, peak current density, source powers, efficiency coefficients, NBI
slowing-down metadata, and the explicit external-claim boundary.

Report artefacts:

- `validation/reports/current_drive_claims.json`
- `validation/reports/current_drive_claims.md`

Ray-traced, Fokker-Planck, or measured-deposition current-drive claims still
require strict reference artifacts for total power, driven current, deposition
centroid, peak current density, and NBI slowing-down agreement.

## Mu-synthesis Claim-Admission Benchmark

`validation/benchmark_mu_synthesis_claims.py` publishes bounded repository
regression evidence for the static D-scaled structured-singular-value analysis
claim boundary. The generated report records plant dimensions, uncertainty
blocks, mu upper bound, robustness margin, controller gain norm, D-scalings,
closed-loop spectral abscissa, and the explicit validated-claim boundary.

Report artefacts:

- `validation/reports/mu_synthesis_claims.json`
- `validation/reports/mu_synthesis_claims.md`

Full frequency-dependent D-K synthesis claims still require documented public,
external mu-toolbox, or measured control replay references for mu upper bound,
robustness margin, controller gain, D-scaling, and closed-loop spectral-abscissa
agreement.

## Disruption-mitigation Claim-Admission Benchmark

`validation/benchmark_disruption_mitigation_claims.py` publishes deterministic
bounded ensemble evidence for the halo-current and runaway-electron mitigation
model. The generated report records ensemble seed, run count, prevention rate,
P95 halo current, P95 runaway current, mean toroidal-peaking-factor product,
ITER-limit summary, and the explicit mitigation-claim admission status.

Generated artefacts:

- `validation/reports/disruption_mitigation_claims.json`
- `validation/reports/disruption_mitigation_claims.md`

Measured disruption-mitigation claims remain blocked until strict measured,
external-benchmark, or documented public reference artefacts validate warning
lead time, mitigation outcome, halo-current envelope, runaway-beam envelope,
and tritium-breeding-ratio metrics inside stated tolerances.

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

## Multi-Shot Campaign Local Regression Evidence (2026-06-04)

The CON-C.6 multi-shot campaign orchestrator was measured on the local workstation with soft CPU affinity on cores 4 and 5. These runs are regression evidence only; they are not production hard-real-time claims because the workstation was not booted with hard core isolation, IRQ shielding, or a PREEMPT_RT kernel.

| Surface | Evidence | Samples | Warmup | Median | p95 | p99 | Max | Evidence class |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Python | `validation/reports/multi_shot_campaign_soft_isolated_20260604T131105Z.md` | 2000 | 200 | 136.581 us | 166.287 us | 193.620 us | 1589.225 us | `local_regression` |
| Rust | `validation/reports/multi_shot_campaign_rust_soft_isolated_20260604T131112Z.md` | 2000 | 200 | 2.558 us | 3.030 us | 4.666 us | 15.440 us | `local_regression` |

Digest-bound pulsed-MPC replay evidence was remeasured after adding per-shot
`pulsed_mpc_admission_digest` propagation. Each run carried two admitted MPC
decision digests through the campaign report.

| Surface | Evidence | Samples | Warmup | Median | p95 | p99 | Max | Evidence class |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Python | `validation/reports/multi_shot_campaign_pulsed_mpc_evidence_python_pyo3_20260604T172543Z.md` | 2000 | 200 | 144.769 us | 196.827 us | 244.097 us | 2333.763 us | `local_regression` |
| PyO3 | `validation/reports/multi_shot_campaign_pulsed_mpc_evidence_python_pyo3_20260604T172543Z.md` | 2000 | 200 | 10.6215 us | 14.885 us | 21.042 us | 41.502 us | `local_regression` |
| Rust | `validation/reports/multi_shot_campaign_pulsed_mpc_evidence_rust_20260604T172604Z.md` | 2000 | 200 | 2.794 us | 3.573 us | 4.536 us | 20.459 us | `local_regression` |
