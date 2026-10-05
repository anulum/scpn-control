# Benchmarks

## Current native runtime evidence package

## How to read benchmark numbers responsibly

Every benchmark in this repository carries an evidence class:

- `local_regression`: valid for comparison and engineering iteration, not a PCS
  deployment claim.
- `production_benchmark`: requires the report metadata to justify host isolation,
  scheduler context, and explicit admission conditions.

The same command can therefore be technically correct but materially insufficient
for deployment planning if the evidence class and environment metadata are not in
the right channel for the audience. Use local-reported numbers to prioritize
implementation paths; use production-admitted reports for procurement, safety, or
facility discussions.

Every persistent benchmark must run through
`tools/run_recorded_benchmark.py`. The wrapper reserves a collision-resistant
campaign before executing the producer, preserves any legacy fixed-name output,
copies every declared output into `artifacts/benchmarks/records/runs/`, and
updates a digest-bound `latest` index only after a complete successful run.
Failed runs remain available with their exit code and partial artifacts. Direct
producer execution is limited to stdout or temporary scratch; persistent
destinations fail closed when the wrapper does not own the campaign.

The fixed paths shown in commands are compatibility materialisations, not the
sole evidence copy. Each `--artifact ROLE=PATH` declaration must name every file
or directory the producer persists. Running the same command twice therefore
creates two immutable run directories and never discards the first result.

### Recorded command lifecycle

The runner accepts an executable argument vector after `--` and executes it
without a shell. Its working directory is the resolved `--repository-root`;
relative output paths use that root. `--records-root` defaults to
`artifacts/benchmarks/records` and must resolve inside the repository. Every child
inherits the native environment and standard streams, with
`SCPN_BENCHMARK_CAMPAIGN_ID` set to the reserved identifier. This same Python
runner can launch Python, Rust or other native producers.

Before launch, the runner reserves the campaign and cooperating writers' output
destinations, then retains and displaces old materializations. After the direct
child exits, it seals every recreated declared artifact and releases its output
lease. Missing artifacts cause a failed manifest; their old materializations are
restored. Failed runs retain available partial artifacts and never advance
`latest`. A successful no-op cannot reuse stale output as fresh evidence.

| Result | Runner exit | Immutable record |
| --- | --- | --- |
| Producer exits zero and recreates every output | `0` | Successful; advances `latest` |
| Producer exits nonzero | Native producer code | Failed; retains available artifacts |
| Producer exits zero with a missing output | `1` | Failed; records the missing role |
| Native executable cannot start | `127` | Failed; records the launch result |
| Interrupt caught while waiting for the child | `130` | Failed; uses native child cleanup before sealing |
| Help or invalid wrapper arguments | argparse `0` or `2` | No campaign is reserved |

POSIX signal termination gives a negative child code through the imported
`main(argv)` API; the script's `SystemExit` uses the platform's command-line
mapping. Reservation, archival or finalization errors propagate rather than
becoming the launch code. An unresolved finalization failure may retain a lease
for explicit recovery. The runner provides no timeout or descendant-process
supervisor; use it with producers whose direct child owns their lifecycle.
An interrupt does not guarantee that the OS has immediately reaped the child.
Native Windows console interruption and POSIX SIGINT have different delivery
rules, so interruption evidence must come from the actual platform.

The optional JSON object in `--measurement-json` overrides labels inferred from
`--steps`, `--iterations`, `--warmup`, `--repeats`, `--samples` and `--n-bench`.
The last spelling wins; integer spellings become integers and other values stay
strings. These labels describe the command, not the measured sample population.
The manifest's source HEAD is repository metadata, not a digest of dirty source
bytes. Successful custody and an evidence-class label alone do not admit a
scientific result, a model or a production deployment.

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
PYTHONPATH=src .venv/bin/python tools/run_recorded_benchmark.py \
  --family native-formal-modes \
  --artifact report=validation/reports/native_formal_modes.json \
  --artifact markdown=validation/reports/native_formal_modes.md \
  -- .venv/bin/python scripts/benchmark_native_formal_modes.py \
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
PYTHONPATH=src .venv/bin/python tools/run_recorded_benchmark.py \
  --family native-formal-pacing \
  --artifact report=validation/reports/native_formal_modes.json \
  --artifact markdown=validation/reports/native_formal_modes.md \
  -- .venv/bin/python scripts/benchmark_native_formal_modes.py \
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
literal `workspace_dirty=false`, and a declared yes/no value for concurrent heavy
jobs. These are report declarations: the reader does not probe the host, reject
a declared heavy-job value of true, reopen certificate bytes, authenticate the
producer or grant certified control. Required nonblank context strings and
nonempty version objects do not independently establish realtime qualification.

The standard-library reader hashes the same exact bytes it decodes. Duplicate
keys, nonfinite floating JSON tokens and overflowing exponents are refused at
any depth. A non-boolean, positive finite numeric p99 threshold is required;
invalid runtime API thresholds return FAIL before case-limit evaluation.
Unconvertible huge numeric declarations become findings rather than tracebacks.
Other latency fields, per-tick measurements, execution rows and isolation/load
contents are not recomputed. The threshold applies to each declared AOT
`avg_cycle_us.p99`, which the producer computes across repeated summaries.

The result preserves case-level admitted labels even when a global schema,
context or cross-case digest finding makes overall status FAIL. Its single
observed certificate digest may come from a rejected case. Consumers must check
overall status and independent artifact custody. Declared production flags also
remain visible on FAIL; they are not qualification. Read/decode/non-object
refusals have no report digest. Caller-relative paths and symlinks are followed;
there is no containment or input size/depth budget. Additional metadata is
unchecked apart from decoder refusals.

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
PYTHONPATH=src .venv/bin/python tools/run_recorded_benchmark.py \
  --family native-handoff \
  --artifact report=validation/reports/native_handoff_comparison.json \
  --artifact markdown=validation/reports/native_handoff_comparison.md \
  -- .venv/bin/python scripts/benchmark_native_handoff.py \
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

### Control-loop latency (canonical, reproducible)

All latency figures below are regenerated by committed benchmarks and published
side by side for the fixed CI runner and the local workstation, each with full
provenance. Reproduce with:

```bash
make bench-native-handoff

PYTHONPATH=src .venv/bin/python tools/run_recorded_benchmark.py \
  --family controller-latency \
  --artifact report=artifacts/benchmarks/controller_latency.json \
  --artifact markdown=artifacts/benchmarks/controller_latency.md \
  -- .venv/bin/python benchmarks/controller_latency.py \
    --json-out artifacts/benchmarks/controller_latency.json \
    --markdown-out artifacts/benchmarks/controller_latency.md
```

Artefacts: `validation/reports/native_handoff_comparison.json` (+ `.local.json`)
and `validation/reports/controller_latency.json` (+ `.local.json`). GitHub-hosted
runner hardware varies between runs, so the recorded `cpu_model` is part of the
provenance; the CI column below is the AMD EPYC 7763 runner of CI run
`27916153302`, the local column is an Intel i5-11600K.

**Integrated active control cycle** (SNN + MPC handoff + formal safety check + UDP
transport; 5000 steps × 7 repeats, P50/P99 µs):

| Path | CI (EPYC 7763) | Local (i5-11600K) |
| --- | ---: | ---: |
| Python-orchestrated | 9.36 / 10.45 | 4.36 / 4.73 |
| Native (Rust) | 5.62 / 6.11 | 2.85 / 3.41 |

**Per-controller step latency** (isolated step, every available backend; P50/P99
µs). Backends whose dependency/binding is absent are listed so nothing is omitted:

| Controller | Backend | CI (EPYC 7763) | Local (i5-11600K) |
| --- | --- | ---: | ---: |
| PID | numpy | 0.42 / 0.52 | 0.25 / 0.56 |
| PID | rust | unavailable (PyPIDController not built) | unavailable |
| SNN | numpy | 23.90 / 43.14 | 13.29 / 26.56 |
| SNN | rust | 0.88 / 0.99 | 0.56 / 1.08 |
| MPC (Np=10) | acados | 10679 / 11393 | 6128 / 10417 |
| MPC (Np=10) | osqp | 16846 / 18090 | 9202 / 15622 |
| MPC (Np=10) | internal QP | 18847 / 25172 | 10460 / 25058 |
| MPC (Np=10) | scipy | 23302 / 24416 | 12750 / 21701 |
| MPC (Np=10) | casadi | 78887 / 82806 | 54573 / 88107 |
| H-infinity | numpy (historical general state-space) | 29.43 / 53.42 | 13.27 / 29.81 |
| H-infinity | rust (historical 2-state approximation) | 1.39 / 1.49 | 1.05 / 5.28 |

The historical PID row labelled `numpy` measured a pure Python fallback. The
current benchmark names that backend `python`; the optional `PyPIDController`
binding is still unavailable. A separate local, non-isolated 2026-09-23 run of
the current PID arithmetic used the same sine-error sequence and gains for
seven repeats of 5,000 steps after 500 warm-up steps. Median-across-repeat
P50/P99 was **1.014/2.449 µs** for Python and **0.055/0.073 µs** for the direct
release-built native Rust PID. The native result comes from
`control-control/examples/bench_pid.rs`, not from a PyO3 call; these local
numbers do not replace the CI table or establish a real-time bound.

The retained H-infinity rows above are historical records from unlike plants
and unlike controller algorithms; they do **not** support a Python/Rust speedup
claim. Current benchmark code supplies both runtimes with one normalized
two-state plant and the exact same Python-admitted DGKF realization. A new
side-by-side number becomes current evidence only after an isolated recorded
campaign; ordinary workstation runs remain orientation-only and do not
overwrite these historical records.

All five MPC QP backends are now measured (acados is built in the
benchmark-nightly workflow and locally). Every backend's real-time-iteration tick
is millisecond-scale, with acados the fastest (≈10.7 ms CI / 6.1 ms local). The
microsecond MPC regime is not reachable through the Python `step_rti` path with
any QP backend: per-stage set/get marshalling across the Python↔C boundary
dominates the tick. A microsecond MPC tick would require a non-Python (C/Rust)
control loop, not merely a compiled QP solver.

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
campaign="$(date -u +%Y%m%dT%H%M%S.%NZ)-capacitor-bank"

PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family capacitor-bank-python \
  --campaign-id "$campaign" \
  --artifact report=validation/reports/capacitor_bank_energy_python.json \
  --artifact markdown=validation/reports/capacitor_bank_energy_python.md \
  -- python benchmarks/bench_capacitor_bank_energy.py \
  --steps 500 \
  --warmup 50 \
  --discharge-steps 200 \
  --dt-s 1.0e-7 \
  --json-out validation/reports/capacitor_bank_energy_python.json \
  --markdown-out validation/reports/capacitor_bank_energy_python.md

PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family capacitor-bank-rust \
  --campaign-id "$campaign" \
  --artifact report=validation/reports/capacitor_bank_energy_rust.json \
  --artifact markdown=validation/reports/capacitor_bank_energy_rust.md \
  -- cargo run --release --manifest-path scpn-control-rs/Cargo.toml \
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

The persisted reader validates declarations without rerunning this campaign:
finite numeric/token domains, digests, drift, spectra and named coverage. A
reader refusal test or copied historical artifact does not establish new
backend timing or physical parity. Its digests bind canonical JSON, not raw
bytes or authenticated device/source metadata. Physical solver kernels,
producer inputs and the recorded CPU/GPU artifacts remain the timing sources;
standalone reader IO/refusal corrections do not change those computations.

Run:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family jax-gk-parity-auto \
  --artifact parity=validation/reports/jax_gk_parity \
  --artifact report=validation/reports/jax_gk_parity_benchmark.json \
  --artifact markdown=validation/reports/jax_gk_parity_benchmark.md \
  -- python validation/benchmark_jax_gk_parity.py --json-out

JAX_PLATFORM_NAME=cpu PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family jax-gk-parity-cpu \
  --artifact parity=validation/reports/jax_gk_parity \
  --artifact report=validation/reports/jax_gk_parity_benchmark.json \
  --artifact markdown=validation/reports/jax_gk_parity_benchmark.md \
  -- python validation/benchmark_jax_gk_parity.py --json-out
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

Recorded local CPU run, generated with `JAX_PLATFORM_NAME=cpu`, regenerated the
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
taskset -c 4,5,6,7 env PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family runtime-admission \
  --artifact report=validation/reports/runtime_admission_release_20260605T000000Z.json \
  --artifact markdown=validation/reports/runtime_admission_release_20260605T000000Z.md \
  -- python benchmarks/bench_runtime_admission.py \
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

- `src/scpn_control/control/fusion_neural_mpc.py`
- `src/scpn_control/control/pulsed_scenario_scheduler_v2.py`
- `src/scpn_control/control/capacitor_bank_state.py`
- `scpn-control-rs/crates/control-control/src/mpc.rs`
- `scpn-control-rs/crates/control-python/src/lib.rs`

Run the local regression harness with explicit output paths:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family pulsed-mpc-adapter \
  --artifact report=validation/reports/pulsed_mpc_adapter_local_regression.json \
  --artifact markdown=validation/reports/pulsed_mpc_adapter_local_regression.md \
  -- python benchmarks/bench_pulsed_mpc_adapter.py \
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
taskset -c 4,5 env PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family pulsed-mpc-adapter-soft-affinity \
  --artifact report=validation/reports/pulsed_mpc_adapter_soft_isolated.json \
  --artifact markdown=validation/reports/pulsed_mpc_adapter_soft_isolated.md \
  -- python benchmarks/bench_pulsed_mpc_adapter.py \
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
campaign="$(date -u +%Y%m%dT%H%M%S.%NZ)-multi-shot"
taskset -c 4,5 env PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family multi-shot-python \
  --campaign-id "$campaign" \
  --artifact report=validation/reports/multi_shot_campaign_soft_isolated.json \
  --artifact markdown=validation/reports/multi_shot_campaign_soft_isolated.md \
  -- python benchmarks/bench_multi_shot_campaign.py \
  --steps 2000 \
  --warmup 200 \
  --evidence-class local_regression \
  --json-out validation/reports/multi_shot_campaign_soft_isolated.json \
  --md-out validation/reports/multi_shot_campaign_soft_isolated.md
```

Rust:

```bash
taskset -c 4,5 env PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family multi-shot-rust \
  --campaign-id "$campaign" \
  --artifact report=validation/reports/multi_shot_campaign_rust_soft_isolated.json \
  --artifact markdown=validation/reports/multi_shot_campaign_rust_soft_isolated.md \
  -- cargo run --manifest-path scpn-control-rs/Cargo.toml \
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

## Manual nonlinear GK software check

This short source-checkout example uses the actual JAX CPU backend and the
runner's existing initial-state seed. It checks saved diagnostics without
executing the fixed long ES/EM or Dimits presets:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' \
PYTHONPATH=.:src python - <<'PY'
from tools.em_and_dimits import run_jax

result = run_jax(
    "short CPU software check",
    n_kx=8, n_ky=4, n_theta=8, n_vpar=4, n_mu=2,
    n_steps=2, save_interval=1,
)
assert result["time_final"] > 0
assert result["chi_i_gB"] is not None
print(result)
PY
```

An installed JAX backend and this checkout's Python dependencies are required.
The two saved samples can set `converged=True`; this only reflects the solver's
finite-sample flag. The displayed transport ratio and endpoint growth are raw
diagnostics, not validated physical transport, a Dimits-shift measurement or
facility admission. Wall-clock timings vary with runtime/device state. Calling
`run_jax` writes no report; running the module's fixed `main()` selects much
larger workloads and replaces `gpu_results/em_and_dimits.json` after all cases
finish. That raw output has no authenticated source or campaign custody.

The separate fixed comparisons use different numerical presets:
`tools.dimits_256_fixed` uses 256 kx modes, 10000 steps and hyper coefficient
0.02; `tools.dimits_long` uses 128 kx modes, 20000 steps and coefficient 0.2.
Both compare drives 3.0 and 6.9 with save interval 200. Their entry points
cannot be substituted for the short configurable example above, and these
different workloads do not establish a controlled timing or convergence
comparison. See their [API and output contracts](api.md#manual-nonlinear-jax-gk-experiments).

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

## Vacuum diagnostic report software check

This source-checkout example runs the actual public writer with a temporary
working directory. Its report paths are outside the checkout's persistent
evidence roots, so the existing scratch-path policy permits the write. It does
not replace the canonical scientific reports:

```bash
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python - <<'PY'
import json
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory

from validation.benchmark_free_boundary import main

with TemporaryDirectory() as scratch, chdir(scratch):
    main()
    payload = json.loads(Path("validation/reports/free_boundary_benchmark.json").read_text())
    assert payload["helmholtz"]["pass"] is None
    assert payload["helmholtz"]["assessment"] == "diagnostic_only_off_axis_sample"
    print(payload["helmholtz"])
PY
```

The Helmholtz sample is off-axis and its reference is on-axis. Their differing
values are diagnostic, with no same-point tolerance or quantitative PASS.
Single-coil equality uses the solver's own expression; the reported X-point is
a grid-gradient minimum with a Z-only test. These checks provide no independent
flux normalization or validated magnetic-null witness. The raw diagnostic API
retains its historical qualitative marker; the report writer explicitly records
that marker as unassessed. The [API contract](api.md#vacuum-software-diagnostic-reports)
describes output order, temporary config cleanup and native error propagation.

## Free-boundary Tracking Claim-Admission Benchmark

`validation/benchmark_free_boundary_tracking_claims.py` publishes bounded
repository-regression evidence for the direct free-boundary tracking claim
boundary. The generated report records true objective residuals, response-rank
health, actuator bounds, latency-compensation status, supervisor actions, and
the explicit facility-claim boundary.

Report artefacts:

- `validation/reports/free_boundary_tracking_claims.json`
- `validation/reports/free_boundary_tracking_claims.md`

The shipped producer uses a fixed 8-by-4 linear current response, with a zero
Psi grid. It runs five real controller steps with gain 0.5, one-step measurement
latency and gain-0.75 compensation. Its config filename is only a report label;
no configuration file or Grad-Shafranov solver is used by this fixture.

The reference metadata validator checks declared fields and tolerances; it does
not verify the digest against artifact bytes or independently replay the metrics.
Facility free-boundary tracking claims remain blocked pending independently
verified reference evidence and device-specific admission.

## EFIT-lite Claim-Admission Benchmark

`validation/benchmark_efit_lite_claims.py` publishes bounded synthetic
regression evidence for the fixed-boundary EFIT-lite reconstruction path. The
generated report records diagnostic provenance, grid shape, flux-loop and
B-probe counts, Rogowski radius, reconstructed current, q95, beta_pol, li, and
the explicit facility-claim boundary. Schema 2 also records the geometric or
Picard termination status and the final Picard relative change when present.

Report artefacts:

- `validation/reports/efit_lite_claims.json`
- `validation/reports/efit_lite_claims.md`

The current builder does not admit facility claims from caller-declared
references, even when all metric tolerances pass. Facility equilibrium claims
require independently obtained and byte-pinned EFIT/P-EFIT or measured
references, diagnostic/shot/time provenance, and fixed comparison conventions
for psi, Ip, q95, beta_pol, and li.

## Kinetic EFIT Claim-Admission Benchmark

`validation/benchmark_kinetic_efit_claims.py` publishes bounded synthetic
regression evidence for kinetic pressure, q-profile, anisotropy, diagnostic
provenance, profile provenance, fast-ion provenance, MSE calibration, and
normalised elliptic-rho interpolation geometry.

Report artefacts:

- `validation/reports/kinetic_efit_claims.json`
- `validation/reports/kinetic_efit_claims.md`

The producer passes empty magnetic measurements. The synthetic flux loop and
radial probe at (6, 1) m lie inside its 33-by-33 R=4..8 m, Z=-3..3 m grid.
The fixed temperature/density constraints produce 50-point pressure profiles;
the prescribed 5-degree MSE pitch gives q_axis=1+5/90 and q_edge=q_axis+2.
Reconstruction chi-squared, iteration count and wall-time fields are constants,
not measured diagnostics or performance. No reference is supplied.

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


### Temporary manufactured mesh and differentiable reader examples

Run these source-checkout examples from the repository root. The first calls
the actual independent manufactured mesh API; it does not invoke FusionKernel
or overwrite reports. The full fixed command additionally runs65/129 grids
and writes sequential caller-relative reports; execute it in an isolated cwd.

```python
from validation.mesh_convergence_study import run_solovev_benchmark

empty = run_solovev_benchmark(5, 7, max_iter=0)
assert empty["iterations"] == 0
coarse = run_solovev_benchmark(17, 17, max_iter=2000)
fine = run_solovev_benchmark(33, 33, max_iter=4000)
assert fine["nrmse"] < coarse["nrmse"]
print("Manufactured stencil only:", coarse["nrmse"], fine["nrmse"])
```

The second runs the complete byte-identical installed-CPU-JAX producer, with
actual audited timing and software-only immutable report custody. Output paths
resolve from the copied source location; external/reference/formal evidence is
absent, so full-fidelity readiness remains blocked. The explicitly authored
invalid-audit derivative is reader input, not a new observation. Timing values
are local diagnostics, not controlled comparisons or hardware guarantees.

```python
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

from validation.validate_differentiable_transport_latency import validate_differentiable_transport_latency

root = Path.cwd()
with tempfile.TemporaryDirectory(prefix="control-differentiable-reader-") as directory:
    temporary = Path(directory)
    source = temporary / "validation/benchmark_differentiable_transport_latency.py"
    source.parent.mkdir()
    original = root / "validation/benchmark_differentiable_transport_latency.py"
    shutil.copyfile(original, source)
    assert source.read_bytes() == original.read_bytes()
    names = ["differentiable_transport_latency",
             "differentiable_transport_rollout_latency",
             "differentiable_transport_full_fidelity_readiness"]
    command = [sys.executable, str(root / "tools/run_recorded_benchmark.py"),
        "--repository-root", str(temporary), "--records-root", "records",
        "--family", "differentiable-reader-example", "--campaign-id", "local-software",
        "--evidence-class", "software_boundary_test"]
    for role, name in zip(["one", "rollout", "readiness"], names, strict=True):
        command += ["--artifact", f"{role}=validation/reports/{name}.json"]
    command += ["--", sys.executable, str(source)]
    subprocess.run(command, cwd=temporary, env=dict(os.environ,
        PYTHONPATH=str(root / "src"), JAX_PLATFORMS="cpu",
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True, text=True, check=True)
    one, rollout, readiness = (temporary / "validation/reports" / (name + ".json") for name in names)
    result = validate_differentiable_transport_latency(
        one, rollout, readiness_report=readiness, require_admitted=True)
    assert result["status"] == "pass" and result["admitted_reports"] == 2
    assert result["full_fidelity_ready"] is False
    payload = json.loads(one.read_text())
    payload["audit"]["passed"] = False
    derivative = temporary / "authored-invalid-audit.json"
    derivative.write_text(json.dumps(payload), encoding="utf-8")
    refused = validate_differentiable_transport_latency(derivative, rollout)
    assert refused["status"] == "fail" and refused["admitted_reports"] == 1
    assert refused["entries"][0]["status"] == "fail"
    print("Local declarations only; full-fidelity readiness remains blocked")
```

require_admitted concerns valid local latency declarations and refuses valid
blocked latency reports. It does not require full-fidelity readiness. Invalid
entries have statusfail and do not count as admitted; aggregate readiness is
False on errors. Runtime and readiness hashes are declarations checked for
syntax/presence, without replay or source/operator authentication.

## Differentiable Scenario Readiness Evidence

The coupled differentiable scenario facade records bounded evidence for an
analytic Solov'ev-form flux surface coupled to the four-channel differentiable
transport rollout. The report audits gradients with respect to both equilibrium
parameters and source schedules, records local non-isolated timing context, and
keeps the claim blocked until the physics-traceability gate is satisfied.

Report artefacts:

- `validation/reports/differentiable_scenario_readiness.json`
- `validation/reports/differentiable_scenario_readiness.md`

Admission:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family differentiable-scenario \
  --artifact report=validation/reports/differentiable_scenario_readiness.json \
  --artifact markdown=validation/reports/differentiable_scenario_readiness.md \
  -- python validation/benchmark_differentiable_scenario.py
python validation/validate_differentiable_scenario.py --json-out
```

The timing field is local admission evidence only. It is not an isolated
hardware benchmark, real-time guarantee, external integrated-modelling result,
or facility-control claim.

## TORAX Code-to-Code External-Reference Evidence

The source-checkout command runs actual CONTROL transport and optionally the
installed TORAX provider. Shared declared linear initial profiles and fixed dt
now map consistently; n_rho sets the local radial solver length. Configured
transport closures, current/geometry and source channels still differ, so
finite diagnostics do not admit a physical transport reference. Schema v3
records scenario/payload digests, model limitations, status and declared
diagnostic_comparison_available separately from physical admission.

Report artefacts:

- `validation/reports/code_to_code_benchmark.json`
- `validation/reports/code_to_code_benchmark.md`

Admission commands:

```bash
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family torax-code-to-code \
  --artifact report=validation/reports/code_to_code_benchmark.json \
  --artifact markdown=validation/reports/code_to_code_benchmark.md \
  -- python validation/code_to_code_benchmark.py --with-torax

PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family torax-code-to-code-required \
  --artifact report=validation/reports/code_to_code_benchmark.json \
  --artifact markdown=validation/reports/code_to_code_benchmark.md \
  -- python validation/code_to_code_benchmark.py --with-torax --require-external
```

The ordinary command writes both outputs and exits zero for a successful local
diagnostic even when the optional provider is missing. With
`--require-external`, exit one follows both outputs while physical admission
remains blocked. A real provider run would supply diagnostic metrics but would
not resolve the current source-backed model limitations. Historical report
numbers are retained in their original artifacts; they are not a fresh run of
this adapter or a controlled cross-code performance comparison.

The following temporary source-checkout example observes actual initialization,
executes two real transport steps and compares those local observations as
declared reader inputs. It does not invoke or substitute TORAX:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from validation.code_to_code_benchmark import (
    ITER_SCENARIO,
    build_comparison_report,
    compare_transport_profiles,
    run_local_transport,
)
from validation.code_to_code_torax import write_torax_config

initial = run_local_transport(dict(ITER_SCENARIO, n_rho=13, n_steps=0, t_final=0.0))
evolved = run_local_transport(dict(ITER_SCENARIO, n_rho=13, n_steps=2, t_final=0.02))
comparison = compare_transport_profiles(evolved, initial)
report = build_comparison_report(comparison, ITER_SCENARIO, requested_torax=False)
assert report["external_reference"]["admitted"] is False
assert comparison["comparison"]["Te_rmse_keV"] > 0
with TemporaryDirectory() as directory:
    write_torax_config(Path(directory) / "config.py", ITER_SCENARIO)
```

## End-to-End Control Latency Evidence

`benchmarks/e2e_control_latency.py` records the full sensor, equilibrium,
transport, controller, and actuator-clamp path.  Use `--output-json` when
publishing evidence, and always supply `--target-hardware-id`,
`--target-hardware-class`, and `--rt-kernel` for Raspberry Pi, Jetson,
industrial PC, or other qualified target-hardware runs.  Reports without those
operator-qualified fields remain local latency evidence only and do not support
hardware-in-the-loop or sub-millisecond real-time claims.

Persisted reports use the `scpn-control.e2e-latency.v1` schema and include a
canonical `payload_sha256` over the latency payload. Each persisted report must
record the benchmark command, generated UTC timestamp, inherited CPU affinity,
isolation method, host load before and after the run, CPU governor state when
available, and whether concurrent heavy jobs were observed. The admission
validator rejects digest tampering, non-positive iteration counts, unordered
percentiles, non-finite timing values, mismatched E2E/kernel overhead factors,
missing benchmark context, unqualified hardware metadata, and reports that alter
the local-evidence claim boundary.

Before a report is cited as target-hardware evidence, run:

```bash
python validation/validate_e2e_latency_evidence.py validation/reports/e2e_control_latency.json \
  --max-e2e-p95-us 1000 --json-out
```

The validator rejects unqualified local-host metadata, missing RT-kernel
evidence, non-finite percentile data, missing claim-boundary text, and optional
P95 latency threshold regressions.


### Temporary E2E reader example

Run this source-checkout example from the repository root. It invokes the actual
Python sensor, one-SOR-step equilibrium, transport, H-infinity and actuator-clamp
path, retaining its report outside canonical scientific destinations. The
record wrapper identifies software-only custody; unqualified target labels
remain unchanged. Three samples with the producer's floor-index percentiles
are insufficient to establish a tail distribution, hardware deadline or
comparative performance. No historical report is replaced.

The reader requires an explicitly zero-offset UTC timestamp and refuses NaN,
infinite, negative, boolean or nonnumeric budgets before opening the report.
A zero-microsecond budget is valid and rejects positive p95; equality passes.
Report checks do not authenticate declared target labels, operator approval,
clocks, actual isolation or raw samples. The parsed-payload self-checksum can
be recomputed by any author; it is not a signature.

```python
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

from validation.validate_e2e_latency_evidence import validate_e2e_latency_evidence

root = Path.cwd()
with tempfile.TemporaryDirectory(prefix="control-e2e-reader-") as temporary:
    output = Path(temporary)
    report = output / "actual.json"
    command = [
        sys.executable, str(root / "tools/run_recorded_benchmark.py"),
        "--repository-root", str(output), "--records-root", "records",
        "--family", "e2e-reader-example", "--campaign-id", "local-software",
        "--evidence-class", "software_boundary_test", "--artifact", f"report={report}",
        "--", sys.executable, str(root / "benchmarks/e2e_control_latency.py"),
        "--iterations", "3", "--warmup", "1", "--output-json", str(report), "--json",
    ]
    process = subprocess.run(command, cwd=root,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True, text=True, check=True)
    observed = json.loads(report.read_text(encoding="utf-8"))
    result = validate_e2e_latency_evidence(report, require_target_hardware=False)
    assert result.status == "pass" and result.p95_us is not None
    assert observed["production_claim_allowed"] is False
    assert validate_e2e_latency_evidence(
        report, require_target_hardware=False, max_e2e_p95_us=result.p95_us
    ).status == "pass"
    cli = subprocess.run([
        sys.executable, str(root / "validation/validate_e2e_latency_evidence.py"),
        str(report), "--allow-local-unqualified", "--json-out",
    ], cwd=root, capture_output=True, text=True, check=True)
    assert json.loads(cli.stdout)["status"] == "pass"
    print("Local software observation only:", result.p95_us, "microseconds")
```

## VMEC-lite Claim-Admission Benchmark

`validation/benchmark_vmec_lite_claims.py` publishes bounded synthetic
regression evidence for the fixed-boundary VMEC-lite spectral facade. The
generated report records Fourier truncation, field periods, pressure and
rotational-transform profile provenance, current-assumption provenance,
positive sampled major-radius bounds, force residual, q-domain, the local
non-isolated benchmark context, and the sealed VMEC-lite geometry-validation
payload SHA-256.

Report artefacts:

- `validation/reports/vmec_lite_geometry.json`
- `validation/reports/vmec_lite_geometry.md`
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
payload visibility, storage-host storage-only execution policy, and fail-closed pre-run
admission status. The companion result-template report binds the launch digest
and declares the holdout, latency, GPU-cost, and admission-certificate outputs
that a future workstation or cloud execution must publish before strict
predictive admission is requested.

The trainer now checks current plan/source self-digests, selected NPZ shape
and shot-held-out splits before fitting. Both preserved MAST source audits
currently have stale self-digests, and the declared physical tensor payload is
absent locally; current source/execute admission remains FAIL. These checks do
not rerun or refresh historical scientific benchmark numbers. Small retained
format regressions exercise the actual NumPy baseline but establish no MAST
throughput, GPU billing or physical predictive performance. The ridge intercept
count is assigned directly to preserve the unpenalised intercept at large
finite regularisation. The launch format has no measured numerical benchmark;
PCA/ridge arithmetic remains a Python implementation.

The MAST dataset producer verifies converter declarations and selected NPZ
custody before assembling inputs. Hash verification and decoding share one
compressed-file byte capture rather than two pathname observations. This
adds temporary storage for one compressed bundle and closes input substitution
between checks; no performance measurement is claimed. The supervised trainer
uses the same verified-NPZ reader for its input dataset. The producer and trainer
also share final N × 12 feature validation. Stored wider-dtype observations
must remain finite after float64 conversion; this is input admission, not a
change to the PCA/ridge numerical formulas or a measured speed improvement. Its stable pressure/RMS mean and median,
descending-grid normalisation and valid LCFS compaction are covered through
actual public format regressions. This producer has no measured benchmark or
native language counterpart; these changes do not refresh historical physical
performance numbers. The reported external corpus remains unavailable locally.

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

The preset fixes I_p=15 MA, B_t=5.3 T, P_heat=50 MW, n_e=10.1 in 1e19 m^-3,
R=6.2 m, A=3.1, kappa=1.7 and M=2.5 AMU, with equal D/T fractions and no
dilution. A local NumPy Generator uses seed 31 for 256 samples. The chain samples
scaling-law and transport/pedestal/boundary proxies; it runs no equilibrium or
transport PDE solver, and supplies no calibration reference. Central tau_E is
in seconds, fusion power in MW and Q is dimensionless.

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

Numerically matched measured-discharge, documented-public, external
particle-balance, or facility-replay references remain bounded comparison
evidence when supplied by the caller. The schema version 2 report records the
comparison result separately. Facility admission remains closed until an
independent reference witness can be verified.

## Bounded particle report software check

From this source checkout, the following example copies the complete producers
to a temporary output root and runs their actual recorded wrapper. It exercises
bounded declarations without replacing the checkout's scientific reports. The
copied root's custody metadata does not establish the canonical package source
identity or an external physical reference.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python - <<'PY'
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

checkout = Path.cwd()
with TemporaryDirectory() as scratch:
    root = Path(scratch)
    (root / "validation").mkdir()
    claims = {
        "density_control_claims": "facility_density_claim_allowed",
        "current_drive_claims": "external_claim_allowed",
        "burn_control_claims": "reactor_claim_allowed",
        "volt_second_claims": "facility_claim_allowed",
        "orbit_following_claims": "external_orbit_claim_allowed",
    }
    for stem, claim in claims.items():
        producer = root / "validation" / f"benchmark_{stem}.py"
        shutil.copyfile(checkout / "validation" / producer.name, producer)
        subprocess.run(
            [sys.executable, str(checkout / "tools/run_recorded_benchmark.py"),
             "--repository-root", str(root), "--records-root", "records",
             "--family", stem, "--evidence-class", "software_boundary_example",
             "--artifact", f"report=validation/reports/{stem}.json",
             "--artifact", f"markdown=validation/reports/{stem}.md",
             "--", sys.executable, str(producer)],
            env=dict(os.environ, PYTHONPATH=str(checkout / "src")),
            check=True,
        )
        payload = json.loads((root / "validation/reports" / f"{stem}.json").read_text())
        assert payload[claim] is False
        print(stem, payload["claim_status"], payload[claim])
PY
```

The density case takes an imposed-source CFL-limited step and computes its
controller command separately from the initial profile. It is not a closed-loop
replay. Current-drive samples fixed bounded source formulae without external
reference bytes. Both remain declarations with closed physical admission; the
copied reports and records disappear when the example's temporary scope ends.

Burn adds static profile diagnostics and one controller update. Volt-second
adds scalar scenario flux accounting. Orbit uses a first-loss formula and
fixed declared particle counts; it executes no trajectory ensemble. These
outputs likewise keep reactor/facility/external orbit admission closed. Their
[specific API contracts](api.md#bounded-burn-volt-second-and-orbit-report-producers)
identify units, fixed presets, state and sequential write order.

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
margin agreement. A caller-supplied reference dictionary does not meet that
gate: the current public builder records its metadata as unverified and keeps
facility admission false until source bytes and comparison metrics are bound.

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

## Static Structured-Mu Claim-Admission Benchmark

`validation/benchmark_static_mu_analysis_claims.py` publishes bounded repository
regression evidence for the static D-scaled structured-singular-value analysis
claim boundary. The generated report records plant dimensions, uncertainty
blocks, the zero-frequency mu upper bound, its reciprocal, controller gain norm,
D-scalings, closed-loop spectral abscissa, and the explicit validated-claim
boundary. Legacy schema-v1 field names remain in the JSON for compatibility.

Report artefacts:

- `validation/reports/static_mu_analysis_claims.json`
- `validation/reports/static_mu_analysis_claims.md`

The previous `mu_synthesis_claims` pair remains retained as historical evidence;
new runs do not overwrite it.

Full frequency-dependent D-K synthesis claims still require documented public,
external mu-toolbox, or measured control replay references for frequency-wise mu
bounds, controller synthesis, D-scale fitting, and closed-loop agreement.

## Disruption-mitigation Claim-Admission Benchmark

`validation/benchmark_disruption_mitigation_claims.py` publishes deterministic
bounded ensemble evidence for the halo-current and runaway-electron mitigation
model. The generated report records ensemble seed, run count, prevention rate,
P95 halo current, P95 runaway current, mean toroidal-peaking-factor product,
ITER-limit summary, and the explicit mitigation-claim admission status.

Generated artefacts:

- `validation/reports/disruption_mitigation_claims.json`
- `validation/reports/disruption_mitigation_claims.md`

Measured disruption-mitigation claims remain blocked until an independent
comparison against measured, external-benchmark, or documented public source
data validates warning lead time, mitigation outcome, halo-current envelope,
runaway-beam envelope, and tritium-breeding-ratio metrics. A metadata-only
reference artefact cannot admit a claim.

The phase-ordering side of the same tracker is covered by the bounded validation
artefacts `validation/reports/disruption_sequence.json` and
`validation/reports/disruption_sequence.md`, generated by
`validation/validate_disruption_sequence.py`. Those artefacts are not performance
benchmarks and keep facility claims blocked pending labelled disruption-window
evidence.

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

The multi-shot campaign orchestrator was measured on the local workstation with
soft CPU affinity on cores 4 and 5. These runs are regression evidence only;
they are not production hard-real-time claims because the workstation was not
booted with hard core isolation, IRQ shielding, or a PREEMPT_RT kernel.

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

## How to interpret benchmark evidence

Benchmark figures should be treated as a three-value statement: measured latency, reproducibility context, and admissibility.

- Measured latency comes from the reported timing columns.
- Reproducibility context comes from host, governor, and core-affinity context in the artifact.
- Admissibility comes from whether the run appears in a validator-linked release note or validation page.

Do not merge local-regression and isolated-target reports without marking the context difference in your interpretation.

## Evidence-first execution checklist

Before sharing any benchmark in planning material, check each report against:

1. **Scope**: the benchmark class matches the command surface you are discussing
   (`native_formal_*`, `runtime_admission`, `multi_shot_campaign`, etc.).
2. **Context**: host load, isolation, core assignment, and timing-mode metadata are
   present and readable.
3. **Admission**: claim level is mapped to `local_regression`, `production_benchmark`,
   `reference_validated`, or stronger where available.
4. **Version lock**: package version and artifact SHA are retained.

If all four checks pass, the benchmark can enter decision discussions.
If any check fails, keep it in implementation-only iteration mode.

## Benchmark regression gate

The polyglot suite runner (`tools/run_benchmark_suite.py`) executes the
registered Python/Rust comparison benchmarks — currently the capacitor-bank
discharge ledger — and emits a `scpn-control.benchmark-regression.v1` report:
per-benchmark, per-language p50/p95/p99 latency and throughput, plus provenance
(CPU model, Rust release profile read from the workspace manifest, commit digest,
CPU affinity, load average, peak RSS, and whether the Rust backend was built).

The gate (`tools/benchmark_regression_gate.py`) compares a fresh report against
the tracked baseline (`benchmarks/baselines/capacitor_bank.json`) under an
explicit threshold policy (`benchmarks/regression_thresholds.toml`). Latency and
memory metrics are upper-bounded (`current <= baseline * ratio`); throughput is
lower-bounded (`current >= baseline * ratio`). It fails closed: a missing report
or baseline, a tampered baseline (its `baseline_sha256` no longer matches its
metrics), a tampered report (`payload_sha256` mismatch), a benchmark/language
metric present in the baseline but absent from the report, or a metric with no
threshold policy are all failures.

These are local declared-record checks. Matching digests do not authenticate
the producer or host; declared CPU equality does not prove comparable execution
conditions. The command's policy and verdict helpers retain their original
arithmetic in `tools.benchmark_gate_policy` and `tools.benchmark_gate_verdict`.
The original `tools.benchmark_regression_gate` Python imports remain available.
Selected report, baseline and threshold-file aliases cannot receive JSON verdict
output, including resolved symlinks and existing hard links. Unrelated existing
outputs may be replaced. Supported input/output/custody failures return 1 even
with `--evidence-only`; generated rejection verdicts alone may return 0 in that
mode. Output is finite, sorted UTF-8 JSON with a trailing newline. Sequential
checks provide no coherent snapshot, lock or atomic write.

```bash
# Record a fresh run. The command prints its immutable manifest path.
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family polyglot-suite \
  --artifact report=artifacts/benchmarks/report.json \
  -- python tools/run_benchmark_suite.py \
    --steps 400 --warmup 40 \
    --json-out artifacts/benchmarks/report.json

# Gate the compatibility materialisation against the promoted baseline.
PYTHONPATH=src python tools/run_recorded_benchmark.py \
  --family benchmark-regression-gate \
  --artifact verdict=artifacts/benchmarks/gate_verdict.json \
  -- python tools/benchmark_regression_gate.py \
    --report artifacts/benchmarks/report.json \
    --baseline benchmarks/baselines/capacitor_bank.json \
    --json-out artifacts/benchmarks/gate_verdict.json

# Baseline promotion is separate and requires the exact immutable source digest.
PYTHONPATH=src python tools/promote_benchmark_baseline.py \
  --source-manifest artifacts/benchmarks/records/runs/polyglot-suite/<campaign-id>/manifest.json \
  --artifact-role report \
  --expected-source-sha256 <sha256-from-run-manifest> \
  --baseline benchmarks/baselines/capacitor_bank.json \
  --suite capacitor-bank \
  --authority-ref <review-or-change-reference> \
  --hardware-compatibility matched
```

Absolute-latency comparison is only valid on the same CPU as the baseline: the
gate emits a `hardware_mismatch` failure when the report and baseline CPU models
differ. The nightly `Benchmark nightly` workflow therefore runs the gate in
`--evidence-only` mode on GitHub-hosted runners — it collects and uploads the
report and verdict without blocking — while real gating is intended on declared
or self-hosted hardware whose baseline was captured on the same CPU. The
committed baseline carries `production_claim_allowed: false`; it is a local
regression guard, not a published performance claim.

## Practical use and scope

Use this page for benchmark interpretation and replay evidence, not as a substitute for deployment qualification.

- Record benchmark context when comparing results across Python, Rust, and benchmark modes.
- Use this with `docs/validation.md` for claim boundaries and with `docs/physics_traceability.md` for evidence lineage.
- Treat every number as environment- and mode-dependent unless explicitly documented otherwise.

### Output freshness and concurrent campaigns

The recorded runner reserves every declared output before starting the producer.
Existing files and directories are first copied into the verified legacy archive,
then moved into the invocation's `prior-output` directory. The producer must
recreate every declared destination. Writing identical deterministic bytes is
valid; merely leaving an older file in place cannot establish a successful run.
Directory producers must recreate the directory and its complete result set.

Disjoint output destinations can run concurrently. Identical or overlapping
file/directory destinations are rejected across recorded campaigns in the same
canonical repository, even when they use different records roots. The reservation
registry is `artifacts/benchmarks/output-leases`; it coordinates cooperating
recorded producers, not unrelated processes writing directly to those paths.
Each invocation records its command, reservation and original/prior-output paths
in `invocation.json` before any old destination is moved.

A failed or incomplete run keeps its immutable partial artifacts and cannot
advance the digest-bound `latest` index. Missing destinations are restored from
the run's prior outputs. If the producer exits zero but an output is absent, the
runner returns exit code 1. Nonzero producer exit codes are preserved.

If finalisation or the process is interrupted, its reservation remains in place.
Recover the invocation and its original/prior-output paths before releasing that
reservation; a PID becoming absent or being reused is not proof of recovery.
Historical manifests remain unchanged. New manifests identify this contract as
`reserved-empty-destination.v1` in `output_custody`; it establishes output
freshness, not source-tree reproducibility or permission for production claims.

The [output lease API][scpn_control.benchmark_output_lease.BenchmarkOutputLease]
documents reservation acquisition and release.

A failed preparation releases its reservation only after all displaced outputs
are restored. If restoration is denied or a destination has become occupied,
the reservation and prior bytes remain available for explicit recovery; a new
campaign cannot claim those paths.

Relative output paths are resolved once against the working directory at
`BenchmarkRun.begin`. Later directory changes cannot redirect reservation,
archival, sealing or restoration to another file.

## Bounded UQ and equilibrium report software check

From this source checkout, this example runs byte-identical producer copies and
the actual recorded wrapper in a temporary output root. It creates software
custody evidence with the physical/calibrated claim flags closed. Copied-root
metadata does not identify the canonical package build or an external reference.
The reports and records disappear when the temporary scope ends.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python - <<'PY'
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

checkout = Path.cwd()
with TemporaryDirectory() as scratch:
    root = Path(scratch)
    (root / "validation").mkdir()
    claims = {
        "uq_claims": "calibrated_uq_claim_allowed",
        "kinetic_efit_claims": "facility_claim_allowed",
        "free_boundary_tracking_claims": "facility_claim_allowed",
    }
    for stem, claim in claims.items():
        producer = root / "validation" / f"benchmark_{stem}.py"
        shutil.copyfile(checkout / "validation" / producer.name, producer)
        subprocess.run(
            [sys.executable, str(checkout / "tools/run_recorded_benchmark.py"),
             "--repository-root", str(root), "--records-root", "records",
             "--family", stem, "--evidence-class", "software_boundary_example",
             "--artifact", f"report=validation/reports/{stem}.json",
             "--artifact", f"markdown=validation/reports/{stem}.md",
             "--", sys.executable, str(producer)],
            env=dict(os.environ, PYTHONPATH=str(checkout / "src")),
            check=True,
        )
        payload = json.loads((root / "validation/reports" / f"{stem}.json").read_text())
        assert payload[claim] is False
        print(stem, payload["claim_status"], payload[claim])
PY
```

## Formal declaration reader and bounded Z3 software check

From this source checkout, the Lean example writes an authored declaration in a
temporary directory and calls the actual reader. It executes no Lean proof and
asserts no artifact admission. The second example invokes installed Z3 on the
fixed two-step model using temporary destinations.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from tests.formal_validator_declaration_fixtures import declared_lean_case
from validation.validate_scpn_lean_formal import validate_lean_formal_evidence

with TemporaryDirectory() as temporary:
    named, other, artifact = declared_lean_case(Path(temporary))
    result = validate_lean_formal_evidence(named)
    assert result.status == "pass" and result.artifact_admitted is False
    wrong = validate_lean_formal_evidence(
        other, artifact_path=artifact, formal_report_root=Path(temporary)
    )
    assert wrong.status == "fail" and wrong.artifact_admitted is False
```

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from validation.validate_scpn_z3_formal import publish_report

with TemporaryDirectory() as temporary:
    root = Path(temporary)
    result = publish_report(
        json_path=root / "proof.json",
        markdown_path=root / "proof.md",
        require_z3=True,
    )
    assert result["status"] == "pass" and result["max_depth"] == 2
```


### Temporary synthetic resilience, ROC and Kuramoto examples

Run this source-checkout example from the repository root with its Python
environment and `PYTHONPATH=src:.`. It exercises actual public producers in
a temporary directory and evaluates five actual synthetic ROC shots.
No canonical reports or controlled latency comparisons are created.

```python
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from validation.disruption_roc_analysis import evaluate_batch, generate_scenario_batch

root = Path.cwd()
env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + str(root / "src"))
with TemporaryDirectory() as temporary:
    directory = Path(temporary)
    phase = directory / "phase.json"
    resilience = directory / "resilience.json"
    markdown = directory / "resilience.md"
    subprocess.run([
        sys.executable, str(root / "validation/benchmark_kuramoto_runtime_evidence.py"),
        "--output-json", str(phase), "--oscillators", "8",
        "--deployment-target-oscillators", "8", "--psi-mode", "mean_field",
    ], cwd=directory, env=env, check=True)
    subprocess.run([
        sys.executable, str(root / "validation/control_resilience_campaign.py"),
        "--episodes", "2", "--window", "16", "--noise-std", ".01", "--strict",
        "--output-json", str(resilience), "--output-md", str(markdown),
    ], cwd=directory, env=env, check=True)
    assert json.loads(phase.read_text())["oscillator_count"] == 8
    assert json.loads(resilience.read_text())["campaign"]["passes_thresholds"]
    shots = generate_scenario_batch(5)
    assert sum(shot["label"] for shot in shots) == 2
    print(evaluate_batch(shots, 0.0))
```

Kuramoto evidence measures numerical one-step refinement/parity, not runtime
latency or a Python/Rust speedup. The actual existing native extension can
satisfy the optional core parity/target checks; the default command remains
bounded. Resilience uses measured duration only as a local campaign diagnostic.
ROC alarms use absolute observed event indices and dimensionless risk rates.
AUC from the fixed mixed synthetic cohort is not a held-out facility score.
The Rust/PyO3 Kuramoto kernels are unchanged by these command contracts, so
the historical timing table above is retained as historical measurement;
this work makes no new comparative timing claim.

## Synthetic disturbance rejection

This Python-only benchmark compares actual available controllers on one
synthetic two-state vertical Euler proxy. Its scenario names describe forcing
patterns; density and beta_N are not simulated. Schema v2 reports actual
requested/completed time, stop reason, settling band and missing providers.
H-infinity uses its defining positive-measurement convention; the SNN provider
runs its own per-call clock. The MPC gradient is approximate; its original
q_weight, r_weight and iterations arguments remain available. Results do not
admit physical accuracy, facility control or a cross-language speedup.

The following source-checkout example checks a real initial/terminal trace:
```python
from validation.benchmark_disturbance_rejection import PIDController, SCENARIOS, run_scenario

scenario = dict(SCENARIOS["VDE"])
scenario["duration_s"] = 0.005
metrics, trace = run_scenario("PID", PIDController(), "VDE", scenario)
assert metrics.stable and metrics.completed_steps == 50
assert len(trace.times) == 51
assert trace.times[-1] == metrics.completed_duration_s == 0.005
assert (trace.errors == -trace.positions).all()
```

Use the [recorded benchmark runner][tools.run_recorded_benchmark] for
CLI campaigns. Its declared artifacts and campaign manifest retain command
and output custody. The API example above produces an in-memory observation.

Default durations remain 2/4/3 seconds at 100 microsecond dt. Unstable runs
stop at the actual boundary crossing, without a padded tail. Ordinary exit
zero means reporting completed; require-complete exits two after output when
any expected controller is missing or a run fails bounded completion. The
CLI protects selected source files and all output aliases before execution;
persistent evidence requires a recorded campaign. The wall_clock_s field
measures the conditional Euler/control loop, including forcing, finite checks
and trace accumulation. It excludes configuration, plant/controller construction
and reset, metric reduction, report writes and plotting. These local times
remain uncontrolled diagnostics, and reports are sequential writes.
Each plot figure is released after rendering or writing, including failures;
plots already written remain when a later plot fails.


## Manual JAX CBC grid study

`tools/gk_convergence_benchmark.py` retains the original full nonlinear JAX
calibration, adiabatic/kinetic runs and kx64/256 grid cases. Its standalone main
is a large manual workload; there are no parsed CLI arguments or help/reduced
mode. Do not substitute a short software check for that campaign or infer grid
convergence from its saved finite-sample flag.

The [full API example](api.md#manual-jax-cbc-grid-study) executes the actual public
configurable runner on a short caller-specified grid and writes caller-owned
scratch. It preserves the seed, config, numerical provider and returned raw
fields. Local qualification compares zero/one/two-sample results to independent
actual provider replays. The provider's one-sample second-half mean is empty and
maps NaN to null; this is not admitted transport data. The original large main
configuration/branch body remains unchanged and is not newly benchmarked here.

`wall_s` includes construction and first-use compilation, uses the system clock
and rounds to one decimal. It excludes import, result printing and file writes.
Device/precision, cache and shared-load differences must remain explicit; these
measurements do not establish a controlled speed comparison across grids or
languages. No matching study wrapper exists in Rust/TypeScript. The backend's
numerical source and existing numerical comparison evidence remain unchanged.

`RESULTS_FILE` defaults to `/tmp/gk_convergence.json` and can be assigned a
caller-owned path before `save`. This overwritable raw mapping is explicitly
classified temporary scratch, with legacy nonstandard-float handling and no
campaign/source digest or reference admission. It must not replace the separate
nonlinear CBC saturation evidence above. See the API contract for exact grids,
step selection, time/flux units and write/error/concurrency limits.


## Fixed CBC model comparison

`tools/gpu_cbc_benchmark.py` retains the original manual linear, native SAT1,
NumPy500/JAX500 and NumPy2000/JAX2000 campaign. All nonlinear stages use the
same16x16x64x16x8 grid with two species, dt0.02,save50 and CFL adaptation off.
Each begins with fresh seed42 state. The500-step calls are complete fresh runs;
there is no separately measured warmup, retained-state continuation or increased
resolution at2000 steps. CPU JAX is permitted; no CUDA device check is imposed.

Public per-stage functions return raw in-memory model reports and allow an
explicit requested nonlinear count. Zero requests exercise allocation with empty
history; one saved sample returns NaN means. `chi_i`/`chi_e` from nonlinear calls
are raw code-unit flux means, whereas native SAT1 reports m^2/s. The linear
adiabatic/period2 spectrum and SAT1 kinetic/period1 spectrum also differ.
No independent CBC/TGLF physical reference, backend parity or saturation test
is performed by this producer. Its flags and requested counts do not authenticate
complete integration. Read the [API contract](api.md#fixed-cbc-model-comparison)
for every field, fixed parameter, actual error and executable short example.

`elapsed_s` clocks have different boundaries: nonlinear run-only time excludes
solver/config construction but includes initialization and JAX first-use JIT/
synchronization; SAT1 includes construction/solve, linear includes solve alone.
Console/imports/writes are excluded. Host contention, caches, device and precision
prevent interpreting these raw fields as an isolated throughput comparison.
No speedup or physical-performance claim follows from a short local observation.

Main writes cwd-relative `gpu_results/gk_nonlinear_cbc_gpu.json` once after all
six stages, with raw NaN/Infinity serialization and no partial-stage checkpoint.
Canonical persistent use requires `tools/run_recorded_benchmark.py` and the
matching declared artifact; outside-source scratch follows the existing guard
exemption. Existing output replacement/partial I/O and missing-JAX skipped reports
retain their original conventions. The raw payload itself has no source/config/
backend digest or physical admission. The standalone script may invoke its
unpinned CUDA-JAX installer before main checks campaign custody; main's API does
not install. Provisioning and the original large CPU/GPU campaign require an
explicitly prepared environment and are separate from the short documented API.


## JarvisLabs remote PPO recipe

`tools/jarvislabs_train.py` requests one A5000 GPU with the PyTorch template,
while forcing CPU training. It reuses/clones the remote repository and uploads
seven recipe sources. The remaining source and dependencies are not pinned to
local HEAD. Fresh remote candidates use `artifacts/rl/<campaign>`; the local
`--output-dir` option selects an existing directory for nine flat downloads.

Before provider requests, main checks fresh local targets and recorded campaign
input. Every SSH/SCP step must exit zero. The full uppercase MPC/PID/PPO schema
requires finite metrics, valid ranges and equal positive episode counts. Existing
artifacts cannot count as new deliveries. Cleanup requires an explicit successful
SDK response; acknowledgement does not independently establish stopped billing.
Successful provider creation, training, transport and cleanup need actual execution
evidence. Local refusals are not training-performance measurements.

## Stored PPO comparison and explicit seed recipe

The corrected shell passes `--seed 42`, `--seed 123` and `--seed 456` to the real
trainer. `--dry-run` prints those three constructor plans without creating a model
or weights. Actual learning records requested and completed steps separately;
SB3 can round the request up to a complete rollout. Candidate selection requires
explicit seed identity in each metrics file, accepts finite rewards below -99999,
and resolves ties in listed seed order. The final table reads actual uppercase
MPC/PID/PPO keys.

Retained seed-labelled weights and metrics predate this correction. The old trainer
used seed 42 for all labels. Those files and the retained 50-episode report remain
historical artifacts; their names cannot establish independent training runs or
hardware provenance. A declared seed in new metrics is also not authenticated
training provenance.

`benchmarks/rl_vs_classical.py` evaluates the stored PPO policy on CPU, the original
proportional baseline and the original one-step 11x5 grid controller on the same
500-step reduced-order model and reset seeds 1000 through 1000+N-1. All controllers
must complete valid summaries; missing/unloadable PPO cannot silently yield a
PID/MPC-only success. The output is an exclusive fresh candidate JSON with reward
mean/population deviation, mean length, termination fraction and episode count.
Persistent report locations require recorded campaign input. The model comparison
does not establish experimental accuracy, stability or controller-safety acceptance.
The benchmark performs inference and does not learn or save weights. Runtime
measurements on a shared host are functional evidence only.

The tutorial loads the actual retained `weights/ppo_tokamak.zip`, reports observed
outcomes and renders retained artifact values dynamically. Default execution does
not train. `--train-demo` explicitly selects the separate 5000-step learning demo;
no hardcoded zero-disruption outcome replaces the observed rollout.


The current stored-policy comparison on 2026-10-02 used the retained policy ZIP
SHA-256 `53e7481a440845db01378018a21ec15fe0fde5bbff5f7413be8ea577d1dce736`
and the current `TokamakEnv` source SHA-256
`725447a489b5384169f62e0bb4b7bbf6c8b3f895f42caae583c6878406abb510`.
All three controllers ran 50 paired episodes, with mean length 500 and observed
termination fraction zero:

| Controller | Mean reward | Population deviation |
| --- | ---: | ---: |
| PPO | -5002.553057 | 160.755944 |
| PID | -4982.946570 | 161.906390 |
| MPC | -5119.341406 | 156.845936 |

The report SHA-256 is
`c6d993256a59f1caad598fde011926dfc5a5a3c4c035b99d4a409fcfbd8e881b`.
These outcomes do not reproduce the retained historical report, and PPO has a
lower mean reward than the proportional baseline in this run. The existing policy
is evaluated without learning or weight changes; the result does not determine
how a newly trained policy would behave. The one-step grid retains its legacy
fixed temperature-response surrogate, which differs from the current environment's
energy-balance evolution. This is a comparison of those concrete implementations.
The shared host was not isolated; no training-speed or production-latency claim
is derived from this run.

To reproduce model inference with an immutable custody record and a fresh temporary report:

```bash
python tools/run_recorded_benchmark.py --family rl-stored-model --artifact comparison=/tmp/scpn-rl-model-comparison.json -- python benchmarks/rl_vs_classical.py --episodes 50 --output /tmp/scpn-rl-model-comparison.json
```

The destination must not already exist. `benchmarks/rl_vs_classical.json` remains
the historical artifact; it is not overwritten or presented as this new result.


## Native oscillator capacity sweep

The standalone `tools/stress_test_oscillators.py` measures repeated calls to the
installed Rust/PyO3 phase kernel. Its [API contract](api.md#native-oscillator-capacity-sweep)
provides the exact sizes, parameters, repetition counts and exit behaviour. The
maximum allocation is 16×32768 = 524288 oscillators.

Each call receives the same phase and frequency arrays for that size, and its
returned tick is discarded. The table therefore describes single-call
throughput on those allocations. The displayed frequency is the reciprocal of
mean call latency. The ten-calls-per-second stop rule is a measurement threshold
chosen by this script.

The default CLI draws unseeded inputs from NumPy's global RNG. A caller can
choose the input seed before invoking the API:

```python
import numpy as np
from tools.stress_test_oscillators import stress_test

np.random.seed(42)
stress_test()
```

This consumes the caller's RNG state. Timing also depends on host contention,
native build, CPU affinity and processor state. Record those conditions for
comparisons; a shared-host table supplies no isolated production-latency or
physical-control admission. The command prints to stdout without writing an
evidence file or installing its required native module.
