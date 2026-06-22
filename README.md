<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Public project overview and quick start. -->

<p align="center">
  <img src="docs/scpn_control_header.png" alt="SCPN-CONTROL — Formal Stochastic Petri Net Engine" width="100%">
</p>

<p align="center">
  <a href="https://github.com/anulum/scpn-control/actions"><img src="https://github.com/anulum/scpn-control/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/anulum/scpn-control/actions/workflows/docs-pages.yml"><img src="https://github.com/anulum/scpn-control/actions/workflows/docs-pages.yml/badge.svg" alt="Docs Pages"></a>
  <a href="https://pypi.org/project/scpn-control/"><img src="https://img.shields.io/pypi/v/scpn-control" alt="PyPI version"></a>
  <a href="https://pypi.org/project/scpn-control/"><img src="https://img.shields.io/pypi/pyversions/scpn-control" alt="Python versions"></a>
  <a href="https://pepy.tech/project/scpn-control"><img src="https://static.pepy.tech/badge/scpn-control" alt="All-time downloads"></a>
  <a href="https://scpn-control.streamlit.app"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License: AGPL-3.0-or-later"></a>
  <a href="https://www.bestpractices.dev/projects/12176"><img src="https://www.bestpractices.dev/projects/12176/badge" alt="OpenSSF Best Practices"></a>
  <a href="https://orcid.org/0009-0009-3560-0851"><img src="https://img.shields.io/badge/ORCID-0009--0009--3560--0851-green.svg" alt="ORCID"></a>
  <a href="https://arxiv.org/abs/2004.06344"><img src="https://img.shields.io/badge/arXiv-2004.06344-b31b1b.svg" alt="arXiv"></a>
  <a href="docs/REVIEWER_PAPER27_INTEGRATION.pdf"><img src="https://img.shields.io/badge/Paper_27-PDF-informational.svg" alt="Paper 27 PDF"></a>
  <a href="https://codecov.io/gh/anulum/scpn-control"><img src="https://codecov.io/gh/anulum/scpn-control/branch/main/graph/badge.svg" alt="codecov"></a>
  <a href="https://doi.org/10.5281/zenodo.18804939"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18804939.svg" alt="DOI"></a>
  <a href="https://buy.stripe.com/4gM00kbiMdjAberaYz5J601"><img src="https://img.shields.io/badge/sponsor-Stripe-635bff.svg" alt="Sponsor"></a>
</p>

---

# SCPN Control

**SCPN Control** is a research-grade control and validation package for fusion
plasma control loops. It turns stochastic Petri-net logic into executable
neuro-symbolic controllers, surrounds those controllers with formal contracts,
and connects them to equilibrium, transport, disruption, digital-twin, and
hardware-in-the-loop evidence gates.

The practical purpose is simple: help fusion teams decide whether a controller
idea is safe enough, fast enough, reproducible enough, and well-evidenced enough
to move from notebook experiments toward a facility control-system review.

## Who it is for

- Fusion control researchers prototyping PCS logic, NMPC, SNN, disruption, and
  mitigation controllers.
- Tokamak and stellarator programmes that need replayable validation artefacts
  before promoting algorithms toward hardware tests.
- Fusion startups that need a compact control package rather than a broad solver
  laboratory.
- Universities teaching plasma control, formal methods, differentiable physics,
  and control-system safety cases.
- Investors, grant reviewers, and collaborators who need to understand which
  claims are already evidenced and which claims are still blocked by external
  data, external codes, target hardware, or facility access.

## What the package does

- Compiles stochastic Petri nets into spiking-neural control artefacts with
  bounded marking, transition, and temporal-logic evidence.
- Runs control-facing equilibrium and transport facades for controller tuning,
  replay, and validation admission.
- Provides NMPC, robust control, phase-dynamics, reinforcement-learning,
  digital-twin, WebSocket, CODAC/EPICS-facing, and hardware evidence surfaces.
- Captures validation results as checksum-bound JSON/Markdown reports instead
  of unverifiable claims.
- Keeps facility-grade promotion fail-closed until the required measured-shot,
  public-reference, external-code, hardware, or independent-review artefacts
  are supplied and admitted.

## Read this first

| If you are... | Start with | Why |
| --- | --- | --- |
| New to the package | [Quick Start](#quick-start), then [Onboarding](docs/onboarding.md) | Confirms installation and explains the three-layer mental model. |
| Evaluating market or collaboration value | [Why it matters](#why-it-matters), [Use Cases](docs/use_cases.md), [Compute Validation Funding](docs/compute_validation_financing.md) | Shows the practical control-evidence workflow and the open support needs. |
| Reviewing claims | [Limitations & Honest Scope](#limitations--honest-scope), [Production Readiness](docs/production_readiness.md), [Validation](docs/validation.md) | Separates bounded repository evidence from facility, external-code, and deployment claims. |
| Building a controller | [Python in 30 Seconds](#python-in-30-seconds), [Tutorials](docs/tutorials.md), [API Reference](docs/api.md) | Moves from SPN/SNN basics to control and validation surfaces. |
| Preparing a release or audit | [Benchmarks](docs/benchmarks.md), [Validation Summary](docs/validation_summary.md), [Changelog](docs/changelog.md) | Shows persisted evidence, benchmark boundaries, and release history. |

## Relationship to SCPN Fusion Core

`scpn-control` is the compact controller-facing package in the SCPN ecosystem.
It owns the admission contracts, replay metadata, runtime safety boundaries,
control APIs, documentation, and release evidence needed by controller users.

[`scpn-fusion-core`](https://github.com/anulum/scpn-fusion-core) is the broader
solver and physics laboratory. Solver experiments and broad physics kernels
mature there first; `scpn-control` ports or wraps the subset that has a clear
control-loop contract.

The split avoids double work: FUSION-CORE advances physics breadth, while
CONTROL turns selected physics into auditable controller surfaces.

[`scpn-quantum-control`](https://github.com/anulum/scpn-quantum-control) is the
third repository in the ecosystem: it owns quantum disruption classifiers,
Qiskit/PennyLane execution, and quantum phase-dynamics variants. `scpn-control`
consumes a bounded control adapter for the quantum disruption path rather than
re-implementing it. The full three-repository ownership split and the contracts
between the repositories are documented in
[the architecture guide](docs/architecture.md#scpn-ecosystem-and-cross-repository-contracts).

## Why it matters

Fusion control software is usually split across offline modelling codes,
facility-specific PCS infrastructure, and ad-hoc research notebooks. SCPN
Control aims to fill the missing middle layer: an installable control package
that can express novel neuro-symbolic controllers, run fast local validation,
and produce evidence suitable for review.

The current differentiators are formal Petri-net safety evidence,
differentiable physics/control facades, local-first LLM-assisted physics-gap
triage, quantum-disruption bridge contracts, strict public-data admission, and
release artefact gates. These are valuable only when the evidence boundary is
honest, so this repository distinguishes library readiness from facility
certification throughout the documentation.

## What it is in one sentence

SCPN Control is the controller-facing evidence layer for fusion software: it
helps teams convert controller ideas into executable artefacts, attach physics
and runtime assumptions, run bounded validation, and decide what can or cannot
be claimed before facility promotion.

## The product boundary

SCPN Control is intentionally not a monolithic replacement for established
facility tools. It sits between four worlds that are often disconnected:
research notebooks, physics solvers, plant-control infrastructure, and safety
review. The package provides the contract layer between those worlds.

| Boundary | SCPN Control role | What still belongs elsewhere |
| --- | --- | --- |
| Controller design | SPN/SNN, NMPC, robust control, phase dynamics, digital-twin control contracts | Site-specific PCS implementation and operator procedures |
| Physics coupling | Control-grade facades, differentiable paths, replay metadata, validation admission | Broad solver development and full physics campaigns, primarily in SCPN Fusion Core or external codes |
| Evidence | Schema-versioned JSON/Markdown artefacts, checksums, unit contracts, strict validators | Facility sign-off, independent V&V, and regulator or plant acceptance |
| Deployment preparation | Runtime security boundaries, target-hardware evidence hooks, CODAC/EPICS/HIL artefact admission | Commissioned plant deployment and machine-protection qualification |

## What is new in v0.21.0

This release strengthens validation evidence. Twenty control and physics models
gained tests that compare their outputs against independently derived exact
closed-form, analytic, eigenvalue, or conservation-law references, and the
differentiable-transport facade reached near-complete statement coverage. It
does not relax the facility, target-hardware, P-EFIT, PREEMPT_RT, or
external-code evidence gates.

- Twenty models are now checked against exact references rather than
  self-consistency fixtures, including the Grad-Shafranov solver against the
  Solov'ev equilibrium, the Kuramoto runtime against published synchronisation
  results, structured singular value against exact mu identities, the
  guiding-centre orbit integrator against conservation laws, and the
  Modified Rutherford, RZIP, resistive-wall-mode, sawtooth, scrape-off-layer,
  current-drive, ideal-MHD, EPED, ELM, momentum-transport, runaway-avalanche,
  halo-current, volt-second, density-control, and DT burn-control models.
- The differentiable-transport facade reached 99.5% statement coverage with
  module-specific input-validation, evidence-guard, and JAX-path tests.
- Strict mypy typing now covers the admission, configuration, current-drive,
  and real-time EFIT modules.
- A Rust advisory was cleared (pyo3/numpy 0.25 to 0.29) and numpy, osqp,
  tornado, hypothesis, pip-audit, ruff, sha2, and socket2 were bumped.
- Release documentation keeps all new evidence classified as repository
  validation coverage. No new production timing or facility validation claim is
  admitted by this release.

## Why this has market value

Fusion organisations spend significant time proving that a controller result is
not just an attractive plot. They need reproducibility, claim discipline,
security boundaries, and a path from local validation to external review. SCPN
Control packages that work into a reusable layer. The immediate market value is
shorter review cycles, clearer due-diligence artefacts, better collaboration
with facilities and external-code owners, and a concrete compute-validation
funding plan instead of vague claims.

The strongest near-term applications are controller concept review, formal
safety-case preparation, public-data validation campaigns, differentiable
controller tuning, target-hardware latency evidence, and local or air-gapped
physics debugging. The project deliberately keeps broader facility claims
blocked until the required external artefacts exist.

> **Neuro-symbolic control with formal safety contracts.** SCPN Control compiles
> Stochastic Petri Nets into spiking neural network controllers and runs them
> behind pre/post-condition contracts checked on every control action, in a
> runtime-selectable stack alongside PID, nonlinear MPC, and H∞ — no GPU required.
> The control compute meets the 1–10 kHz real-time budget with margin (native
> integrated control cycle ~5 µs P50 on CI / ~3 µs local, reproducible via
> `scripts/benchmark_native_handoff.py`); in a fielded loop the bottleneck is
> diagnostics, equilibrium reconstruction, and actuation, not the controller.
> Benchmark methodology and per-backend tables: [benchmarks](docs/benchmarks.md).
> See [competitive analysis](docs/competitive_analysis.md) for methodology and
> [production readiness](docs/production_readiness.md) for deployment limits.
>
> **Status: Alpha / Research.** This is not a commissioned plant PCS. Public
> full-fidelity facility claims remain blocked unless the corresponding strict
> validation gate admits real measured-shot, public-reference, external-code,
> target-hardware, and review artefacts.

## Capability Inventory

<!-- capability-snapshot:start -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Generated capability snapshot. -->

**Capability Inventory**

| Surface | Count |
| --- | ---: |
| Package version | 0.21.0 |
| Python requirement | >=3.10 |
| Project scripts | 2 |
| Public API exports | 44 |
| Python control/physics modules | 146 |
| Python public classes | 526 |
| Rust source files | 64 |
| Rust PyO3 exports | 39 |
| Validation scripts | 119 |
| Optional extras | 17 |
| Python test files | 379 |
| Public documentation pages | 53 |
| GitHub Actions workflows | 10 |

**Evidence roots:** `src/scpn_control/{core,control,phase,scpn}`, `scpn-control-rs/crates`, `validation`, `tests`, `docs`, and `.github/workflows`.

Refresh with `python tools/capability_manifest.py`; enforce with `python tools/capability_manifest.py --check`.
<!-- capability-snapshot:end -->

## Quick Start

```bash
pip install scpn-control                        # core (numpy, scipy, click)
pip install "scpn-control[dashboard,ws]"        # + Streamlit dashboard + WebSocket
scpn-control demo --steps 1000
scpn-control benchmark --n-bench 5000
```

For development (editable install):

```bash
git clone https://github.com/anulum/scpn-control.git
cd scpn-control
pip install -e ".[dev]"
```

### Python in 30 Seconds

```python
from scpn_control.core.jax_gs_solver import jax_gs_solve
psi = jax_gs_solve(NR=33, NZ=33, Ip_target=1e6, n_picard=40, n_jacobi=100)

from scpn_control.scpn.structure import StochasticPetriNet
from scpn_control.scpn.compiler import FusionCompiler
net = StochasticPetriNet()
net.add_place("idle", initial_tokens=1.0)
net.add_place("heating"); net.add_transition("ignite")
net.add_arc("idle", "ignite"); net.add_arc("ignite", "heating")
net.compile()
artifact = FusionCompiler().compile(net)  # SPN -> SNN
```

Full walkthrough: `python examples/quickstart.py`

## Documentation and Tutorials

- Documentation site: https://anulum.github.io/scpn-control/
- Local docs index: `docs/index.md`
- Benchmark guide: `docs/benchmarks.md`
- Notebook tutorials:
  - `examples/neuro_symbolic_control_demo.ipynb`
  - `examples/q10_breakeven_demo.ipynb`
  - `examples/snn_compiler_walkthrough.ipynb`
  - `examples/paper27_phase_dynamics_demo.ipynb` — Knm/UPDE + ζ sin(Ψ−θ)
  - `examples/snn_pac_closed_loop_demo.ipynb` — SNN-PAC-Kuramoto closed loop
  - `examples/streamlit_ws_client.py` — live WebSocket phase sync dashboard

Build docs locally:

```bash
python -m pip install mkdocs
mkdocs serve
```

Execute all notebooks:

```bash
python -m pip install "scpn-control[viz]" jupyter nbconvert
jupyter nbconvert --to notebook --execute --output-dir artifacts/notebook-exec examples/q10_breakeven_demo.ipynb
jupyter nbconvert --to notebook --execute --output-dir artifacts/notebook-exec examples/snn_compiler_walkthrough.ipynb
```

Optional notebook (requires `sc_neurocore` available in environment):

```bash
jupyter nbconvert --to notebook --execute --output-dir artifacts/notebook-exec examples/neuro_symbolic_control_demo.ipynb
```

## Features

### Control and formal methods

- **Petri Net to SNN compilation** -- translates stochastic Petri nets into
  spiking neural controller artefacts with LIF neurons and bitstream metadata.
- **Bounded formal verification** -- covers exact finite Petri-net reachability,
  marking-bound checks, place invariants, transition liveness, bounded temporal
  response specifications, optional Z3 bounded model checking, Lean proof
  evidence admission, and fail-closed proof-manifest validation.
- **Controller families** -- PID, MPC, NMPC, H-infinity, mu-synthesis,
  gain-scheduled, sliding-mode, fault-tolerant, SNN, and reinforcement-learning
  research controllers.
- **Robust control** -- H-infinity DARE synthesis, bounded static mu-analysis,
  degraded-mode operation, and shape-control surfaces with explicit admission
  boundaries.

### Physics and differentiable facades

- **Grad-Shafranov facades** -- fixed-boundary and experimental free-boundary
  equilibrium paths with L/H-mode profiles and JAX-differentiable research
  surfaces where JAX is available.
- **Transport and scenario management** -- integrated transport, current
  diffusion, sawteeth, NTM, SOL, scenario scheduling, and ITER/NSTX-U presets.
- **Gyrokinetic research surfaces** -- nonlinear delta-f GK, local dispersion,
  TGLF-like native approximations, kinetic electron/species tooling,
  collision/operator research paths, and JAX backend parity evidence. Full
  quantitative GK claims still require external-code validation on identical
  inputs.
- **Neural transport and equilibrium** -- QLKNN-style and neural-equilibrium
  surrogate paths with strict reference-admission boundaries.

### Runtime, evidence, and deployment preparation

- **Rust acceleration** -- PyO3 bindings for SCPN activation, marking update,
  Boris integration, SNN pools, and MPC kernels.
- **Benchmark admission** -- persisted benchmark-regression reports validate
  digests, thresholds, sample counts, hardware context, and claim text before
  release preflight can treat local latency evidence as current.
- **Native-build security** -- optional C++ solver compilation is gated,
  checksum-bound, compiler-admitted, environment-reduced, symlink-hardened, and
  atomically published.
- **Digital twin integration** -- telemetry ingest, replay metadata,
  closed-loop simulation, EFIT-facing contracts, and flight-simulator surfaces.
- **Disruption prediction** -- ML disruption and mitigation research surfaces.
  Facility-facing disruption claims remain blocked without measured databases
  and strict admission.

## Architecture

```
src/scpn_control/
+-- scpn/              # Petri net -> SNN compiler
|   +-- structure.py   #   StochasticPetriNet graph builder
|   +-- compiler.py    #   FusionCompiler -> CompiledNet (LIF + bitstream)
|   +-- contracts.py   #   ControlObservation, ControlAction, ControlTargets
|   +-- controller.py  #   NeuroSymbolicController (main entry point)
+-- core/              # Physics solvers + plant models (73 modules)
|   +-- fusion_kernel.py           # Grad-Shafranov equilibrium (fixed + free boundary)
|   +-- integrated_transport_solver.py  # Multi-species transport PDE
|   +-- gyrokinetic_transport.py   # Quasilinear TGLF-10 (ITG/TEM/ETG)
|   +-- ballooning_solver.py       # s-alpha ballooning eigenvalue ODE
|   +-- sawtooth.py                # Kadomtsev crash + Porcelli trigger
|   +-- ntm_dynamics.py            # Modified Rutherford NTM + ECCD stabilization
|   +-- current_diffusion.py       # Parallel current evolution PDE
|   +-- current_drive.py           # ECCD, NBI, LHCD deposition models
|   +-- sol_model.py               # Two-point SOL + Eich heat-flux width
|   +-- rzip_model.py              # Linearised vertical stability (RZIp)
|   +-- integrated_scenario.py     # Full scenario simulator (ITER/NSTX-U presets)
|   +-- stability_mhd.py           # 5 MHD stability criteria
|   +-- scaling_laws.py            # IPB98y2 confinement scaling
|   +-- neural_transport.py        # QLKNN-10D trained surrogate
|   +-- neural_equilibrium.py      # PCA+MLP GS surrogate (research-path speedup, not P-EFIT validated)
|   +-- ...                        # 14 more (eqdsk, uncertainty, pedestal, ...)
+-- control/           # Controllers (50 modules, optional deps guarded)
|   +-- h_infinity_controller.py   # H-inf robust control (DARE)
|   +-- mu_synthesis.py            # Static D-scaled structured singular value bound
|   +-- nmpc_controller.py         # Nonlinear MPC (SQP, 20-step horizon)
|   +-- gain_scheduled_controller.py  # PID scheduled on operating regime
|   +-- sliding_mode_vertical.py   # Sliding-mode vertical stabilizer
|   +-- fault_tolerant_control.py  # Fault detection + degraded-mode operation
|   +-- shape_controller.py        # Plasma shape via boundary Jacobian
|   +-- safe_rl_controller.py      # PPO + MHD constraint checker
|   +-- scenario_scheduler.py      # Shot timeline + actuator scheduling
|   +-- realtime_efit.py           # Streaming equilibrium reconstruction
|   +-- control_benchmark_suite.py # Standardised benchmark scenarios
|   +-- disruption_predictor.py    # ML disruption prediction + SPI
|   +-- tokamak_digital_twin.py    # Digital twin
|   +-- ...                        # 24 more (MPC, flight sim, HIL, ...)
+-- phase/             # Paper 27 Knm/UPDE phase dynamics (9 modules)
|   +-- kuramoto.py    #   Kuramoto-Sakaguchi step + order parameter
|   +-- knm.py         #   Paper 27 Knm coupling matrix builder
|   +-- upde.py        #   UPDE multi-layer solver
|   +-- lyapunov_guard.py          # Sliding-window stability monitor
|   +-- realtime_monitor.py        # Tick-by-tick UPDE + TrajectoryRecorder
|   +-- ws_phase_stream.py         # Async WebSocket live stream server
+-- cli.py             # Click CLI

scpn-control-rs/       # Rust workspace (5 crates)
+-- control-types/     # PlasmaState, EquilibriumConfig, ControlAction
+-- control-math/      # LIF neuron, Boris pusher, Kuramoto, upde_tick
+-- control-core/      # GS solver, transport, confinement scaling
+-- control-control/   # PID, MPC, H-inf, SNN controller
+-- control-python/    # PyO3 bindings (PyRealtimeMonitor, PySnnPool, ...)

tests/                 # 3,700+ collected Python tests
+-- mock_diiid.py      # Synthetic DIII-D shot generator (NOT real MDSplus data)
+-- test_e2e_phase_diiid.py  # E2E: shot-driven monitor + HDF5/NPZ export
+-- test_phase_kuramoto.py   # 50 Kuramoto/UPDE/Guard/Monitor tests
+-- test_rust_realtime_parity.py  # Rust PyRealtimeMonitor parity
+-- ...                # 170+ more test files
```

## Paper 27 Phase Dynamics (Knm/UPDE Engine)

Implements the generalized Kuramoto-Sakaguchi mean-field model with exogenous
global field driver `ζ sin(Ψ − θ)`, per arXiv:2004.06344 and SCPN Paper 27.

**Modules:** `src/scpn_control/phase/` (9 modules)

| Module | Purpose |
|--------|---------|
| `kuramoto.py` | Kuramoto-Sakaguchi step, order parameter R·e^{iΨ}, Lyapunov V/λ |
| `knm.py` | Paper 27 16×16 coupling matrix (exponential decay + calibration anchors) |
| `upde.py` | UPDE multi-layer solver with PAC gating |
| `lyapunov_guard.py` | Sliding-window stability monitor (mirrors DIRECTOR_AI CoherenceScorer) |
| `realtime_monitor.py` | Tick-by-tick UPDE + TrajectoryRecorder (HDF5/NPZ export) |
| `ws_phase_stream.py` | Async WebSocket server streaming R/V/λ per tick |

**Rust acceleration:** `upde_tick()` in `control-math` + `PyRealtimeMonitor` PyO3 binding.

**Live phase sync convergence** ([GIF fallback](docs/phase_sync_live.gif)):

<p align="center">
  <video src="docs/phase_sync_live.mp4" autoplay loop muted playsinline width="100%">
    <img src="docs/phase_sync_live.gif" alt="Phase Sync Convergence — 16 layers × 50 osc, ζ=0.5" width="100%">
  </video>
</p>

> 500 ticks, 16 layers × 50 oscillators, ζ=0.5. R converges to 0.92,
> V→0, λ settles to −0.47 (stable). Generated by `tools/generate_phase_video.py`.

**WebSocket live stream:**

```bash
# Terminal 1: start server (CLI)
scpn-control live --host 127.0.0.1 --port 8765 --zeta 0.5 --api-key "$SCPN_PHASE_WS_API_KEY"

# Remote exposure requires an API key and should use TLS.
scpn-control live --host 0.0.0.0 --port 8765 --api-key "$SCPN_PHASE_WS_API_KEY" \
  --tls-cert phase.pem --tls-key phase-key.pem --require-tls

# Terminal 2: Streamlit WS client (live R/V/λ plots, guard status, control)
pip install "scpn-control[dashboard,ws]"
streamlit run examples/streamlit_ws_client.py

# Or embedded mode (server + client in one process)
streamlit run examples/streamlit_ws_client.py -- --embedded
```

**E2E test with mock DIII-D shot data:**

```bash
pytest tests/test_e2e_phase_diiid.py -v
```

## Dependencies

| Required | Optional |
|----------|----------|
| numpy >= 1.24 | sc-neurocore >= 3.8.0 (`pip install "scpn-control[neuro]"`) |
| scipy >= 1.10 | matplotlib (`pip install "scpn-control[viz]"`) |
| click >= 8.0 | streamlit (`pip install "scpn-control[dashboard]"`) |
| | torch (`pip install "scpn-control[ml]"`) |
| | ~~nengo~~ (removed — pure LIF+NEF engine, no external dependency) |
| | h5py (`pip install "scpn-control[hdf5]"`) |
| | websockets (`pip install "scpn-control[ws]"`) |

## CLI

```bash
scpn-control demo --scenario combined --steps 1000   # Closed-loop control demo
scpn-control benchmark --n-bench 5000                 # PID vs SNN timing benchmark
scpn-control validate                                 # RMSE validation dashboard
scpn-control validate-eped-reference --require-reference-artifacts --json-out  # EPED pedestal reference gate
scpn-control validate-marfe-reference --require-reference-artifacts --json-out # MARFE density-limit reference gate
scpn-control validate-ntm-reference --require-reference-artifacts --json-out   # NTM island-dynamics reference gate
scpn-control live --host 127.0.0.1 --port 8765 --zeta 0.5 --api-key "$SCPN_PHASE_WS_API_KEY"  # Real-time WS phase sync server
scpn-control hil-test --shots-dir ...                 # HIL test campaign
```

## Benchmarks

Python micro-benchmark:

```bash
scpn-control benchmark --n-bench 5000 --json-out
```

Rust Criterion benchmarks:

```bash
cd scpn-control-rs
cargo bench --workspace
```

Benchmark docs: `docs/benchmarks.md`

## Dashboard

```bash
pip install "scpn-control[dashboard]"
streamlit run dashboard/control_dashboard.py
```

Six tabs: Trajectory Viewer, RMSE Dashboard, Timing Benchmark, Shot Replay,
Phase Sync Monitor (live R/V/λ plots), Benchmark Plots (interactive Vega).

### Streamlit Cloud

**Live dashboard:** [scpn-control.streamlit.app](https://scpn-control.streamlit.app)

The phase sync dashboard runs on Streamlit Cloud with embedded server mode
(no external WS server needed). Entry point: `streamlit_app.py`.

To deploy your own instance:
1. Fork to your GitHub
2. [share.streamlit.io](https://share.streamlit.io) > New app > select `streamlit_app.py`
3. Deploy (auto-starts embedded PhaseStreamServer)

## Rust Acceleration

```bash
cd scpn-control-rs
cargo test --workspace --exclude scpn-control-rs

# Build Python bindings
cd crates/control-python
pip install maturin
maturin develop --release
cd ../../

# Verify
python -c "import importlib.util; from scpn_control.core._rust_compat import _rust_available; print(bool(importlib.util.find_spec('scpn_control_rs') and _rust_available()))"
```

The Rust backend provides PyO3 bindings for:
- `PyFusionKernel` -- Grad-Shafranov solver
- `PySnnPool` / `PySnnController` -- Spiking neural network pools
- `PyMpcController` -- Model Predictive Control
- `PyPlasma2D` -- Digital twin
- `PyTransportSolver` -- Chang-Hinton + Sauter bootstrap
- `PyRealtimeMonitor` -- Multi-layer Kuramoto UPDE tick (phase dynamics)
- SCPN kernels -- `dense_activations`, `marking_update`, `sample_firing`

## Citation

```bibtex
@software{sotek2026scpncontrol,
  title   = {SCPN Control: Neuro-Symbolic Stochastic Petri Net Controller},
  author  = {Sotek, Miroslav and Reiprich, Michal},
  year    = {2026},
  url     = {https://github.com/anulum/scpn-control},
  license = {AGPL-3.0-or-later}
}
```

## Release and PyPI

**Local publish script:**

```bash
# Dry run (build + check, no upload)
python tools/publish.py --dry-run

# Publish to TestPyPI
python tools/publish.py --target testpypi

# Bump version + publish to PyPI
python tools/publish.py --bump minor --target pypi --confirm
```

**CI workflow** (tag-triggered trusted publishing):

```bash
git tag v0.21.0
git push --tags
# → .github/workflows/publish-pypi.yml runs automatically
```

## Limitations & Honest Scope

> These are not future roadmap items — they are current architectural
> constraints that users must understand.

- **No facility deployment**: DIII-D replay evidence is limited to immutable
  repository reference artefacts with manifest checksums. Synthetic fixtures
  remain for CI plumbing only, not public physics evidence. No live MDSplus,
  no experimental control-room replay, and no real-world validation.
- **No peer-reviewed fusion-control publication yet**: Paper 27
  (arXiv:2004.06344) is public, but this repository still needs a dedicated
  peer-reviewed fusion-control software paper for the current stack.
- **Not a production PCS**: Alpha-stage research software. CODAC/EPICS support
  and WebSocket control-stream support are research adapters and
  evidence-admission contracts, not a certified ITER plant deployment. No
  safety certification and no real hardware deployment.
- **Formal verification is bounded evidence**: The repository includes
  Petri-net reachability, temporal-logic, Z3-backed, certificate-bundle, and
  Lean proof-admission paths. Those reports are bounded software evidence, not
  facility safety certification.
- **Benchmark comparisons are not apples-to-apples**: The ~5 µs figure is the
  integrated control cycle on a loopback-UDP campaign, not a fielded plant loop.
  DIII-D PCS timings include I/O, diagnostics, and actuator commands over real
  hardware. A fair comparison requires equivalent end-to-end measurement on
  comparable hardware. Publish E2E control-latency evidence with
  `benchmarks/e2e_control_latency.py --output-json ... --target-hardware-id ...
  --target-hardware-class ... --rt-kernel ...`; admitted reports must be
  schema-versioned and digest-bound, and unqualified local runs do not support
  hardware-in-the-loop real-time claims.
- **Equilibrium solver**: Two variants exist: stable fixed-boundary GS, plus an
  experimental free-boundary external-coil scaffold. The free-boundary path is
  not yet sufficient for full shape control, X-point geometry, or divertor
  configuration. No stellarator geometry.
- **Transport and GK**: 1.5D flux-surface-averaged transport and nonlinear
  delta-f GK research surfaces exist. Native TGLF-like approximations and
  nonlinear solvers remain bounded local models until cross-validated against
  production TGLF, GENE, GS2, CGYRO, QuaLiKiz, or documented public references
  on identical equilibria.
- **Disruption predictor**: Synthetic training data only. Not validated on
  experimental disruption databases.
- **No GPU equilibrium**: P-EFIT is faster on GPU hardware. JAX neural equilibrium
  runs on GPU if available. Public MAST EFM prediction evidence is available as
  fail-closed flux and derived-geometry evaluation with exact public EFM
  coordinate grids, but it is not cross-validated against matched P-EFIT
  pressure, q-profile, and exact-input artefacts.
- **Rust acceleration**: Optional. Pure-Python fallback is complete but 5-10x
  slower for GS solve and Kuramoto steps at N > 1000.

## Support the Project

**scpn-control** is open-source (AGPL-3.0-or-later | commercial license available).
Funding goes to compute, validation data, and development time. See the GitHub Pages [compute validation funding plan](https://anulum.github.io/scpn-control/compute_validation_financing/) for GPU-hour, storage, public-data, and external-code validation needs.

| | | |
|---|---|---|
| [Sponsor via Stripe](https://buy.stripe.com/4gM00kbiMdjAberaYz5J601) | [Donate via PayPal](https://www.paypal.com/donate?hosted_button_id=4X5F6DNT934HY) | [Pay via TWINT](https://go.twint.ch/1/e/tw?tw=acq.lJTAypb8SL2s8vPg7fL0ubi2C220ajOH0BEQn1aKfEJIiIakLpt8jlEv8XdQ9tCp.) |

**Crypto:** BTC `bc1qg48gdmrjrjumn6fqltvt0cf0w6nvs0wggy37zd` ·
ETH `0xd9b07F617bEff4aC9CAdC2a13Dd631B1980905FF` ·
LTC `ltc1q886tmvtlnj86kmg2urd8f5td3lmfh32xtpdrut`

**Bank:** CHF IBAN CH14 8080 8002 1898 7544 1 · EUR IBAN CH66 8080 8002 8173 6061 8 · BIC RAIFCH22

Full tier details (Pro, Academic, Enterprise, Sponsorships): [docs/pricing.md](docs/pricing.md)

## Authors

- **Miroslav Sotek** — ANULUM CH & LI — [ORCID](https://orcid.org/0009-0009-3560-0851)
- **** — ANULUM CH & LI

## License

- Concepts: Copyright 1996-2026
- Code: Copyright 2024-2026
- License: AGPL-3.0-or-later

Commercial licensing available — contact protoscience@anulum.li.
