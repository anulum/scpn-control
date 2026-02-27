<p align="center">
  <img src="docs/scpn_control_header.png" alt="SCPN-CONTROL — Formal Stochastic Petri Net Engine" width="100%">
</p>

<p align="center">
  <a href="https://github.com/anulum/scpn-control/actions"><img src="https://github.com/anulum/scpn-control/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/anulum/scpn-control/actions/workflows/docs-pages.yml"><img src="https://github.com/anulum/scpn-control/actions/workflows/docs-pages.yml/badge.svg" alt="Docs Pages"></a>
  <a href="https://pypi.org/project/scpn-control/"><img src="https://img.shields.io/pypi/v/scpn-control" alt="PyPI version"></a>
  <a href="https://pypi.org/project/scpn-control/"><img src="https://img.shields.io/pypi/pyversions/scpn-control" alt="Python versions"></a>
  <a href="https://scpn-control.streamlit.app"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg" alt="License: MIT OR Apache-2.0"></a>
  <a href="https://orcid.org/0009-0009-3560-0851"><img src="https://img.shields.io/badge/ORCID-0009--0009--3560--0851-green.svg" alt="ORCID"></a>
  <a href="https://arxiv.org/abs/2004.06344"><img src="https://img.shields.io/badge/arXiv-2004.06344-b31b1b.svg" alt="arXiv"></a>
  <a href="docs/REVIEWER_PAPER27_INTEGRATION.pdf"><img src="https://img.shields.io/badge/Paper_27-PDF-informational.svg" alt="Paper 27 PDF"></a>
  <a href="https://codecov.io/gh/anulum/scpn-control"><img src="https://codecov.io/gh/anulum/scpn-control/branch/main/graph/badge.svg" alt="codecov"></a>
  <a href="https://doi.org/10.5281/zenodo.18804940"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18804940.svg" alt="DOI"></a>
</p>

---

**scpn-control** is a standalone neuro-symbolic control engine that compiles
Stochastic Petri Nets into spiking neural network controllers with formal
verification guarantees. Extracted from
[scpn-fusion-core](https://github.com/anulum/scpn-fusion-core) — 48 source
modules, 50 test files, **701 tests**, 5 Rust crates, 13 CI jobs.

> **11.9 µs P50 control loop** — faster than any open-source fusion code,
> competitive with the DIII-D PCS (4–10 kHz physics loops).
> With `sc-neurocore` installed, the SNN compiler uses `VectorizedSCLayer` +
> Rust backend for **512× real-time** spiking neural network execution.
> See [competitive analysis](docs/competitive_analysis.md) for full benchmarks.

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

- **Petri Net to SNN compilation** -- Translates Stochastic Petri Nets into spiking neural network controllers with LIF neurons and bitstream encoding
- **Formal verification** -- Contract-based pre/post-condition checking on all control observations and actions
- **Sub-millisecond latency** -- <1ms control loop with optional Rust-accelerated kernels
- **Rust acceleration** -- PyO3 bindings for SCPN activation, marking update, Boris integration, SNN pools, and MPC
- **Multiple controller types** -- PID, MPC, H-infinity, SNN, neuro-cybernetic dual R+Z
- **Grad-Shafranov solver** -- Free-boundary equilibrium solver with L-mode/H-mode profile support
- **Digital twin integration** -- Real-time telemetry ingest, closed-loop simulation, and flight simulator
- **RMSE validation** -- CI-gated regression testing against DIII-D and SPARC experimental reference data
- **Disruption prediction** -- ML-based predictor with SPI mitigation and halo/RE physics

## Architecture

```
src/scpn_control/
+-- scpn/              # Petri net -> SNN compiler
|   +-- structure.py   #   StochasticPetriNet graph builder
|   +-- compiler.py    #   FusionCompiler -> CompiledNet (LIF + bitstream)
|   +-- contracts.py   #   ControlObservation, ControlAction, ControlTargets
|   +-- controller.py  #   NeuroSymbolicController (main entry point)
+-- core/              # Solver + plant model (clean init, no import bombs)
|   +-- fusion_kernel.py           # Grad-Shafranov equilibrium solver
|   +-- integrated_transport_solver.py  # Multi-species transport
|   +-- scaling_laws.py            # IPB98y2 confinement scaling
|   +-- eqdsk.py                   # GEQDSK/EQDSK file I/O
|   +-- uncertainty.py             # Monte Carlo UQ
+-- control/           # Controllers (optional deps guarded)
|   +-- h_infinity_controller.py   # H-inf robust control
|   +-- fusion_sota_mpc.py         # Model Predictive Control
|   +-- disruption_predictor.py    # ML disruption prediction
|   +-- tokamak_digital_twin.py    # Digital twin
|   +-- tokamak_flight_sim.py      # IsoFlux flight simulator
|   +-- neuro_cybernetic_controller.py  # Dual R+Z SNN
+-- phase/             # Paper 27 Knm/UPDE phase dynamics (NEW)
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

tests/                 # 701 tests (50 files)
+-- mock_diiid.py      # Synthetic DIII-D shot generator (14-field NPZ)
+-- test_e2e_phase_diiid.py  # E2E: shot-driven monitor + HDF5/NPZ export
+-- test_phase_kuramoto.py   # 50 Kuramoto/UPDE/Guard/Monitor tests
+-- test_rust_realtime_parity.py  # Rust PyRealtimeMonitor parity
+-- ...                # 44 more test files
```

## Paper 27 Phase Dynamics (Knm/UPDE Engine)

Implements the generalized Kuramoto-Sakaguchi mean-field model with exogenous
global field driver `ζ sin(Ψ − θ)`, per arXiv:2004.06344 and SCPN Paper 27.

**Modules:** `src/scpn_control/phase/` (7 files)

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
scpn-control live --port 8765 --zeta 0.5

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
| | nengo (`pip install "scpn-control[nengo]"`) |
| | h5py (`pip install "scpn-control[hdf5]"`) |
| | websockets (`pip install "scpn-control[ws]"`) |

## CLI

```bash
scpn-control demo --scenario combined --steps 1000   # Closed-loop control demo
scpn-control benchmark --n-bench 5000                 # PID vs SNN timing benchmark
scpn-control validate                                 # RMSE validation dashboard
scpn-control live --port 8765 --zeta 0.5              # Real-time WS phase sync server
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
cargo test --workspace

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
  license = {MIT OR Apache-2.0}
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
git tag v0.2.0
git push --tags
# → .github/workflows/publish-pypi.yml runs automatically
```

## Limitations

- **Equilibrium solver**: Fixed-boundary Grad-Shafranov only. Free-boundary with
  external coil currents is not implemented. No stellarator geometry.
- **Transport**: 1.5D flux-surface-averaged. No turbulence micro-instability
  models (TGLF/QuaLiKiz) — uses Chang-Hinton neoclassical + scaling-law anomalous.
- **Validation scope**: Benchmarked against analytic Solov'ev solutions and
  published DIII-D/SPARC equilibria (GEQDSK). No real MDSplus shot data yet.
- **Disruption predictor**: Synthetic training data only. Not validated on
  experimental disruption databases.
- **Rust acceleration**: Optional. Pure-Python fallback is complete but 5-10x
  slower for GS solve and Kuramoto steps at N > 1000.
- **Deployment**: Research-grade. Not hardened for real-time PCS integration.
  No ITER CODAC or EPICS interface.

## Authors

- **Miroslav Sotek** — ANULUM CH & LI — [ORCID](https://orcid.org/0009-0009-3560-0851)
- **Michal Reiprich** — ANULUM CH & LI

## License

- Concepts: Copyright 1996-2026
- Code: Copyright 2024-2026
- License: MIT OR Apache-2.0

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
