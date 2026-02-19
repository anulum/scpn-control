<p align="center">
  <img src="docs/scpn_control_header.png" alt="SCPN-CONTROL — Formal Stochastic Petri Net Engine" width="100%">
</p>

<p align="center">
  <a href="https://github.com/anulum/scpn-control/actions"><img src="https://github.com/anulum/scpn-control/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/anulum/scpn-control/actions/workflows/docs-pages.yml"><img src="https://github.com/anulum/scpn-control/actions/workflows/docs-pages.yml/badge.svg" alt="Docs Pages"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg" alt="License: AGPL v3"></a>
  <a href="https://orcid.org/0009-0009-3560-0851"><img src="https://img.shields.io/badge/ORCID-0009--0009--3560--0851-green.svg" alt="ORCID"></a>
</p>

---

**scpn-control** is a standalone neuro-symbolic control engine that compiles
Stochastic Petri Nets into spiking neural network controllers with formal
verification guarantees. Extracted from
[scpn-fusion-core](https://github.com/anulum/scpn-fusion-core) as the minimal
41-file transitive closure of the control pipeline.

## Quick Start

```bash
pip install -e "."
scpn-control demo --steps 1000
scpn-control benchmark --n-bench 5000
```

## Documentation and Tutorials

- Documentation site: https://anulum.github.io/scpn-control/
- Local docs index: `docs/index.md`
- Benchmark guide: `docs/benchmarks.md`
- Release and PyPI guide: `docs/release.md`
- Notebook tutorials:
  - `examples/neuro_symbolic_control_demo.ipynb`
  - `examples/q10_breakeven_demo.ipynb`
  - `examples/snn_compiler_walkthrough.ipynb`

Build docs locally:

```bash
python -m pip install mkdocs
mkdocs serve
```

Execute all notebooks:

```bash
python -m pip install -e ".[viz]" jupyter nbconvert
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
+-- cli.py             # Click CLI

scpn-control-rs/       # Rust workspace (5 crates)
+-- control-types/     # PlasmaState, EquilibriumConfig, ControlAction
+-- control-math/      # LIF neuron, Boris pusher, matrix ops
+-- control-core/      # GS solver, transport, confinement scaling
+-- control-control/   # PID, MPC, H-inf, SNN controller
+-- control-python/    # Slim PyO3 bindings (~474 LOC)
```

## Dependencies

| Required | Optional |
|----------|----------|
| numpy >= 1.24 | matplotlib (`pip install -e ".[viz]"`) |
| scipy >= 1.10 | streamlit (`pip install -e ".[dashboard]"`) |
| click >= 8.0 | torch (`pip install -e ".[ml]"`) |
| | nengo (`pip install -e ".[nengo]"`) |

## CLI

```bash
scpn-control demo --scenario combined --steps 1000   # Closed-loop control demo
scpn-control benchmark --n-bench 5000                 # PID vs SNN timing benchmark
scpn-control validate                                 # RMSE validation dashboard
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
pip install -e ".[dashboard]"
streamlit run dashboard/control_dashboard.py
```

Four tabs: Trajectory Viewer, RMSE Dashboard, Timing Benchmark, Shot Replay.

## Rust Acceleration

```bash
cd scpn-control-rs
cargo test --workspace

# Build Python bindings
pip install maturin
maturin develop --release

# Verify
python -c "import scpn_control_rs; print('Rust backend active')"
```

The Rust backend provides PyO3 bindings for:
- `PyFusionKernel` -- Grad-Shafranov solver
- `PySnnPool` / `PySnnController` -- Spiking neural network pools
- `PyMpcController` -- Model Predictive Control
- `PyPlasma2D` -- Digital twin
- `PyTransportSolver` -- Chang-Hinton + Sauter bootstrap
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

Release checklist and publishing guide:

- `docs/release.md`
- Workflow: `.github/workflows/publish-pypi.yml`

## Authors

- **Miroslav Sotek** — ANULUM CH & LI — [ORCID](https://orcid.org/0009-0009-3560-0851)
- **Michal Reiprich** — ANULUM CH & LI

## License

- Concepts: Copyright 1996-2026
- Code: Copyright 2024-2026
- License: GNU AGPL v3

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).

For commercial licensing inquiries, contact: [protoscience@anulum.li](mailto:protoscience@anulum.li)
