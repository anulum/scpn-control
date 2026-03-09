# Architecture

SCPN Control is a two-tier neuro-symbolic control engine: a Python
simulation layer with optional Rust acceleration via PyO3.

## Directory Map

```
scpn-control/
├── src/scpn_control/          Python package
│   ├── scpn/                  Petri net -> SNN compiler
│   │   ├── structure.py       StochasticPetriNet graph builder
│   │   ├── compiler.py        FusionCompiler -> CompiledNet (LIF + bitstream)
│   │   ├── contracts.py       ControlObservation, ControlAction, ControlTargets
│   │   └── controller.py      NeuroSymbolicController (main entry point)
│   ├── core/                  Solver + plant model
│   │   ├── fusion_kernel.py   Grad-Shafranov equilibrium solver
│   │   ├── integrated_transport_solver.py  Multi-species transport
│   │   ├── scaling_laws.py    IPB98y2 confinement scaling
│   │   ├── eqdsk.py           GEQDSK/EQDSK file I/O
│   │   ├── uncertainty.py     Monte Carlo UQ
│   │   └── _rust_compat.py    Rust backend wrapper (PyO3 fallback)
│   ├── control/               Controllers (optional deps guarded)
│   │   ├── h_infinity_controller.py   H-inf robust control
│   │   ├── fusion_sota_mpc.py         Model Predictive Control
│   │   ├── disruption_predictor.py    ML disruption prediction
│   │   ├── spi_mitigation.py          SPI mitigation actuator
│   │   ├── tokamak_digital_twin.py    Digital twin
│   │   ├── tokamak_flight_sim.py      IsoFlux flight simulator
│   │   └── neuro_cybernetic_controller.py  Dual R+Z SNN
│   ├── phase/                 Paper 27 Knm/UPDE phase dynamics
│   │   ├── kuramoto.py        Kuramoto-Sakaguchi step + order parameter
│   │   ├── knm.py             16x16 coupling matrix builder
│   │   ├── upde.py            UPDE multi-layer solver
│   │   ├── lyapunov_guard.py  Sliding-window stability monitor
│   │   ├── realtime_monitor.py Tick-by-tick UPDE + trajectory recorder
│   │   └── ws_phase_stream.py Async WebSocket live stream
│   └── cli.py                 Click CLI
│
├── scpn-control-rs/           Rust workspace (5 crates)
│   ├── crates/
│   │   ├── control-types/     PlasmaState, EquilibriumConfig, ControlAction
│   │   ├── control-math/      LIF neuron, Boris pusher, Kuramoto, upde_tick
│   │   ├── control-core/      GS solver, transport, confinement scaling
│   │   ├── control-control/   PID, MPC, H-inf, SNN controller
│   │   └── control-python/    PyO3 bindings
│   └── deny.toml              cargo-deny supply-chain policy
│
├── tests/                     2019 tests (118 files)
├── validation/                RMSE dashboards + reference data
├── examples/                  Jupyter notebooks + demo scripts
├── dashboard/                 Streamlit control dashboard
├── tools/                     CI gates, preflight, benchmarks
└── docs/                      MkDocs site source
```

## Data Flow

```
Diagnostics (DIII-D/SPARC)
    │
    ▼
ControlObservation  ──►  NeuroSymbolicController
    │                         │
    │                    ┌────┴────┐
    │                    │ SPN     │  Petri net state machine
    │                    │ Compiler│  (structure → LIF network)
    │                    └────┬────┘
    │                         │
    │                    ┌────┴────┐
    │                    │ SNN Pool│  Spiking neural network
    │                    │ (Rust)  │  (LIF neurons + bitstream)
    │                    └────┬────┘
    │                         │
    ▼                         ▼
ControlAction  ◄──  Controllers (PID/MPC/H-inf/SNN)
    │
    ▼
Actuators (coils, gas valves, SPI)
```

## Build Targets

| Target | Command |
|--------|---------|
| Python package | `pip install -e ".[dev]"` |
| Rust engine | `cd scpn-control-rs && cargo build --release` |
| Rust bindings | `cd scpn-control-rs/crates/control-python && maturin develop --release` |
| Tests (Python) | `pytest tests/ -v` |
| Tests (Rust) | `cd scpn-control-rs && cargo test --workspace` |
| Docs | `mkdocs serve` |
| Benchmarks (Rust) | `cd scpn-control-rs && cargo bench --workspace` |
| Benchmarks (Python) | `scpn-control benchmark --n-bench 5000` |
