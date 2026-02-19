# scpn-control

**Neuro-symbolic Stochastic Petri Net controller for plasma control.**

Extracted from [scpn-fusion-core](https://github.com/anulum/scpn-fusion-core) — the minimal 40-file transitive closure of the control pipeline.

## Quick Start

```bash
pip install -e "."
scpn-control demo --steps 1000
scpn-control benchmark --n-bench 5000
```

## Features

- **Petri Net → SNN compilation**: Translates Stochastic Petri Nets into spiking neural network controllers
- **Formal verification**: Contract-based pre/post-condition checking on all control actions
- **Sub-millisecond latency**: <1ms control loop with Rust-accelerated kernels
- **Rust acceleration**: Optional PyO3 bindings for SCPN activation, marking update, and Boris integration
- **Multiple controller types**: PID, MPC, H-infinity, SNN, neuro-cybernetic
- **Digital twin integration**: Real-time telemetry ingest and closed-loop simulation
- **RMSE validation**: CI-gated regression testing against experimental reference data

## Architecture

```
src/scpn_control/
├── scpn/              # Petri net → SNN compiler (structure, compiler, contracts, controller)
├── core/              # Grad-Shafranov solver, transport, scaling laws (clean __init__)
├── control/           # Controllers (PID, MPC, H-inf, SNN, digital twin, disruption)
└── cli.py             # Click CLI (demo, benchmark, validate, hil-test)

scpn-control-rs/       # Rust workspace (5 crates)
├── control-types/     # Type definitions
├── control-math/      # LIF neuron, boris pusher, matrix ops
├── control-core/      # GS solver, transport
├── control-control/   # PID, MPC, H-inf, SNN
└── control-python/    # Slim PyO3 bindings (~400 LOC)
```

## Dependencies

| Required | Optional |
|----------|----------|
| numpy >= 1.24 | matplotlib (`[viz]`) |
| scipy >= 1.10 | streamlit (`[dashboard]`) |
| click >= 8.0 | torch (`[ml]`) |
| | nengo (`[nengo]`) |

## CLI Commands

```bash
scpn-control demo --scenario combined --steps 1000   # Closed-loop control demo
scpn-control benchmark --n-bench 5000                 # PID vs SNN timing
scpn-control validate                                 # RMSE validation
scpn-control hil-test --shots-dir ...                 # HIL test campaign
```

## Dashboard

```bash
pip install -e ".[dashboard]"
streamlit run dashboard/control_dashboard.py
```

## Rust Acceleration

```bash
cd scpn-control-rs
cargo test --workspace
# Build Python bindings:
pip install maturin
maturin develop --release
```

## Citation

```bibtex
@software{sotek2026scpncontrol,
  title = {SCPN Control},
  author = {Sotek, Miroslav and Reiprich, Michal},
  year = {2026},
  url = {https://github.com/anulum/scpn-control},
  license = {AGPL-3.0-or-later}
}
```

## License

GNU AGPL v3. See [LICENSE](LICENSE).
