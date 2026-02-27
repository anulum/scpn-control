# SCPN Control

SCPN Control is a standalone neuro-symbolic control engine that compiles
stochastic Petri nets into spiking neural network controllers with formal
contract checks.

![SCPN Control Header](scpn_control_header.png)

## Visual overview

![SCPN Control Logic Map](SCPN%20CONTROL.PNG)
![SCPN Control Logic Map (compact)](SCPN%20CONTROL%20small.png)

## What is in this repository

- Python package: `src/scpn_control/`
- Rust workspace: `scpn-control-rs/`
- Tutorial notebooks: `examples/`
- Validation and benchmark assets: `validation/`, `tests/`, `tools/`

## Quick start

```bash
pip install scpn-control                        # core (numpy, scipy, click)
pip install "scpn-control[dashboard,ws]"        # + Streamlit + WebSocket
scpn-control demo --steps 1000
scpn-control benchmark --n-bench 5000
scpn-control validate
```

## Build and verification baseline

Use this baseline before release:

```bash
# Python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p hypothesis.extra.pytestplugin tests/ -q

# Rust workspace
cd scpn-control-rs
cargo build --workspace
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

For notebook execution, see [Tutorials](tutorials.md). For parity and CI checks,
see [Validation and QA](validation.md). For performance methodology, see
[Benchmarks](benchmarks.md).
