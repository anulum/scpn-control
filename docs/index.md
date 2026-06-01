<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# SCPN Control

SCPN Control is a standalone neuro-symbolic control engine that compiles
stochastic Petri nets into spiking neural network controllers with formal
contract checks and bounded Petri-net reachability proofs.

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

## Compute validation campaign

SCPN Control now publishes a dedicated [compute validation funding](compute_validation_financing.md)
page for GPU, storage, public-data, and external-code validation needs. The page
keeps the evidence boundary explicit: support funds reproducible validation
artefacts, and full-fidelity claims remain blocked until those artefacts pass
the repository admission gates.
