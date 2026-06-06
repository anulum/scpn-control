<!-- SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Project: SCPN Control -->
<!-- Description: Development guide. -->

# Development

## Local Setup

### Python

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .\.venv\Scripts\activate      # Windows
pip install -e ".[dev]"
```

### Rust (optional)

Requires Rust 1.70+ and maturin:

```bash
pip install maturin
cd scpn-control-rs
cargo build --release
maturin develop --release -m crates/control-python/Cargo.toml
```

After `maturin develop`, `RUST_BACKEND` becomes `True` automatically.

---

## Architecture

Four Python packages under `src/scpn_control/`:

| Package | Purpose |
|---------|---------|
| `core` | GS equilibrium solver, transport, scaling laws, TokamakConfig presets |
| `control` | H-infinity, MPC, SNN, flight sim, disruption predictor, digital twin |
| `scpn` | Stochastic Petri Net → SNN compiler with formal contracts |
| `phase` | Paper 27 Kuramoto-Sakaguchi engine, UPDE, Lyapunov guard, WebSocket stream |

Five Rust crates under `scpn-control-rs/crates/`:

| Crate | Purpose |
|-------|---------|
| `control-types` | PlasmaState, EquilibriumConfig |
| `control-math` | LIF neurons, Boris pusher, Kuramoto |
| `control-core` | Rust GS solver, transport |
| `control-control` | Rust PID, MPC, H-inf, SNN |
| `control-python` | PyO3 bindings |

---

## Running Tests

```bash
pytest tests/ -v                         # full suite
pytest tests/test_h_infinity_controller.py  # single file
pytest -m "not slow"                     # skip slow markers
pytest --cov=scpn_control --cov-report=term  # coverage
```

Coverage gate: 93% (configured in `pyproject.toml`). Coverage claims should
come from the latest local coverage run or the GitHub coverage lane, not from
static documentation text.

---

## Type Checking

```bash
mypy
```

Scope: all modules under `src/scpn_control/` (`disallow_untyped_defs = true`, `warn_return_any = true`).
PEP 561 marker: `src/scpn_control/py.typed`.

---

## Release Process

1. Bump version in `pyproject.toml`, `CITATION.cff`, and `.zenodo.json`
2. Tag and push:

    ```bash
    git tag vX.Y.Z && git push origin vX.Y.Z
    ```

3. CI publishes to PyPI via `publish-pypi.yml`
4. Verify: `pip install scpn-control==X.Y.Z`

---

## Docs

```bash
pip install -e ".[docs]"
mkdocs serve     # preview at http://127.0.0.1:8000
mkdocs build     # static site in site/
```

CI deploys to GitHub Pages on push to `main` via `.github/workflows/docs-pages.yml`.

## How to use this guide in practice

This page defines the engineering path to stable work, not the path for first contact.
Use it after onboarding is complete when you need to:

- reproduce an existing result,
- add a new module behind an existing interface,
- or prepare a release candidate with validation and documentation updates.

Each section is intentionally scoped: setup to make the stack runnable, test and
type gates to keep it safe, and release steps to keep claim and version metadata
aligned.

The docs site includes:

- Full API reference via mkdocstrings, including a complete module index for
  every tracked Python module under `src/scpn_control/`
- Theory page with rendered MathJax equations
- Architecture diagrams via Mermaid
- Notebook gallery with execution instructions
- Changelog, benchmarks, and validation reports

## Practical use and scope

Use this guide for change workflow in `scpn-control` itself.

- Run setup and local checks here before editing core modules.
- Use the workflow before opening implementation tasks that affect CI or packaging.
- Keep claim-boundary and admission checks in lockstep with this guide when production-relevant files change.

