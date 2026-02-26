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

Minimum coverage: 50% (configured in `pyproject.toml`).

---

## Type Checking

```bash
mypy
```

Scope: `scpn/`, select `core/` and `control/` modules.
PEP 561 marker: `src/scpn_control/py.typed`.

---

## Release Process

1. Bump version in `pyproject.toml` and `src/scpn_control/__init__.py`
2. Tag and push:

    ```bash
    git tag v0.2.1 && git push origin v0.2.1
    ```

3. CI publishes to PyPI via `publish-pypi.yml`
4. Verify: `pip install scpn-control==0.2.1`

---

## Docs

```bash
pip install mkdocs-material
mkdocs serve     # preview at http://127.0.0.1:8000
mkdocs build     # static site in site/
```

CI deploys to GitHub Pages on push to `main`.
