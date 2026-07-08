<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Development guide. -->

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

Coverage gate: 99% (configured in `pyproject.toml`). Coverage claims should
come from the latest local coverage run or the GitHub coverage lane, not from
static documentation text.

---

## Type Checking

```bash
mypy
```

Scope: all modules under `src/scpn_control/` (`disallow_untyped_defs = true`, `warn_return_any = true`).
PEP 561 marker: `src/scpn_control/py.typed`.

## Wiring Checks

Two local guards keep source modules connected to real repository surfaces:

```bash
python tools/check_test_module_linkage.py
python tools/check_runtime_wiring.py
```

`check_test_module_linkage.py` confirms production modules have direct test
linkage or an explicit allowlist entry. `check_runtime_wiring.py` parses imports
across `src`, `tests`, `benchmarks`, `examples`, `tools`, and `validation`; it
fails if a source module is referenced by no package API, test, tool, or
pipeline file. Both gates run through `tools/preflight.py`.

## Coverage Exclusions

Every source `# pragma: no cover` exclusion must explain why the line is not
covered by the local Python coverage job:

```bash
python tools/check_coverage_pragmas.py
```

Use a one-line reason such as an optional dependency path, native backend path,
or defensive invariant branch. The gate runs through `tools/preflight.py` and
fails on bare exclusions.

## GitHub Token Format Guard

The security lane runs `python tools/check_github_token_format_readiness.py` in
CI. The guard scans tracked text files and workflow files for brittle GitHub
installation-token assumptions: exact-width `ghs_` regexes, fixed token-length
checks, undersized storage columns, and installation-token endpoint calls that
omit `X-GitHub-Stateless-S2S-Token`.

Treat installation tokens as opaque strings. Code may check for presence,
prefixes needed for routing, or provider errors, but it must not assume a fixed
length or storage width. Test fixtures live under `tests/` and are intentionally
excluded from the repository scan so negative examples remain possible without
making the gate flag itself.

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
