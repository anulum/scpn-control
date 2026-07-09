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
pytest --cov=scpn_control --cov-report=term --cov-fail-under=100
```

Coverage gate: 100% (configured in `pyproject.toml`). Coverage claims should
come from the latest local coverage run or the GitHub coverage lane, not from
static documentation text.

---

## Type Checking

```bash
mypy
python tools/run_mypy_strict.py
```

Scope: all modules under `src/scpn_control/` (`disallow_untyped_defs = true`, `warn_return_any = true`).
PEP 561 marker: `src/scpn_control/py.typed`.

`tools/run_mypy_strict.py` is the local preflight gate for strict-typing debt. It
first runs the configured repository mypy check, then runs a whole-package
`mypy --strict src/scpn_control/` probe and compares the result with
`tools/mypy_strict_debt.json`. The ratchet may stay flat or fall; increases need
an explicit baseline update with the increase flag so added strict debt remains a
reviewed change.

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

Native-dependent modules use a two-environment coverage matrix because the
authoritative Python job runs without the optional `scpn_control_rs` extension
while the interop job builds it. CI uploads `coverage-data-python` and
`coverage-data-rust`, then `native-coverage-combine` runs:

```bash
coverage combine --keep artifacts/coverage/python artifacts/coverage/rust
coverage report --fail-under=100
```

`scpn-native-coverage-matrix` checks that the workflow, threshold, and public
docs still describe the `scpn-control.native-coverage-matrix.v1` contract. The
same guard runs through `tools/preflight.py`.

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

## Public Surface Hygiene

```bash
python tools/check_public_surface_hygiene.py
```

This guard scans tracked outward-facing text files and fails on bare
self-applied promotion terms. Internal planning surfaces under `docs/internal/`
and `.coordination/` are excluded; bounded negative language and candidate
labels remain allowed because they do not assert an achieved public claim.

## Changelog Mirror

```bash
python tools/check_changelog_sync.py
```

`CHANGELOG.md` is the authoritative release history. `docs/changelog.md` is the
rendered MkDocs mirror and must stay byte-identical to the root file. The guard
runs in CI, local preflight, pre-commit, and `make lint`.

## Public API Docstrings

```bash
python tools/run_docstring_gate.py
```

The docstring gate runs ruff's public API pydocstyle rules for classes,
functions, methods, packages, and nested classes. The recorded debt is zero, so
new public APIs without docstrings fail CI, local preflight, and `make lint`.
Docstrings should name the technical contract, units, failure modes, and claim
boundaries where those details matter.

## Studio Custody Guards

```bash
python tools/check_studio_deploy_key.py
python tools/check_studio_offline_sealing.py
```

`check_studio_deploy_key.py` validates the tracked Studio deploy public key, the
CI rsync deploy workflow, and private deploy-key exclusion.
`check_studio_offline_sealing.py` keeps Studio publication signing custody
offline: workflows, Studio surfaces, docs, and tools must not reference
Hub/Studio sealing or signing private-key secrets, and tracked policy surfaces
must not contain private-key blocks. The guard deliberately allows deploy-only
SSH credentials because they do not sign evidence; sealed evidence keys stay
with the Studio keeper.

---

## Release Process

1. Bump version in `pyproject.toml`, `CITATION.cff`, and `.zenodo.json`
   and run `python tools/check_version_sync.py` to verify release notes,
   README PyPI/Python-version badges, the Pepy all-time downloads badge, and
   local version metadata.
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

## JOSS Submission Review

Run `python tools/check_joss_submission.py` before sending the paper to an
external JOSS workflow. The guard checks the canonical `paper.md` front matter,
the root `paper.bib` bibliography, bracketed citation coverage, the
`docs/joss_paper.md` mirror, and the claim-boundary/editorial text that keeps
benchmark and validation statements tied to admitted evidence.

The guard runs in local preflight and CI lint so the paper, docs mirror, and
bibliography drift together instead of relying on a manual editorial pass.

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
