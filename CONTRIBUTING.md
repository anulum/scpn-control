# Contributing to scpn-control

## Development setup

```bash
git clone https://github.com/anulum/scpn-control.git
cd scpn-control
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

Rust toolchain (stable) is required for the PyO3 crates:

```bash
cd scpn-control-rs
cargo test
cargo clippy -- -D warnings
```

To build the Python extension module from Rust:

```bash
pip install maturin
cd scpn-control-rs/crates/control-python
maturin develop --release
```

## Running tests

Pre-commit hooks (ruff, mypy, yaml, whitespace):

```bash
pre-commit run --all-files    # check everything
pre-commit run ruff            # single hook
```

Python (701 tests):

```bash
pytest                       # full suite
pytest tests/ -k "not slow"  # skip long-running regression tests
pytest --cov=scpn_control    # with coverage (minimum: 50%)
```

Rust (5 crates: control-core, control-math, control-types, control-control, control-python):

```bash
cd scpn-control-rs
cargo test --workspace
```

Type checking (scoped to configured files only):

```bash
mypy
```

## Code style

This project enforces an anti-slop policy. Read the full rules in `CLAUDE.md`
at the repository root. Summary:

**Comments** exist only to cite papers/equations, state non-obvious assumptions,
or mark `TODO(gh-ISSUE):` items. Delete anything that restates what the code does.

**Names** must describe what code does, not what it aspires to be.
A 2-layer MLP is `DynamicsMLP`, not `NeuralODEDynamics`.
Gradient descent on an action sequence is `trajectory_optimizer`, not `NMPC`.

**Magic numbers** require a source citation or a named constant with units:
```python
COULOMB_LOG = 15.0  # dimensionless, Wesson Ch.14
```

**No trivial wrappers.** If a function body can be inlined at the call site, inline it.

**No cargo-cult typing.** Type hints are mandatory on public API signatures.
Internal helpers do not need annotations nobody checks.

**Commit messages**: imperative mood, under 72 characters, no filler.
`Fix Riccati sign error in H-inf observer` -- not a paragraph about what
was comprehensively addressed.

## Submitting changes

1. Fork the repository and create a feature branch off `main`.
2. Keep commits atomic. One logical change per commit.
3. Open a pull request against `main`.
4. CI must pass (14 jobs: lint, type-check, pre-commit, pytest matrix,
   cargo test/clippy, maturin build, mkdocs build, coverage gate).
5. At least one maintainer review is required before merge.

If your change touches Rust code, ensure `cargo clippy -- -D warnings` is clean.
If your change touches Python public API, add or update tests.

## Reporting issues

File issues at <https://github.com/anulum/scpn-control/issues>.

Include:
- Python version, OS, Rust toolchain version (if relevant)
- Minimal reproduction steps
- Full traceback or error output

For security vulnerabilities, email protoscience@anulum.li directly.
Do not open a public issue.

## License

scpn-control is licensed under AGPL-3.0-or-later. All contributions are
released under the same license. No CLA is required, but contributions
must be AGPL-compatible (no proprietary or GPL-incompatible dependencies).

By submitting a pull request, you agree that your contribution is licensed
under AGPL-3.0-or-later.
