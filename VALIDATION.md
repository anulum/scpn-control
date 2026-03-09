# Validation

## Test Matrix

| Suite | Count | Scope |
|-------|------:|-------|
| Python unit/integration | 1969 | `pytest tests/` across 115 files |
| Rust engine | 108+ | `cargo test --workspace` in `scpn-control-rs/` |
| Rust-Python interop | 3 files | PyO3 parity tests via maturin |
| Notebooks | 5 | Executed in CI via `nbconvert` |
| E2E (DIII-D mock) | 1 file | Full shot-driven control loop |
| RMSE gate | 1 file | Regression against DIII-D/SPARC reference |

CI runs tests on Python 3.9-3.13 (Ubuntu), Rust on Ubuntu.

## CI Validation Gates

All gates must pass before merge to `main`.

| Gate | Workflow | Enforces |
|------|----------|----------|
| ruff check | `ci.yml` | Import hygiene, code quality |
| ruff format | `ci.yml` | Consistent formatting |
| bandit | `ci.yml` | Security static analysis (SAST) |
| test + coverage | `ci.yml` | `pytest --cov-fail-under=85` on Python 3.12 |
| mypy | `ci.yml` | Type checking (scoped files) |
| notebook smoke | `ci.yml` | All tutorial notebooks execute |
| package quality | `ci.yml` | `twine check` on built sdist/wheel |
| pip-audit | `ci.yml` | No known vulnerabilities in deps |
| RMSE gate | `ci.yml` | Regression bounds vs. reference data |
| E2E DIII-D | `ci.yml` | End-to-end mock shot test |
| rust-tests | `ci.yml` | `cargo test --workspace` + clippy + fmt |
| rust-python-interop | `ci.yml` | Maturin build + parity tests |
| rust-benchmarks | `ci.yml` | Criterion benchmarks (no regression gate) |
| cargo-deny | `ci.yml` | License + advisory supply-chain policy |
| cargo-audit | `ci.yml` | Rust dependency vulnerability scan |
| pre-commit | `pre-commit.yml` | Trailing whitespace, YAML, formatting |
| CodeQL | `codeql.yml` | GitHub CodeQL security analysis |
| Scorecard | `scorecard.yml` | OpenSSF Scorecard supply-chain audit |

## Coverage Policy

- Threshold: 85% (enforced by `pytest --cov-fail-under=85`)
- Excluded lines: `pragma: no cover`, `if __name__`, `raise NotImplementedError`,
  conditional imports (`HAS_TORCH`, `HAS_NENGO`, `HAS_SC_NEUROCORE`, `except ImportError`)

## RMSE Validation

The `validation/rmse_dashboard.py` script computes pointwise RMSE against
reference GEQDSK equilibria (DIII-D, SPARC) and enforces bounds in CI
via `tools/ci_rmse_gate.py`.

## Benchmark Gates

| Benchmark | Budget | Scope |
|-----------|--------|-------|
| Kuramoto step (N=1000) | < 5 ms P50 | Single-step latency |
| Kuramoto step (N=4096) | < 5 ms P50 | DIII-D scale |
| RealtimeMonitor tick | < 50 ms P50 | 16 layers x 50 oscillators |
