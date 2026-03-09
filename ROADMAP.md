# Roadmap

## Shipped

### v0.1.0 — 2026-02-19
- [x] Extract 41 modules + 5 Rust crates from scpn-fusion-core
- [x] Click CLI (demo, benchmark, validate, hil-test)
- [x] Streamlit dashboard
- [x] CI: 6 jobs, 482 tests

### v0.2.0 — 2026-02-26
- [x] Paper 27 phase dynamics engine (Kuramoto, Knm, UPDE, Lyapunov guard)
- [x] WebSocket live phase stream
- [x] Streamlit Cloud deployment
- [x] CI: 12 jobs, 680 tests

### v0.3.x — 2026-02-27
- [x] Ruff linter + CI job
- [x] Property-based tests (Hypothesis)
- [x] Gymnasium TokamakEnv
- [x] IMAS/OMAS adapter
- [x] VectorizedSCLayer + Rust SNN path
- [x] MIT OR Apache-2.0 dual license
- [x] Pre-commit hooks, Docker, devcontainer
- [x] MkDocs site (theory, architecture, mkdocstrings)
- [x] CI: 13 jobs, 701 tests

### v0.4.0 — 2026-03-01
- [x] Real-time adaptive Knm engine (beta, MHD, coherence PI, rate limit, Lyapunov veto)
- [x] Zenodo metadata (.zenodo.json, CITATION.cff)
- [x] CI: 15 jobs, 1888 tests

### v0.5.x — 2026-03-02
- [x] PEP 621 `__version__`, cargo-deny supply-chain policy
- [x] UNDERDEVELOPED_REGISTER tracking gaps
- [x] SECURITY.md, CODE_OF_CONDUCT.md
- [x] Shared validators (core/_validators.py)
- [x] Coverage gate: 62% (actual: 93%)
- [x] Python 3.13 in CI matrix
- [x] CI: 15 jobs, ~1900 tests

### v0.6.0 — 2026-03-02
- [x] .editorconfig, CODEOWNERS, copyright headers on CI workflows
- [x] Typing modernization (`from __future__ import annotations`, `Optional[X]` → `X | None`)
- [x] Shared test fixtures (conftest.py deduplication)
- [x] 63 new tests (validators + Hypothesis property tests)
- [x] Paper 27 citations on OMEGA_N_16 and build_knm_paper27
- [x] Anti-slop cleanup (dead constants, narration comments, print→logging)
- [x] Rust Cargo.toml `repository` fields
- [x] CI: 15 jobs, 1969 tests

### v0.7.0 — 2026-03-02
- [x] Coverage gate ratcheted 62% → 85%
- [x] Nengo SNN wrapper tests (14 mocked)
- [x] E2E compile-to-control integration tests
- [x] `require_range` validator, doc comments on 11 Rust functions
- [x] `print()` → `logging` in 13 control modules (58 sites)
- [x] TokamakTopology rename, stale doc sweep

### v0.7.1 — 2026-03-09
- [x] Enterprise repo hardening (18 new files, 5 new workflows)
- [x] SHA-pinned GitHub Actions across all workflows
- [x] Top-level `permissions: {}`, concurrency groups, SPDX headers
- [x] Typos checker with domain-specific allowlist (46 terms)
- [x] OpenSSF Scorecard, CodeQL, pre-commit, release, stale workflows
- [x] 17 GitHub labels, squash-only merge, discussions enabled
- [x] Dependabot PR triage (1 merged, 4 closed as incompatible)

### v0.8.0 — 2026-03-09
- [x] Python fallback for `RustSPIMitigation` ([gh-13](https://github.com/anulum/scpn-control/issues/13), U-003)
- [x] Python fallback for `rust_multigrid_vcycle` ([gh-14](https://github.com/anulum/scpn-control/issues/14), U-004)
- [x] Python fallback for `rust_svd_optimal_correction` ([gh-15](https://github.com/anulum/scpn-control/issues/15), U-005)
- [x] Factor `np.isfinite` validation boilerplate ([gh-17](https://github.com/anulum/scpn-control/issues/17), U-007)
- [x] `require_bounded_float`, `require_finite_array` validators
- [x] 16 new fallback tests

### v0.8.1 — 2026-03-10
- [x] Rust H-inf: Euler → Padé(6,6) matrix exponential ZOH + DARE ([gh-10](https://github.com/anulum/scpn-control/issues/10), U-001)
- [x] E2E control latency benchmark (`benchmarks/e2e_control_latency.py`)
- [x] Honesty sweep: fix overstated claims in README, pitch, use_cases, VALIDATION, competitive_analysis
- [x] Stale doc counts updated (2019 tests, 118 files, 54 modules)

## Next

### v0.9.0 — CI expansion + data ingest
- [ ] mypy `strict = true` (incremental, per-module)
- [ ] Nengo Loihi emulator CI test ([gh-11](https://github.com/anulum/scpn-control/issues/11), U-002)
- [ ] JAX tracing in optional CI matrix ([gh-12](https://github.com/anulum/scpn-control/issues/12), U-006)
- [ ] IMAS IDS round-trip tests with real `omas` install
- [ ] Real MDSplus data ingest (DIII-D, JET public shots)
- [ ] Codecov badge + token setup ([gh-17 TODO](https://github.com/anulum/scpn-control/issues/17))

### v1.0.0 — Production readiness
- [ ] Streamlit dashboard v2 (shot replay + multi-machine selector)
- [ ] GPU-accelerated Grad-Shafranov (CuPy / JAX backend)
- [ ] Neural equilibrium pre-trained weights (SPARC, ITER)
- [ ] TORAX hybrid transport coupling
- [ ] Coordinated Rust dep upgrade (ndarray 0.16+, ndarray-linalg 0.18+, rand 0.9)

## Future
- [ ] Stellarator geometry support (Wendelstein 7-X)
- [ ] Federated learning for multi-machine disruption prediction
- [ ] FPGA bitstream export from SNN compiler
- [ ] ITER CODAC interface prototype
