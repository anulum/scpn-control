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
- [x] Stale doc counts updated

### v0.9.0 — 2026-03-10
- [x] mypy `disallow_untyped_defs`, `warn_return_any` across all 54 modules
- [x] Nengo Loihi CI job (Job 17, [gh-11](https://github.com/anulum/scpn-control/issues/11), U-002)
- [x] JAX parity CI job (Job 16, [gh-12](https://github.com/anulum/scpn-control/issues/12), U-006)
- [x] Cross-platform CI: Windows + macOS runners
- [x] Codecov `fail_ci_if_error: true`
- [x] PEP 561 `py.typed` marker
- [x] Complete API docs (16 modules added)
- [x] RMSE threshold documentation
- [x] `[jax]` and `[loihi]` optional dependency groups
- [x] All 7 UNDERDEVELOPED_REGISTER items RESOLVED

### v0.10.0 — 2026-03-10
- [x] JAX-accelerated transport primitives (Thomas solver, Crank-Nicolson, vmap batching)
- [x] Autodifferentiation through transport solver (`jax.grad`)
- [x] Real DIII-D shot validation (17 shots, 95 tests, CI job)
- [x] JOSS paper draft (`paper.md`, `paper.bib`)
- [x] 25 CI jobs, 2024+ tests

### v0.11.0 — 2026-03-10
- [x] JAX neural equilibrium MLP (`jax_neural_equilibrium.py`)
- [x] GPU-accelerated equilibrium inference via jaxlib
- [x] `jax.grad` through equilibrium predict (adjoint-based shape optimization)
- [x] `jax.vmap` batch equilibrium (100+ simultaneous)
- [x] NumPy fallback when JAX unavailable

### v0.12.0 — 2026-03-10
- [x] QLKNN-10D trained transport model (synthetic critical-gradient, Zenodo-ready)
- [x] Training script `tools/train_neural_transport_qlknn.py` + `--synthetic` CI mode
- [x] Auto-discovery: `NeuralTransportModel()` loads weights from `weights/`
- [x] 20 QLKNN tests (weight validation, inference, profile, training E2E)
- [x] PPO agent on TokamakEnv (stable-baselines3 PPO, 50K timesteps)
- [x] Gymnasium wrapper (`GymTokamakEnv`) with proper `spaces.Box`
- [x] RL vs PID vs MPC benchmark (`benchmarks/rl_vs_classical.py`)
- [x] 14 RL tests (wrapper, PID, agent loading, training E2E)
- [x] `[rl]` optional dependency group
- [x] TokamakEnv q95 physics fix (elongation factor)

### v0.13.0 — 2026-03-10
- [x] JAX-differentiable Grad-Shafranov solver (Picard + Jacobi via `lax.fori_loop`)
- [x] `jax.grad` through full equilibrium solve (closes autodiff depth gap)
- [x] Reward shaping for PPO training (survival bonus, progress, Ng et al. 1999)
- [x] `examples/quickstart.py` — 30-second Python demo
- [x] README quickstart block
- [x] JOSS paper updated to v0.13.0 metrics (57 modules, 2,201 tests)
- [x] 20 new JAX GS solver tests

### v0.14.0 — 2026-03-10
- [x] PPO 500K training on JarvisLabs RTX5000 (3 seeds, best-seed selection)
- [x] PPO beats MPC and PID: reward=143.7 vs MPC=58.1 vs PID=-912.3
- [x] 0% disruption rate across all controllers
- [x] Reproducible: 3 seeds within +-0.2 mean reward
- [x] Cloud training script (`tools/train_rl_upcloud.sh`)
- [x] JarvisLabs automation script (`tools/jarvislabs_train.py`)
- [x] 100% test coverage: 2,404 tests, 9,652 statements, 0 missed
- [x] Coverage gate ratcheted: 85% → 99%
- [x] 25 CI jobs all green

## Next

### v0.15.0 — JOSS submission
- [ ] JOSS paper submission (fact-checked, final claims)

### v1.0.0 — Production readiness
- [ ] IMAS IDS round-trip tests with real `omas` install
- [ ] Streamlit dashboard v2 (shot replay + multi-machine selector)
- [ ] Neural equilibrium pre-trained weights (SPARC, ITER)
- [ ] TORAX hybrid transport coupling
- [ ] Coordinated Rust dep upgrade (ndarray 0.16+, ndarray-linalg 0.18+, rand 0.9)

## Future
- [ ] Stellarator geometry support (Wendelstein 7-X)
- [ ] Federated learning for multi-machine disruption prediction
- [ ] FPGA bitstream export from SNN compiler
- [ ] ITER CODAC interface prototype
