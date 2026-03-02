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

### v0.5.x — 2026-03-02 (current)
- [x] PEP 621 `__version__`, cargo-deny supply-chain policy
- [x] UNDERDEVELOPED_REGISTER tracking gaps
- [x] SECURITY.md, CODE_OF_CONDUCT.md
- [x] Shared validators (core/_validators.py)
- [x] Coverage gate: 62% (actual: 93%)
- [x] Python 3.13 in CI matrix
- [x] CI: 15 jobs, ~1900 tests

## Next

### v0.6.0
- [ ] Codecov badge ≥ 65% (from 62% gate)
- [ ] mypy `strict = true` (incremental, per-module)
- [ ] IMAS IDS round-trip tests with real `omas` install
- [ ] Streamlit dashboard v2 (shot replay + multi-machine)

### v0.7.0
- [ ] GPU-accelerated Grad-Shafranov (CuPy / JAX backend)
- [ ] Real MDSplus data ingest (DIII-D, JET public shots)
- [ ] Neural equilibrium pre-trained weights (SPARC, ITER)
- [ ] TORAX hybrid transport coupling

## Future
- [ ] Stellarator geometry support (Wendelstein 7-X)
- [ ] Federated learning for multi-machine disruption prediction
- [ ] FPGA bitstream export from SNN compiler
- [ ] ITER CODAC interface prototype
