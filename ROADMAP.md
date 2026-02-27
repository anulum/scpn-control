# Roadmap

## v0.3.x (current)

- [x] Ruff linter integration + CI job
- [x] Property-based tests (Hypothesis)
- [x] Gymnasium `TokamakEnv`
- [x] IMAS/OMAS equilibrium adapter
- [x] Docker + devcontainer
- [x] Pre-commit hooks
- [x] MkDocs: theory, architecture, mkdocstrings
- [x] Zenodo archival metadata
- [ ] Codecov badge > 65%

## v0.4.0

- [ ] Pydantic v2 for all config/contract types
- [ ] mypy `strict = true` (incremental, per-module)
- [ ] IMAS IDS round-trip tests with real `omas` install
- [ ] Streamlit dashboard v2 (shot replay + multi-machine)

## v0.5.0

- [ ] GPU-accelerated Grad-Shafranov (CuPy / JAX backend)
- [ ] Real MDSplus data ingest (DIII-D, JET public shots)
- [ ] Neural equilibrium pre-trained weights (SPARC, ITER)
- [ ] TORAX hybrid transport coupling

## Future

- [ ] Stellarator geometry support (Wendelstein 7-X)
- [ ] Federated learning for multi-machine disruption prediction
- [ ] FPGA bitstream export from SNN compiler
- [ ] ITER CODAC interface prototype
