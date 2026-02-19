# Changelog

## [0.1.0] — 2026-02-19

### Added
- Initial extraction from scpn-fusion-core v3.4.0
- 41 Python source files (minimal control transitive closure)
- 5 Rust crates (control-types, control-math, control-core, control-control, control-python)
- Slim PyO3 bindings (~474 LOC, control-only)
- Clean `__init__.py` files (no matplotlib/GPU/ML import bombs)
- Click CLI with 4 commands (demo, benchmark, validate, hil-test)
- Streamlit dashboard (optional, `[dashboard]` extra)
- CI workflow (6 jobs: python-tests, rmse-gate, rust-tests, rust-python-interop, rust-benchmarks, rust-audit)
- 37 test files (482 passing, 122 skipped, 0 failures)
- 16 disruption shot reference data files
- 8 SPARC EFIT equilibria
- 3 pretrained weight files
- Validation scripts and tools
- Stochastic Petri Net logic map header (`docs/scpn_control_header.png`)

### Changed (vs scpn-fusion-core)
- Required deps: numpy, scipy, click ONLY (was: numpy, scipy, matplotlib, streamlit)
- matplotlib, streamlit, torch, nengo moved to optional extras
- All imports renamed: `scpn_fusion` -> `scpn_control`, `scpn_fusion_rs` -> `scpn_control_rs`
- Rust workspace reduced from 11 crates to 5
- CI reduced from 13 jobs to 6
- `hpc_bridge.py` relocated from `hpc/` to `core/` subpackage
- Import guards added for excluded modules (stability_analyzer, global_design_scanner, imas_connector, diagnostics.forward, fusion_ignition_sim)
