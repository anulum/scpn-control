# Changelog

## [0.3.2] — 2026-02-27

### Added
- VectorizedSCLayer + Rust backend path in SNN compiler (512× real-time)
- Two-tier import: v3.8.0+ preferred → legacy bit-ops → numpy float fallback
- Test for v3.8 detection and VectorizedSCLayer forward-path benchmark
- sc-neurocore listed first in optional deps table (crown jewel)

### Changed
- README: engine callout and dep table updated for sc-neurocore

## [0.3.0] — 2026-02-27

### Added
- Ruff linter (E/F/W/I/B rules) — CI job + pyproject.toml config
- Property-based tests for phase/ module (Hypothesis, 11 properties)
- Gymnasium-compatible TokamakEnv (control/gym_tokamak_env.py, 10 tests)
- IMAS/OMAS equilibrium adapter (core/imas_adapter.py)
- CLI `scpn-control info` command (version, Rust status, weights, Python/NumPy)
- Weight provenance manifest (reproduction commands, hardware, training config)
- Paper 27 + H-infinity notebooks in CI smoke tests

### Fixed
- API docs: wrong snapshot keys (R→R_global, V→V_global, lambda→lambda_exp)
- API docs: wrong UPDESystem constructor and LyapunovGuard API examples
- README/CHANGELOG: "14 CI jobs" → actual count, test counts updated
- `_rust_compat.py`: calculate_thermodynamics/vacuum_field now delegate to Python
- 163 ruff auto-fixes (whitespace, import sorting, unused imports)
- 26 manual ruff fixes (raise-from, unused variables, one-liners, E402)
- Bandit now fails on medium+ severity (was --exit-zero)
- TeX build artifacts (.aux/.log/.out/.toc) excluded from repo

### Changed
- Coverage threshold raised from 50% → 55% (actual: 61%)
- CI: 12 → 13 jobs (added python-lint)
- Test suite: 680 → 701 tests (50 test files)

## [0.2.0] — 2026-02-26

### Added
- Paper 27 phase dynamics engine (`src/scpn_control/phase/`, 7 modules)
- Kuramoto-Sakaguchi step with global field driver (kuramoto.py)
- 16x16 Knm coupling matrix builder with calibration anchors (knm.py)
- UPDE multi-layer solver with PAC gating (upde.py)
- LyapunovGuard sliding-window stability monitor (lyapunov_guard.py)
- RealtimeMonitor tick-by-tick UPDE + trajectory recorder (realtime_monitor.py)
- PhaseStreamServer async WebSocket live stream (ws_phase_stream.py)
- CLI `scpn-control live` command for real-time WS phase sync server
- Streamlit WS client (`examples/streamlit_ws_client.py`)
- Streamlit Cloud deployment (`streamlit_app.py`, `.streamlit/config.toml`)
- Mock DIII-D shot generator (`tests/mock_diiid.py`)
- E2E phase sync with shot data tests (`tests/test_e2e_phase_diiid.py`)
- Phase sync convergence video (MP4 + GIF) and generator script
- PyPI publish script (`tools/publish.py`)
- Rust `upde_tick()` in control-math + PyRealtimeMonitor PyO3 binding

### Changed
- CI expanded from 6 to 12 jobs
- Test suite expanded from 482 to 680 tests (680 passing, 94 skipped)
- README updated with `<video>` MP4 embed, Streamlit Cloud badge
- `.gitignore` updated to allow docs GIF/PNG and Streamlit config

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
