# Underdeveloped Register — scpn-control

Tracks known gaps, stubs, and incomplete implementations.

| ID | Component | Gap | Severity | Tracking |
|----|-----------|-----|----------|----------|
| U-001 | `control-control/h_infinity.rs` | `update_discretization()` uses Euler; ZOH + DARE not implemented | P1 | [gh-10](https://github.com/anulum/scpn-control/issues/10) |
| U-002 | `control/nengo_snn_wrapper.py` | Nengo Loihi backend untested in CI (requires `nengo_loihi` hardware or emulator) | P2 | [gh-11](https://github.com/anulum/scpn-control/issues/11) |
| U-003 | `core/_rust_compat.py` `RustSPIMitigation` | No Python fallback; Rust-only. Users without Rust get `ImportError` | P2 | [gh-13](https://github.com/anulum/scpn-control/issues/13) |
| U-004 | `core/_rust_compat.py` `rust_multigrid_vcycle` | No Python fallback; Rust-only | P2 | [gh-14](https://github.com/anulum/scpn-control/issues/14) |
| U-005 | `core/_rust_compat.py` `rust_svd_optimal_correction` | No Python fallback; Rust-only | P2 | [gh-15](https://github.com/anulum/scpn-control/issues/15) |
| U-006 | `control/jax_traceable_runtime.py` | JAX tracing gated on `jax` install; untested in CI matrix | P3 | [gh-12](https://github.com/anulum/scpn-control/issues/12) |
| U-007 | 21 files (124 occurrences) | Repeated `np.isfinite` validation boilerplate | P3 | [gh-17](https://github.com/anulum/scpn-control/issues/17) |
