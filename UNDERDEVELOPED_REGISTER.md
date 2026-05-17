<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Underdeveloped Register — scpn-control

Tracks known gaps, stubs, and incomplete implementations.

| ID | Component | Gap | Severity | Tracking |
|----|-----------|-----|----------|----------|
| ~~U-001~~ | `control-control/h_infinity.rs` | ~~`update_discretization()` uses Euler; ZOH + DARE not implemented~~ **RESOLVED v0.8.1**: Padé(6,6) scaling-and-squaring `matrix_exp` + `zoh_discretize`, 6 new tests | — | [gh-10](https://github.com/anulum/scpn-control/issues/10) |
| ~~U-002~~ | `control/nengo_snn_wrapper.py` | ~~Nengo Loihi backend untested in CI~~ **RESOLVED v0.9.0**: dedicated `nengo-loihi` CI job (Job 17) with `nengo>=4.0` | — | [gh-11](https://github.com/anulum/scpn-control/issues/11) |
| ~~U-003~~ | `core/_rust_compat.py` `RustSPIMitigation` | ~~No Python fallback~~ **RESOLVED v0.8.0**: Python fallback matching Rust spi.rs constants | — | [gh-13](https://github.com/anulum/scpn-control/issues/13) |
| ~~U-004~~ | `core/_rust_compat.py` `rust_multigrid_vcycle` | ~~No Python fallback~~ **RESOLVED v0.8.0**: delegates to FusionKernel._multigrid_vcycle | — | [gh-14](https://github.com/anulum/scpn-control/issues/14) |
| ~~U-005~~ | `core/_rust_compat.py` `rust_svd_optimal_correction` | ~~No Python fallback~~ **RESOLVED v0.8.0**: NumPy SVD pseudoinverse fallback | — | [gh-15](https://github.com/anulum/scpn-control/issues/15) |
| ~~U-006~~ | `control/jax_traceable_runtime.py` | ~~JAX tracing untested in CI~~ **RESOLVED v0.9.0**: `[jax]` optional dep group + dedicated `jax-parity` CI job (Job 16) | — | [gh-12](https://github.com/anulum/scpn-control/issues/12) |
| ~~U-007~~ | 21 files (124 occurrences) | ~~Repeated `np.isfinite` validation boilerplate~~ **RESOLVED v0.8.0**: shared validators (`require_bounded_float`, `require_finite_array`, etc.) in `core/_validators.py`; 3 P1 modules converted | — | [gh-17](https://github.com/anulum/scpn-control/issues/17) |
| ~~U-008~~ | `control/rzip_model.py` `VerticalStabilityAnalysis.compute_n_index` | ~~Returned hard-coded `-1.0` instead of deriving the vertical field index from the flux grid~~ **RESOLVED v0.18.x**: computes `B_Z = (1/R) dpsi/dR`, evaluates `n = -(R0/B_Z) dB_Z/dR` at the midplane, and rejects degenerate grids | — | local |
| ~~U-009~~ | `control/fault_tolerant_control.py` `ReconfigurableController.handle_sensor_fault` | ~~Sensor fault handler was a no-op after FDI isolation~~ **RESOLVED v0.18.x**: records isolated sensors, removes their allocation weight, recomputes the gain, masks faulted residuals during control allocation, and rejects invalid sensor indices | — | local |
| ~~U-010~~ | `core/sol_model.py` `detachment_threshold` | ~~Returned `False` for every SOL state~~ **RESOLVED v0.18.x**: evaluates a two-point Spitzer/sheath target-temperature criterion and reports detachment onset when `T_target <= 5 eV`, with fail-fast validation for non-positive inputs | — | local |
| ~~U-011~~ | `control/realtime_efit.py` `RealtimeEFIT.find_lcfs` | ~~Returned a zero-array LCFS stub~~ **RESOLVED v0.18.x**: extracts finite LCFS boundary points from the positive closed-flux region on the EFIT R/Z grid and angle-sorts the contour | — | local |
| ~~U-012~~ | `control/realtime_efit.py` `DiagnosticResponse.simulate_measurements` | ~~Returned a hard-coded Rogowski current independent of the flux grid~~ **RESOLVED v0.18.x**: derives plasma current by integrating `J_phi = -Delta*psi / (mu0 R)` over the diagnostic R/Z grid with fail-fast flux validation | — | local |
| ~~U-013~~ | `control/realtime_efit.py` `RealtimeEFIT._solve_gs_with_sources` | ~~Returned a generic clipped ellipse instead of solving the Grad-Shafranov source equation~~ **RESOLVED v0.18.x**: solves the fixed-boundary GS equation on the EFIT grid with a sparse finite-difference operator and polynomial `p'` / `FF'` source profiles | — | local |
