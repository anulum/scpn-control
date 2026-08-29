# Known limitations and resolved implementation history

This page states current capability boundaries and preserves resolved
implementation history. It is not a roadmap, priority queue, or task registry.
Repository planning remains private; public users need the limitation itself,
its effect on claims, and the evidence that resolves a historical issue.

## Current limitation

### Fixed-weight disruption-risk baseline

`scpn_control.control.disruption_predictor.predict_disruption_risk` computes a
deterministic, hand-tuned sigmoid over toroidal-asymmetry observables. It is a
heuristic baseline with synthetic sanity checks, not a model trained or
validated on an experimental disruption database. Its score must not be
represented as facility-validated probability, experimental accuracy, or a
production protection-system decision.

The public claim boundary is enforced in the predictor metadata, Studio
evidence adapter, README limitation disclosure, and validation tests. Optional
Transformer training on synthetic shots does not close the experimental-data
boundary.

## Resolved implementation history

The entries below are historical engineering facts, not open work items.

| Component | Former limitation | Resolution evidence | Tracking |
|---|---|---|---|
| Rust H-infinity control | `update_discretization()` used Euler integration and lacked zero-order-hold discretisation | v0.8.1 added Padé(6,6) scaling-and-squaring `matrix_exp`, `zoh_discretize`, and six focused tests | [gh-10](https://github.com/anulum/scpn-control/issues/10) |
| Nengo Loihi wrapper | The Loihi backend was not exercised in CI | v0.9.0 added the dedicated `nengo-loihi` job with `nengo>=4.0` | [gh-11](https://github.com/anulum/scpn-control/issues/11) |
| Rust SPI compatibility | `RustSPIMitigation` had no Python fallback | v0.8.0 added a Python fallback matching the Rust SPI phase constants | [gh-13](https://github.com/anulum/scpn-control/issues/13) |
| Rust multigrid compatibility | `rust_multigrid_vcycle` had no Python fallback | v0.8.0 delegated the fallback to `FusionKernel._multigrid_vcycle` | [gh-14](https://github.com/anulum/scpn-control/issues/14) |
| Rust SVD correction compatibility | `rust_svd_optimal_correction` had no Python fallback | v0.8.0 added the NumPy truncated-SVD pseudoinverse fallback | [gh-15](https://github.com/anulum/scpn-control/issues/15) |
| JAX traceable runtime | JAX tracing was not exercised in CI | v0.9.0 added the dedicated `jax-parity` job | [gh-12](https://github.com/anulum/scpn-control/issues/12) |
| Finite-value validation | Repeated `np.isfinite` validation was duplicated across modules | v0.8.0 introduced shared bounded-float and finite-array validators and migrated the initial owners | [gh-17](https://github.com/anulum/scpn-control/issues/17) |
| Vertical stability index | `VerticalStabilityAnalysis.compute_n_index` returned a hard-coded value | v0.18.x derives the vertical field index from the flux grid and rejects degenerate grids | local regression suite |
| Sensor-fault reconfiguration | `ReconfigurableController.handle_sensor_fault` did not change allocation after isolation | v0.18.x records isolated sensors, removes their allocation weight, recomputes gain, masks residuals, and rejects invalid indices | local regression suite |
| SOL detachment threshold | `detachment_threshold` returned `False` for every state | v0.18.x evaluates a two-point Spitzer/sheath target-temperature criterion with fail-fast input validation | local regression suite |
| LCFS extraction | `RealtimeEFIT.find_lcfs` returned a zero-array stub | v0.18.x extracts and angle-sorts finite boundary points from the positive closed-flux region | local regression suite |
| Rogowski measurement | `DiagnosticResponse.simulate_measurements` returned a hard-coded current | v0.18.x integrates the reconstructed toroidal current density over the diagnostic grid | local regression suite |
| Grad-Shafranov source solve | `RealtimeEFIT._solve_gs_with_sources` returned a generic clipped ellipse | v0.18.x solves the fixed-boundary source equation with a sparse finite-difference operator and polynomial source profiles | local regression suite |
| EFIT reconstruction | `RealtimeEFIT.reconstruct` did not perform a real inverse reconstruction | 2026-06-22 work added weighted least squares, Tikhonov regularisation, Picard geometry updates, free-boundary coil fitting, and closure tests; magnetic-only data still do not identify the pressure/current split | commits `a640bf0`, `13d5417` |
| Director integration | An implicit `director_module` import kept the native branch permanently unavailable | commit `75568e0` replaced it with injected-director or rule-based-fallback contracts and removed the stale typing override | commit `75568e0` |

The current repository head, tests, and public API documentation are the
authority for present behaviour. Historical version statements describe the
point at which each limitation was resolved; they do not independently claim
facility validation, deployed performance, or certification.
