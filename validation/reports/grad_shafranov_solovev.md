<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Grad-Shafranov Solov'ev Validation

- Schema: `scpn-control.grad-shafranov-solovev-validation.v1`
- Generated (UTC): 2026-06-14T09:26:56Z
- Target: `local-grad-shafranov-solovev`
- Status: **pass**

## Discrete operator (`FusionKernel._apply_gs_operator`)

- Order of accuracy: 2.000 (gate ≥ 1.8)
- Finest-grid max truncation error: 6.104e-05 (gate < 5.0e-04)
- Passed: True

| resolution | h | max |Δ* error| |
| --- | --- | --- |
| 33 | 4.6875e-02 | 5.4932e-04 |
| 49 | 3.1250e-02 | 2.4414e-04 |
| 65 | 2.3438e-02 | 1.3733e-04 |
| 97 | 1.5625e-02 | 6.1035e-05 |

## SOR reconstruction (`FusionKernel._sor_step`)

- Order of accuracy: 2.019 (gate ≥ 1.8)
- Finest-grid NRMSE: 1.194e-06 (gate < 1.0e-04)
- Passed: True

| resolution | h | NRMSE | sweeps | converged |
| --- | --- | --- | --- | --- |
| 33 | 4.6875e-02 | 1.0969e-05 | 651 | True |
| 49 | 3.1250e-02 | 4.8253e-06 | 1451 | True |
| 65 | 2.3438e-02 | 2.7003e-06 | 2551 | True |
| 97 | 1.5625e-02 | 1.1939e-06 | 5701 | True |

## Rust multigrid backend (`scpn_control_rs.py_multigrid_solve`)

- Resolution: 97
- NRMSE vs analytic: 1.5638e-01
- Residual (inf-norm): 1.3852e+01
- Injected Dirichlet data preserved: True
- Meets analytic tolerance: False

The Rust binding's fixed-cycle multigrid does not converge on this forcing — it preserves the injected Dirichlet boundary but leaves a large interior residual — so it does not reproduce the Solov'ev equilibrium; recorded for transparency and not part of the pass/fail gate. See the Rust/Python SOR parity gap in `tests/test_rust_python_parity.py`.
