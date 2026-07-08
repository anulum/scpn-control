<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Grad-Shafranov Solov'ev Validation

- Schema: `scpn-control.grad-shafranov-solovev-validation.v2`
- Generated (UTC): 2026-07-08T19:14:13Z
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

## Python multigrid reconstruction (`FusionKernel._multigrid_vcycle`)

- Order of accuracy: 2.019 (gate ≥ 1.8)
- Finest-grid NRMSE: 1.194e-06 (gate < 1.0e-04)
- Passed: True

| resolution | h | NRMSE | cycles | residual | initial residual | converged |
| --- | --- | --- | --- | --- | --- | --- |
| 33 | 4.6875e-02 | 1.0970e-05 | 10 | 1.4718e-07 | 4.1596e+03 | True |
| 49 | 3.1250e-02 | 4.8255e-06 | 10 | 4.0732e-07 | 9.5070e+03 | True |
| 65 | 2.3438e-02 | 2.7005e-06 | 10 | 7.7709e-07 | 1.7032e+04 | True |
| 97 | 1.5625e-02 | 1.1939e-06 | 11 | 2.2604e-07 | 3.8615e+04 | True |

## Rust multigrid backend (`scpn_control_rs.py_multigrid_solve`)

- Resolution: 97
- NRMSE vs analytic: 1.1939e-06
- Residual (inf-norm): 3.5578e-09
- Injected Dirichlet data preserved: True
- Meets analytic tolerance: True

The Rust binding reproduces the Solov'ev analytic field under the shared solver-stack sign convention. The record is informational; Python operator, SOR, and multigrid paths remain the pass/fail gate.
