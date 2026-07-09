# Transport Heat-Diffusion Validation

- Schema: `scpn-control.transport-diffusion-validation.v1`
- Generated (UTC): 2026-06-14T13:43:34Z
- Target: `local-transport-diffusion`
- chi = 0.7 m^2/s; a = 0.5 m; lambda = 2.404826 (first J0 zero)
- Status: **pass**

## Diffusion operator vs Bessel eigenvalue (`_explicit_diffusion_rhs`)

- Order of accuracy: 2.000 (gate >= 1.8)
- Finest-grid relative error: 3.309e-05 (gate < 5.0e-03)

| resolution | h | relative error |
| --- | --- | --- |
| 33 | 3.1250e-02 | 5.2941e-04 |
| 65 | 1.5625e-02 | 1.3236e-04 |
| 129 | 7.8125e-03 | 3.3091e-05 |

## Manufactured steady state (`_build_cn_tridiag` + `_thomas_solve`)

- Order of accuracy: 1.928 (gate >= 1.8)
- Finest-grid NRMSE: 2.422e-05 (gate < 1.0e-03)

| resolution | h | NRMSE |
| --- | --- | --- |
| 33 | 3.1250e-02 | 3.5054e-04 |
| 65 | 1.5625e-02 | 9.2618e-05 |
| 129 | 7.8125e-03 | 2.4225e-05 |

## Polyglot Thomas parity (`scpn_control_rs.py_thomas_solve`)

- max |Python - Rust|: 0.000e+00
- max |Rust - dense numpy|: 7.649e-14
- Matches within tolerance: True
