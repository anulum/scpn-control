# NTM Island Dynamics Validation (Modified Rutherford Equation)

- Schema: `scpn-control.ntm-island-dynamics-validation.v1`
- Generated (UTC): 2026-06-14T13:55:25Z
- Target: `local-ntm-island-dynamics`
- Surface: 2/1 at r_s=0.3 m (a=0.5 m, R0=1.7 m, B0=2.0 T)
- Status: **pass**

## Classical-only trajectory vs exact separable solution

- Max dw/dt relative error: 1.488e-16 (gate < 1.0e-10)
- Max trajectory relative error: 1.492e-14 (gate < 1.0e-08)

| case | Delta'_0 | w0 [m] | dw/dt rel err | trajectory rel err |
| --- | --- | --- | --- | --- |
| decaying_strong | -3.0 | 0.05 | 1.488e-16 | 4.025e-15 |
| decaying_weak | -1.5 | 0.08 | 1.488e-16 | 7.112e-15 |
| growing | 2.0 | 0.01 | 1.116e-16 | 1.492e-14 |

## Classical + bootstrap saturated width (stable attractor)

- Delta'_0 = -6.0; analytic w_sat = 2.19407e-02 m
- Fixed-point residual |dw/dt(w_sat)| / |dw/dt(0.5 w_sat)|: 1.302e-16 (gate < 1.0e-09)
- Convergence from below: rel err 4.094e-07, monotonic True
- Convergence from above: rel err 1.153e-06, monotonic True
- Gate: rel err < 5.0e-03 with monotonic approach
