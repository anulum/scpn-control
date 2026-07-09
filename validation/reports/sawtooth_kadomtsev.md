# Sawtooth Kadomtsev Crash Validation

- Schema: `scpn-control.sawtooth-kadomtsev-validation.v1`
- Generated (UTC): 2026-06-14T14:46:56Z
- Target: `local-sawtooth-kadomtsev`
- Profile: q(rho) = 0.8 + 2.2 rho^2 on 201 points
- q=1 radius rho_1 = 0.30150; mixing radius rho_mix = 0.44248
- Status: **pass**

## Exact conservation and structure (gate < 1.0e-10)

| reference | value |
| --- | --- |
| temperature volume-integral rel error | 0.000e+00 |
| density volume-integral rel error | 1.363e-16 |
| helical-flux residual psi*(rho_mix) | 3.666e-18 |
| inner temperature flatness | 0.000e+00 |
| inner density flatness | 0.000e+00 |
| inner q value | 1.0100 |
| outside max abs change | 0.000e+00 |

## q=1 radius convergence

- Coarse-grid rel error: 2.890e-05
- Fine-grid rel error: 8.225e-06
- Convergence order: 1.813
- No-crash guard (q>1 everywhere leaves profiles unchanged): True
