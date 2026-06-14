<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Resistive-Wall-Mode Feedback Validation

- Schema: `scpn-control.rwm-feedback-validation.v1`
- Generated (UTC): 2026-06-14T14:33:26Z
- Target: `local-rwm-feedback`
- Window: beta_nw=2.0, beta_w=4.0, tau_wall=0.005 s
- Status: **pass**

## Exact closed-form references (relative error, gate < 1.0e-09)

| reference | max rel error / residual |
| --- | --- |
| Bondeson-Ward growth rate | 0.000e+00 |
| wall-gap tau_eff = tau_wall (b/d)^2 | 0.000e+00 |
| rotation stabilisation | 0.000e+00 |
| critical-rotation marginality | 1.421e-16 |
| required feedback gain | 0.000e+00 |
| feedback closed-loop marginality | 0.000e+00 |
| 1/tau_wall scaling | 0.000e+00 |

## Stability-window boundaries

- Stable below no-wall limit: growth rate = 0.000e+00
- Ideal kink at/above wall limit: infinite growth = True
