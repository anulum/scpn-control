<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# VMEC-lite Spectral-Geometry Validation

- Schema: `scpn-control.vmec-lite-geometry-validation.v1`
- Generated (UTC): 2026-06-15T10:40:02Z
- Target: `local-vmec-lite-geometry`
- Claim status: `bounded_vmec_lite_evidence`
- Full VMEC claim allowed: `False`
- Truncation: m_pol=2, n_tor=1, n_fp=1
- Geometry: R0=6.2 m, a=2.0 m, kappa=1.7, delta=0.33
- Status: **pass**

## Exact geometry checks (gate < 1.0e-12)

| check | value |
| --- | --- |
| Basis mode-count error | 0 |
| Cosine Fourier evaluation error | 0.000e+00 |
| Sine Fourier evaluation error | 0.000e+00 |
| Axisymmetric boundary coefficient error | 0.000e+00 |
| Fixed-boundary radial scaling error | 0.000e+00 |
| q = 1 / iota reciprocal error | 1.110e-16 |
| B-coefficient construction error | 0.000e+00 |

- Minimum sampled major radius: 3.87 m
- Initial-geometry force residual: 1.8263

This is bounded local-regression evidence for the repository VMEC-lite facade.
Full VMEC-grade 3D MHD equilibrium claims remain blocked until matched external VMEC or public references pass admission.
