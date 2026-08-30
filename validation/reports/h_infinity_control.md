<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — normalized DGKF H-infinity validation report. -->

# Normalized DGKF H-infinity validation

- Generated: `2026-08-30T04:32:35.273706Z`
- Source commit: `bbbb1c9567d45fd45432c5a378425323f1f9a7ab`
- Schema: `scpn-control.h-infinity-validation.v1`
- Payload seal: `feb0e790ecc60e941e30bb5101acab12e233a413d87a3508756f859b6a94e1c7`
- Overall: `PASS`

| Check | Result |
|---|---:|
| Admitted gamma | 51.4592957065 |
| Maximum normalization residual | 0.000e+00 |
| X Riccati relative residual | 9.959e-11 |
| Y Riccati relative residual | 2.665e-15 |
| Central-controller formula relative error | 0.000e+00 |
| Spectral feasibility margin | 31.6633821513 |
| Dominant augmented closed-loop pole real part | -1.41514923841 |
| Frequency-sweep peak | 51.458655504 |
| Frequency-sweep peak / gamma | 0.999987559051 |
| Frequency samples | 20002 |

## Claim boundary

This admits only `normalized continuous-time standard plant; D11=D22=0`. The frequency sweep is
`finite numerical corroboration, not exact norm proof`. Production admission is
`false`. Excluded claims:

- facility or reactor validation
- saturated H-infinity guarantee
- arbitrary sampled-data stability
- structured uncertainty or D-K synthesis
- classical gain margin
