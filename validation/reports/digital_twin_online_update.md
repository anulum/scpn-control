<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Digital twin online update report -->

# Digital twin online update benchmark

- Evidence kind: bounded synthetic online update
- Source: synthetic_digital_twin_reference
- Evaluated points: 18
- Baseline loss: 2.54856349e+01
- Best loss: 5.13466226e-03
- Evidence result digest: `4f2489fe1becc9ee88678b7a248349a3957a0a3c11780c4b430375440f124794`
- Evidence observation digest: `0e3e34618c86c155b214bf0b21ec87d57ccb494566af9bd960a28a9de8149bc6`
- Evidence prior digest: `614cdfe3b5a2f01bdcf8f0595c60ca843017f5cf1ac99808609a9784a9c35e90`
- Evidence simulator digest: `93a330469667c310962054339af9150742d7be007804ff60eeb37098d604e7f2`

Best parameters:

- `Z_eff`: 2.91629139e+00
- `actuator_rate_limit`: 2.48661549e-01
- `actuator_tau_steps`: 2.93837517e+00
- `n_e`: 8.76200895e+19

Claim boundary: this benchmark exercises deterministic Bayesian
model updating against a synthetic digital-twin reference. External
TRANSP/TSC coupling is fail-closed behind validated simulator
artifact metadata and the digital-twin reference gate.
