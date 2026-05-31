<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — JAX GK Parity Evidence -->

# JAX GK Parity Evidence

- Status: `pass`
- Artifacts: `3`
- Claim boundary: backend parity only; external GK validation remains required.

| Case | Backend | Dtype | X64 | Native dominant | JAX dominant | Gamma relative error | Omega absolute error |
|---|---|---|---:|---|---|---:|---:|
| cyclone_base_case | gpu | float32 | False | ITG | ITG | 2.20946e-07 | 2.49803e-07 |
| stable_mode | gpu | float32 | False | ITG | ITG | 1.53861e-06 | 1.74585e-07 |
| tem_kinetic_electron | gpu | float32 | False | ITG | ITG | 2.13612e-07 | 2.96581e-07 |
