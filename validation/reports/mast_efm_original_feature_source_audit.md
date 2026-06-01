<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — MAST EFM original feature-source audit report -->

# MAST EFM Original Feature-Source Audit

Schema: `scpn-control.mast-efm-original-feature-source-audit.v1`
Status: `blocked`
Can rebuild dataset now: `False`
Reference dataset: `mast-efm-30419-30420-30421-30422-30423-30424-2cb0254259360c05`
Shot count: 6

## Feature source status

| Feature | Status | Selected source | Transform | Resolution |
|---|---|---|---|---|
| `Bt_T` | `source_found_requires_rebuild` | `bphi_rmag` | `identity_T` | total toroidal field at the magnetic axis is available in original public EFM metadata |
| `Ip_MA` | `source_found_requires_rebuild` | `plasma_current_x` | `A_to_MA` | measured total plasma current is available in original public EFM metadata |
| `ffprime_scale` | `source_found_requires_policy` | `ffprime` | `profile_to_training_scalar` | FF-prime profile is available, but the scalar training-feature reduction must be specified before rebuild |

## Next processing steps

- define ffprime profile reduction before rebuilding the supervised dataset
- rebuild the supervised dataset only after every fallback feature has an admitted original public source
- record the source-variable policy in the dataset report before training
