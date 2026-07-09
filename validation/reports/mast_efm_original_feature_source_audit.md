# MAST EFM Original Feature-Source Audit

Schema: `scpn-control.mast-efm-original-feature-source-audit.v1`
Status: `source_ready`
Can rebuild dataset now: `True`
Reference dataset: `mast-efm-30419-30420-30421-30422-30423-30424-a50277960ea1ada9`
Shot count: 6

## Feature source status

| Feature | Status | Selected source | Transform | Resolution |
|---|---|---|---|---|
| `Ip_MA` | `source_found_requires_rebuild` | `plasma_current_x` | `A_to_MA` | measured total plasma current is available in original public EFM metadata |
| `Bt_T` | `source_found_requires_rebuild` | `bphi_rmag` | `identity_T` | total toroidal field at the magnetic axis is available in original public EFM metadata |
| `ffprime_scale` | `source_found_requires_rebuild` | `ffprime` | `profile_rms_to_campaign_median_normalised_scalar` | FF-prime profile is available; the dataset policy uses per-time-slice RMS magnitude, campaign-median normalisation, and [0.25, 4.0] clipping |

## Next processing steps

- rebuild or verify the supervised dataset with all former fallback features sourced from public EFM metadata
- keep the source-variable policy fixed while training and holdout evaluation are performed
