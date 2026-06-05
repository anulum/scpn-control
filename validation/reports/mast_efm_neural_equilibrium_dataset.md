# MAST EFM Neural-Equilibrium Supervised Dataset

Schema: `scpn-control.mast-efm-neural-equilibrium-supervised-dataset.v1`
Status: `blocked`
Reference dataset: `mast-efm-30419-30420-30421-30422-30423-30424-a50277960ea1ada9`
Dataset path: `processed/neural_equilibrium/mast_efm_supervised_dataset.npz`
Dataset SHA-256: `3206bd530efdd6fc73bae57b2ac18646aff39e130533c7d5167abe1ae7d136f3`
Equilibria: 527
Grid shape: 65 x 129
Maximum LCFS points: 157
Split counts: train=340, validation=80, test=107

## Split policy

- Train shots: 30419, 30420, 30421, 30422
- Validation shots: 30423
- Test shots: 30424
- Policy: shot-held-out deterministic split; no random time-slice leakage across holdout shots

## Targets

- `psirz_Wb_per_rad`
- `psirz_valid_mask`
- `psi_axis_Wb_per_rad`
- `psi_boundary_Wb_per_rad`
- `pprime_Pa_per_Wb_rad`
- `pprime_valid_mask`
- `q_profile`
- `q_profile_valid_mask`
- `lcfs_r_m`
- `lcfs_z_m`
- `lcfs_valid_mask`
- `magnetic_axis_r_m`
- `magnetic_axis_z_m`

LCFS coordinates are padded with NaN values, LCFS validity masks are padded with False values, and `lcfs_point_count` records the real point count per slice.

## Admission boundary

This is a supervised public-MAST-EFM dataset for training and holdout evaluation. Predictive EFIT/P-EFIT admission remains blocked because no trained full-output pressure/q-profile/LCFS predictive artefact has passed tolerances.

Fallback features: none

## Feature source policy

- `Ip_MA` from `Ip_MA` using `A_to_MA during reference conversion; identity_MA during dataset build`
- `Bt_T` from `Bt_T` using `identity_T`
- `ffprime_scale` from `ffprime_rms_T_rad` using `campaign_median_normalised_rms`

## Next processing steps

- train a full-output model on the train split and evaluate only once on validation/test shot splits
- keep public-source feature policy fixed while training and holdout evaluation are performed
- emit compact holdout metrics and keep large weights/predictions on storage-host storage by SHA-256
- run validate_neural_equilibrium_reference.py only after full predictive artefacts and tolerances exist
