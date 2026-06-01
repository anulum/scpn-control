# MAST EFM Neural-Equilibrium Training Launch

Schema: `scpn-control.mast-efm-neural-equilibrium-training.v1`
Status: `prepared`
Execution mode: `dry_run`
Dataset path: `/mnt/data_sas/DATASETS/SCPN-CONTROL/processed/neural_equilibrium/mast_efm_supervised_dataset.npz`
Dataset SHA-256: `e5d3bb4bbf426b489f8f6b51ae44a17c7cfcbde15d91da18db4329c7a772605e`
Dataset exists on this host: `False`
Weights path: `/mnt/data_sas/DATASETS/SCPN-CONTROL/models/neural_equilibrium/mast_efm_full_output_baseline_weights.npz`

## Claim boundary

This report prepares or executes a deterministic repository baseline. It is not predictive EFIT/P-EFIT admission evidence.

## Run command

```bash
python validation/train_mast_efm_neural_equilibrium.py --execute --dataset-path /mnt/data_sas/DATASETS/SCPN-CONTROL/processed/neural_equilibrium/mast_efm_supervised_dataset.npz --weights-out /mnt/data_sas/DATASETS/SCPN-CONTROL/models/neural_equilibrium/mast_efm_full_output_baseline_weights.npz
```

## Required targets

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

## Admission blockers

- replace fallback Ip_MA, Bt_T, and ffprime_scale with acquired or documented public inputs
- run --execute on admitted storage and publish holdout metrics for train, validation, and test splits
- validate the exact trained weight checksum through the strict neural-equilibrium reference gate
