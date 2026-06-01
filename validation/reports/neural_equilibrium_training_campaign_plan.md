# Neural-Equilibrium Training Campaign Plan

Schema: `scpn-control.neural-equilibrium-training-campaign-plan.v1`
Status: `prepared`
SAS root: `/mnt/data_sas/DATASETS/SCPN-CONTROL`

## Claim boundary

This report prepares data-processing and training campaigns. It is not predictive EFIT/P-EFIT admission evidence and does not launch GPU training.

## MAST EFM dataset

- Dataset status: `prepared`
- Equilibria: 527
- Grid shape: 65 x 129
- Split counts: `{"test": 107, "train": 340, "validation": 80}`
- Dataset SHA-256: `e5d3bb4bbf426b489f8f6b51ae44a17c7cfcbde15d91da18db4329c7a772605e`
- SAS payload: `/mnt/data_sas/DATASETS/SCPN-CONTROL/processed/neural_equilibrium/mast_efm_supervised_dataset.npz`
- Exists on this host: `False`
- Verified available: `True`

## Prepared dataset lanes

| Lane | Status | Next action |
|---|---|---|
| `mast_efm_neural_equilibrium` | `prepared_on_sas` | run the dry-run trainer locally, then execute explicitly on admitted storage when compute is reserved |
| `qlknn_qualikiz_neural_transport` | `manifested_large_payloads_deferred` | download deferred payloads to SAS, verify checksums, then build processed transport tensors |
| `external_efit_pefit_or_diiid_equilibrium` | `external_material_required` | acquire matched public EFIT/P-EFIT, GEQDSK, or MDSplus-derived reconstruction artefacts |
| `sparc_or_public_geqdsk_equilibrium` | `external_material_required` | seal redistributable GEQDSK/EQDSK artefacts with source policy and SHA-256 manifests |

## GPU budget estimates

| Scenario | GPU class | Minimum GPU-h | Nominal GPU-h | Upper GPU-h | Storage TB | Blocking condition |
|---|---|---:|---:|---:|---:|---|
| `mast_efm_readiness_smoke` | single 16-24 GB CUDA GPU or CPU fallback | 0.0 | 1.0 | 3.0 | 0.05 | full-output trainer still required before predictive admission |
| `mast_efm_single_seed_full_output` | single 24-48 GB CUDA GPU | 2.0 | 6.0 | 12.0 | 0.1 | requires implementation of full-output trainer and admitted input-feature provenance |
| `mast_efm_multiseed_ablation` | one to four 24-80 GB CUDA GPUs | 30.0 | 80.0 | 180.0 | 0.5 | requires single-seed trainer and stable holdout metric schema |
| `qlknn_qualikiz_payload_processing` | single A10/A100-class GPU for first pass; A100/H100 for sweeps | 100.0 | 350.0 | 900.0 | 2.0 | large numeric payloads must be pulled to SAS and checksum-verified first |
| `external_efit_pefit_or_diiid_equilibrium_set` | single 24-80 GB CUDA GPU after CPU-side conversion | 20.0 | 120.0 | 400.0 | 1.0 | requires acquired public or collaborator-provided matched reconstruction artefacts |
| `publication_grade_equilibrium_campaign` | multi-GPU A100/H100-class allocation | 500.0 | 1500.0 | 4000.0 | 4.0 | requires at least one admitted external equilibrium reference set beyond MAST EFM candidate data |

## Run order

1. Re-run the MAST EFM dataset readiness check before any campaign.
2. Run the MAST EFM trainer in dry-run mode and inspect the launch report.
3. Use explicit --execute only on admitted storage and reserved compute.
4. Run a smoke campaign and publish compact metrics before spending multi-seed GPU budget.
5. Pull QLKNN/QuaLiKiz large payloads to SAS only when storage and GPU allocation are reserved.
6. Keep all predictive and facility claims blocked until strict admission reports pass.
