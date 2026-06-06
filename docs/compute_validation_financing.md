<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Compute Validation Financing -->

# Compute validation financing

SCPN Control has the software infrastructure for bounded fusion-control
experiments: formal Petri-net safety checks, differentiable physics surfaces,
neural transport and equilibrium surrogates, digital-twin update contracts,
quantum disruption facades, and strict evidence admission gates. The remaining
step is not a claim-writing exercise. It is a compute, data, and validation
campaign.

We are seeking support for GPU time, storage, public-data processing, external
code comparisons, and documented validation artefacts. The goal is to convert
implemented research infrastructure into reproducible evidence that can be
reviewed, repeated, and challenged.

!!! warning "Claim boundary"
    Funding supports validation work. It does not by itself create facility
    deployment evidence, safety approval, or quantitative public claims. SCPN
    Control keeps full-fidelity claims blocked until the corresponding external
    artefacts pass the repository admission gates.

## Current support request

The immediate need is funding and collaboration for validation, not marketing
language. The prepared lanes need public-data acquisition, GPU execution,
external-code comparisons, target-hardware timing, and independent review. The
repository already records blockers as evidence gates; financing pays to run the
campaigns that can close those gates or produce useful negative evidence.

## Why compute financing matters

The repository already separates bounded local evidence from external validation
requirements. That discipline is useful, but it also makes the bottleneck clear:
several high-value physics surfaces need large public datasets, repeated GPU
campaigns, and external solver comparisons before they can support broader
claims.

Current blockers include:

| Validation lane | What is needed | Why financing helps |
|---|---|---|
| Neural transport | QLKNN/QuaLiKiz tensors, preprocessing, training, holdout reports | Public payloads are hundreds of GB before processed tensors and repeats |
| Differentiable physics | JAX transport/equilibrium gradient campaigns and latency evidence | GPU runs make end-to-end gradient validation practical |
| Gyrokinetic comparison | Real TGLF, GENE, GS2, CGYRO, or QuaLiKiz artefacts for identical cases | External-code runs require compute, executable access, and evidence packaging |
| Neural equilibrium | Synthetic pretraining plus EFIT/P-EFIT reference comparison | Real reconstruction artefacts and repeated training runs are required |
| Digital twin | TRANSP/TSC or equivalent integrated-modelling evidence | External simulator artefacts must be converted into strict admission reports |
| Disruption prediction | Measured multi-facility databases and baseline comparisons | Synthetic benchmarks do not replace measured disruption evidence |

## Active public-data acquisition

The current public acquisition lane targets QLKNN and QuaLiKiz data hosted on
Zenodo. The repository stores normalised acquisition manifests; the large tensor
payloads are downloaded into a local data workspace rather than committed to git.

| Dataset | DOI | Files | Payload size |
|---|---|---:|---:|
| QLKNN10D training set | `10.5281/zenodo.3497066` | 5 | 32,080,102,848 bytes |
| QuaLiKiz v2.6.2 JET spectra | `10.5281/zenodo.7418108` | 1 | 29,655,790,232 bytes |
| QLKNN11D training set | `10.5281/zenodo.8017522` | 46 | 247,952,755,894 bytes |
| **Total** |  | **52** | **309,688,648,974 bytes** |

These downloads are necessary for neural-transport training and validation, but
they are not sufficient by themselves. After acquisition, the next steps are
checksum verification, format inspection, preprocessing, feature/target schema
binding, train/validation/test splits, model training, holdout prediction, and
strict reference-artifact admission.

## GPU and storage budget tiers

The estimates below are planning budgets. They are intentionally expressed as
GPU-hours and storage rather than vendor-specific prices, because cloud rates and
local hardware availability change.

| Tier | Compute | Storage | What it unlocks |
|---|---:|---:|---|
| Local validation start | 100-300 GPU-h on one 24 GB GPU | 1-2 TB | QLKNN10D preprocessing, compact surrogate retraining, JAX smoke campaigns, neural-equilibrium synthetic pretraining |
| External-evidence campaign | 1,000-3,000 GPU-h on A100/H100-class GPUs | 4-8 TB | QLKNN11D training runs, QuaLiKiz holdout validation, differentiable transport/equilibrium reports, digital-twin surrogate campaigns |
| Publication-grade campaign | 5,000-20,000 GPU-h on multi-GPU A100/H100-class systems | 10-30 TB | Multi-seed training, uncertainty sweeps, nonlinear GK campaigns, end-to-end differentiable control evidence, repeatable benchmark tables |

CPU/HPC resources are also needed. Formal verification is mostly CPU/RAM-bound,
and external tools such as TRANSP, TSC, TGLF, GENE, GS2, CGYRO, and QuaLiKiz may
require CPU clusters, facility licences, or code-specific runtime environments.

## Prepared neural-equilibrium campaign budget

The MAST EFM neural-equilibrium dataset is prepared on storage-host dataset storage and can be
checked without launching long training jobs:

```bash
python validation/plan_neural_equilibrium_training_campaign.py --require-storage-payload --verified-storage-payload
```

The generated reports
`validation/reports/neural_equilibrium_training_campaign_plan.json` and
`validation/reports/neural_equilibrium_training_campaign_plan.md` record the
prepared MAST EFM payload, deferred QLKNN/QuaLiKiz public-data lane, external
EFIT/P-EFIT dataset requirements, and GPU-hour estimates. Current planning
budgets are:

| Scenario | Minimum GPU-h | Nominal GPU-h | Upper GPU-h | Storage |
|---|---:|---:|---:|---:|
| MAST EFM readiness smoke | 0 | 1 | 3 | 0.05 TB |
| MAST EFM single-seed full-output trainer | 2 | 6 | 12 | 0.1 TB |
| MAST EFM multi-seed ablation and uncertainty | 30 | 80 | 180 | 0.5 TB |
| QLKNN/QuaLiKiz payload processing and baseline training | 100 | 350 | 900 | 2 TB |
| External EFIT/P-EFIT or DIII-D equilibrium set | 20 | 120 | 400 | 1 TB |
| Publication-grade equilibrium campaign | 500 | 1,500 | 4,000 | 4 TB |

These are estimates for planning and financing. They are not benchmark claims.
The MAST EFM dataset is prepared, but predictive EFIT/P-EFIT admission remains
blocked until a full-output trainer and strict holdout admission artefacts exist.
The launch path is also dry-run prepared through
`validation/train_mast_efm_neural_equilibrium.py`; long training requires an
explicit `--execute` flag and must run only on this workstation or external
cloud compute after storage access and GPU capacity are reserved. storage host is a
storage host for this lane, not a training machine.
storage host dry-run evidence now confirms the prepared MAST EFM storage-host dataset is visible
from the storage host. The current supervised dataset has no fallback feature
columns: `Ip_MA` derives from public `plasma_current_x`, `Bt_T` derives from
public `bphi_rmag`, and `ffprime_scale` derives from public `ffprime` profile
RMS magnitude normalised by the campaign median. The remaining no-compute
blocker has moved from feature provenance to execution evidence: long training
must be launched with explicit `--execute` on workstation or cloud compute, then
compact holdout metrics and strict reference-admission artefacts must be
published. The compute execution package is now prepared in the campaign report:
it fixes the expected dataset SHA-256, keeps default weights under
`artifacts/neural_equilibrium/`, declares storage host as a forbidden training host,
requires `workstation` or `external_cloud` host admission, and points to
repository-published templates for holdout metrics, latency metrics, GPU-cost
accounting, and the final admission certificate.

## What support pays for

Funding is used for concrete, auditable work products:

- Public dataset acquisition with size and checksum verification.
- Reproducible preprocessing scripts and immutable processed-data manifests.
- GPU training runs with pinned seeds, hardware metadata, and holdout reports.
- External-code comparison artefacts with input decks, raw outputs, parsed
  outputs, units, tolerances, and SHA-256 digests.
- Hardware timing evidence for control-relevant latency budgets.
- Documentation that keeps bounded evidence separate from facility claims.
- Issue-linked validation reports that reviewers can inspect without trusting
  unpublished assertions.

## Evidence produced by a funded campaign

A successful campaign should produce repository artefacts such as:

| Artefact | Purpose |
|---|---|
| `public_data_acquisition_report.json` | Shows which public files were acquired and verified |
| `neural_transport_reference` reports | Binds QLKNN/QuaLiKiz reference tensors, weights, predictions, units, and metrics |
| `jax_gk_parity` reports | Shows CPU/GPU parity for selected native JAX gyrokinetic cases |
| `gk_crosscode` reports | Compares native outputs against real external-code runs |
| `digital_twin_reference` reports | Binds TRANSP/TSC or equivalent integrated-modelling evidence |
| `physics_traceability` reports | Shows which claims remain blocked and which have admitted evidence |

Every admitted artefact should be schema-versioned, digest-bound, and tied to the
claim it supports. This makes the funding result useful even when a campaign
finds disagreement: a failed validation run is still actionable evidence.

## Support the validation campaign

SCPN Control is developed as research infrastructure. Compute sponsorship,
cloud credits, hardware access, facility-data collaboration, external-code run
artefacts, and direct financing all help move the project from bounded local
evidence toward broader public validation.

If you can support the campaign, use the sponsor link in the repository README
or contact `protoscience@anulum.li` with the resource you can provide:

- GPU or HPC credits.
- Storage for public and processed datasets.
- Access to real shot data with a clear data policy.
- External-code runs for matched validation cases.
- Review of validation reports and claim boundaries.
- Direct financing for compute, data engineering, and documentation work.

## Current priority order

1. Finish QLKNN/QuaLiKiz public-payload acquisition and checksum verification.
2. Build processed neural-transport datasets with immutable manifests.
3. Train and validate QLKNN-class neural transport surrogates against holdout
   data.
4. Run JAX differentiable transport and equilibrium gradient campaigns.
5. Package external-code and facility-data artefacts into strict admission
   reports as collaborators provide them.

The public page will be updated as new evidence is admitted. Until then, the
claim boundary remains explicit: support funds the validation campaign; the
validation campaign must earn the claims.

## Funding-to-evidence translation

This page links budget tiers to evidence outputs, not to speculative outcomes.

- allocate budget first to unblock blocked claim gates,
- run one evidence pack per funded tier,
- then only expand scope when the prior tier outputs are admitted by validators.

This keeps spend aligned with claim clarity rather than model output volume.

## Practical use and scope

Use this plan to align funding decisions with measurable validation milestones.

- Convert each funding tranche into one or more concrete benchmark or hardening outcomes.
- Track cost estimates against observable outputs in `docs/validation.md`.
- Do not spend this budget on un-measured work; every line here should map to executable evidence.
