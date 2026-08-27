# Compute and validation collaboration

SCPN Control is research infrastructure for bounded fusion-control experiments.
It includes formal Petri-net checks, differentiable physics surfaces, neural
transport and equilibrium facades, digital-twin contracts, and strict evidence
admission gates. External compute, data, hardware, and technical review can
strengthen the evidence available through those public interfaces.

!!! warning "Claim boundary"
    Sponsorship, credits, data access, or collaboration do not create facility
    deployment evidence, safety approval, or a quantitative scientific claim.
    Evidence is admitted only when the corresponding public validator passes.

## Evidence interfaces

Collaborators can supply reviewable artifacts in these forms:

| Evidence class | Reviewable contribution |
|---|---|
| Public datasets | Stable source URI, licence, size, checksum, retrieval date, feature/target schema, and units |
| External-code comparison | Input decks, raw output, parsed output, code/version identity, units, tolerances, and digests |
| Model training | Dataset manifest, preprocessing contract, seeds, hardware metadata, weights, holdout predictions, and uncertainty |
| Hardware timing | Exact source revision, build identity, host and accelerator metadata, sample definition, scheduler/load context, and raw samples |
| Facility replay | Data-policy authority, immutable shot/signal identity, calibration and units, replay configuration, output metrics, and claim boundary |
| Independent review | Reproducer, observed result, environment details, and a clear distinction between confirmation and disagreement |

Failed comparisons are useful evidence when their inputs and outputs are
preserved. They remain failures; they are not rewritten into positive claims.

## Public dataset references

The repository contains normalized acquisition manifests for these public
neural-transport sources. Large payloads are not committed to Git.

| Dataset | DOI | Files | Published payload size |
|---|---|---:|---:|
| QLKNN10D training set | `10.5281/zenodo.3497066` | 5 | 32,080,102,848 bytes |
| QuaLiKiz v2.6.2 JET spectra | `10.5281/zenodo.7418108` | 1 | 29,655,790,232 bytes |
| QLKNN11D training set | `10.5281/zenodo.8017522` | 46 | 247,952,755,894 bytes |

Possession of these payloads does not admit a neural-transport claim. The
public claim report also requires bound preprocessing, weights, holdout
predictions, metrics, provenance, and an admitted evidence class.

## Current claim boundaries

- Neural transport uses an analytic critical-gradient fallback unless admitted
  weights and reference evidence are supplied.
- Neural-equilibrium assets are synthetic pretraining inputs until an
  identical-input EFIT/P-EFIT holdout report passes.
- External gyrokinetic agreement is absent until a strict matched-input report
  binds real TGLF, GENE, GS2, CGYRO, or QuaLiKiz output.
- Synthetic disruption fixtures exercise data and prediction plumbing; they do
  not establish a measured facility ROC.
- Local and CI timing observations do not establish deterministic PCS, HIL, or
  production behavior.

The generated [physics traceability report](physics_traceability.md) is the
public authority for component-level claim status. The [validation guide](validation.md)
explains evidence classes and the [benchmark guide](benchmarks.md) explains
timing context.

## Contact

Compute sponsorship, cloud credits, storage, facility-data collaboration,
external-code artifacts, target-hardware measurements, and independent review
are welcome. Contact
[protoscience@anulum.li](mailto:protoscience@anulum.li) with the evidence or
resource class, applicable licence or access policy, and the public interface
you want reviewed.

Public issue trackers [#46](https://github.com/anulum/scpn-control/issues/46)
through [#53](https://github.com/anulum/scpn-control/issues/53) provide neutral
contribution intake by evidence domain. They do not disclose or prescribe an
internal work sequence.
