<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — Production readiness boundary -->

# Production Readiness Boundary

SCPN Control is not certified facility-control software. It is a research and
control-integration library with production-oriented engineering gates around
installation, API contracts, bounded validation, provenance, tests, and
fail-closed claim admission.

This page separates the current library-readiness status from the much stricter
requirements for tokamak plant operation.

## Readiness summary

| Scope | Current status | Meaning |
| --- | --- | --- |
| Library packaging and APIs | Production-oriented | Public Python package, documented APIs, optional dependencies, generated capability inventory, and compatibility checks are maintained. |
| Local software quality gates | Production-oriented | Module-specific tests, lint/type checks, pre-commit gates, generated traceability checks, and security scans are part of normal development. |
| Bounded physics/control evidence | Bounded research evidence | Repository benchmarks, deterministic fixtures, formal checks, and checksum/provenance gates support limited claims only inside declared domains. |
| Facility physics validation | Not complete | Many physics surfaces still require measured-shot, external-code, or documented public reference artifacts before quantitative or facility claims are admissible. |
| Real plant deployment | Not ready | No commissioned PCS deployment, no CODAC/EPICS plant acceptance, no qualified WebSocket control-stream runtime artifact, no hardware interlock qualification, no operator procedure package, and no safety certification evidence pack. |
| Safety certification | Not ready | Certification would require hazard analysis, requirements traceability, independent V&V, configuration management, audit trails, and device-specific acceptance evidence. |

## Claim status levels

| Level | Allowed wording | Required evidence | Current use in this repository |
| --- | --- | --- | --- |
| Bounded | Bounded model, bounded regression, deterministic fixture, local acceptance | Unit/module tests, deterministic benchmark reports, declared assumptions, provenance metadata, and fail-closed APIs | Most controller facades and many physics adapters are here. |
| Reference-validated | Validated against named documented/public/reference artefacts | Immutable artifacts, SHA-256 digests, unit contracts, declared tolerances, and passing strict validators | Available only for surfaces with matching reference artifacts. |
| External-code validated | Matched to GENE, TGLF, CGYRO, EFIT/P-EFIT, VMEC, TRANSP/TSC, or equivalent | Same case definitions, code/version provenance, numerical tolerances, and reproducible reports | Required for quantitative solver claims; not assumed by default. |
| Measured-facility validated | Matched to measured discharges or facility replay data | Shot IDs, diagnostics, acquisition manifests, facility data policy, checksums, and error metrics | Required for facility-control claims. |
| Deployment-certified | Safe for commissioned plant operation | Facility safety case, interlock qualification, operator procedures, latency envelope, cyber review, independent V&V, and acceptance sign-off | Not claimed. |

## What is production-oriented now

- API and package surfaces are treated as production software contracts.
- Optional external dependencies are guarded and fail closed where practical.
- Claim-admission helpers reject unsupported facility claims unless reference
  artifacts pass strict validators.
- Physics traceability records validity domains, public-claim status, required
  actions, evidence paths, and external-validation blockers.
- Module-specific tests are preferred over generic coverage buckets.
- Benchmark reports are persisted only as bounded evidence unless their matching
  validator admits the stronger claim.

## What is not production-grade for plant operation

- No real tokamak actuator authority is claimed.
- No safety interlock or machine-protection acceptance is claimed.
- No ITER CODAC, EPICS, MARTe, or site PCS commissioning evidence is claimed.
- No operator training, alarm response, or procedure package is included.
- No measured-facility validation is implied by synthetic, deterministic, or
  repository-local benchmarks.
- No external-code agreement is implied unless the exact artifact gate for that
  surface passes.

## Reading the evidence correctly

Use `validation/physics_traceability.json` and `docs/physics_traceability.md` as
the authority for public claim boundaries. A benchmark report means the bounded
path is reproducible. It does not by itself upgrade a model to measured-facility
validation or deployment certification.

Before presenting a stronger claim for any surface, require all of the following:

1. The traceability entry allows the claim level.
2. The strict validator for that surface passes on the relevant artifacts.
3. The artifact declares source provenance, model identity, units, checksums,
   case count, and tolerances.
4. The report states the same validity domain as the public documentation.
5. The change is reflected in the changelog, validation docs, and generated
   traceability report.

## Decision tree for users

- If you need a Python package for research, bounded controller experiments,
  formal checks, and reproducible validation reports, SCPN Control is usable
  today inside the documented claim boundaries.
- If you need publication-grade quantitative physics claims, use only surfaces
  whose strict validators admit the required public, external-code, or measured
  artefacts.
- If you need target-hardware or HIL claims, require target-specific timing,
  interlock, backpressure, replay, and hardware metadata reports.
- If you need plant deployment, treat this repository as supporting evidence
  infrastructure only; facility-specific safety engineering remains required.

## Narrative for enterprise decision-makers

For enterprise programs, the important signal is not only what is implemented but
what is *admissible for decision gates*. SCPN Control provides:

- explicit separation between bounded/local and external/measured evidence levels,
- deterministic artefact metadata,
- strict default refusal for over-claims,
- and a documented path to move a surface from bounded → validated → measured.

The current value is strongest in governance and reproducibility: teams can
decide what they can responsibly claim to investors, collaborators, or internal
review boards, and where each claim still needs external physics or hardware
evidence.

## Current concise classification

SCPN Control is a production-oriented research and control-integration library.
It is not production-grade software for real tokamak operation or certified
industrial deployment.

## How to use this boundary in release planning

Treat this page as the first checkpoint before any external claim. A change is ready for internal use when it remains within the current readiness level and all required checks are recorded in the matching validator output.

Before promoting a surface:

- confirm the claim level in the table remains true for that surface,
- confirm the linked validator run records provenance, units, and tolerances,
- confirm timing claims were produced under the matching benchmark context.

A release can include local capability upgrades while keeping facility claims blocked, as long as each change is anchored to the exact evidence lane that admitted it.
