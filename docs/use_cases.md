<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Use Cases and Market Value

SCPN Control is aimed at the gap between fusion-control research and facility
promotion. Its value is not only that it can run controllers; it can attach
claim boundaries, checksums, validators, formal evidence, and release artefacts
to the result.

## Primary users

| User | Pain point | SCPN Control value |
| --- | --- | --- |
| Fusion control researchers | Controller prototypes are hard to replay and review. | Executable controller artefacts with formal and validation metadata. |
| Fusion startups | Need credible control evidence before hardware exists. | Digital-twin, public-data, and strict admission workflows for staged review. |
| Facility teams | New algorithms need safety and timing boundaries before PCS integration. | Fail-closed runtime contracts, HIL evidence hooks, and target-hardware gates. |
| Academic groups | Students need one package spanning control, physics facades, and safety evidence. | Tutorials, notebooks, API docs, and reproducible validators. |
| Funders and collaborators | Need to see which claims are real and which require external resources. | Public roadmap, compute-validation budget categories, and honest blocked gates. |

## Application areas

### Controller concept development

Use stochastic Petri nets, robust control, NMPC, SNN controllers, and phase
monitoring to express candidate plasma-control policies. The formal-verification
surface can check bounded marking, liveness, reachability, and temporal
properties before the controller is promoted to replay or HIL workflows.

### Safety-case preparation

The package emits hash-bound evidence for safety-critical controller artefacts,
formal certificates, WebSocket runtime evidence, CODAC runtime evidence, FPGA
export evidence, and target-hardware timing. These reports are designed to be
inputs to a larger safety case, not replacements for facility approval.

### Public-data and external-code validation

Recent validation tooling converts and evaluates public MAST EFM data for neural
equilibrium reference-candidate work, while keeping predictive EFIT/P-EFIT
claims blocked until all strict inputs are available. Similar validators exist
for gyrokinetic, transport, MHD, disruption, and digital-twin surfaces.

### Differentiable controller tuning

JAX-backed equilibrium, transport, and gyrokinetic parity surfaces make gradient
information available to controller-tuning workflows. The repository admits
backend parity and bounded gradients separately from full facility physics
claims.

### Digital twin with online updating

The digital-twin surface supports strict simulator metadata, Bayesian update
records, and external TRANSP/TSC bridge contracts. The market value is staged:
first reproducible replay, then target-hardware timing, then facility-specific
admission.

### Local-first physics debugging

The physics-debug assistant is designed for onsite use. It supports local
providers first, optional hallucination guardrails, strict prompt-injection
neutralisation, tamper-evident report digests, and human-review caps. Its output
is advisory evidence, never automatic controller promotion.

### Quantum-enhanced disruption research

The quantum-disruption bridge contracts prepare controller-side integration for
SCPN Quantum Control without duplicating that repository. SCPN Control owns the
admission boundary and disruption-control interface; quantum-kernel development
can mature in the quantum repository and return as an evidence-bound provider.

## Market position

SCPN Control is not trying to replace major physics suites. Its commercial and
collaboration value is the control-evidence layer:

- faster iteration between controller idea and reviewable artefact;
- fewer unbounded claims in papers, grant proposals, and demos;
- clearer separation between local research evidence and facility readiness;
- explicit funding targets for data, hardware, and external-code validation;
- reusable interfaces for labs that need local or air-gapped workflows.

## Current limitations

- No commissioned plant deployment is claimed.
- Full predictive EFIT/P-EFIT claims remain blocked until matched public or
  facility reference artefacts pass admission.
- Full cross-code gyrokinetic and integrated-modelling claims remain blocked
  where external artefacts are still missing.
- Hardware timing evidence must be collected on target platforms before
  real-time deployment claims are admissible.

For claim language, use [Production Readiness](production_readiness.md) as the
source of truth.
