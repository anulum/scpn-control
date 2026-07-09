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

## What a collaborator can fund or contribute

| Contribution | Repository outcome | Why it matters |
| --- | --- | --- |
| GPU or HPC allocation | Executed training, latency, and uncertainty reports | Converts prepared pipelines into measured evidence |
| Public or facility data access | Immutable acquisition manifests and reference reports | Moves claims from bounded to reference-validated |
| External-code runs | Matched input decks, raw outputs, parsed metrics, and checksums | Closes gyrokinetic, transport, equilibrium, and digital-twin gaps |
| Target hardware | P50/P95/P99 timing and HIL replay reports | Separates local timing from deployment-relevant timing |
| Safety review | Independent review artefacts and requirement traceability | Makes formal evidence useful for facility processes |
| Documentation review | Better onboarding, tutorials, and claim wording | Lowers adoption friction and prevents overstated claims |

## Why this is different from a solver repository

A solver repository tries to improve physical models. SCPN Control tries to make
controller claims auditable. It can wrap or port selected solver contracts, but
its distinctive value is the evidence path around controllers: fail-closed
runtime boundaries, proof artefacts, replay metadata, benchmark admission, and
release documentation that tells users what is ready, what is bounded, and what
is blocked.

## Current limitations

- No commissioned plant deployment is claimed.
- Full predictive EFIT/P-EFIT claims remain blocked until matched public or
  facility reference artefacts pass admission.
- Full cross-code gyrokinetic and integrated-modelling claims remain blocked
  where external artefacts are still missing.
- Hardware timing evidence must be collected on target platforms before
  real-time deployment claims are admissible.

## Read this page as a decision map

For each user profile above, SCPN Control should be considered in three levels:

1. **Internal iteration:** use local and local-regression evidence to reduce risk and
   increase repeatability.
2. **Pre-facility argument:** add reference adapters, stricter validation gates,
   and publication-quality reproducibility artifacts.
3. **Deployment-ready argument:** pair validated controllers with external-code and
   facility pathways that already have operator and safety governance.

This structure keeps the same codebase useful across research and collaboration
contexts while avoiding over-claiming before each governance boundary is met.

For claim language, use [Production Readiness](production_readiness.md) as the
source of truth.

## How to apply this page to planning

Use this page to decide what to build next in a constrained budget.

- Pick the user profile that matches your immediate goal.
- Choose one primary surface and one blocked surface for that cycle.
- Collect only the evidence required for that profile before broadening scope.

This keeps the project from mixing research iteration with deployment commitment in the same cycle.

## Practical use and scope

Use this page to map target users and practical deployment scenarios.

- Use it when prioritizing which workflows to optimize next.
- Use this with pricing and readiness pages when planning support, proof, and roadmaps.
- Keep use-case claims tied to validated evidence instead of generic capability statements.
