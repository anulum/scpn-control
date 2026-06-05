<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# SCPN Control

SCPN Control is a neuro-symbolic fusion-control package for turning controller
ideas into auditable, replayable, evidence-bound software artefacts. It combines
stochastic Petri nets, spiking neural control, robust and nonlinear MPC,
phase-dynamics monitoring, differentiable physics facades, public-data
validation gates, and runtime safety contracts.

![SCPN Control Header](scpn_control_header.png)

## Plain-language summary

A fusion control team needs more than a simulator. It needs a way to answer
five questions before any controller moves toward hardware:

1. What physical contract is this controller allowed to claim?
2. Which data, reference code, or hardware evidence supports that claim?
3. Does the control logic satisfy bounded safety invariants?
4. Can the runtime fail closed when evidence, authentication, or configuration
   is missing?
5. Can another lab replay the result and see the same admission decision?

SCPN Control is built around those questions. It is not a replacement for a
facility PCS, EFIT deployment, TRANSP workflow, or external gyrokinetic code.
It is the controller-facing evidence layer that makes those integrations safer
to discuss, test, fund, and review.

## Current release focus

The current release focus is native execution evidence and formal-runtime claim
discipline. SCPN Control now separates Python-orchestrated campaigns from fused
Rust/PyO3 campaigns, reports formal-verification coverage explicitly, and uses
digest-bound AOT certificate evidence for the hot path. Non-isolated
workstation timings remain local-regression evidence; production benchmark
claims still require isolated cores, host-load metadata, governor context, and
target-hardware or PREEMPT_RT admission.

## Main surfaces

| Surface | Purpose |
| --- | --- |
| Petri-net to SNN compiler | Build neuro-symbolic controllers from explicit places, transitions, and firing contracts. |
| Formal verification | Produce bounded reachability, marking, liveness, CTL/LTL, and certificate-bundle evidence for control logic. |
| NMPC and robust control | Prototype constrained plasma-control policies with strict solver lifecycle and admission checks. |
| Differentiable physics facades | Expose JAX-backed controller-tuning paths without claiming facility fidelity before admission. |
| Digital twin and replay | Couple controller logic to replayable plant and external-simulator metadata. |
| Public-data validation | Convert and evaluate public datasets while keeping predictive claims blocked until full evidence exists. |
| Runtime security | Harden WebSocket, native-code, config, and artefact-loading boundaries for fail-closed operation. |
| Documentation and release evidence | Publish enough context for users, collaborators, reviewers, and funders to understand the product boundary. |

## What the software is for

SCPN Control is for the stage where a fusion-control idea needs to become a
reviewable software artefact. The package helps you answer: what controller was
run, what physics/replay contract fed it, what assumptions were declared, what
validation artefact was produced, what checksum binds that artefact, and what
stronger claims are still blocked.

The core product is not a single algorithm. It is the control-evidence workflow:
controller construction, bounded verification, physics/replay integration,
benchmarking, report generation, strict admission, documentation, and release
tracking. That workflow is useful to research groups, startups, facility teams,
and funders because it turns ambiguous demonstrations into auditable evidence.

## Application and value map

| Application | Immediate value | Evidence still required for stronger claims |
| --- | --- | --- |
| Controller concept review | Prototype SPN/SNN, NMPC, robust, phase, and digital-twin controllers with explicit assumptions | HIL replay, target-hardware timing, independent safety review |
| Neural equilibrium preparation | Prepare public MAST EFM datasets, launch reports, and result templates without hiding blockers | Executed full-output training, holdout metrics, latency/GPU-cost reports, EFIT/P-EFIT admission |
| Differentiable control tuning | Use JAX transport/equilibrium paths and gradient evidence for optimisation experiments | External physics validation and target-hardware timing evidence |
| Gyrokinetic validation | Persist CPU/GPU backend parity and local bounded reports | Real GENE, TGLF, GS2, CGYRO, or QuaLiKiz comparison artefacts |
| Safety-case infrastructure | Emit formal proof and certificate bundles for bounded controller logic | Facility hazard analysis, independent V&V, requirements traceability |
| Collaboration and financing | Show exactly what data, compute, and hardware evidence are missing | Funded acquisition, external-code runs, facility-data permissions |

## Visual overview

![SCPN Control Logic Map](SCPN%20CONTROL.PNG)
![SCPN Control Logic Map (compact)](SCPN%20CONTROL%20small.png)

## Fast start

```bash
pip install scpn-control
scpn-control demo --steps 1000
scpn-control validate
```

For a guided path, start with [Onboarding](onboarding.md), then run the
[First Steps tutorial](tutorials/first_steps.md), then choose a workflow from
[Use Cases](use_cases.md).

## Evidence boundary

SCPN Control publishes many validators and reports. A passing bounded validator
means the repository contract for that surface passed. It does not mean that a
facility, regulator, ITER safety case, or external physics authority has
accepted the result.

A full-fidelity public claim requires all required artefacts for that surface:
measured-shot or documented public-reference data, external-code or independent
reference where applicable, exact unit contracts, checksums, declared tolerances,
target hardware timing where relevant, and human review.

## Support and collaboration

The project needs collaborators and financing for target-hardware benchmarks,
external-code validation, public-data curation, cloud GPU campaigns, safety-case
review, and facility replay access. See [Compute Validation Funding](compute_validation_financing.md)
for concrete budget categories and why each category matters.
