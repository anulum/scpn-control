<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Control — API reference -->

# API Reference

## Top-Level Exports

```python
import scpn_control

scpn_control.__version__       # "0.19.2"
scpn_control.FusionKernel      # Grad-Shafranov equilibrium solver
scpn_control.RUST_BACKEND      # True if Rust acceleration available
scpn_control.TokamakConfig     # Preset tokamak geometries
scpn_control.StochasticPetriNet
scpn_control.FusionCompiler
scpn_control.CompiledNet
scpn_control.NeuroSymbolicController
scpn_control.kuramoto_sakaguchi_step
scpn_control.order_parameter
scpn_control.KnmSpec
scpn_control.build_knm_paper27
scpn_control.UPDESystem
scpn_control.LyapunovGuard
scpn_control.RealtimeMonitor
scpn_control.PhysicsDebugAssistant
```

---

## Physics Debug Assistance

`scpn_control.physics_debug` provides a local-first advisory assistant for
physics validation gaps. The default provider policy admits loopback endpoints
only; facility or external gateways must be explicitly allowlisted.
`build_local_provider()` supplies loopback profiles for common onsite gateway
protocols: chat-completions-compatible, Ollama-style chat, direct JSON, and
text-generation endpoints. Reports are schema-versioned advisory evidence with
secret redaction, falsifiable hypothesis checks, campaign risk controls, and
risk-bound prompt-injection neutralization for untrusted evidence text before
provider prompting. Prompt-guard findings are recorded in the tamper-evident
payload digest. `build_guardrail_provider()` adds an optional hallucination
guardrail gateway with a `director-ai` default profile and explicit alternate
profiles for lab-owned guardrail solutions. Guardrail block decisions fail
closed before report persistence; allow findings are bound into the report
digest together with the SHA-256 digest of the reviewed provider draft.
The guardrail request also binds the provider metadata, safety policy, and
guardrail policy digests so reviews cannot be replayed across another provider
or relaxed policy. High-severity guardrail findings must use block actions, and
risk controls must meet the configured guardrail policy before persistence.
They are not validated physics truth, controller-parameter promotion, or
facility safety approval.
`run_provider_quorum()` runs multiple providers in local-first order and admits
only hypotheses corroborated by the required provider count over the same gap
and evidence set.
`PhysicsDebugSafetyPolicy` binds the human-review requirement, caps advisory
confidence, and rejects provider text that attempts controller promotion,
actuation, review bypass, or approval claims.

::: scpn_control.physics_debug.ProviderPolicy

::: scpn_control.physics_debug.PhysicsDebugGuardrailPolicy

::: scpn_control.physics_debug.PhysicsDebugEvidence

::: scpn_control.physics_debug.PhysicsDebugGap

::: scpn_control.physics_debug.PhysicsDebugSafetyPolicy

::: scpn_control.physics_debug.HTTPChatProvider

::: scpn_control.physics_debug.PhysicsDebugGuardrailProvider

::: scpn_control.physics_debug.PhysicsDebugAssistant

::: scpn_control.physics_debug.build_local_provider

::: scpn_control.physics_debug.build_guardrail_provider

::: scpn_control.physics_debug.build_physics_debug_report

::: scpn_control.physics_debug.run_provider_quorum

::: scpn_control.physics_debug.validate_physics_debug_report

::: scpn_control.physics_debug.validate_physics_debug_quorum_report

::: scpn_control.physics_debug.write_physics_debug_report

::: scpn_control.physics_debug.write_physics_debug_quorum_report

---

## Control — Federated Disruption Prediction

`scpn_control.control.federated_disruption` supports FedAvg and FedProx
training across named tokamak clients without centralising facility arrays.
`create_facility_clients_from_arrays()` is the production ingestion boundary
for per-facility `X_train`, `y_train`, `X_test`, and `y_test` arrays. It
enforces the shared 8-feature disruption contract and binary labels before a
client joins the federation.

`DifferentialPrivacyConfig` enables facility-update clipping, Gaussian noise,
and a serialisable privacy ledger. The shipped benchmark
`validation/benchmark_federated_disruption.py` publishes deterministic
synthetic multi-facility evidence in
`validation/reports/federated_disruption_benchmark.json` and
`validation/reports/federated_disruption_benchmark.md`. Those artefacts test
federation, heterogeneity, and differential privacy contracts; they do not
claim measured cross-facility validation without external shot databases.

::: scpn_control.control.federated_disruption.DifferentialPrivacyConfig

::: scpn_control.control.federated_disruption.PrivacyLedgerEntry

::: scpn_control.control.federated_disruption.FacilityBenchmarkSummary

::: scpn_control.control.federated_disruption.FederatedConfig

::: scpn_control.control.federated_disruption.MachineClient

::: scpn_control.control.federated_disruption.FederatedServer

::: scpn_control.control.federated_disruption.create_facility_clients_from_arrays

::: scpn_control.control.federated_disruption.run_synthetic_multifacility_benchmark

---

## Control — Quantum Disruption Bridge

`scpn_control.control.quantum_disruption_bridge` is a fail-closed facade for
optional quantum-enhanced disruption prediction. Quantum circuit and provider
ownership stays in `scpn-quantum-control`; SCPN-CONTROL owns the control
feature contract, lazy optional import boundary, bounded claim metadata, and
tamper-evident advisory reports. The bridge maps the CONTROL 8-feature
disruption vector to the ITER 11-feature contract only when missing ITER fields
are either supplied explicitly or declared as bounded centre defaults. Reports
are not facility validation, controller promotion, or publication-safe evidence
without external disruption databases and benchmark artefacts.

`quantum_disruption_kernel_matrix()` emits a bounded amplitude-encoding kernel
report with symmetry, diagonal, and `[0, 1]` admission checks. The callable
quantum owner path uses `scpn_quantum_control.control.q_disruption_iter`
lazily; when that optional dependency is unavailable the report fails closed
with `status="quantum-unavailable"` and no quantum score. Every bridge report
also records advisory admission evidence: CONTROL feature digests, ITER mapping
digests, default-use reasons, and the external evidence still required before
facility or publication claims are admissible. Bridge and kernel reports carry
schema-versioned advisory certificates that bind report kind, repository
ownership, claim boundary, downstream non-admission policy, and the content
digest before the outer payload digest is accepted. The facade also publishes a
machine-readable dependency contract that names the `scpn-quantum-control`
backend module, required classifier surface, Qiskit core dependencies, optional
provider dependency families, report schemas, feature contract, and
non-admission policy for future backend hardening. Generated bridge and kernel
reports embed that dependency contract and bind its digest into the advisory
certificate so archived reports cannot be replayed against a different quantum
backend contract. When the optional backend exposes its own
`scpn_control_bridge_dependency_contract()` callable, CONTROL compares it
against the expected contract, records backend-contract attestation evidence,
and fails closed before report creation if the backend advertises a conflicting
contract. Bridge reports also include schema-versioned advisory decision
evidence that records whether the score came from the quantum backend or the
classical fallback, applies deterministic low/elevated/high risk-band
thresholds, records backend-contract validation state, fixes the control action
to `blocked`, and binds the decision digest into the advisory certificate.

::: scpn_control.control.quantum_disruption_bridge.QuantumDisruptionBridgeConfig

::: scpn_control.control.quantum_disruption_bridge.QuantumFeatureMapping

::: scpn_control.control.quantum_disruption_bridge.map_control_features_to_iter

::: scpn_control.control.quantum_disruption_bridge.quantum_disruption_kernel_matrix

::: scpn_control.control.quantum_disruption_bridge.quantum_disruption_dependency_contract

::: scpn_control.control.quantum_disruption_bridge.run_quantum_disruption_bridge

::: scpn_control.control.quantum_disruption_bridge.validate_quantum_disruption_dependency_contract

::: scpn_control.control.quantum_disruption_bridge.validate_quantum_disruption_bridge_report

::: scpn_control.control.quantum_disruption_bridge.validate_quantum_disruption_kernel_report

---

## Core — Physics Solvers

### FusionKernel

`FusionKernel` validates its JSON configuration before grid construction:
the root must be an object, duplicate JSON keys are rejected, dimensions and
grid resolution must be physical, `physics.plasma_current_target` must be
positive finite, and `physics.vacuum_permeability` must be positive finite
when supplied.

::: scpn_control.core.fusion_kernel.FusionKernel

### TokamakConfig

::: scpn_control.core.tokamak_config.TokamakConfig

### TransportSolver

::: scpn_control.core.integrated_transport_solver.TransportSolver

### Scaling Laws

::: scpn_control.core.scaling_laws.ipb98y2_tau_e

::: scpn_control.core.scaling_laws.compute_h_factor

### GEQDSK I/O

::: scpn_control.core.eqdsk.GEqdsk

::: scpn_control.core.eqdsk.read_geqdsk

::: scpn_control.core.eqdsk.write_geqdsk

### Uncertainty Quantification

::: scpn_control.core.uncertainty.quantify_uncertainty

::: scpn_control.core.uncertainty.quantify_full_chain

::: scpn_control.core.uncertainty.UQClaimEvidence

::: scpn_control.core.uncertainty.uq_claim_evidence

::: scpn_control.core.uncertainty.assert_uq_calibrated_claim_admissible

::: scpn_control.core.uncertainty.save_uq_claim_evidence

### JAX-Accelerated Transport Primitives

Requires `pip install "scpn-control[jax]"`. GPU execution automatic when jaxlib has CUDA/ROCm.

::: scpn_control.core.jax_solvers.thomas_solve

::: scpn_control.core.jax_solvers.diffusion_rhs

::: scpn_control.core.jax_solvers.crank_nicolson_step

::: scpn_control.core.jax_solvers.batched_crank_nicolson

### Differentiable Transport Facade

Requires `pip install "scpn-control[jax]"` for gradient evaluation. The NumPy
path is deterministic for parity checks and non-JAX deployments, but
`transport_loss_gradient()` fails closed without JAX.
`transport_parameter_gradients()` extends the same traced Crank-Nicolson
contract to source schedules, returning JAX gradients for both transport
coefficients and additive heating, fuelling, or impurity-source inputs.
`differentiable_transport_rollout()` advances a bounded multi-step source
schedule with the same four-channel boundary contract.
`transport_rollout_source_gradients()` returns fail-closed JAX gradients for
that full source schedule so controller tuning can optimise time-distributed
heating, fuelling, and impurity-source inputs without finite differences. The
rollout gradient path keeps the loss inside the traced JAX graph and enables
JAX x64 before importing `jax.numpy`, so persisted dtype evidence is not
silently downgraded.
`audit_transport_rollout_source_gradients()` and
`assert_transport_rollout_source_gradients_consistent()` compare those rollout
gradients against sampled NumPy finite-difference perturbations.
`audit_transport_parameter_gradients()` and
`assert_transport_parameter_gradients_consistent()` compare those JAX gradients
against sampled independent finite-difference perturbations before
controller-tuning admission.
`transport_coefficients_from_neural_closure()` maps bounded neural transport
closure outputs into the four-channel coefficient order used by the facade:
electron heat, ion heat, electron particle diffusivity, and a declared impurity
diffusivity fraction.
`gyrokinetic_transport_closure_profiles()` wraps the reduced gyrokinetic
transport profile evaluator as bounded closure provenance, and
`transport_coefficients_from_gyrokinetic_closure()` maps that closure into the
same four-channel coefficient order without promoting the reduced GK model to
an externally validated transport claim.
`transport_campaign_metadata()` records backend, dtype, radial grid, timestep,
boundary conditions, closure provenance, gradient tolerance, and optional
equilibrium-grid shape for reproducible controller-tuning campaigns.
`save_transport_campaign_metadata()` and `load_transport_campaign_metadata()`
persist the same contract as schema-versioned JSON and fail closed on malformed
or physically inconsistent replay metadata.
`assert_transport_campaign_metadata_replay()` compares archived campaign
metadata with a candidate setup and raises on backend, grid, boundary, closure,
gradient-tolerance, or equilibrium-shape drift before controller tuning reruns.
`transport_differentiability_evidence()` and
`assert_transport_differentiability_claim_admissible()` add a tamper-evident
admission envelope over campaign metadata and gradient-audit results. That
envelope requires JAX backend evidence, passed sampled finite-difference
gradient audit, stable SHA-256 digests for the campaign and audit payloads, and
an optional link to the safety-critical controller proof artifact digest.
Admission revalidates finite non-negative audit losses and errors, tolerance
agreement with campaign metadata, unique in-domain sampled audit indices,
pass/fail consistency with maximum audit error, and ordered latency percentiles
before persisted controller-tuning evidence is accepted.
`equilibrium_weighted_transport_rollout_tracking_loss()` extends the optional
Grad-Shafranov flux-map weighting from one transport step to a full source
rollout. `equilibrium_weighted_transport_rollout_source_gradient()` returns
fail-closed JAX gradients with respect to both the source schedule and the
equilibrium flux map for controller-tuning studies.

::: scpn_control.core.differentiable_transport.differentiable_transport_step

::: scpn_control.core.differentiable_transport.transport_tracking_loss

::: scpn_control.core.differentiable_transport.transport_loss_gradient

::: scpn_control.core.differentiable_transport.transport_parameter_gradients

::: scpn_control.core.differentiable_transport.TransportParameterGradients

::: scpn_control.core.differentiable_transport.differentiable_transport_rollout

::: scpn_control.core.differentiable_transport.transport_rollout_tracking_loss

::: scpn_control.core.differentiable_transport.transport_rollout_source_gradients

::: scpn_control.core.differentiable_transport.TransportRolloutSourceGradients

::: scpn_control.core.differentiable_transport.audit_transport_rollout_source_gradients

::: scpn_control.core.differentiable_transport.assert_transport_rollout_source_gradients_consistent

::: scpn_control.core.differentiable_transport.TransportRolloutGradientAudit

::: scpn_control.core.differentiable_transport.audit_transport_parameter_gradients

::: scpn_control.core.differentiable_transport.assert_transport_parameter_gradients_consistent

::: scpn_control.core.differentiable_transport.TransportGradientAudit

::: scpn_control.core.differentiable_transport.benchmark_transport_parameter_gradient_latency

::: scpn_control.core.differentiable_transport.TransportGradientLatencyReport

::: scpn_control.core.differentiable_transport.save_transport_gradient_latency_report

::: scpn_control.core.differentiable_transport.benchmark_transport_rollout_source_gradient_latency

::: scpn_control.core.differentiable_transport.TransportRolloutGradientLatencyReport

::: scpn_control.core.differentiable_transport.save_transport_rollout_gradient_latency_report

::: scpn_control.core.differentiable_transport.transport_coefficients_from_neural_closure

::: scpn_control.core.differentiable_transport.gyrokinetic_transport_closure_profiles

::: scpn_control.core.differentiable_transport.transport_coefficients_from_gyrokinetic_closure

::: scpn_control.core.differentiable_transport.GyrokineticTransportClosureResult

::: scpn_control.core.differentiable_transport.TransportCampaignMetadata

::: scpn_control.core.differentiable_transport.transport_campaign_metadata

::: scpn_control.core.differentiable_transport.save_transport_campaign_metadata

::: scpn_control.core.differentiable_transport.load_transport_campaign_metadata

::: scpn_control.core.differentiable_transport.assert_transport_campaign_metadata_replay

::: scpn_control.core.differentiable_transport.TransportDifferentiabilityEvidence

::: scpn_control.core.differentiable_transport.transport_differentiability_evidence

::: scpn_control.core.differentiable_transport.assert_transport_differentiability_claim_admissible

::: scpn_control.core.differentiable_transport.equilibrium_radial_weights

::: scpn_control.core.differentiable_transport.equilibrium_weighted_transport_tracking_loss

::: scpn_control.core.differentiable_transport.equilibrium_weighted_transport_loss_gradient

::: scpn_control.core.differentiable_transport.EquilibriumWeightedTransportGradient

::: scpn_control.core.differentiable_transport.equilibrium_weighted_transport_rollout_tracking_loss

::: scpn_control.core.differentiable_transport.equilibrium_weighted_transport_rollout_source_gradient

::: scpn_control.core.differentiable_transport.EquilibriumWeightedTransportRolloutGradient

### Neural Equilibrium

`NeuralEquilibriumAccelerator.pretrain_from_synthetic_equilibria()` trains
JAX-compatible PCA plus MLP weights on bounded synthetic Solovev-like
equilibria for pretraining. The corresponding real EFIT fine-tuning entry point
`fine_tune_from_efit_reconstructions()` fails closed unless the persisted
P-EFIT or documented-public-reference artefact validator passes.

::: scpn_control.core.neural_equilibrium.NeuralEquilibriumAccelerator

::: scpn_control.core.neural_equilibrium.NeuralEquilibriumClaimEvidence

::: scpn_control.core.neural_equilibrium.generate_synthetic_equilibrium_dataset

::: scpn_control.core.neural_equilibrium.neural_equilibrium_claim_evidence

::: scpn_control.core.neural_equilibrium.assert_neural_equilibrium_facility_claim_admissible

::: scpn_control.core.neural_equilibrium.save_neural_equilibrium_claim_evidence

::: scpn_control.core.neural_equilibrium.pretrain_neural_equilibrium_synthetic

::: scpn_control.core.neural_equilibrium.PretrainingResult

::: scpn_control.core.neural_equilibrium.SyntheticEquilibriumCampaign

### JAX-Accelerated Neural Equilibrium

Requires `pip install "scpn-control[jax]"`. GPU and autodiff via `jax.grad`.

::: scpn_control.core.jax_neural_equilibrium.jax_neural_eq_predict

::: scpn_control.core.jax_neural_equilibrium.jax_neural_eq_predict_batched

::: scpn_control.core.jax_neural_equilibrium.load_weights_as_jax

### Neural Transport

`cross_validate_neural_transport()` benchmarks the active surrogate against the
analytic critical-gradient reference across fixed regime cases and a canonical
profile, so shipped weights can be checked against a deterministic local
baseline instead of only reporting synthetic training RMSE.

`neural_transport_closure_profiles()` packages profile transport coefficients
for controller and differentiable-transport coupling.  It validates finite
strictly ordered profile inputs, fails closed when neural weights are required
but unavailable, and records whether coefficients came from loaded weights or
the analytic fallback.

::: scpn_control.core.neural_transport.NeuralTransportModel

::: scpn_control.core.neural_transport.NeuralTransportClosureResult

::: scpn_control.core.neural_transport.NeuralTransportClaimEvidence

::: scpn_control.core.neural_transport.neural_transport_closure_profiles

::: scpn_control.core.neural_transport.cross_validate_neural_transport

::: scpn_control.core.neural_transport.neural_transport_claim_evidence

::: scpn_control.core.neural_transport.assert_neural_transport_quantitative_claim_admissible

::: scpn_control.core.neural_transport.save_neural_transport_claim_evidence

### MHD Stability

::: scpn_control.core.stability_mhd.run_full_stability_check

### IMAS Adapter

::: scpn_control.core.imas_adapter.EquilibriumIDS

::: scpn_control.core.imas_adapter.from_geqdsk

::: scpn_control.core.imas_adapter.from_kernel

### HPC Bridge

`HPCBridge` loads compiled Grad-Shafranov solver libraries only from absolute
dynamic-library paths. Package-local solver libraries are trusted by default.
External paths provided through `SCPN_SOLVER_LIB` require the additional
operator gate `SCPN_ALLOW_EXTERNAL_SOLVER_LIB=1`; without that gate the bridge
fails closed before calling the dynamic loader.

::: scpn_control.core.hpc_bridge.HPCBridge

### Gyrokinetic Transport (v0.16.0)

::: scpn_control.core.gyrokinetic_transport.GyrokineticTransportModel

### GK Solver Interface (v0.17.0)

::: scpn_control.core.gk_interface.GKSolverBase

::: scpn_control.core.gk_interface.GKLocalParams

::: scpn_control.core.gk_interface.GKOutput

### Native Linear GK Solver (v0.17.0)

::: scpn_control.core.gk_eigenvalue.solve_linear_gk

::: scpn_control.core.gk_quasilinear.quasilinear_fluxes_from_spectrum

### GK Hybrid Validation (v0.17.0)

::: scpn_control.core.gk_ood_detector.OODDetector

::: scpn_control.core.gk_scheduler.GKScheduler

::: scpn_control.core.gk_corrector.GKCorrector

### Ballooning Solver (v0.16.0)

::: scpn_control.core.ballooning_solver.BallooningEquation

::: scpn_control.core.ballooning_solver.BallooningStabilityAnalysis

::: scpn_control.core.ballooning_solver.find_marginal_stability

### Current Diffusion (v0.16.0)

::: scpn_control.core.current_diffusion.CurrentDiffusionSolver

### Current Drive (v0.16.0)

::: scpn_control.core.current_drive.ECCDSource

::: scpn_control.core.current_drive.NBISource

::: scpn_control.core.current_drive.CurrentDriveMix

### NTM Dynamics (v0.16.0)

::: scpn_control.core.ntm_dynamics.NTMController

### Sawtooth Model (v0.16.0)

::: scpn_control.core.sawtooth.SawtoothCycler

::: scpn_control.core.sawtooth.kadomtsev_crash

### SOL Model (v0.16.0)

::: scpn_control.core.sol_model.TwoPointSOL

### Integrated Scenario (v0.16.0)

::: scpn_control.core.integrated_scenario.IntegratedScenarioSimulator

::: scpn_control.core.integrated_scenario.audit_scenario_coupling

::: scpn_control.core.integrated_scenario.save_scenario_coupling_report

::: scpn_control.core.integrated_scenario.iter_baseline_scenario

---

## SCPN — Petri Net Compiler

### StochasticPetriNet

::: scpn_control.scpn.structure.StochasticPetriNet

### Formal Verification

::: scpn_control.scpn.formal_verification.FormalPetriNetVerifier

::: scpn_control.scpn.formal_verification.verify_formal_contracts

::: scpn_control.scpn.formal_verification.PlaceInvariant

::: scpn_control.scpn.formal_verification.SafetyCertificatePolicy

::: scpn_control.scpn.formal_verification.SafetyCertificateBundlePolicy

::: scpn_control.scpn.formal_verification.CTLFormula

::: scpn_control.scpn.formal_verification.LTLFormula

::: scpn_control.scpn.formal_verification.AlwaysBounded

::: scpn_control.scpn.formal_verification.AlwaysEventuallyMarked

::: scpn_control.scpn.formal_verification.EventuallyFires

::: scpn_control.scpn.formal_verification.FireLeadsToMarking

::: scpn_control.scpn.formal_verification.NeverCoMarked

::: scpn_control.scpn.formal_verification.build_safety_certificate_payload

::: scpn_control.scpn.formal_verification.build_safety_certificate_bundle_payload

::: scpn_control.scpn.formal_verification.build_safety_certificate_bundle_artifact

::: scpn_control.scpn.formal_verification.generate_safety_certificate

::: scpn_control.scpn.formal_verification.validate_safety_certificate_payload

::: scpn_control.scpn.formal_verification.validate_safety_certificate_bundle_payload

::: scpn_control.scpn.formal_verification.validate_safety_certificate_bundle_artifact

::: scpn_control.scpn.formal_verification.admit_safety_certificate_bundle_artifact

::: scpn_control.scpn.formal_verification.write_safety_certificate

::: scpn_control.scpn.formal_verification.write_safety_certificate_bundle

::: scpn_control.scpn.z3_model_checking.Z3BoundedModelChecker

::: scpn_control.scpn.z3_model_checking.verify_z3_formal_contracts

::: scpn_control.scpn.z3_model_checking.build_z3_formal_report_payload

::: scpn_control.scpn.z3_model_checking.build_blocked_z3_formal_report_payload

::: scpn_control.scpn.z3_model_checking.validate_z3_formal_report_payload

::: scpn_control.scpn.z3_model_checking.write_z3_formal_report

### FusionCompiler

::: scpn_control.scpn.compiler.FusionCompiler

### CompiledNet

::: scpn_control.scpn.compiler.CompiledNet

### NeuroSymbolicController

`NeuroSymbolicController` rejects nonzero `sc_bitflip_rate` unless
`allow_fault_injection=True` is supplied explicitly and the process environment
sets `SCPN_ALLOW_CONTROLLER_FAULT_INJECTION=1`. Bit-flip mutation is a
double-gated fault-injection test mode, not a production control default.

Controller JSONL logging requires an explicit `log_root` whenever `log_path` is
provided. Relative and absolute log paths must resolve under that root and use a
`.jsonl` suffix before any file is opened. Log appends use a constrained append
helper that rejects symlink targets where the platform exposes no-follow open
semantics.

::: scpn_control.scpn.controller.NeuroSymbolicController

### Contracts

::: scpn_control.scpn.contracts.ControlObservation

::: scpn_control.scpn.contracts.ControlAction

::: scpn_control.scpn.contracts.ControlTargets

::: scpn_control.scpn.contracts.extract_features

::: scpn_control.scpn.contracts.decode_actions

### Artifacts

Safety-critical controller admission must call `load_artifact(...,
require_formal_verification=True)` or `validate_safety_critical_artifact()`.
That gate rejects missing, blocked, failed, malformed, or unbounded proof
evidence and accepts only hash-addressed bounded proof manifests tied to the
compiled controller artifact. The proof manifest must include the canonical
artifact payload SHA-256, report SHA-256, bounded proof depth, checked
specification names, backend/solver metadata, and a safe relative report URI.
When callers provide `formal_report_root`, the loader resolves the report URI
under that root and verifies the report bytes against the declared SHA-256. Z3
reports are additionally schema-versioned as
`scpn-control.z3-formal-report.v1`, carry a canonical payload SHA-256 over the
proof payload, and must match the manifest status, solver, proof depth, and
checked specification list before a safety-critical artifact is admitted.

::: scpn_control.scpn.artifact.Artifact

::: scpn_control.scpn.artifact.FormalVerificationEvidence

::: scpn_control.scpn.artifact.compute_artifact_payload_sha256

::: scpn_control.scpn.artifact.save_artifact

::: scpn_control.scpn.artifact.load_artifact

::: scpn_control.scpn.artifact.validate_safety_critical_artifact

---

## Phase — Paper 27 Dynamics

### Kuramoto-Sakaguchi Step

::: scpn_control.phase.kuramoto.kuramoto_sakaguchi_step

::: scpn_control.phase.kuramoto.order_parameter

::: scpn_control.phase.kuramoto.lyapunov_v

::: scpn_control.phase.kuramoto.lyapunov_exponent

::: scpn_control.phase.kuramoto.wrap_phase

::: scpn_control.phase.kuramoto.GlobalPsiDriver

### Knm Coupling Matrix

::: scpn_control.phase.knm.KnmSpec

::: scpn_control.phase.knm.build_knm_paper27

### UPDE Multi-Layer Solver

::: scpn_control.phase.upde.UPDESystem

### Lyapunov Guard

::: scpn_control.phase.lyapunov_guard.LyapunovGuard

### Realtime Monitor

::: scpn_control.phase.realtime_monitor.RealtimeMonitor

::: scpn_control.phase.realtime_monitor.TrajectoryRecorder

### Adaptive Knm Engine

::: scpn_control.phase.adaptive_knm.AdaptiveKnmEngine

::: scpn_control.phase.adaptive_knm.AdaptiveKnmConfig

::: scpn_control.phase.adaptive_knm.DiagnosticSnapshot

### Plasma Knm

::: scpn_control.phase.plasma_knm.build_knm_plasma

### WebSocket Stream

`PhaseStreamServer` binds to loopback by default and requires authenticated
clients by default.  Operators must supply an API key or explicitly disable
client authentication for local development.  Non-loopback binds require an API
key, command frames are capped by `max_payload_bytes`, accepted commands are
rate-limited per connection, and production remote exposure should enable TLS
with `require_tls=True`.  Query-string token authentication and plaintext
non-loopback binds are disabled by default and require explicit operator
opt-ins for constrained development or isolated lab environments.  Browser
clients that send an `Origin` header are rejected unless the origin is
allowlisted, and deployments may restrict command authority with
`allowed_actions`.

::: scpn_control.phase.ws_phase_stream.PhaseStreamServer

---

## Control — Controllers

### H-infinity (Riccati DARE)

::: scpn_control.control.h_infinity_controller.HInfinityController

::: scpn_control.control.h_infinity_controller.get_radial_robust_controller

### Model Predictive Control

::: scpn_control.control.fusion_sota_mpc.NeuralSurrogate

::: scpn_control.control.fusion_sota_mpc.ModelPredictiveController

### Optimal Control

::: scpn_control.control.fusion_optimal_control.OptimalController

### Digital Twin

`run_digital_twin()` now supports persistent sensor calibration bias and drift
in addition to dropout and white-noise corruption, and it can now stress the
command path with deterministic actuator bias, drift, first-order lag, and
rate limiting. The returned summary exposes both commanded and applied actions
plus actuator-lag telemetry so replay tests can see what the plant actually
received. Density and effective-charge knobs are explicit model-update
parameters.

`digital_twin_online_update` adds fail-closed TRANSP/TSC simulator artifact
metadata validation and deterministic Bayesian optimisation over bounded twin
parameters. The shipped benchmark is synthetic online-update evidence only;
external simulator replay claims require validated artifact metadata and the
strict digital-twin reference gate.
`digital_twin_update_evidence()` and
`assert_digital_twin_update_claim_admissible()` bind a bounded Bayesian update
to TRANSP and TSC simulator metadata digests, observation and prior digests,
result digest, baseline-improvement evidence, and an optional
safety-critical controller proof artifact digest. Admission also revalidates
source binding, finite non-negative loss history, minimum-loss consistency,
best-parameter bounds, strict integer campaign settings, and simulator unit
coverage for every observation target.

::: scpn_control.control.tokamak_digital_twin.run_digital_twin

::: scpn_control.control.digital_twin_online_update.validate_external_simulator_artifact

::: scpn_control.control.digital_twin_online_update.bayesian_update_digital_twin

::: scpn_control.control.digital_twin_online_update.DigitalTwinUpdateEvidence

::: scpn_control.control.digital_twin_online_update.digital_twin_update_evidence

::: scpn_control.control.digital_twin_online_update.assert_digital_twin_update_claim_admissible

::: scpn_control.control.digital_twin_online_update.synthetic_online_update_benchmark

### Controller Safety-Case Evidence

`control.safety_case` defines the bounded safety-case workflow contract that
links a passing safety-critical controller proof manifest, audited JAX
differentiable-transport evidence, and TRANSP/TSC-backed bounded digital-twin
online-update evidence. The bundle is tamper-evident and fails closed unless
all evidence items bind to the same canonical controller artifact SHA-256. This
is a repository safety-package admission boundary, not an external
certification claim. `save_controller_safety_case_evidence()` persists the
bundle with a manifest integrity digest, and
`load_controller_safety_case_evidence()` rejects schema drift, malformed fields,
or edited evidence payloads before replay admission.
`evaluate_controller_safety_case_readiness()` separates linked bounded evidence
from promotion readiness: external physics validation, target-hardware timing
evidence, and independent safety-review digests are all required before
`assert_controller_safety_case_readiness_admissible()` accepts the package.
`ReadinessArtifactEvidence` and
`evaluate_controller_safety_case_readiness_from_artifacts()` provide the normal
promotion path: each required readiness input must be a typed artifact with a
known kind, SHA-256 digest, safe relative artifact URI, producer, and generation
timestamp before it can satisfy the promotion gate. The evaluator also requires
an explicit `artifact_root`: each URI must resolve below that root and match the
declared bytes. `target_hardware_timing` artifacts must additionally pass the
schema-versioned E2E latency evidence validator with qualified target hardware
and the configured p95 limit.
`save_controller_safety_case_readiness()` and
`load_controller_safety_case_readiness()` persist that readiness decision with
the same schema-versioned integrity-digest semantics as the safety-case bundle.

::: scpn_control.control.safety_case.ControllerSafetyCaseEvidence

::: scpn_control.control.safety_case.SafetyCaseReadinessEvidence

::: scpn_control.control.safety_case.ReadinessArtifactEvidence

::: scpn_control.control.safety_case.controller_safety_case_evidence

::: scpn_control.control.safety_case.assert_controller_safety_case_admissible

::: scpn_control.control.safety_case.save_controller_safety_case_evidence

::: scpn_control.control.safety_case.load_controller_safety_case_evidence

::: scpn_control.control.safety_case.evaluate_controller_safety_case_readiness

::: scpn_control.control.safety_case.evaluate_controller_safety_case_readiness_from_artifacts

::: scpn_control.control.safety_case.assert_controller_safety_case_readiness_admissible

::: scpn_control.control.safety_case.save_controller_safety_case_readiness

::: scpn_control.control.safety_case.load_controller_safety_case_readiness

### Flight Simulator

::: scpn_control.control.tokamak_flight_sim.IsoFluxController

::: scpn_control.control.tokamak_flight_sim.run_flight_sim

### Free-Boundary Tracking

Experimental closed-loop free-boundary tracking that keeps the full
`FusionKernel` in the loop and re-identifies the local coil-response map from
repeated solves. Safe-current fallback targets can be supplied through the
`free_boundary_tracking.fallback_currents` config block when supervisor
rejection should ramp the coils toward a predefined safe state. Persistent
objective residuals can also be accumulated with the config-driven
`free_boundary_tracking.observer_gain` and `observer_max_abs` settings. When
free-boundary objective tolerances are configured, the controller also uses
them directly in its correction and accept/reject logic so tighter X-point or
divertor targets take precedence over looser shape goals, and it refuses trial
steps that would push an already-satisfied objective back outside tolerance. If
the identified coil-response map loses authority entirely, the controller marks
that degraded state explicitly and drops into the safe-state recovery path
instead of silently accepting a zero-action step. Residuals already inside the
configured tolerances are also treated as deadband, so the controller stops
chattering the coils once the protected objectives are met. Coil allocation is
also headroom-aware, so the regularized solve prefers actuators that still have
current authority instead of leaning equally on a nearly saturated coil.
Deterministic objective-space sensor bias and per-step drift can be injected
through `free_boundary_tracking.measurement_bias` and
`measurement_drift_per_step`, and known calibration corrections can be applied
with `measurement_correction_bias` and `measurement_correction_drift_per_step`.
The run summary reports both measured and hidden true objective errors so
calibration faults cannot masquerade as control success in acceptance tests.

```python
from scpn_control.control.free_boundary_tracking import run_free_boundary_tracking

summary = run_free_boundary_tracking(
    "iter_config.json",
    shot_steps=5,
    gain=0.8,
    verbose=False,
    coil_slew_limits=2.5e5,
    supervisor_limits={"x_point_position": 0.15, "max_abs_actuator_lag": 1.0e5},
    hold_steps_after_reject=2,
)

print(summary["shape_rms"], summary["objective_converged"], summary["supervisor_intervention_count"])
```

::: scpn_control.control.free_boundary_tracking.FreeBoundaryTrackingController

::: scpn_control.control.free_boundary_tracking.run_free_boundary_tracking

### Disruption Predictor

`predict_disruption_risk_safe()` still returns a bounded scalar risk, but its
metadata now includes deterministic sigma-point uncertainty summaries
(`risk_p05`, `risk_p50`, `risk_p95`, `risk_std`, `risk_interval`) for both
fallback and checkpoint inference paths.

::: scpn_control.control.disruption_predictor.DisruptionTransformer

::: scpn_control.control.disruption_predictor.predict_disruption_risk

::: scpn_control.control.disruption_predictor.predict_disruption_risk_safe

### Disruption Contracts

::: scpn_control.control.disruption_contracts.run_disruption_episode

::: scpn_control.control.disruption_contracts.predict_disruption_risk

### SPI Mitigation

::: scpn_control.control.spi_mitigation.ShatteredPelletInjection

::: scpn_control.control.spi_mitigation.run_spi_mitigation

### Fusion Control Room

::: scpn_control.control.fusion_control_room.run_control_room

::: scpn_control.control.fusion_control_room.TokamakPhysicsEngine

### Gymnasium Environment

::: scpn_control.control.gym_tokamak_env.TokamakEnv

### Analytic Solver

::: scpn_control.control.analytic_solver.AnalyticEquilibriumSolver

### Bio-Holonomic Controller

::: scpn_control.control.bio_holonomic_controller.BioHolonomicController

### Digital Twin Ingest

::: scpn_control.control.digital_twin_ingest.RealtimeTwinHook

### Director Interface

::: scpn_control.control.director_interface.DirectorInterface

### Fueling Mode Controller

::: scpn_control.control.fueling_mode.IcePelletFuelingController

### Halo RE Physics

::: scpn_control.control.halo_re_physics.HaloCurrentModel

::: scpn_control.control.halo_re_physics.DisruptionMitigationClaimEvidence

::: scpn_control.control.halo_re_physics.disruption_mitigation_claim_evidence

::: scpn_control.control.halo_re_physics.assert_disruption_mitigation_claim_admissible

::: scpn_control.control.halo_re_physics.save_disruption_mitigation_claim_evidence

### HIL Test Harness

::: scpn_control.control.hil_harness.HILControlLoop

### JAX Traceable Runtime

Requires `pip install "scpn-control[jax]"`.

::: scpn_control.control.jax_traceable_runtime.TraceableRuntimeSpec

### LIF+NEF SNN Controller

::: scpn_control.control.nengo_snn_wrapper.NengoSNNController

### Neuro-Cybernetic Controller

::: scpn_control.control.neuro_cybernetic_controller.NeuroCyberneticController

### TORAX Hybrid Loop

::: scpn_control.control.torax_hybrid_loop.run_nstxu_torax_hybrid_campaign

### Advanced SOC Learning

::: scpn_control.control.advanced_soc_fusion_learning.run_advanced_learning_sim

### NMPC Controller (v0.16.0)

`NonlinearMPC` validates the NMPC quadratic program contract before
optimization: `Q`, `R`, and optional terminal `P` must be finite symmetric
positive-definite matrices with tokamak state/input dimensions; state, input,
and slew bounds must be finite and ordered; and plant-model evaluations must
return finite state vectors. Invalid math contracts fail closed instead of
propagating undefined SQP or PGD iterates.
The public `compute_cost()` evaluator includes the finite-horizon terminal
penalty, using configured `P` when supplied and the controller's conservative
fallback terminal weight otherwise.
Production plant models may provide an analytic `linearization_model(x, u)`
contract returning finite `(6, 6)` state and `(6, 3)` input Jacobians. The
controller validates those matrices before use and records
`last_linearization_source == "analytic"`. If no analytic provider is supplied,
the controller falls back to bounded central finite differences and records
`last_linearization_source == "finite_difference"`.
DARE-derived terminal matrices are accepted only when finite, symmetric, and
positive definite; invalid solver output falls back to the conservative terminal
weight.
Explicit terminal state sets are configured with paired `terminal_x_min` and
`terminal_x_max` vectors. These bounds must lie inside the configured physics
state envelope and currently require `qp_backend="scipy"`, `qp_backend="osqp"`,
`qp_backend="casadi"`, or `qp_backend="acados"` so the coupled terminal-state
inequality is enforced inside the constrained QP solve rather than checked
after the fact.  `casadi` is a repository-local optional dependency path.
The `acados` backend is a full optional OCP interface: deployments may inject a
pre-built acados OCP/solver factory, or provide `symbolic_dynamics_model(ca, x,
u)` so the controller builds a discrete augmented-state acados model. The
augmented state stores the previous actuator vector, making `|Δu| <= du_max`
a native acados path constraint instead of a post-solve clamp. The default
builder configures SQP, partial-condensing HPIPM, exact Hessian mode, linear
least-squares stage/terminal costs, state/input bounds, terminal state bounds,
warm starts, fail-closed solver-status handling, and a runtime plant-consistency
gate. The returned acados state trajectory must start from the commanded state,
remain inside configured state bounds, satisfy any terminal admissible set, and
match `plant_model` transitions within `acados_dynamics_residual_tol` before the
first actuator command is admitted.
The previous input supplied to `step()` must already satisfy actuator bounds so
the slew-rate projection cannot propagate an unsafe actuator state.
The accepted `horizon=1` case is handled as a valid one-step receding-horizon
controller and warm-starts from the bounded previous input.
Each QP solve records `last_qp_iterations` and `last_qp_converged`, making
projection-tolerance convergence distinguishable from iteration-budget
exhaustion.
The projected-gradient QP iteration budget is configured by `qp_max_iter`
instead of being an unobservable hard-coded loop bound.
Linearization perturbations are clipped to the configured state/input domain:
interior points use central differences, while boundary points use one-sided
finite differences.
`tune_transport_coefficients_for_tracking()` connects NMPC controller tuning to
the differentiable transport facade. It updates four-channel transport
coefficients from the JAX gradient of the transport tracking loss, applies
non-negative coefficient bounds and fractional update caps, and fails closed
when JAX gradients are unavailable. By default, coefficient tuning also runs the
differentiable-transport finite-difference gradient audit before admission and
stores the audit result beside the validated transport campaign metadata for
backend, dtype, radial grid, boundary conditions, closure provenance, and
gradient tolerance.
`tune_neural_transport_closure_for_tracking()` initialises the same tuning path
from a bounded neural transport closure, preserving the differentiable facade's
four-channel coefficient contract, the explicit JAX-gradient requirement, and
the default gradient-audit admission gate.
`tune_transport_sources_for_tracking()` applies the audited JAX gradient path to
additive heating, fuelling, and impurity-source schedules. Source lower and
upper bounds are explicit because replay studies may include physically valid
sink terms, and every accepted update carries campaign metadata plus the
gradient-audit result.
`tune_transport_source_rollout_for_tracking()` extends that admission boundary
from a single transport step to a complete `(n_steps, 4, n_rho)` source
schedule. It uses JAX for the multi-step rollout gradient, requires a sampled
NumPy finite-difference audit by default, clips per-entry source updates when
configured, and records bounded campaign metadata before the schedule can enter
NMPC tuning.

::: scpn_control.control.nmpc_controller.NonlinearMPC

::: scpn_control.control.nmpc_controller.TransportCoefficientTuningResult

::: scpn_control.control.nmpc_controller.TransportSourceScheduleTuningResult

::: scpn_control.control.nmpc_controller.TransportSourceRolloutGradientAudit

::: scpn_control.control.nmpc_controller.TransportSourceRolloutTuningResult

::: scpn_control.control.nmpc_controller.tune_transport_coefficients_for_tracking

::: scpn_control.control.nmpc_controller.tune_transport_sources_for_tracking

::: scpn_control.control.nmpc_controller.tune_transport_source_rollout_for_tracking

::: scpn_control.control.nmpc_controller.tune_neural_transport_closure_for_tracking

### Mu-Synthesis (v0.16.0)

::: scpn_control.control.mu_synthesis.MuSynthesisController

::: scpn_control.control.mu_synthesis.compute_mu_upper_bound

### Real-Time EFIT (v0.16.0)

::: scpn_control.control.realtime_efit.RealtimeEFIT

::: scpn_control.control.realtime_efit.EFITLiteClaimEvidence

::: scpn_control.control.realtime_efit.efit_lite_claim_evidence

::: scpn_control.control.realtime_efit.assert_efit_lite_facility_claim_admissible

::: scpn_control.control.realtime_efit.save_efit_lite_claim_evidence

### Gain-Scheduled Controller (v0.16.0)

::: scpn_control.control.gain_scheduled_controller.GainScheduledController

### Shape Controller (v0.16.0)

::: scpn_control.control.shape_controller.PlasmaShapeController

### Safe RL Controller (v0.16.0)

::: scpn_control.control.safe_rl_controller.LagrangianPPO

### Sliding-Mode Vertical (v0.16.0)

::: scpn_control.control.sliding_mode_vertical.VerticalStabilizer

### Scenario Scheduler (v0.16.0)

::: scpn_control.control.scenario_scheduler.ScenarioOptimizer

### Fault-Tolerant Control (v0.16.0)

::: scpn_control.control.fault_tolerant_control.ReconfigurableController

### RZIp Model (v0.16.0)

::: scpn_control.control.rzip_model.RZIPModel

::: scpn_control.control.rzip_model.RZIPCalibrationEvidence

::: scpn_control.control.rzip_model.rzip_calibration_evidence

::: scpn_control.control.rzip_model.assert_rzip_facility_claim_admissible

::: scpn_control.control.rzip_model.save_rzip_calibration_evidence

### RWM Feedback (v0.16.0)

::: scpn_control.control.rwm_feedback.RWMFeedbackController

::: scpn_control.control.rwm_feedback.RWMClaimEvidence

::: scpn_control.control.rwm_feedback.rwm_claim_evidence

::: scpn_control.control.rwm_feedback.assert_rwm_facility_claim_admissible

::: scpn_control.control.rwm_feedback.save_rwm_claim_evidence

---

## Complete Module Index

This index keeps the published API reference aligned with every tracked Python module under `src/scpn_control/`. Domain pages above highlight primary entry points; this section exposes the remaining module surfaces through mkdocstrings so public signatures and docstrings stay visible as the codebase grows.

### Top-Level CLI

#### Cli

::: scpn_control.cli

### Control Modules

#### Burn Controller

::: scpn_control.control.burn_controller

#### Codac Interface

::: scpn_control.control.codac_interface

#### Controller Tuning

::: scpn_control.control.controller_tuning

#### Density Controller

::: scpn_control.control.density_controller

::: scpn_control.control.density_controller.DensityControlClaimEvidence

::: scpn_control.control.density_controller.density_control_claim_evidence

::: scpn_control.control.density_controller.assert_density_control_facility_claim_admissible

::: scpn_control.control.density_controller.save_density_control_claim_evidence

#### Detachment Controller

::: scpn_control.control.detachment_controller

#### Federated Disruption

::: scpn_control.control.federated_disruption

#### State Estimator

::: scpn_control.control.state_estimator

#### Volt Second Manager

::: scpn_control.control.volt_second_manager

### Core Support and Physics Modules

#### Rust Compatibility

::: scpn_control.core._rust_compat

#### Validators

::: scpn_control.core._validators

#### Alfven Eigenmodes

::: scpn_control.core.alfven_eigenmodes

#### Blob Transport

::: scpn_control.core.blob_transport

#### Checkpoint

::: scpn_control.core.checkpoint

#### Disruption Sequence

::: scpn_control.core.disruption_sequence

#### Elm Model

::: scpn_control.core.elm_model

#### Eped Pedestal

::: scpn_control.core.eped_pedestal

#### GK CGYRO

::: scpn_control.core.gk_cgyro

#### GK GENE

::: scpn_control.core.gk_gene

#### GK Geometry

::: scpn_control.core.gk_geometry

#### GK GS2

::: scpn_control.core.gk_gs2

#### GK Nonlinear

::: scpn_control.core.gk_nonlinear

#### GK Online Learner

`OnlineLearner` admits finite nonnegative transport targets only when the
caller-supplied OOD score is inside the configured threshold. Retraining uses a
validation holdout, rolls back on non-improvement, and can persist an auditable
JSON report containing every accepted or rejected update decision.

::: scpn_control.core.gk_online_learner

#### GK QuaLiKiz

::: scpn_control.core.gk_qualikiz

#### GK Species

::: scpn_control.core.gk_species

#### GK TGLF

::: scpn_control.core.gk_tglf

#### GK TGLF Native

::: scpn_control.core.gk_tglf_native

#### GK Verification Report

::: scpn_control.core.gk_verification_report

#### Impurity Transport

::: scpn_control.core.impurity_transport

#### JAX GK Nonlinear

::: scpn_control.core.jax_gk_nonlinear

#### JAX GK Solver

The linear JAX GK solver includes a schema-versioned parity artifact producer
for backend reproducibility. `build_jax_gk_parity_artifact()` and
`write_jax_gk_parity_artifact()` bind the native local-dispersion comparison,
backend metadata, dtype/X64 state, solver kwargs, tolerances, and canonical
payload SHA-256 digest while preserving the backend-parity-only claim boundary.

::: scpn_control.core.jax_gk_solver

::: scpn_control.core.jax_gk_solver.gk_stiffness_chi_i_profile_jax

::: scpn_control.core.jax_gk_solver.build_jax_gk_parity_artifact

::: scpn_control.core.jax_gk_solver.write_jax_gk_parity_artifact

#### JAX GS Solver

::: scpn_control.core.jax_gs_solver

#### Kinetic EFIT

::: scpn_control.core.kinetic_efit

::: scpn_control.core.kinetic_efit.KineticEFITClaimEvidence

::: scpn_control.core.kinetic_efit.kinetic_efit_claim_evidence

::: scpn_control.core.kinetic_efit.assert_kinetic_efit_facility_claim_admissible

::: scpn_control.core.kinetic_efit.save_kinetic_efit_claim_evidence

#### L-H Transition

::: scpn_control.core.lh_transition

#### Locked Mode

::: scpn_control.core.locked_mode

#### MARFE

::: scpn_control.core.marfe

#### MDSplus Acquisition

::: scpn_control.core.mdsplus_acquisition

#### Momentum Transport

::: scpn_control.core.momentum_transport

#### Neoclassical

::: scpn_control.core.neoclassical

#### Neural Turbulence

::: scpn_control.core.neural_turbulence

::: scpn_control.core.neural_turbulence.NeuralTurbulenceClaimEvidence

::: scpn_control.core.neural_turbulence.cross_validate_neural_turbulence

::: scpn_control.core.neural_turbulence.neural_turbulence_claim_evidence

::: scpn_control.core.neural_turbulence.assert_neural_turbulence_quantitative_claim_admissible

::: scpn_control.core.neural_turbulence.save_neural_turbulence_claim_evidence

#### Orbit Following

::: scpn_control.core.orbit_following

::: scpn_control.core.orbit_following.OrbitFollowingClaimEvidence

::: scpn_control.core.orbit_following.orbit_following_claim_evidence

::: scpn_control.core.orbit_following.assert_orbit_following_external_claim_admissible

::: scpn_control.core.orbit_following.save_orbit_following_claim_evidence

#### Pedestal

::: scpn_control.core.pedestal

#### Pellet Injection

::: scpn_control.core.pellet_injection

#### Plasma Startup

::: scpn_control.core.plasma_startup

#### Plasma Wall Interaction

::: scpn_control.core.plasma_wall_interaction

#### Real Data Manifest

::: scpn_control.core.real_data_manifest

#### Runaway Electrons

::: scpn_control.core.runaway_electrons

#### Stellarator Geometry

::: scpn_control.core.stellarator_geometry

#### Tearing Mode Coupling

::: scpn_control.core.tearing_mode_coupling

#### Vessel Model

::: scpn_control.core.vessel_model

#### VMEC Lite

::: scpn_control.core.vmec_lite

::: scpn_control.core.vmec_lite.VMECLiteClaimEvidence

::: scpn_control.core.vmec_lite.vmec_lite_claim_evidence

::: scpn_control.core.vmec_lite.assert_vmec_lite_full_vmec_claim_admissible

::: scpn_control.core.vmec_lite.save_vmec_lite_claim_evidence

### Phase Modules

#### GK UPDE Bridge

::: scpn_control.phase.gk_upde_bridge

### SCPN Compiler and Replay Modules

#### FPGA Export

::: scpn_control.scpn.fpga_export

#### Geometry Neutral Contracts

::: scpn_control.scpn.geometry_neutral_contracts

#### Geometry Neutral Replay

::: scpn_control.scpn.geometry_neutral_replay

---

## CLI

```bash
scpn-control demo --scenario combined --steps 1000
scpn-control benchmark --n-bench 5000 --json-out
scpn-control validate --json-out
scpn-control info --json-out
scpn-control live --port 8765 --zeta 0.5 --layers 16
scpn-control hil-test --shots-dir path/to/shots
```

| Command | Description |
|---------|------------|
| `demo` | Closed-loop control demonstration (PID, SNN, combined) |
| `benchmark` | PID vs SNN timing benchmark with JSON output option |
| `validate` | Import hygiene plus repository data-manifest provenance gate |
| `info` | Version, Rust backend status, weight provenance, Python/NumPy versions |
| `live` | Real-time WebSocket phase sync server |
| `hil-test` | Hardware-in-the-loop test campaign against shot data |

---

## Rust Acceleration

When `scpn-control-rs` is built via maturin, all core solvers use Rust backends automatically:

```python
from scpn_control import RUST_BACKEND
print(RUST_BACKEND)  # True if Rust available

# Transparent acceleration — same Python API, Rust execution
kernel = FusionKernel(R0=6.2, a=2.0, B0=5.3)
```

Build Rust bindings:

```bash
cd scpn-control-rs/crates/control-python
maturin develop --release
```

### PyO3 Bindings

| Python Class | Rust Binding | Crate |
|-------------|-------------|-------|
| `FusionKernel` | `PyFusionKernel` | control-core |
| `RealtimeMonitor` | `PyRealtimeMonitor` | control-math |
| `SnnPool` | `PySnnPool` | control-control |
| `MpcController` | `PyMpcController` | control-control |
| `Plasma2D` | `PyPlasma2D` | control-core |
| `TransportSolver` | `PyTransportSolver` | control-core |
