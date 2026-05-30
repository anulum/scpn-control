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
```

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
`audit_transport_parameter_gradients()` and
`assert_transport_parameter_gradients_consistent()` compare those JAX gradients
against sampled independent finite-difference perturbations before
controller-tuning admission.
`transport_coefficients_from_neural_closure()` maps bounded neural transport
closure outputs into the four-channel coefficient order used by the facade:
electron heat, ion heat, electron particle diffusivity, and a declared impurity
diffusivity fraction.
`transport_campaign_metadata()` records backend, dtype, radial grid, timestep,
boundary conditions, closure provenance, gradient tolerance, and optional
equilibrium-grid shape for reproducible controller-tuning campaigns.
`save_transport_campaign_metadata()` and `load_transport_campaign_metadata()`
persist the same contract as schema-versioned JSON and fail closed on malformed
or physically inconsistent replay metadata.
`assert_transport_campaign_metadata_replay()` compares archived campaign
metadata with a candidate setup and raises on backend, grid, boundary, closure,
gradient-tolerance, or equilibrium-shape drift before controller tuning reruns.

::: scpn_control.core.differentiable_transport.differentiable_transport_step

::: scpn_control.core.differentiable_transport.transport_tracking_loss

::: scpn_control.core.differentiable_transport.transport_loss_gradient

::: scpn_control.core.differentiable_transport.transport_parameter_gradients

::: scpn_control.core.differentiable_transport.TransportParameterGradients

::: scpn_control.core.differentiable_transport.audit_transport_parameter_gradients

::: scpn_control.core.differentiable_transport.assert_transport_parameter_gradients_consistent

::: scpn_control.core.differentiable_transport.TransportGradientAudit

::: scpn_control.core.differentiable_transport.transport_coefficients_from_neural_closure

::: scpn_control.core.differentiable_transport.TransportCampaignMetadata

::: scpn_control.core.differentiable_transport.transport_campaign_metadata

::: scpn_control.core.differentiable_transport.save_transport_campaign_metadata

::: scpn_control.core.differentiable_transport.load_transport_campaign_metadata

::: scpn_control.core.differentiable_transport.assert_transport_campaign_metadata_replay

::: scpn_control.core.differentiable_transport.equilibrium_radial_weights

::: scpn_control.core.differentiable_transport.equilibrium_weighted_transport_tracking_loss

::: scpn_control.core.differentiable_transport.equilibrium_weighted_transport_loss_gradient

::: scpn_control.core.differentiable_transport.EquilibriumWeightedTransportGradient

### Neural Equilibrium

::: scpn_control.core.neural_equilibrium.NeuralEquilibriumAccelerator

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

::: scpn_control.core.neural_transport.neural_transport_closure_profiles

::: scpn_control.core.neural_transport.cross_validate_neural_transport

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

::: scpn_control.core.integrated_scenario.iter_baseline_scenario

---

## SCPN — Petri Net Compiler

### StochasticPetriNet

::: scpn_control.scpn.structure.StochasticPetriNet

### Formal Verification

::: scpn_control.scpn.formal_verification.FormalPetriNetVerifier

::: scpn_control.scpn.formal_verification.verify_formal_contracts

::: scpn_control.scpn.formal_verification.AlwaysBounded

::: scpn_control.scpn.formal_verification.EventuallyFires

::: scpn_control.scpn.formal_verification.NeverCoMarked

### FusionCompiler

::: scpn_control.scpn.compiler.FusionCompiler

### CompiledNet

::: scpn_control.scpn.compiler.CompiledNet

### NeuroSymbolicController

`NeuroSymbolicController` rejects nonzero `sc_bitflip_rate` unless
`allow_fault_injection=True` is supplied explicitly. Bit-flip mutation is a
fault-injection test mode, not a production control default.

Controller JSONL logging requires an explicit `log_root` whenever `log_path` is
provided. Relative and absolute log paths must resolve under that root and use a
`.jsonl` suffix before any file is opened.

::: scpn_control.scpn.controller.NeuroSymbolicController

### Contracts

::: scpn_control.scpn.contracts.ControlObservation

::: scpn_control.scpn.contracts.ControlAction

::: scpn_control.scpn.contracts.ControlTargets

::: scpn_control.scpn.contracts.extract_features

::: scpn_control.scpn.contracts.decode_actions

### Artifacts

::: scpn_control.scpn.artifact.Artifact

::: scpn_control.scpn.artifact.save_artifact

::: scpn_control.scpn.artifact.load_artifact

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
received.

::: scpn_control.control.tokamak_digital_twin.run_digital_twin

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
state envelope and currently require `qp_backend="scipy"` or
`qp_backend="osqp"` so the coupled
terminal-state inequality is enforced inside the constrained QP solve rather
than checked after the fact.
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

::: scpn_control.control.nmpc_controller.NonlinearMPC

::: scpn_control.control.nmpc_controller.TransportCoefficientTuningResult

::: scpn_control.control.nmpc_controller.TransportSourceScheduleTuningResult

::: scpn_control.control.nmpc_controller.tune_transport_coefficients_for_tracking

::: scpn_control.control.nmpc_controller.tune_transport_sources_for_tracking

::: scpn_control.control.nmpc_controller.tune_neural_transport_closure_for_tracking

### Mu-Synthesis (v0.16.0)

::: scpn_control.control.mu_synthesis.MuSynthesisController

::: scpn_control.control.mu_synthesis.compute_mu_upper_bound

### Real-Time EFIT (v0.16.0)

::: scpn_control.control.realtime_efit.RealtimeEFIT

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

### RWM Feedback (v0.16.0)

::: scpn_control.control.rwm_feedback.RWMFeedbackController

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

::: scpn_control.core.jax_gk_solver

::: scpn_control.core.jax_gk_solver.gk_stiffness_chi_i_profile_jax

#### JAX GS Solver

::: scpn_control.core.jax_gs_solver

#### Kinetic EFIT

::: scpn_control.core.kinetic_efit

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

#### Orbit Following

::: scpn_control.core.orbit_following

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
