# API Reference

## Top-Level Exports

```python
import scpn_control

scpn_control.__version__       # "0.15.0"
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

::: scpn_control.core.neural_transport.NeuralTransportModel

::: scpn_control.core.neural_transport.cross_validate_neural_transport

### MHD Stability

::: scpn_control.core.stability_mhd.run_full_stability_check

### IMAS Adapter

::: scpn_control.core.imas_adapter.EquilibriumIDS

::: scpn_control.core.imas_adapter.from_geqdsk

::: scpn_control.core.imas_adapter.from_kernel

### HPC Bridge

::: scpn_control.core.hpc_bridge.HPCBridge

---

## SCPN — Petri Net Compiler

### StochasticPetriNet

::: scpn_control.scpn.structure.StochasticPetriNet

### FusionCompiler

::: scpn_control.scpn.compiler.FusionCompiler

### CompiledNet

::: scpn_control.scpn.compiler.CompiledNet

### NeuroSymbolicController

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

### Nengo SNN Controller

Requires `pip install "scpn-control[nengo]"`.

::: scpn_control.control.nengo_snn_wrapper.NengoSNNController

### Neuro-Cybernetic Controller

::: scpn_control.control.neuro_cybernetic_controller.NeuroCyberneticController

### TORAX Hybrid Loop

::: scpn_control.control.torax_hybrid_loop.run_nstxu_torax_hybrid_campaign

### Advanced SOC Learning

::: scpn_control.control.advanced_soc_fusion_learning.run_advanced_learning_sim

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
| `validate` | RMSE validation against reference shots |
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
