# API Reference

## Top-Level Exports

```python
import scpn_control

scpn_control.__version__       # "0.3.3"
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

### IMAS Adapter

::: scpn_control.core.imas_adapter.EquilibriumIDS

::: scpn_control.core.imas_adapter.from_geqdsk

::: scpn_control.core.imas_adapter.from_kernel

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

### Disruption Predictor

::: scpn_control.control.disruption_predictor.DisruptionTransformer

::: scpn_control.control.disruption_predictor.predict_disruption_risk

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

### Nengo SNN Controller

Requires `pip install "scpn-control[nengo]"`.

```python
from scpn_control.control import get_nengo_controller

NengoSNNController = get_nengo_controller()
ctrl = NengoSNNController()
u = ctrl.step(np.array([0.05, -0.02]))
```

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
