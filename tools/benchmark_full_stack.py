import time
import numpy as np
import scpn_control_rs as rs
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.contracts import ControlTargets, ControlScales
from scpn_control.scpn.controller import NeuroSymbolicController
from scpn_control.scpn.structure import StochasticPetriNet


def benchmark_full_stack():
    print("SCPN-CONTROL: Full-Stack Physics Benchmark")
    print("-" * 50)

    # 1. Rust Math Kernel (Equilibrium)
    print("1. Grad-Shafranov Equilibrium (Rust Multigrid)")
    try:
        kernel = rs.PyFusionKernel("iter_config.json")
        t0 = time.perf_counter()
        res = kernel.solve_equilibrium()
        t1 = time.perf_counter()
        print(f"   Solve time:   {(t1 - t0) * 1000:.2f} ms")
        print(f"   Iterations:   {res.iterations}")
        print(f"   Residual:     {res.residual:.2e}")
    except Exception as e:
        print(f"   Error: {e}")

    # 2. Neuro-Symbolic Controller (SNN)
    print("\n2. Neuro-Symbolic Controller (SNN Step)")
    # Build a standard net
    net = StochasticPetriNet()
    net.add_place("x_R_pos")
    net.add_place("x_R_neg")
    net.add_transition("t_PF", threshold=0.2)
    net.add_arc("x_R_pos", "t_PF", weight=1.0)

    compiler = FusionCompiler()
    compiled_net = compiler.compile(net)

    # Minimal readout config
    readout = {
        "action_names": ["ctrl"],
        "pos_places": [net._place_idx["x_R_pos"]],
        "neg_places": [net._place_idx["x_R_neg"]],
        "gains": [1.0],
        "abs_max": [10.0],
        "slew_per_s": [100.0],
    }
    # Dummy injection mapping R_axis_m -> x_R_pos
    injections = [{"place_id": net._place_idx["x_R_pos"], "source": "R_axis_m", "scale": 1.0, "offset": 0.0}]

    artifact = compiled_net.export_artifact(
        name="benchmark_artifact", readout_config=readout, injection_config=injections
    )

    targets = ControlTargets(R_target_m=6.2, Z_target_m=0.0)
    scales = ControlScales(R_scale_m=1.0, Z_scale_m=1.0)

    controller = NeuroSymbolicController(artifact, seed_base=42, targets=targets, scales=scales, runtime_backend="rust")

    obs = {"R_axis_m": 6.1, "Z_axis_m": 0.05}

    # Warmup
    for k in range(10):
        _ = controller.step(obs, k)

    n_steps = 1000
    t0 = time.perf_counter()
    for k in range(n_steps):
        _ = controller.step(obs, k)
    t1 = time.perf_counter()

    avg_us = (t1 - t0) / n_steps * 1e6
    print(f"   Backend:      {controller.runtime_backend_name}")
    print(f"   Control step: {avg_us:.2f} us/tick")
    print(f"   Throughput:   {1e6 / avg_us:.1f} Hz")

    # 3. Raw Kuramoto Sync (Rust)
    print("\n3. Raw Kuramoto Sync (Rust)")
    L, N = 16, 50
    theta = np.random.uniform(0, 2 * np.pi, L * N)
    omega = np.random.uniform(10, 100, L * N)

    t0 = time.perf_counter()
    # rs.kuramoto_step expects scalar k, alpha, zeta
    _ = rs.kuramoto_step(theta, omega, 0.001, 1.0, 0.0, 0.5, 0.0)
    t1 = time.perf_counter()
    print(f"   Single step (L={L}, N={N}): {(t1 - t0) * 1e6:.2f} us")


if __name__ == "__main__":
    benchmark_full_stack()
