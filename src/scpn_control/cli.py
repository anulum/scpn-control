# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI Entry Point

"""
CLI for scpn-control.

Usage::

    scpn-control demo --scenario combined --steps 1000
    scpn-control benchmark --n-bench 5000
    scpn-control validate
    scpn-control validate-manifest real_manifest.json --json-out
    scpn-control validate-data-manifests --json-out
    scpn-control validate-release-evidence artifacts/release_evidence_report.json --json-out
    scpn-control validate-physics-traceability --json-out
    scpn-control validate-gk-geometry-reference --json-out
    scpn-control validate-gk-species-reference --json-out
    scpn-control validate-jax-gk-parity --require-parity-artifacts --json-out
    scpn-control validate-gk-ood-calibration --require-campaign-artifacts --json-out
    scpn-control validate-gk-interface-artifacts --require-interface-artifacts --json-out
    scpn-control validate-blob-transport-reference --require-reference-artifacts --json-out
    scpn-control validate-elm-reference --require-reference-artifacts --json-out
    scpn-control validate-eped-reference --require-reference-artifacts --json-out
    scpn-control validate-marfe-reference --require-reference-artifacts --json-out
    scpn-control validate-ntm-reference --require-reference-artifacts --json-out
    scpn-control validate-neural-equilibrium-reference --require-reference-artifacts --json-out
    scpn-control validate-neural-transport-reference --require-reference-artifacts --json-out
    scpn-control validate-neural-turbulence-reference --require-reference-artifacts --json-out
    scpn-control validate-orbit-reference --require-reference-artifacts --json-out
    scpn-control validate-uncertainty-reference --require-reference-artifacts --json-out
    scpn-control validate-vmec-reference --require-reference-artifacts --json-out
    scpn-control validate-rzip-reference --require-reference-artifacts --json-out
    scpn-control validate-density-reference --require-reference-artifacts --json-out
    scpn-control validate-burn-reference --require-reference-artifacts --json-out
    scpn-control validate-volt-second-reference --require-reference-artifacts --json-out
    scpn-control validate-current-drive-reference --require-reference-artifacts --json-out
    scpn-control validate-mu-synthesis-reference --require-reference-artifacts --json-out
    scpn-control validate-free-boundary-reference --require-reference-artifacts --json-out
    scpn-control validate-disruption-reference --require-reference-artifacts --json-out
    scpn-control validate-digital-twin-reference --require-reference-artifacts --json-out
    scpn-control validate-soc-reference --require-reference-artifacts --json-out
    scpn-control acquire-mdsplus-shot --spec-json validation/reference_data/diiid/acquisition_specs/shot_163303_mdsplus.json
    scpn-control live --port 8765 --zeta 0.5
    scpn-control hil-test --shots-dir validation/reference_data/diiid/disruption_shots
    scpn-control run-hardware-campaign --neurons 64 --core-snn 1 --core-z3 2 --core-net 3 --core-hb 4
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, cast

import click
import numpy as np

from scpn_control.cli_reference_validators import REFERENCE_VALIDATOR_COMMANDS

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(
        0, str(_REPO_ROOT)
    )  # pragma: no cover - bootstrap; repo root already on path under editable install/PYTHONPATH

try:
    _VERSION = version("scpn-control")
except PackageNotFoundError:  # pragma: no cover - package is installed in every supported run context
    _VERSION = "0.0.0.dev"

if TYPE_CHECKING:
    from scpn_control.core.mdsplus_acquisition import MDSplusSignalSpec

_LOGGER = logging.getLogger("SCPN.Control.CLI")


def _coerce_float(name: str, value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise click.ClickException(f"invalid float for {name}: {value!r}") from exc


def _emit_campaign_result(summary: dict[str, object], *, json_out: bool) -> None:
    if json_out:
        click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(f"Hardware campaign completed: {summary.get('status', 'unknown')}")
        if "execution" in summary:
            execution = summary["execution"]
            if isinstance(execution, dict):
                click.echo(
                    "Execution: "
                    f"mode={execution.get('mode')} "
                    f"targets=(r={summary.get('target_r')}, z={summary.get('target_z')}) "
                    f"steps={summary.get('steps')} "
                    f"elapsed_s={summary.get('elapsed_s')}"
                )
        native_summary = summary.get("native")
        if isinstance(native_summary, dict):
            click.echo(f"Native summary keys: {sorted(native_summary.keys())}")
        runtime_admission = summary.get("runtime_admission")
        if isinstance(runtime_admission, dict):
            click.echo(
                "Runtime admission: "
                f"{runtime_admission.get('status')} "
                f"production_claim_allowed={runtime_admission.get('production_claim_allowed')}"
            )
        if summary.get("status") != "normal":
            click.echo(f"Anomaly details: status={summary.get('status')}", err=True)


@click.group()
@click.version_option(version=_VERSION, prog_name="scpn-control")
def main() -> None:
    """SCPN Control — Neuro-symbolic Stochastic Petri Net controller."""


@main.command()
@click.option("--scenario", default="combined", type=click.Choice(["pid", "snn", "combined"]))
@click.option("--steps", default=1000, type=int, help="Simulation time steps")
@click.option("--json-out", is_flag=True, help="Emit JSON instead of text")
def demo(scenario: str, steps: int, json_out: bool) -> None:
    """Run a closed-loop control demo."""
    from scpn_control.scpn.compiler import FusionCompiler
    from scpn_control.scpn.contracts import ControlScales, ControlTargets, FeatureAxisSpec
    from scpn_control.scpn.controller import NeuroSymbolicController
    from scpn_control.scpn.structure import StochasticPetriNet

    # Build a simple control Petri Net
    net = StochasticPetriNet()
    # Observation places (axis-based error mapping)
    net.add_place("x_err_pos", initial_tokens=0.0)  # idx 0
    net.add_place("x_err_neg", initial_tokens=0.0)  # idx 1
    # Action places
    net.add_place("a_ctrl_pos", initial_tokens=0.0)  # idx 2
    net.add_place("a_ctrl_neg", initial_tokens=0.0)  # idx 3

    # Logic: if error is positive, fire positive control
    net.add_transition("t_pos", threshold=0.01)
    net.add_arc("x_err_pos", "t_pos", weight=0.1)
    net.add_arc("t_pos", "a_ctrl_pos", weight=0.1)

    # Logic: if error is negative, fire negative control
    net.add_transition("t_neg", threshold=0.01)
    net.add_arc("x_err_neg", "t_neg", weight=0.1)
    net.add_arc("t_neg", "a_ctrl_neg", weight=0.1)

    compiler = FusionCompiler()
    compiled = compiler.compile(net)

    # Create the controller with mapping
    artifact = compiled.export_artifact(
        name="demo_controller",
        injection_config=[
            {"place_id": 0, "source": "x_err_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 1, "source": "x_err_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ],
        readout_config={
            "actions": [{"name": "ctrl", "pos_place": 2, "neg_place": 3}],
            "gains": [0.2],
            "abs_max": [1.0],
        },
    )

    # Map a generic 'error' feature to our places
    feat_axes = [
        FeatureAxisSpec(
            obs_key="err",
            target=1.0,  # target state
            scale=1.0,
            pos_key="x_err_pos",
            neg_key="x_err_neg",
        )
    ]

    controller = NeuroSymbolicController(
        artifact,
        seed_base=42,
        targets=ControlTargets(R_target_m=1.0, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=1.0, Z_scale_m=1.0),
        feature_axes=feat_axes,
        allow_runtime_backend_fallback=True,
        allow_legacy_runtime_backend_fallback=True,
        sc_binary_margin=0.0,
    )

    rng = np.random.default_rng(42)
    target = 1.0
    state = 0.5
    trajectory = []

    for step in range(steps):
        # The controller internal logic:
        # 1. Takes 'err' observation
        # 2. Subtracts target (1.0) -> if state is 0.5, err_val is 0.5.
        #    Wait, obs is usually the measured value.
        #    Controller computes: feature_err = target - measurement

        obs = {"err": float(state)}
        actions = controller.step(obs, step)
        action = float(cast(Mapping[str, float], actions).get("ctrl", 0.0))

        if scenario == "pid":
            # Override with PID for comparison
            action = 0.1 * (target - state) + 0.01 * rng.standard_normal()
        elif scenario == "combined":
            action += 0.05 * (target - state)

        state += action
        state = np.clip(state, 0.0, 2.0)
        error = target - state
        trajectory.append({"step": step, "state": float(state), "error": float(error)})

    final_error = abs(trajectory[-1]["error"])
    result: dict[str, object] = {
        "scenario": scenario,
        "steps": steps,
        "final_state": trajectory[-1]["state"],
        "final_error": final_error,
        "converged": final_error < 0.05,
        "net_places": net.n_places,
        "net_transitions": net.n_transitions,
        "compiled_neurons": compiled.n_neurons if hasattr(compiled, "n_neurons") else "N/A",
    }
    if scenario == "combined":
        from scpn_control.control.closed_loop_scenario import (
            closed_loop_scenario_result_to_dict,
            run_integrated_scenario_closed_loop,
        )

        closed_loop = run_integrated_scenario_closed_loop(max_steps=min(max(1, steps), 5))
        result["integrated_scenario_closed_loop"] = closed_loop_scenario_result_to_dict(closed_loop)

    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Scenario: {scenario}")
        click.echo(f"Steps: {steps}")
        click.echo(f"Final state: {float(cast(float, result['final_state'])):.4f}")
        click.echo(f"Final error: {float(cast(float, result['final_error'])):.4f}")
        click.echo(f"Converged: {result['converged']}")
        if scenario == "combined":
            closed_loop_summary = cast(dict[str, object], result["integrated_scenario_closed_loop"])
            click.echo(
                "Integrated scenario closed loop: "
                f"steps={len(cast(list[object], closed_loop_summary['steps']))} "
                f"audit_passed={cast(dict[str, object], closed_loop_summary['coupling_audit'])['passed']}"
            )


@main.command()
@click.option("--n-bench", default=5000, type=int, help="Number of benchmark iterations")
@click.option("--n-warmup", default=100, type=int, help="Warm-up iterations excluded from timing")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def benchmark(n_bench: int, n_warmup: int, json_out: bool) -> None:
    """Benchmark PID versus SNN control-step latency.

    A warm-up phase is executed but excluded from timing to isolate steady-state
    micro-benchmark behavior from cold-start effects (JIT, allocator, cache, init).
    """
    kp, ki, kd = 1.0, 0.1, 0.01
    integral, prev_error = 0.0, 0.0
    n_neurons = 50
    v = np.zeros(n_neurons)
    threshold = 1.0
    decay = 0.9

    def _run_benchmark(iterations: int, op: Callable[[], None]) -> list[float]:
        timings_ns: list[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            op()
            timings_ns.append(time.perf_counter_ns() - t0)
        return timings_ns

    def _percentile(sorted_values: list[float], frac: float) -> float:
        if (
            not sorted_values
        ):  # pragma: no cover - defensive; benchmark computes statistics.mean first, so the sample is never empty here
            return 0.0
        idx = int(len(sorted_values) * frac)
        if idx >= len(
            sorted_values
        ):  # pragma: no cover - defensive clamp; callers only pass frac=0.95 so int(n*frac) < n
            idx = len(sorted_values) - 1
        return float(sorted_values[idx]) / 1000.0

    def pid_step() -> None:
        nonlocal integral, prev_error
        error = np.random.randn()
        integral += error
        derivative = error - prev_error
        _ = kp * error + ki * integral + kd * derivative
        prev_error = error

    def snn_step() -> None:
        nonlocal v
        error = np.random.randn()
        v[:] = v * decay + error * np.random.randn(n_neurons) * 0.1
        spikes = v > threshold
        v[spikes] = 0.0
        _ = np.mean(spikes.astype(float))

    # Warm-up phase: compile/freeze hot paths before timing.
    for _ in range(n_warmup):
        pid_step()
        snn_step()

    pid_ns = _run_benchmark(n_bench, pid_step)
    snn_ns = _run_benchmark(n_bench, snn_step)

    pid_sorted = sorted(pid_ns)
    snn_sorted = sorted(snn_ns)

    pid_mean_us = statistics.mean(pid_ns) / 1000.0
    snn_mean_us = statistics.mean(snn_ns) / 1000.0
    pid_p50_us = statistics.median(pid_ns) / 1000.0
    snn_p50_us = statistics.median(snn_ns) / 1000.0
    pid_p95_us = _percentile(pid_sorted, 0.95)
    snn_p95_us = _percentile(snn_sorted, 0.95)
    pid_p99_us = _percentile(pid_sorted, 0.99)
    snn_p99_us = _percentile(snn_sorted, 0.99)

    result = {
        "n_bench": n_bench,
        "n_warmup": n_warmup,
        "pid_us_per_step": round(pid_mean_us, 2),
        "pid_p50_us": round(pid_p50_us, 2),
        "pid_p95_us": round(pid_p95_us, 2),
        "pid_p99_us": round(pid_p99_us, 2),
        "snn_us_per_step": round(snn_mean_us, 2),
        "snn_p50_us": round(snn_p50_us, 2),
        "snn_p95_us": round(snn_p95_us, 2),
        "snn_p99_us": round(snn_p99_us, 2),
        "speedup_ratio": round(pid_mean_us / max(snn_mean_us, 1e-9), 2),
    }

    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Benchmarked steps: {n_bench} (warmup: {n_warmup})")
        click.echo(f"PID: {result['pid_us_per_step']} us/step")
        click.echo(f"PID p50/p95/p99: {result['pid_p50_us']} / {result['pid_p95_us']} / {result['pid_p99_us']} us")
        click.echo(f"SNN: {result['snn_us_per_step']} us/step")
        click.echo(f"SNN p50/p95/p99: {result['snn_p50_us']} / {result['snn_p95_us']} / {result['snn_p99_us']} us")
        click.echo(f"Ratio: {result['speedup_ratio']}x")


@main.command(name="run-hardware-campaign")
@click.option("--neurons", default=64, type=int, help="Neuro core neuron count used by native controller")
@click.option("--seed", default=7, type=int, help="Native controller seed")
@click.option("--state-init-r", default=None, type=float, help="Initial major radius [m]")
@click.option("--state-init-z", default=None, type=float, help="Initial vertical position [m]")
@click.option("--plant-gain", default=0.0, type=float, help="Plant gain used by hybrid fallback")
@click.option("--target-r", default=6.2, type=float, help="Primary ACADOS/NSC target radius")
@click.option("--target-z", default=0.0, type=float, help="Primary ACADOS/NSC target vertical position")
@click.option("--beta-n-limit", default=2.5, type=float, help="Normalized beta limit")
@click.option("--itpa-cgb", default=None, type=str, help="Optional override for c_gB")
@click.option("--itpa-path", default=None, type=click.Path(exists=True, dir_okay=False), help="ITPA gyro-Bohm JSON")
@click.option(
    "--kuramoto-weights",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="JSON file with Kuramoto weights",
)
@click.option("--transport-endpoint", default="239.0.0.1", help="UDP destination address")
@click.option("--transport-port", default=5555, type=int, help="UDP destination port")
@click.option("--transport-ttl", default=1, type=int, help="UDP TTL / multicast hop count")
@click.option("--transport-max-queue", default=4, type=int, help="Max unprocessed outbound frames")
@click.option("--transport-backend", default="std", help="Transport backend (std / io-uring)")
@click.option(
    "--execution-backend",
    default="auto",
    type=click.Choice(["auto", "native", "python"], case_sensitive=False),
    help="Execution backend (auto / native / python)",
)
@click.option("--heartbeat-port", default=0, type=int, help="Heartbeat listen port used for dead-man switch")
@click.option("--heartbeat-timeout-ms", default=3, type=int, help="Heartbeat timeout threshold (ms)")
@click.option("--core-snn", default=1, type=int, help="Core pinned to SNN execution")
@click.option("--core-z3", default=2, type=int, help="Core pinned to Z3 verification")
@click.option("--core-net", default=3, type=int, help="Core pinned to UDP publisher")
@click.option("--core-hb", default=4, type=int, help="Core pinned to heartbeat monitor")
@click.option("--steps", default=None, type=int, help="Exact number of control iterations")
@click.option("--runtime-s", default=None, type=float, help="Alternative run horizon in seconds")
@click.option("--tick-interval-s", default=0.001, type=float, help="Control tick interval")
@click.option(
    "--pacing-mode",
    default="sleep",
    type=click.Choice(["sleep", "spin"], case_sensitive=False),
    help="Native pacing mode; spin busy-waits instead of yielding to the OS scheduler",
)
@click.option(
    "--runtime-admission-policy",
    default="warn",
    type=click.Choice(["off", "warn", "require"], case_sensitive=False),
    help="PREEMPT_RT/core-affinity admission policy before campaign execution",
)
@click.option("--max-publish-failures", default=None, type=int, help="Python fallback publish fail tolerance")
@click.option(
    "--formal-mode",
    default="async_drop",
    type=click.Choice(
        ["disabled", "async_drop", "sync_stride", "aot_certificate"],
        case_sensitive=False,
    ),
    help="Native formal verification mode",
)
@click.option("--formal-stride", default=30, type=int, help="Formal verification dispatch stride")
@click.option("--formal-depth", default=4, type=int, help="Bounded Z3 verification depth")
@click.option("--formal-max-marking", default=100, type=int, help="Formal marking safety bound")
@click.option("--formal-channel-capacity", default=2, type=int, help="Formal worker queue capacity")
@click.option("--require-native", is_flag=True, help="Abort when native bridge is unavailable")
@click.option("--json-out", is_flag=True, help="Emit JSON instead of text")
def run_hardware_campaign(
    neurons: int,
    seed: int,
    state_init_r: float | None,
    state_init_z: float | None,
    plant_gain: float,
    target_r: float,
    target_z: float,
    beta_n_limit: float,
    itpa_cgb: str | None,
    itpa_path: str | None,
    kuramoto_weights: str | None,
    transport_endpoint: str,
    transport_port: int,
    transport_ttl: int,
    transport_max_queue: int,
    transport_backend: str,
    execution_backend: str,
    heartbeat_port: int,
    heartbeat_timeout_ms: int,
    core_snn: int,
    core_z3: int,
    core_net: int,
    core_hb: int,
    steps: int | None,
    runtime_s: float | None,
    tick_interval_s: float,
    pacing_mode: str,
    runtime_admission_policy: str,
    max_publish_failures: int | None,
    formal_mode: str,
    formal_stride: int,
    formal_depth: int,
    formal_max_marking: int,
    formal_channel_capacity: int,
    require_native: bool,
    json_out: bool,
) -> None:
    """Launch the Rust-native execution plane from Python control layer."""
    from scpn_control.core.rust_engine import NeuroCyberneticEngine

    engine = NeuroCyberneticEngine(
        n_neurons=neurons,
        seed=seed,
        plant_gain=plant_gain,
        state_init_r=6.2 if state_init_r is None else float(state_init_r),
        state_init_z=0.0 if state_init_z is None else float(state_init_z),
    )
    if require_native and not engine.native_backend_available:
        raise click.ClickException("Native bridge not available: _NATIVE_CONTROLLER_AVAILABLE is False.")

    targets = {
        "target_r": float(target_r),
        "target_z": float(target_z),
        "beta_n_limit": float(beta_n_limit),
    }
    if itpa_cgb is not None:
        target_c_gb = _coerce_float("itpa_cgb", itpa_cgb)
        if target_c_gb is None:
            raise click.ClickException("--itpa-cgb must be numeric")
        targets["c_gB"] = target_c_gb
    engine.configure_acados_targets(targets)

    if itpa_path is None:
        default_itpa = _REPO_ROOT / "validation" / "reference_data" / "itpa" / "gyro_bohm_coefficients.json"
        if default_itpa.exists():
            _ = engine.configure_itpa_gyro_bohm(default_itpa)
    else:
        _ = engine.configure_itpa_gyro_bohm(itpa_path)

    if kuramoto_weights is not None:
        _ = engine.configure_kuramoto_weights(kuramoto_weights)

    engine.configure_transport(
        endpoint=transport_endpoint,
        port=transport_port,
        ttl=transport_ttl,
        max_queue=transport_max_queue,
        backend=transport_backend,
        heartbeat_port=heartbeat_port,
        heartbeat_timeout_ms=heartbeat_timeout_ms,
    )

    engine.configure_execution_affinity(core_snn=core_snn, core_z3=core_z3, core_net=core_net, core_hb=core_hb)
    engine.configure_native_formal_verification(
        enabled=formal_mode != "disabled",
        mode=formal_mode,
        max_marking=formal_max_marking,
        max_depth=formal_depth,
        dispatch_interval_steps=formal_stride,
        channel_capacity=formal_channel_capacity,
    )
    if max_publish_failures is not None:
        engine.set_max_publish_failures(max_publish_failures)

    try:
        summary = engine.execute_hardware_loop(
            steps=steps,
            runtime_s=runtime_s,
            tick_interval_s=tick_interval_s,
            max_publish_failures=max_publish_failures,
            core_snn=core_snn,
            core_z3=core_z3,
            core_net=core_net,
            core_hb=core_hb,
            execution_backend=execution_backend,
            pacing_mode=pacing_mode,
            runtime_admission_policy=runtime_admission_policy,
        )
        _emit_campaign_result(summary, json_out=json_out)
    except RuntimeError as exc:
        emergency = engine.execute_emergency_shutdown()
        _LOGGER.critical("Campaign aborted by runtime safety system: %s", exc)
        if not isinstance(emergency, dict):
            emergency = {}
        payload = {"status": "runtime_aborted", "error": str(exc), "emergency": emergency}
        if json_out:
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo(f"Campaign aborted: {exc}", err=True)
            click.echo(f"Emergency telemetry: {emergency}", err=True)
    except KeyboardInterrupt:
        emergency = engine.execute_emergency_shutdown()
        payload = {"status": "keyboard_interrupt", "emergency": emergency}
        if json_out:
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo("Campaign interrupted by operator", err=True)
            click.echo(f"Emergency telemetry: {emergency}", err=True)


@main.command()
@click.option("--json-out", is_flag=True, help="Emit JSON")
@click.option("--data-manifest-root", help="Root directory to scan for repository data manifests")
@click.option("--no-data-manifests", is_flag=True, help="Skip repository data-manifest validation")
@click.option("--no-verify-artifacts", is_flag=True, help="Validate manifest metadata without local checksums")
@click.option(
    "--jax-gk-parity-root", help="Directory or JSON artifact containing persisted JAX/native GK parity evidence"
)
@click.option("--no-jax-gk-parity", is_flag=True, help="Skip persisted JAX GK parity evidence validation")
@click.option("--physics-traceability-registry", help="Physics traceability registry JSON path")
@click.option("--no-physics-traceability", is_flag=True, help="Skip physics traceability validation")
@click.option("--multi-shot-campaign-python-report", help="Python/PyO3 multi-shot campaign benchmark JSON report")
@click.option("--multi-shot-campaign-rust-report", help="Rust multi-shot campaign benchmark JSON report")
@click.option(
    "--multi-shot-min-digest-count",
    default=2,
    show_default=True,
    type=int,
    help="Minimum pulsed-MPC admission digests required in each multi-shot campaign report",
)
@click.option("--no-multi-shot-campaign-evidence", is_flag=True, help="Skip multi-shot campaign evidence validation")
@click.option("--runtime-admission-report", help="Runtime-admission benchmark JSON report")
@click.option("--no-runtime-admission-evidence", is_flag=True, help="Skip runtime-admission evidence validation")
@click.option("--native-formal-certificate-report", help="Native formal AOT certificate benchmark JSON report")
@click.option(
    "--native-formal-max-aot-p99-cycle-us",
    default=10.0,
    show_default=True,
    type=float,
    help="Maximum admitted native formal AOT p99 cycle latency in microseconds",
)
@click.option("--no-native-formal-certificate", is_flag=True, help="Skip native formal certificate evidence validation")
def validate(
    json_out: bool,
    data_manifest_root: str | None,
    no_data_manifests: bool,
    no_verify_artifacts: bool,
    jax_gk_parity_root: str | None,
    no_jax_gk_parity: bool,
    physics_traceability_registry: str | None,
    no_physics_traceability: bool,
    multi_shot_campaign_python_report: str | None,
    multi_shot_campaign_rust_report: str | None,
    multi_shot_min_digest_count: int,
    no_multi_shot_campaign_evidence: bool,
    runtime_admission_report: str | None,
    no_runtime_admission_evidence: bool,
    native_formal_certificate_report: str | None,
    native_formal_max_aot_p99_cycle_us: float,
    no_native_formal_certificate: bool,
) -> None:
    """Run import hygiene, provenance, parity, traceability, and formal evidence validation."""
    try:
        from scpn_control.core.integrated_transport_solver import IntegratedTransportSolver  # noqa: F401

        has_transport = True
    except ImportError:
        has_transport = False

    result = {
        "transport_solver_available": has_transport,
        "import_clean": True,
        "status": "pass",
    }
    validation_failed = False

    for mod in ["matplotlib", "torch", "streamlit"]:
        if mod in sys.modules:
            result["import_clean"] = False
            result["status"] = "fail"
            result["contaminated_module"] = mod
            break

    if no_data_manifests:
        result["data_manifests"] = {"status": "skipped"}
    else:
        from validation.validate_data_manifests import ROOT, validate_manifest_directory

        validation_root = data_manifest_root or str(ROOT / "validation" / "reference_data")
        manifest_report = validate_manifest_directory(validation_root, verify_artifacts=not no_verify_artifacts)
        result["data_manifests"] = manifest_report
        if manifest_report["status"] != "pass":
            result["status"] = "fail"
            validation_failed = True

    if no_jax_gk_parity:
        result["jax_gk_parity"] = {"status": "skipped"}
    else:
        from validation.validate_jax_gk_parity import ROOT, validate_jax_gk_parity

        parity_root = jax_gk_parity_root or str(ROOT / "validation" / "reports" / "jax_gk_parity")
        parity_report = validate_jax_gk_parity(
            parity_root,
            require_parity_artifacts=True,
            require_cases=("cyclone_base_case", "tem_kinetic_electron", "stable_mode"),
            require_backends=("cpu", "gpu"),
        )
        result["jax_gk_parity"] = parity_report
        if parity_report["status"] != "pass":
            result["status"] = "fail"
            validation_failed = True

    if no_physics_traceability:
        result["physics_traceability"] = {"status": "skipped"}
    else:
        from validation.validate_physics_traceability import ROOT, validate_physics_traceability

        registry_path = physics_traceability_registry or str(ROOT / "validation" / "physics_traceability.json")
        traceability_report = validate_physics_traceability(registry_path)
        result["physics_traceability"] = traceability_report
        if traceability_report["status"] != "pass":
            result["status"] = "fail"
            validation_failed = True

    if no_multi_shot_campaign_evidence:
        result["multi_shot_campaign"] = {"status": "skipped"}
    else:
        from validation.validate_multi_shot_campaign_evidence import (
            DEFAULT_PYTHON_REPORT,
            DEFAULT_RUST_REPORT,
            validate_multi_shot_campaign_evidence,
        )

        multi_shot_report = validate_multi_shot_campaign_evidence(
            multi_shot_campaign_python_report or DEFAULT_PYTHON_REPORT,
            multi_shot_campaign_rust_report or DEFAULT_RUST_REPORT,
            minimum_digest_count=multi_shot_min_digest_count,
        ).as_dict()
        result["multi_shot_campaign"] = multi_shot_report
        if multi_shot_report["status"] != "pass":
            result["status"] = "fail"
            validation_failed = True

    if no_runtime_admission_evidence:
        result["runtime_admission"] = {"status": "skipped"}
    else:
        from validation.validate_runtime_admission_evidence import (
            DEFAULT_REPORT as DEFAULT_RUNTIME_ADMISSION_REPORT,
        )
        from validation.validate_runtime_admission_evidence import (
            validate_runtime_admission_evidence,
        )

        runtime_report = validate_runtime_admission_evidence(
            runtime_admission_report or DEFAULT_RUNTIME_ADMISSION_REPORT
        ).as_dict()
        result["runtime_admission"] = runtime_report
        if runtime_report["status"] != "pass":
            result["status"] = "fail"
            validation_failed = True

    if no_native_formal_certificate:
        result["native_formal_certificate"] = {"status": "skipped"}
    else:
        from validation.validate_native_formal_certificate_evidence import (
            DEFAULT_REPORT,
            validate_native_formal_certificate_evidence,
        )

        certificate_report_path = native_formal_certificate_report or str(DEFAULT_REPORT)
        certificate_report = validate_native_formal_certificate_evidence(
            certificate_report_path,
            max_aot_p99_cycle_us=native_formal_max_aot_p99_cycle_us,
        ).as_dict()
        result["native_formal_certificate"] = certificate_report
        if certificate_report["status"] != "pass":
            result["status"] = "fail"
            validation_failed = True

    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Transport solver: {'OK' if has_transport else 'MISSING'}")
        click.echo(f"Import clean: {'OK' if result['import_clean'] else 'FAIL'}")
        data_manifests = result["data_manifests"]
        if isinstance(data_manifests, dict) and data_manifests.get("status") == "skipped":
            click.echo("Data manifests: SKIPPED")
        elif isinstance(data_manifests, dict):
            click.echo(
                "Data manifests: "
                f"{data_manifests['status']} "
                f"total={data_manifests['total']} "
                f"real={data_manifests['real']} "
                f"synthetic={data_manifests['synthetic']}"
            )
            for error in data_manifests["errors"]:
                click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
        jax_gk_parity = result["jax_gk_parity"]
        if isinstance(jax_gk_parity, dict) and jax_gk_parity.get("status") == "skipped":
            click.echo("JAX GK parity: SKIPPED")
        elif isinstance(jax_gk_parity, dict):
            click.echo(f"JAX GK parity: {jax_gk_parity['status']} parity_artifacts={jax_gk_parity['parity_artifacts']}")
            for error in jax_gk_parity["errors"]:
                click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
        physics_traceability = result["physics_traceability"]
        if isinstance(physics_traceability, dict) and physics_traceability.get("status") == "skipped":
            click.echo("Physics traceability: SKIPPED")
        elif isinstance(physics_traceability, dict):
            click.echo(
                "Physics traceability: "
                f"{physics_traceability['status']} "
                f"total={physics_traceability['total']} "
                f"open_fidelity_gaps={physics_traceability['open_fidelity_gaps']} "
                f"public_claim_blocked={physics_traceability['public_claim_blocked']}"
            )
            for error in physics_traceability["errors"]:
                index = error.get("index", "-")
                click.echo(f"ERROR {error['path']}[{index}].{error['field']}: {error['error']}", err=True)
        multi_shot_campaign = result["multi_shot_campaign"]
        if isinstance(multi_shot_campaign, dict) and multi_shot_campaign.get("status") == "skipped":
            click.echo("Multi-shot campaign evidence: SKIPPED")
        elif isinstance(multi_shot_campaign, dict):
            surfaces = multi_shot_campaign.get("admitted_surfaces", ())
            surface_count = len(surfaces) if isinstance(surfaces, list) else 0
            click.echo(
                "Multi-shot campaign evidence: "
                f"{multi_shot_campaign['status']} "
                f"surfaces={surface_count} "
                f"pyo3={multi_shot_campaign.get('pyo3_status') or '-'}"
            )
            for error in multi_shot_campaign["errors"]:
                click.echo(f"ERROR multi_shot_campaign: {error}", err=True)
        runtime_admission = result["runtime_admission"]
        if isinstance(runtime_admission, dict) and runtime_admission.get("status") == "skipped":
            click.echo("Runtime admission evidence: SKIPPED")
        elif isinstance(runtime_admission, dict):
            click.echo(
                "Runtime admission evidence: "
                f"{runtime_admission['status']} "
                f"admission={runtime_admission.get('admission_status') or '-'} "
                f"class={runtime_admission.get('benchmark_evidence_class') or '-'}"
            )
            for error in runtime_admission["errors"]:
                click.echo(f"ERROR runtime_admission: {error}", err=True)
        native_formal_certificate = result["native_formal_certificate"]
        if isinstance(native_formal_certificate, dict) and native_formal_certificate.get("status") == "skipped":
            click.echo("Native formal certificate: SKIPPED")
        elif isinstance(native_formal_certificate, dict):
            digest = native_formal_certificate.get("certificate_assumption_sha256") or "-"
            admitted_cases = native_formal_certificate.get("admitted_cases", ())
            admitted_count = len(admitted_cases) if isinstance(admitted_cases, list) else 0
            click.echo(
                "Native formal certificate: "
                f"{native_formal_certificate['status']} "
                f"admitted_cases={admitted_count} "
                f"digest={digest}"
            )
            for error in native_formal_certificate["errors"]:
                click.echo(f"ERROR native_formal_certificate: {error}", err=True)
        click.echo(f"Status: {result['status']}")
    if validation_failed:
        raise click.exceptions.Exit(1)


@main.command("validate-release-evidence")
@click.argument("report", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_release_evidence_command(report: str, json_out: bool) -> None:
    """Validate top-level release evidence report admission."""
    from validation.validate_release_evidence import (
        RELEASE_EVIDENCE_SCHEMA_VERSION,
        validate_release_evidence,
    )

    result = validate_release_evidence(report)
    payload = {
        "schema_version": RELEASE_EVIDENCE_SCHEMA_VERSION,
        "status": result.status,
        "errors": list(result.errors),
        "report_sha256": result.report_sha256,
        "admitted_gates": list(result.admitted_gates),
    }
    if json_out:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        click.echo(f"Release evidence: {result.status}")
        if result.report_sha256 is not None:
            click.echo(f"Report SHA-256: {result.report_sha256}")
        for error in result.errors:
            click.echo(f"ERROR {error}", err=True)
    if result.status != "pass":
        raise click.exceptions.Exit(1)


@main.command("validate-manifest")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--json-out", is_flag=True, help="Emit JSON")
@click.option("--verify-artifact", is_flag=True, help="Verify local artefact checksum")
def validate_manifest(manifest: str, json_out: bool, verify_artifact: bool) -> None:
    """Validate real-shot or synthetic-shot data manifest provenance."""
    from scpn_control.core.real_data_manifest import RealDataManifestError, load_real_data_manifest

    try:
        validated = load_real_data_manifest(manifest, verify_artifact=verify_artifact)
    except RealDataManifestError as exc:
        result: dict[str, object] = {
            "status": "fail",
            "error": str(exc),
        }
        if json_out:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Status: {result['status']}")
            click.echo(f"Error: {result['error']}")
        raise click.exceptions.Exit(1) from exc

    result = {
        "dataset_id": validated.dataset_id,
        "kind": validated.kind,
        "machine": validated.machine,
        "shot": validated.shot,
        "signals": len(validated.signals),
        "source_kind": validated.source.kind,
        "status": "pass",
    }
    if verify_artifact:
        result["artifact_verified"] = True
    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Dataset: {result['dataset_id']}")
        click.echo(f"Kind: {result['kind']}")
        click.echo(f"Machine: {result['machine']}")
        click.echo(f"Shot: {result['shot']}")
        click.echo(f"Signals: {result['signals']}")
        click.echo(f"Source: {result['source_kind']}")
        click.echo(f"Status: {result['status']}")


@main.command("validate-data-manifests")
@click.option("--root", help="Root directory to scan for repository data manifests")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--no-verify-artifacts", is_flag=True, help="Validate metadata without local checksum verification")
@click.option(
    "--require-real-acquisition",
    is_flag=True,
    help="Fail when an acquisition spec has no corresponding acquired MDSplus manifest",
)
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_data_manifests_command(
    root: str | None,
    output_json: str | None,
    no_verify_artifacts: bool,
    require_real_acquisition: bool,
    json_out: bool,
) -> None:
    """Validate repository data manifests, local artefacts, and acquisition specs."""
    from validation.validate_data_manifests import ROOT, validate_manifest_directory

    validation_root = root or str(ROOT / "validation" / "reference_data")
    report = validate_manifest_directory(
        validation_root,
        verify_artifacts=not no_verify_artifacts,
        require_real_acquisition=require_real_acquisition,
    )
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(
            "Data manifests: "
            f"{report['status']} "
            f"total={report['total']} "
            f"real={report['real']} "
            f"synthetic={report['synthetic']} "
            f"acquisition_specs={report['acquisition_specs']['total']}"
        )
        for error in report["errors"]:
            click.echo(f"ERROR {error['path']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@main.command("validate-physics-traceability")
@click.option("--registry", help="Physics traceability registry JSON path")
@click.option("--output-json", type=click.Path(dir_okay=False), help="Write JSON report to this path")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def validate_physics_traceability_command(
    registry: str | None,
    output_json: str | None,
    json_out: bool,
) -> None:
    """Validate physics fidelity traceability and bounded-claim contracts."""
    from validation.validate_physics_traceability import ROOT, validate_physics_traceability

    registry_path = registry or str(ROOT / "validation" / "physics_traceability.json")
    report = validate_physics_traceability(registry_path)
    if output_json is not None:
        from pathlib import Path as _P

        output_path = _P(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if json_out:
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(
            "Physics traceability: "
            f"{report['status']} "
            f"total={report['total']} "
            f"open_fidelity_gaps={report['open_fidelity_gaps']} "
            f"public_claim_blocked={report['public_claim_blocked']} "
            f"external_validation_trackers={report['external_validation_tracker_count']}"
        )
        for error in report["errors"]:
            index = error.get("index", "-")
            click.echo(f"ERROR {error['path']}[{index}].{error['field']}: {error['error']}", err=True)
    if report["status"] != "pass":
        raise click.exceptions.Exit(1)


@main.command("acquire-mdsplus-shot")
@click.option(
    "--spec-json",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Versioned acquisition spec JSON; replaces --tree/--shot/--signal",
)
@click.option("--tree", help="MDSplus tree name")
@click.option("--shot", type=int, help="Shot number")
@click.option("--signal", "signals_json", multiple=True, help="Signal JSON with name/node/units/timebase")
@click.option("--output-npz", required=True, type=click.Path(dir_okay=False), help="Output NPZ path")
@click.option("--manifest-json", required=True, type=click.Path(dir_okay=False), help="Output manifest JSON path")
@click.option("--source-uri", help="Source URI for provenance; defaults to mdsplus://TREE/SHOT")
@click.option("--access-policy", default="facility-approved", show_default=True, help="Access policy label")
@click.option("--licence", default="facility data policy", show_default=True, help="Licence or facility data policy")
@click.option("--retrieved-at", help="UTC retrieval timestamp for reproducible tests")
@click.option("--json-out", is_flag=True, help="Emit JSON")
def acquire_mdsplus_shot_command(
    spec_json: str | None,
    tree: str | None,
    shot: int | None,
    signals_json: tuple[str, ...],
    output_npz: str,
    manifest_json: str,
    source_uri: str | None,
    access_policy: str,
    licence: str,
    retrieved_at: str | None,
    json_out: bool,
) -> None:
    """Acquire selected MDSplus shot signals and write a validated manifest."""
    from scpn_control.core.mdsplus_acquisition import acquire_mdsplus_shot, load_mdsplus_acquisition_request

    try:
        if spec_json is not None:
            request = load_mdsplus_acquisition_request(spec_json)
            tree = request.tree
            shot = request.shot
            signal_specs = request.signals
            source_uri = source_uri or request.source_uri
            access_policy = request.access_policy
            licence = request.licence
        else:
            if tree is None:
                raise ValueError("--tree is required unless --spec-json is supplied")
            if shot is None:
                raise ValueError("--shot is required unless --spec-json is supplied")
            if not signals_json:
                raise ValueError("at least one --signal is required unless --spec-json is supplied")
            signal_specs = [_parse_mdsplus_signal_spec(raw) for raw in signals_json]

        result = acquire_mdsplus_shot(
            tree=tree,
            shot=shot,
            signals=signal_specs,
            output_npz=output_npz,
            manifest_json=manifest_json,
            source_uri=source_uri or f"mdsplus://{tree}/{shot}",
            access_policy=access_policy,
            licence=licence,
            retrieved_at=retrieved_at,
        )
    except (RuntimeError, ValueError, json.JSONDecodeError, KeyError) as exc:
        payload: dict[str, object] = {"status": "fail", "error": str(exc)}
        if json_out:
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo(f"Status: {payload['status']}")
            click.echo(f"Error: {payload['error']}")
        raise click.exceptions.Exit(1) from exc

    payload = {
        "status": "pass",
        "dataset_id": result.dataset_id,
        "output_npz": str(result.output_npz),
        "manifest_json": str(result.manifest_json),
        "checksum_sha256": result.checksum_sha256,
    }
    if json_out:
        click.echo(json.dumps(payload, indent=2))
    else:
        click.echo(f"Dataset: {payload['dataset_id']}")
        click.echo(f"Output NPZ: {payload['output_npz']}")
        click.echo(f"Manifest: {payload['manifest_json']}")
        click.echo(f"Status: {payload['status']}")


def _parse_mdsplus_signal_spec(raw: str) -> MDSplusSignalSpec:
    from scpn_control.core.mdsplus_acquisition import MDSplusSignalSpec

    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("signal specification must be a JSON object")
    return MDSplusSignalSpec(
        name=str(payload["name"]),
        node=str(payload["node"]),
        units=str(payload["units"]),
        timebase=str(payload["timebase"]),
    )


@main.command("validate-rmse")
@click.option("--json-out", is_flag=True, help="Print JSON report to stdout")
@click.option("--output-json", default="artifacts/rmse_report.json", help="JSON path")
@click.option("--output-md", default="artifacts/rmse_report.md", help="Markdown path")
def validate_rmse(json_out: bool, output_json: str, output_md: str) -> None:
    """Run full RMSE validation dashboard against reference data."""
    from validation.rmse_dashboard import main as rmse_main

    sys.argv = ["rmse_dashboard", "--output-json", output_json, "--output-md", output_md]
    exit_code = rmse_main()
    if json_out:
        from pathlib import Path as _P

        p = _P(output_json)
        if p.exists():
            click.echo(p.read_text(encoding="utf-8"))
    sys.exit(exit_code or 0)


@main.command()
@click.option("--port", default=8765, type=int, help="WebSocket port")
@click.option("--host", default="127.0.0.1", help="Bind address")
@click.option("--layers", default=16, type=int, help="Kuramoto layers (L)")
@click.option("--n-per", default=50, type=int, help="Oscillators per layer")
@click.option("--zeta", default=0.5, type=float, help="Global field coupling zeta")
@click.option("--psi", default=0.0, type=float, help="Initial Psi driver")
@click.option("--tick-interval", default=0.001, type=float, help="Seconds between ticks")
@click.option("--api-key", envvar="SCPN_PHASE_WS_API_KEY", default=None, help="API key required for remote binds")
@click.option("--command-rate-limit", default=20, type=int, help="Maximum commands per connection window")
@click.option("--command-rate-window-s", default=1.0, type=float, help="Command rate-limit window in seconds")
@click.option(
    "--max-payload-bytes",
    default=65536,
    type=int,
    help="Maximum WebSocket command payload size in bytes",
)
@click.option(
    "--max-client-write-buffer-bytes",
    default=262144,
    type=int,
    help="Maximum per-client WebSocket outbound write buffer before ejection",
)
@click.option("--allow-unauthenticated-clients", is_flag=True, help="Allow unauthenticated local development clients")
@click.option("--allow-query-token-auth", is_flag=True, help="Allow token query-string authentication")
@click.option("--require-tls", is_flag=True, help="Require TLS before starting the phase stream")
@click.option(
    "--allow-insecure-remote", is_flag=True, help="Allow plaintext non-loopback binds for isolated lab networks"
)
@click.option("--allowed-origin", multiple=True, help="Allowed browser Origin header; repeat for multiple origins")
@click.option(
    "--allowed-action",
    multiple=True,
    type=click.Choice(["reset", "set_pac_gamma", "set_psi", "stop"]),
    help="Allowed command action; repeat to restrict the control surface",
)
@click.option(
    "--tls-cert", default=None, type=click.Path(exists=True, dir_okay=False), help="TLS certificate for wss://"
)
@click.option(
    "--tls-key", default=None, type=click.Path(exists=True, dir_okay=False), help="TLS private key for wss://"
)
def live(
    port: int,
    host: str,
    layers: int,
    n_per: int,
    zeta: float,
    psi: float,
    tick_interval: float,
    api_key: str | None,
    command_rate_limit: int,
    command_rate_window_s: float,
    max_payload_bytes: int,
    max_client_write_buffer_bytes: int,
    allow_unauthenticated_clients: bool,
    allow_query_token_auth: bool,
    require_tls: bool,
    allow_insecure_remote: bool,
    allowed_origin: tuple[str, ...],
    allowed_action: tuple[str, ...],
    tls_cert: str | None,
    tls_key: str | None,
) -> None:
    """Start real-time WebSocket phase sync server.

    Streams Kuramoto R/V/lambda tick snapshots over ws://<host>:<port>.
    Connect with: examples/streamlit_ws_client.py or any WS client.
    """
    import logging
    import ssl

    from scpn_control.phase.realtime_monitor import RealtimeMonitor
    from scpn_control.phase.ws_phase_stream import PhaseStreamServer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    tls_context = None
    if tls_cert or tls_key:
        if not tls_cert or not tls_key:
            raise click.ClickException("--tls-cert and --tls-key must be provided together")
        tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls_context.load_cert_chain(tls_cert, tls_key)

    scheme = "wss" if tls_context is not None else "ws"
    click.echo(f"Starting phase sync server on {scheme}://{host}:{port}")
    click.echo(f"  L={layers}, N_per={n_per}, zeta={zeta}, psi={psi}")
    click.echo("  Ctrl-C to stop")

    mon = RealtimeMonitor.from_paper27(
        L=layers,
        N_per=n_per,
        zeta_uniform=zeta,
        psi_driver=psi,
    )
    server = PhaseStreamServer(
        monitor=mon,
        tick_interval_s=tick_interval,
        api_key=api_key,
        command_rate_limit=command_rate_limit,
        command_rate_window_s=command_rate_window_s,
        max_payload_bytes=max_payload_bytes,
        max_client_write_buffer_bytes=max_client_write_buffer_bytes,
        require_client_auth=not allow_unauthenticated_clients,
        allow_query_token_auth=allow_query_token_auth,
        require_tls=require_tls,
        allow_insecure_remote=allow_insecure_remote,
        allowed_origins=allowed_origin,
        allowed_actions=allowed_action or ("set_psi", "set_pac_gamma", "reset", "stop"),
    )
    server.serve_sync(host=host, port=port, ssl_context=tls_context)


@main.command(name="hil-test")
@click.option("--shots-dir", default="validation/reference_data/diiid/disruption_shots", type=str)
@click.option("--json-out", is_flag=True, help="Emit JSON")
def hil_test(shots_dir: str, json_out: bool) -> None:
    """Hardware-in-the-loop test campaign against reference shots."""
    from pathlib import Path

    shots_path = Path(shots_dir)
    if not shots_path.exists():
        click.echo(f"ERROR: Shots directory not found: {shots_dir}", err=True)
        sys.exit(1)

    shot_files = sorted(shots_path.glob("*.npz"))
    results = []
    for sf in shot_files:
        data = np.load(sf, allow_pickle=True)
        results.append(
            {
                "shot": sf.stem,
                "keys": list(data.keys()),
                "status": "loaded",
            }
        )

    summary = {
        "shots_dir": str(shots_dir),
        "n_shots": len(results),
        "shots": results,
    }

    if json_out:
        click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(f"Loaded {len(results)} shots from {shots_dir}")
        for r in results:
            click.echo(f"  {r['shot']}: {len(r['keys'])} arrays")


@main.command()
@click.option("--json-out", is_flag=True, help="Emit JSON")
def info(json_out: bool) -> None:
    """Print environment and backend information."""
    from pathlib import Path

    import scpn_control

    rust_status = "available" if scpn_control.RUST_BACKEND else "not installed"

    weights_dir = Path(__file__).resolve().parents[2] / "weights"
    weight_files = sorted(weights_dir.glob("*.npz")) if weights_dir.exists() else []

    result = {
        "version": scpn_control.__version__,
        "rust_backend": scpn_control.RUST_BACKEND,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "weights": [w.name for w in weight_files],
        "source_modules": len(list((Path(__file__).resolve().parent).rglob("*.py"))),
        "test_files": len(list((Path(__file__).resolve().parent.parent.parent.parent / "tests").glob("test_*.py"))),
    }

    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"scpn-control {result['version']}")
        click.echo(f"  Rust backend: {rust_status}")
        click.echo(f"  Python: {result['python']}")
        click.echo(f"  NumPy: {result['numpy']}")
        click.echo(f"  Weights: {len(weight_files)} files")
        for w in weight_files:
            click.echo(f"    {w.name} ({w.stat().st_size / 1024:.0f} KB)")


for _reference_command in REFERENCE_VALIDATOR_COMMANDS:
    main.add_command(_reference_command)


if __name__ == "__main__":
    main()
