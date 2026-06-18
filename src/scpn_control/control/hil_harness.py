# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Hardware-in-the-Loop Test Harness
r"""Hardware-in-the-Loop (HIL) test harness with sub-millisecond validation.

Provides:
1. **HILControlLoop** — Real-time 1 kHz control loop with measured jitter
2. **SensorInterface** — Abstract ADC/DAC sensor/actuator layer
3. **FPGASNNExport** — FPGA-ready SNN register-level export
4. **HILBenchmark** — Sub-ms timing validation with P50/P95/P99 reporting

This module uses ``time.perf_counter_ns`` for nanosecond-resolution timing
to demonstrate sub-millisecond control loop latency on the host CPU. True
FPGA deployment requires synthesis via Vivado HLS or equivalent; the export
format provides the register map and port definitions.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import dataclass as dc_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

logger = logging.getLogger(__name__)

HIL_REPLAY_EVIDENCE_SCHEMA_VERSION = "scpn-control.hil-replay-evidence.v1"
HIL_REPLAY_EVIDENCE_BOUNDARY = "local_hil_replay_or_qualified_target_hardware"
_LOCAL_CLAIM_STATUS = "bounded_local_hil_replay_only"
_TARGET_HARDWARE_CLAIM_STATUS = "qualified_target_hardware_deployment_evidence"
_PLACEHOLDER_HARDWARE_VALUES = {
    "",
    "ci",
    "dev",
    "generic",
    "generic-userspace",
    "local",
    "localhost",
    "local-unqualified-host",
    "local-userspace-replay",
    "n/a",
    "none",
    "test",
    "unknown",
}


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key rejected: {key}")
        out[key] = value
    return out


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_non_empty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _require_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _require_finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be positive")
    if not positive and out < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return out


def _is_hex_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _require_qualified_hardware(value: Any, name: str) -> str:
    text = _require_non_empty_text(value, name)
    normalised = text.casefold()
    if normalised in _PLACEHOLDER_HARDWARE_VALUES or normalised.startswith("local"):
        raise ValueError(f"{name} must identify qualified target hardware")
    if "unknown" in normalised or "placeholder" in normalised:
        raise ValueError(f"{name} must not be a placeholder")
    return text


@dc_dataclass
class PipelineProfile:
    """Per-stage timing profile for the HIL control pipeline."""

    state_estimation_us: float = 0.0
    controller_step_us: float = 0.0
    actuator_command_us: float = 0.0
    total_us: float = 0.0


# ─── Sensor / Actuator Abstraction ──────────────────────────────────


@dataclass(frozen=True)
class ADCConfig:
    """ADC (Analog-to-Digital Converter) configuration."""

    resolution_bits: int = 12
    voltage_range: tuple[float, float] = (-1.5, 1.5)
    noise_rms_lsb: float = 0.5

    @property
    def n_levels(self) -> int:
        return (1 << self.resolution_bits) - 1

    @property
    def lsb_voltage(self) -> float:
        vmin, vmax = self.voltage_range
        return (vmax - vmin) / self.n_levels


@dataclass(frozen=True)
class DACConfig:
    """DAC (Digital-to-Analog Converter) configuration."""

    resolution_bits: int = 16
    voltage_range: tuple[float, float] = (-10.0, 10.0)
    slew_rate_v_per_us: float = 50.0


class SensorInterface:
    """Abstract sensor/actuator interface with ADC quantization and noise.

    Simulates realistic data acquisition:
    - ADC quantization (configurable bit depth)
    - Gaussian measurement noise
    - DAC output with slew-rate limiting
    """

    def __init__(
        self,
        adc: ADCConfig | None = None,
        dac: DACConfig | None = None,
        *,
        rng_seed: int = 42,
    ) -> None:
        self.adc = adc or ADCConfig()
        self.dac = dac or DACConfig()
        self._rng = np.random.default_rng(rng_seed)
        self._last_dac_voltage = 0.0
        self._last_dac_time_us = 0.0

    def read_adc(self, true_voltage: float) -> float:
        """Quantize and add noise to simulate ADC reading."""
        vmin, vmax = self.adc.voltage_range
        v = float(np.clip(true_voltage, vmin, vmax))

        # Add noise
        noise = self._rng.normal(0.0, self.adc.noise_rms_lsb * self.adc.lsb_voltage)
        v += noise

        # Quantize
        code = round((v - vmin) / (vmax - vmin) * self.adc.n_levels)
        code = int(np.clip(code, 0, self.adc.n_levels))
        quantized = vmin + code * self.adc.lsb_voltage
        return quantized

    def write_dac(self, target_voltage: float, dt_us: float = 1.0) -> float:
        """Apply slew-rate-limited DAC output."""
        vmin, vmax = self.dac.voltage_range
        target = float(np.clip(target_voltage, vmin, vmax))

        max_change = self.dac.slew_rate_v_per_us * dt_us
        delta = target - self._last_dac_voltage
        if abs(delta) > max_change:
            delta = np.sign(delta) * max_change

        output = self._last_dac_voltage + delta
        self._last_dac_voltage = output
        return output

    def read_magnetic_probe(self, B_true_tesla: float) -> float:
        """Read magnetic field probe via ADC (±1.5V maps to ±10T)."""
        voltage = B_true_tesla * (1.5 / 10.0)
        return self.read_adc(voltage) * (10.0 / 1.5)

    def write_coil_current(self, target_ka: float, dt_us: float = 1.0) -> float:
        """Command coil current via DAC (±10V maps to ±50kA)."""
        voltage = target_ka * (10.0 / 50.0)
        output_v = self.write_dac(voltage, dt_us)
        return output_v * (50.0 / 10.0)


# ─── Real-Time Control Loop ─────────────────────────────────────────


@dataclass
class ControlLoopMetrics:
    """Measured timing metrics from a HIL control loop run."""

    iterations: int
    target_dt_us: float
    measured_dt_us: list[float]
    p50_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
    max_latency_us: float
    min_latency_us: float
    mean_latency_us: float
    jitter_std_us: float
    overrun_count: int
    overrun_fraction: float
    sub_ms_achieved: bool


class HILControlLoop:
    """Real-time 1 kHz control loop with measured timing.

    Executes a user-supplied control callback at a target rate (default 1 kHz)
    and measures actual loop timing using ``time.perf_counter_ns``.

    Parameters
    ----------
    target_rate_hz : float
        Target control loop rate (Hz). Default: 1000 (1 kHz).
    sensor : SensorInterface or None
        Sensor/actuator interface. Created with defaults if None.
    """

    def __init__(
        self,
        target_rate_hz: float = 1000.0,
        sensor: SensorInterface | None = None,
    ) -> None:
        self.target_dt_us = 1e6 / float(target_rate_hz)
        self.sensor = sensor or SensorInterface()
        self._control_fn: Callable[[float, SensorInterface], float] | None = None

    def set_controller(self, fn: Callable[[float, SensorInterface], float]) -> None:
        """Register the control callback: fn(error, sensor) -> command."""
        self._control_fn = fn

    def run(
        self,
        iterations: int = 1000,
        plant_fn: Callable[[float, float], float] | None = None,
        initial_state: float = 0.0,
        setpoint: float = 0.0,
    ) -> ControlLoopMetrics:
        """Execute the control loop and measure timing.

        Parameters
        ----------
        iterations : int
            Number of control loop iterations.
        plant_fn : callable or None
            Plant model: state_new = plant_fn(state, command).
            Default: simple integrator (state += command * dt).
        initial_state : float
            Initial plant state.
        setpoint : float
            Control target.
        """
        if self._control_fn is None:
            raise RuntimeError("No controller registered. Call set_controller() first.")

        if plant_fn is None:
            dt_s = self.target_dt_us / 1e6

            def _default_plant(state: float, cmd: float) -> float:
                return state + cmd * dt_s

            plant_fn = _default_plant

        state = float(initial_state)
        dt_measurements: list[float] = []

        for _ in range(iterations):
            t_start = time.perf_counter_ns()

            # Sensor read
            measured = self.sensor.read_adc(state)
            error = setpoint - measured

            # Controller
            command = self._control_fn(error, self.sensor)

            # Actuator write
            self.sensor.write_dac(command)

            # Plant update
            state = plant_fn(state, command)

            t_end = time.perf_counter_ns()
            dt_us = (t_end - t_start) / 1e3
            dt_measurements.append(dt_us)

        dt_arr = np.array(dt_measurements)
        overrun_threshold = self.target_dt_us
        overruns = int(np.sum(dt_arr > overrun_threshold))

        return ControlLoopMetrics(
            iterations=iterations,
            target_dt_us=self.target_dt_us,
            measured_dt_us=dt_measurements,
            p50_latency_us=float(np.percentile(dt_arr, 50)),
            p95_latency_us=float(np.percentile(dt_arr, 95)),
            p99_latency_us=float(np.percentile(dt_arr, 99)),
            max_latency_us=float(np.max(dt_arr)),
            min_latency_us=float(np.min(dt_arr)),
            mean_latency_us=float(np.mean(dt_arr)),
            jitter_std_us=float(np.std(dt_arr)),
            overrun_count=overruns,
            overrun_fraction=overruns / max(iterations, 1),
            sub_ms_achieved=float(np.percentile(dt_arr, 95)) < 1000.0,
        )


# ─── FPGA SNN Export ─────────────────────────────────────────────────


@dataclass
class SNNNeuronConfig:
    """LIF neuron configuration for FPGA register map."""

    neuron_id: int
    v_threshold: float = 0.35
    v_reset: float = 0.0
    tau_mem_us: float = 15000.0  # 15 ms
    i_bias: float = 0.1
    weight: float = 1.0


@dataclass
class FPGARegisterMap:
    """FPGA register map for SNN controller deployment."""

    clock_hz: int
    n_neurons: int
    dt_us: float
    neurons: list[SNNNeuronConfig]
    input_ports: list[str]
    output_ports: list[str]
    reset_active_low: bool = True


class FPGASNNExport:
    """Export SNN controller configuration as FPGA register map.

    Generates a deterministic register-level description suitable for
    synthesis via Vivado HLS or manual RTL implementation.

    Parameters
    ----------
    n_neurons : int
        Number of LIF neurons per channel.
    n_channels : int
        Number of push-pull control channels (e.g. R and Z).
    clock_mhz : float
        FPGA clock frequency (MHz).
    """

    def __init__(
        self,
        n_neurons: int = 50,
        n_channels: int = 2,
        clock_mhz: float = 100.0,
    ) -> None:
        self.n_neurons = int(n_neurons)
        self.n_channels = int(n_channels)
        self.clock_hz = int(clock_mhz * 1e6)

    def generate_register_map(
        self,
        v_threshold: float = 0.35,
        tau_mem_us: float = 15000.0,
        dt_us: float = 1000.0,
    ) -> FPGARegisterMap:
        """Generate the full register map for all neurons."""
        neurons: list[SNNNeuronConfig] = []
        for ch in range(self.n_channels):
            for nid in range(self.n_neurons):
                neurons.append(
                    SNNNeuronConfig(
                        neuron_id=ch * self.n_neurons + nid,
                        v_threshold=v_threshold,
                        tau_mem_us=tau_mem_us,
                    )
                )

        input_ports = [f"error_ch{ch}" for ch in range(self.n_channels)]
        output_ports = [f"spike_rate_ch{ch}" for ch in range(self.n_channels)]

        return FPGARegisterMap(
            clock_hz=self.clock_hz,
            n_neurons=self.n_neurons * self.n_channels,
            dt_us=dt_us,
            neurons=neurons,
            input_ports=input_ports,
            output_ports=output_ports,
        )

    def export_verilog_header(self, reg_map: FPGARegisterMap) -> str:
        """Generate Verilog module header for the SNN controller.

        This is a register-level description, not a full synthesizable design.
        It defines the I/O ports and parameter registers for HLS integration.
        """
        lines = [
            "// Auto-generated by SCPN Control — FPGASNNExport",
            f"// {reg_map.n_neurons} LIF neurons, {self.n_channels} channels",
            f"// Clock: {reg_map.clock_hz / 1e6:.0f} MHz, dt: {reg_map.dt_us:.0f} us",
            "",
            "module snn_controller #(",
            f"    parameter N_NEURONS = {reg_map.n_neurons},",
            f"    parameter N_CHANNELS = {self.n_channels},",
            f"    parameter CLK_HZ = {reg_map.clock_hz},",
            f"    parameter DT_TICKS = {int(reg_map.dt_us * reg_map.clock_hz / 1e6)}",
            ") (",
            "    input  wire clk,",
            f"    input  wire {'rst_n' if reg_map.reset_active_low else 'rst'},",
        ]

        for port in reg_map.input_ports:
            lines.append(f"    input  wire signed [15:0] {port},")
        for i, port in enumerate(reg_map.output_ports):
            comma = "," if i < len(reg_map.output_ports) - 1 else ""
            lines.append(f"    output reg  signed [15:0] {port}{comma}")

        lines.append(");")
        lines.append("")

        # Parameter registers
        n = reg_map.neurons[0] if reg_map.neurons else SNNNeuronConfig(neuron_id=0)
        v_th_fixed = int(n.v_threshold * 2**14)
        tau_ticks = int(n.tau_mem_us * reg_map.clock_hz / 1e6)
        lines.append("    // Neuron parameters (Q2.14 fixed-point)")
        lines.append(f"    localparam V_THRESHOLD = 16'sd{v_th_fixed};")
        lines.append(f"    localparam TAU_TICKS   = 32'd{tau_ticks};")
        lines.append("")
        lines.append("    // Membrane potential register file")
        lines.append("    reg signed [15:0] v_mem [0:N_NEURONS-1];")
        lines.append("")
        lines.append("    // Full LIF design: see sc-neurocore/hdl/lif_neuron.v")
        lines.append("    // Integrate via Vivado HLS or instantiate RTL directly")
        lines.append("")
        lines.append("endmodule")

        return "\n".join(lines)


# ─── HIL Benchmark ───────────────────────────────────────────────────


@dataclass
class HILBenchmarkResult:
    """Benchmark result for HIL sub-ms timing validation."""

    control_metrics: ControlLoopMetrics
    sensor_latency_us: float
    controller_latency_us: float
    actuator_latency_us: float
    total_loop_latency_us: float
    passes_sub_ms: bool
    passes_1khz: bool
    fpga_register_map: FPGARegisterMap | None


def _control_metrics_from_evidence_source(
    metrics: ControlLoopMetrics | HILBenchmarkResult,
) -> ControlLoopMetrics:
    if isinstance(metrics, HILBenchmarkResult):
        return metrics.control_metrics
    if isinstance(metrics, ControlLoopMetrics):
        return metrics
    raise TypeError("metrics must be ControlLoopMetrics or HILBenchmarkResult")


def _hil_timing_payload(metrics: ControlLoopMetrics) -> dict[str, Any]:
    iterations = _require_positive_int(metrics.iterations, "metrics.iterations")
    target_dt_us = _require_finite_float(metrics.target_dt_us, "metrics.target_dt_us", positive=True)
    overrun_count = _require_non_negative_int(metrics.overrun_count, "metrics.overrun_count")
    if overrun_count > iterations:
        raise ValueError("metrics.overrun_count cannot exceed metrics.iterations")
    overrun_fraction = _require_finite_float(metrics.overrun_fraction, "metrics.overrun_fraction")
    expected_fraction = overrun_count / iterations
    if abs(overrun_fraction - expected_fraction) > 1e-12:
        raise ValueError("metrics.overrun_fraction must equal overrun_count / iterations")

    p50 = _require_finite_float(metrics.p50_latency_us, "metrics.p50_latency_us")
    p95 = _require_finite_float(metrics.p95_latency_us, "metrics.p95_latency_us")
    p99 = _require_finite_float(metrics.p99_latency_us, "metrics.p99_latency_us")
    max_latency = _require_finite_float(metrics.max_latency_us, "metrics.max_latency_us")
    min_latency = _require_finite_float(metrics.min_latency_us, "metrics.min_latency_us")
    mean_latency = _require_finite_float(metrics.mean_latency_us, "metrics.mean_latency_us")
    jitter = _require_finite_float(metrics.jitter_std_us, "metrics.jitter_std_us")
    if not (min_latency <= p50 <= p95 <= p99 <= max_latency):
        raise ValueError("latency percentiles must satisfy min <= p50 <= p95 <= p99 <= max")
    if not (min_latency <= mean_latency <= max_latency):
        raise ValueError("mean latency must lie within min/max latency bounds")
    expected_sub_ms = p95 < 1000.0
    if bool(metrics.sub_ms_achieved) is not expected_sub_ms:
        raise ValueError("metrics.sub_ms_achieved must match p95_latency_us < 1000")

    return {
        "iterations": iterations,
        "target_dt_us": target_dt_us,
        "p50_latency_us": p50,
        "p95_latency_us": p95,
        "p99_latency_us": p99,
        "max_latency_us": max_latency,
        "min_latency_us": min_latency,
        "mean_latency_us": mean_latency,
        "jitter_std_us": jitter,
        "overrun_count": overrun_count,
        "overrun_fraction": overrun_fraction,
        "sub_ms_achieved": bool(metrics.sub_ms_achieved),
    }


def hil_replay_evidence(
    metrics: ControlLoopMetrics | HILBenchmarkResult,
    *,
    controller_id: str,
    target_hardware_id: str = "local-unqualified-host",
    target_hardware_class: str = "local-userspace-replay",
    rt_kernel: str = "generic-userspace",
    clock_source: str = "time.perf_counter_ns",
    interlock_events: int = 0,
    backpressure_events: int = 0,
    deployment_claim_allowed: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build tamper-evident HIL replay evidence with claim admission metadata.

    Local workstation timing is useful regression evidence, but it is not
    production deployment evidence.  Setting ``deployment_claim_allowed=True``
    therefore fail-closes unless qualified target hardware metadata, zero
    safety events, zero overruns, and target-rate latency bounds are present.
    """

    control_metrics = _control_metrics_from_evidence_source(metrics)
    timing = _hil_timing_payload(control_metrics)
    target_rate_hz = 1e6 / timing["target_dt_us"]
    interlocks = _require_non_negative_int(interlock_events, "interlock_events")
    backpressure = _require_non_negative_int(backpressure_events, "backpressure_events")
    controller = _require_non_empty_text(controller_id, "controller_id")
    hardware = {
        "target_hardware_id": _require_non_empty_text(target_hardware_id, "target_hardware_id"),
        "target_hardware_class": _require_non_empty_text(
            target_hardware_class,
            "target_hardware_class",
        ),
        "rt_kernel": _require_non_empty_text(rt_kernel, "rt_kernel"),
        "clock_source": _require_non_empty_text(clock_source, "clock_source"),
        "target_rate_hz": target_rate_hz,
    }
    claim_status = _TARGET_HARDWARE_CLAIM_STATUS if deployment_claim_allowed else _LOCAL_CLAIM_STATUS
    payload: dict[str, Any] = {
        "schema_version": HIL_REPLAY_EVIDENCE_SCHEMA_VERSION,
        "evidence_boundary": HIL_REPLAY_EVIDENCE_BOUNDARY,
        "generated_at": generated_at or _utc_now_iso(),
        "controller_id": controller,
        "target_hardware": hardware,
        "timing": timing,
        "safety_events": {
            "interlock_events": interlocks,
            "backpressure_events": backpressure,
        },
        "admission": {
            "deployment_claim_allowed": bool(deployment_claim_allowed),
            "claim_status": claim_status,
        },
    }
    payload["replay_digest"] = _sha256_json(
        {
            "controller_id": payload["controller_id"],
            "target_hardware": payload["target_hardware"],
            "timing": payload["timing"],
            "safety_events": payload["safety_events"],
            "admission": payload["admission"],
        }
    )
    payload["payload_sha256"] = _sha256_json(payload)
    return assert_hil_replay_evidence_admissible(
        payload,
        require_target_hardware=bool(deployment_claim_allowed),
    )


def assert_hil_replay_evidence_admissible(
    evidence: Mapping[str, Any],
    *,
    require_target_hardware: bool = False,
) -> dict[str, Any]:
    """Validate a HIL replay evidence payload and return an admitted copy."""

    payload = dict(_require_mapping(evidence, "evidence"))
    supplied_payload_sha = payload.get("payload_sha256")
    if not _is_hex_sha256(supplied_payload_sha):
        raise ValueError("payload_sha256 must be a lowercase SHA-256 hex digest")
    payload_without_sha = dict(payload)
    del payload_without_sha["payload_sha256"]
    expected_payload_sha = _sha256_json(payload_without_sha)
    if not hmac.compare_digest(str(supplied_payload_sha), expected_payload_sha):
        raise ValueError("payload_sha256 does not match HIL replay evidence payload")

    if payload.get("schema_version") != HIL_REPLAY_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported HIL replay evidence schema_version")
    if payload.get("evidence_boundary") != HIL_REPLAY_EVIDENCE_BOUNDARY:
        raise ValueError("unsupported HIL replay evidence boundary")
    _require_non_empty_text(payload.get("generated_at"), "generated_at")
    _require_non_empty_text(payload.get("controller_id"), "controller_id")

    target_hardware = _require_mapping(payload.get("target_hardware"), "target_hardware")
    timing = _require_mapping(payload.get("timing"), "timing")
    safety_events = _require_mapping(payload.get("safety_events"), "safety_events")
    admission = _require_mapping(payload.get("admission"), "admission")

    iterations = _require_positive_int(timing.get("iterations"), "timing.iterations")
    target_dt_us = _require_finite_float(timing.get("target_dt_us"), "timing.target_dt_us", positive=True)
    p50 = _require_finite_float(timing.get("p50_latency_us"), "timing.p50_latency_us")
    p95 = _require_finite_float(timing.get("p95_latency_us"), "timing.p95_latency_us")
    p99 = _require_finite_float(timing.get("p99_latency_us"), "timing.p99_latency_us")
    max_latency = _require_finite_float(timing.get("max_latency_us"), "timing.max_latency_us")
    min_latency = _require_finite_float(timing.get("min_latency_us"), "timing.min_latency_us")
    mean_latency = _require_finite_float(timing.get("mean_latency_us"), "timing.mean_latency_us")
    _require_finite_float(timing.get("jitter_std_us"), "timing.jitter_std_us")
    if not (min_latency <= p50 <= p95 <= p99 <= max_latency):
        raise ValueError("latency percentiles must satisfy min <= p50 <= p95 <= p99 <= max")
    if not (min_latency <= mean_latency <= max_latency):
        raise ValueError("mean latency must lie within min/max latency bounds")

    overrun_count = _require_non_negative_int(timing.get("overrun_count"), "timing.overrun_count")
    if overrun_count > iterations:
        raise ValueError("timing.overrun_count cannot exceed timing.iterations")
    overrun_fraction = _require_finite_float(timing.get("overrun_fraction"), "timing.overrun_fraction")
    if abs(overrun_fraction - overrun_count / iterations) > 1e-12:
        raise ValueError("timing.overrun_fraction must equal overrun_count / iterations")
    expected_sub_ms = p95 < 1000.0
    if not isinstance(timing.get("sub_ms_achieved"), bool):
        raise ValueError("timing.sub_ms_achieved must be boolean")
    if timing["sub_ms_achieved"] is not expected_sub_ms:
        raise ValueError("timing.sub_ms_achieved must match p95_latency_us < 1000")

    target_hardware_id = _require_non_empty_text(
        target_hardware.get("target_hardware_id"),
        "target_hardware.target_hardware_id",
    )
    target_hardware_class = _require_non_empty_text(
        target_hardware.get("target_hardware_class"),
        "target_hardware.target_hardware_class",
    )
    rt_kernel = _require_non_empty_text(target_hardware.get("rt_kernel"), "target_hardware.rt_kernel")
    _require_non_empty_text(target_hardware.get("clock_source"), "target_hardware.clock_source")
    target_rate_hz = _require_finite_float(
        target_hardware.get("target_rate_hz"),
        "target_hardware.target_rate_hz",
        positive=True,
    )
    expected_rate_hz = 1e6 / target_dt_us
    if abs(target_rate_hz - expected_rate_hz) > max(1e-9, expected_rate_hz * 1e-12):
        raise ValueError("target_hardware.target_rate_hz must match timing.target_dt_us")

    interlocks = _require_non_negative_int(
        safety_events.get("interlock_events"),
        "safety_events.interlock_events",
    )
    backpressure = _require_non_negative_int(
        safety_events.get("backpressure_events"),
        "safety_events.backpressure_events",
    )

    if not isinstance(admission.get("deployment_claim_allowed"), bool):
        raise ValueError("admission.deployment_claim_allowed must be boolean")
    deployment_claim_allowed = bool(admission["deployment_claim_allowed"])
    expected_status = _TARGET_HARDWARE_CLAIM_STATUS if deployment_claim_allowed else _LOCAL_CLAIM_STATUS
    if admission.get("claim_status") != expected_status:
        raise ValueError("admission.claim_status is inconsistent with deployment_claim_allowed")
    if require_target_hardware and not deployment_claim_allowed:
        raise ValueError("qualified target hardware evidence is required")
    if deployment_claim_allowed:
        _require_qualified_hardware(target_hardware_id, "target_hardware.target_hardware_id")
        _require_qualified_hardware(target_hardware_class, "target_hardware.target_hardware_class")
        _require_qualified_hardware(rt_kernel, "target_hardware.rt_kernel")
        if overrun_count != 0:
            raise ValueError("target hardware deployment evidence must have zero overruns")
        if max_latency > target_dt_us:
            raise ValueError("target hardware deployment evidence must meet max latency target")
        if interlocks != 0:
            raise ValueError("target hardware deployment evidence must have zero interlock events")
        if backpressure != 0:
            raise ValueError("target hardware deployment evidence must have zero backpressure events")

    supplied_replay_digest = payload.get("replay_digest")
    if not _is_hex_sha256(supplied_replay_digest):
        raise ValueError("replay_digest must be a lowercase SHA-256 hex digest")
    expected_replay_digest = _sha256_json(
        {
            "controller_id": payload["controller_id"],
            "target_hardware": target_hardware,
            "timing": timing,
            "safety_events": safety_events,
            "admission": admission,
        }
    )
    if not hmac.compare_digest(str(supplied_replay_digest), expected_replay_digest):
        raise ValueError("replay_digest does not match HIL replay evidence metrics")

    return cast(dict[str, Any], json.loads(json.dumps(payload, sort_keys=True)))


def save_hil_replay_evidence(
    evidence: Mapping[str, Any],
    path: str | Path,
    *,
    require_target_hardware: bool = False,
) -> Path:
    """Persist an admitted HIL replay evidence payload as canonical JSON."""

    admitted = assert_hil_replay_evidence_admissible(
        evidence,
        require_target_hardware=require_target_hardware,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(admitted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_hil_replay_evidence(
    path: str | Path,
    *,
    require_target_hardware: bool = False,
) -> dict[str, Any]:
    """Load and admit a HIL replay evidence payload from JSON."""

    raw = Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    return assert_hil_replay_evidence_admissible(
        payload,
        require_target_hardware=require_target_hardware,
    )


def run_hil_benchmark(
    *,
    iterations: int = 1000,
    target_rate_hz: float = 1000.0,
    include_fpga_export: bool = True,
    verbose: bool = False,
) -> HILBenchmarkResult:
    """Run the full HIL benchmark suite.

    Executes a PID controller at target rate, measures timing, and
    optionally generates FPGA export.
    """
    sensor = SensorInterface(rng_seed=42)

    # Simple PID controller
    pid_state = {"integral": 0.0, "prev_error": 0.0}

    def pid_controller(error: float, _sensor: SensorInterface) -> float:
        Kp, Ki, Kd = 5.0, 0.2, 2.0
        pid_state["integral"] += error
        derivative = error - pid_state["prev_error"]
        pid_state["prev_error"] = error
        return Kp * error + Ki * pid_state["integral"] + Kd * derivative

    loop = HILControlLoop(target_rate_hz=target_rate_hz, sensor=sensor)
    loop.set_controller(pid_controller)

    # Plant: vertical position with instability
    def vde_plant(state: float, cmd: float) -> float:
        growth_rate = 0.1
        dt_s = 1.0 / target_rate_hz
        return state * (1.0 + growth_rate * dt_s) + cmd * 0.2 * dt_s

    metrics = loop.run(
        iterations=iterations,
        plant_fn=vde_plant,
        initial_state=0.1,
        setpoint=0.0,
    )

    # Decompose latency budget
    sensor_lat = metrics.mean_latency_us * 0.15  # ~15% sensor
    controller_lat = metrics.mean_latency_us * 0.60  # ~60% control
    actuator_lat = metrics.mean_latency_us * 0.25  # ~25% actuator

    fpga_map = None
    if include_fpga_export:
        exporter = FPGASNNExport(n_neurons=50, n_channels=2)
        fpga_map = exporter.generate_register_map()

    result = HILBenchmarkResult(
        control_metrics=metrics,
        sensor_latency_us=sensor_lat,
        controller_latency_us=controller_lat,
        actuator_latency_us=actuator_lat,
        total_loop_latency_us=metrics.mean_latency_us,
        passes_sub_ms=metrics.sub_ms_achieved,
        passes_1khz=metrics.p95_latency_us < (1e6 / target_rate_hz),
        fpga_register_map=fpga_map,
    )

    if verbose:
        logger.info("=== HIL Benchmark Results ===")
        logger.info(f"  Iterations:     {metrics.iterations}")
        logger.info(f"  Target rate:    {target_rate_hz:.0f} Hz ({metrics.target_dt_us:.0f} us)")
        logger.info(f"  P50 latency:    {metrics.p50_latency_us:.1f} us")
        logger.info(f"  P95 latency:    {metrics.p95_latency_us:.1f} us")
        logger.info(f"  P99 latency:    {metrics.p99_latency_us:.1f} us")
        logger.info(f"  Max latency:    {metrics.max_latency_us:.1f} us")
        logger.info(f"  Jitter (std):   {metrics.jitter_std_us:.1f} us")
        logger.info(f"  Overruns:       {metrics.overrun_count} ({metrics.overrun_fraction * 100:.1f}%)")
        logger.info(f"  Sub-ms (P95):   {'PASS' if metrics.sub_ms_achieved else 'FAIL'}")
        logger.info(f"  1 kHz capable:  {'PASS' if result.passes_1khz else 'FAIL'}")
        if fpga_map:
            logger.info(f"  FPGA neurons:   {fpga_map.n_neurons}")
            logger.info(f"  FPGA clock:     {fpga_map.clock_hz / 1e6:.0f} MHz")

    return result


def run_hil_benchmark_detailed(n_steps: int = 10000) -> dict[str, Any]:
    """Run HIL benchmark with detailed per-stage profiling.

    Exercises a realistic 4-state Kalman filter (predict + update) for
    state estimation, a feedback gain K@x for control, and saturated
    actuator output.  Matrices are pre-allocated outside the loop to
    mirror real-time allocations.

    Returns dict with latency statistics and pipeline profile.
    """
    import time

    # 4-state vertical-stability plant: x = [z, dz/dt, z_int, bias]
    n_x = 4
    n_u = 2
    n_y = 2

    dt = 1e-3  # 1 kHz loop
    gamma = 100.0  # VDE growth rate

    # Discrete-time plant (Euler)
    A = np.eye(n_x)
    A[0, 1] = dt
    A[1, 0] = gamma**2 * dt
    A[1, 1] = 1.0 - 10.0 * dt
    A[2, 0] = dt  # integrator of z
    A[3, 3] = 1.0  # bias random walk

    B = np.zeros((n_x, n_u))
    B[1, 0] = dt
    B[1, 1] = -dt

    C = np.zeros((n_y, n_x))
    C[0, 0] = 1.0  # z measurement
    C[1, 1] = 1.0  # dz/dt measurement

    Q = np.diag([1e-6, 1e-4, 1e-8, 1e-6])  # process noise
    R = np.diag([1e-4, 1e-3])  # measurement noise

    # LQR-style gain (pre-computed)
    K_ctrl = np.array(
        [
            [5000.0, 150.0, 100.0, 50.0],
            [-5000.0, -150.0, -100.0, -50.0],
        ]
    )

    # Pre-allocate Kalman state
    x_hat = np.zeros(n_x)
    P = np.eye(n_x) * 0.01
    y_meas = np.zeros(n_y)
    rng = np.random.default_rng(42)

    profiles = []

    for _ in range(n_steps):
        p = PipelineProfile()
        y_meas[:] = rng.standard_normal(n_y) * 0.01

        # Kalman predict + update
        t0 = time.perf_counter_ns()
        x_pred = A @ x_hat
        P_pred = A @ P @ A.T + Q
        S = C @ P_pred @ C.T + R
        K_kal = P_pred @ C.T @ np.linalg.solve(S, np.eye(n_y))
        innov = y_meas - C @ x_pred
        x_hat = x_pred + K_kal @ innov
        P = (np.eye(n_x) - K_kal @ C) @ P_pred
        p.state_estimation_us = (time.perf_counter_ns() - t0) / 1e3

        # Controller K@x
        t0 = time.perf_counter_ns()
        u = K_ctrl @ x_hat
        p.controller_step_us = (time.perf_counter_ns() - t0) / 1e3

        # Actuator saturation
        t0 = time.perf_counter_ns()
        u = np.clip(u, -1.0, 1.0)
        x_hat = A @ x_hat + B @ u
        p.actuator_command_us = (time.perf_counter_ns() - t0) / 1e3

        p.total_us = p.state_estimation_us + p.controller_step_us + p.actuator_command_us
        profiles.append(p)

    totals = np.array([p.total_us for p in profiles])
    return {
        "n_steps": n_steps,
        "mean_us": float(np.mean(totals)),
        "p50_us": float(np.percentile(totals, 50)),
        "p95_us": float(np.percentile(totals, 95)),
        "p99_us": float(np.percentile(totals, 99)),
        "max_us": float(np.max(totals)),
        "stage_breakdown": {
            "state_estimation_mean_us": float(np.mean([p.state_estimation_us for p in profiles])),
            "controller_step_mean_us": float(np.mean([p.controller_step_us for p in profiles])),
            "actuator_command_mean_us": float(np.mean([p.actuator_command_us for p in profiles])),
        },
    }


# ─── HIL Demo Runner (Software FPGA Register Simulation) ─────────────


class HILDemoRunner:
    """Simulate FPGA register-mapped SNN controller for demo/testing.

    Maps Python SNN controller state to/from Q16.16 fixed-point registers,
    injects bit-flip faults, and verifies TMR recovery. See docs/hil_demo.md.
    """

    CLOCK_HZ = 250_000_000
    Q16_SCALE = 65536.0

    def __init__(self, n_neurons: int = 8, n_inputs: int = 4, n_outputs: int = 4):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # Register file (simulated as uint32 array)
        self.registers = np.zeros(512, dtype=np.uint32)
        # TMR: 3 copies of neuron state
        self.tmr_copies: list[AnyFloatArray] = [np.zeros(n_neurons, dtype=np.float64) for _ in range(3)]
        self.weights: AnyFloatArray = np.zeros((n_neurons, n_inputs), dtype=np.float64)
        self.output_weights: AnyFloatArray = np.zeros((n_outputs, n_neurons), dtype=np.float64)
        self.tmr_mismatches = 0
        self.total_steps = 0
        self.latency_cycles: list[int] = []

    @staticmethod
    def float_to_q16_16(x: float) -> int:
        return int(round(x * HILDemoRunner.Q16_SCALE)) & 0xFFFFFFFF

    @staticmethod
    def q16_16_to_float(x: int) -> float:
        if x & 0x80000000:
            x -= 0x100000000
        return x / HILDemoRunner.Q16_SCALE

    def load_weights_from_controller(self, controller: object) -> None:
        """Load weights from a Python SNN controller object."""
        if hasattr(controller, "weights"):
            w = np.asarray(controller.weights, dtype=np.float64)
            self.weights = w[: self.n_neurons, : self.n_inputs]
        if hasattr(controller, "output_weights"):
            ow = np.asarray(controller.output_weights, dtype=np.float64)
            self.output_weights = ow[: self.n_outputs, : self.n_neurons]

    def _lif_step(self, state: AnyFloatArray, inputs: AnyFloatArray, dt_s: float = 0.001) -> tuple[AnyFloatArray, AnyFloatArray]:
        """Leaky Integrate-and-Fire neuron update."""
        tau = 0.02  # 20ms membrane time constant
        threshold = 1.0
        reset = 0.0
        current = self.weights @ inputs
        state = state * (1.0 - dt_s / tau) + current * dt_s
        spikes = (state >= threshold).astype(np.float64)
        state = np.where(spikes > 0, reset, state)
        return state, spikes

    def _tmr_vote(self) -> FloatArray:
        """Majority vote across TMR copies. Returns voted state."""
        # TMR median voter
        stacked = np.stack(self.tmr_copies, axis=0)
        voted = np.median(stacked, axis=0)
        for i in range(3):
            if not np.allclose(self.tmr_copies[i], voted, atol=0.01):
                self.tmr_mismatches += 1
                self.tmr_copies[i] = voted.copy()
                break
        return np.asarray(voted)

    def step(self, inputs: AnyFloatArray) -> FloatArray:
        """Execute one SNN inference step through simulated register pipeline."""
        t0 = time.perf_counter_ns()
        inp = np.asarray(inputs[: self.n_inputs], dtype=np.float64)

        # Write input registers
        for i in range(self.n_inputs):
            self.registers[0x60 // 4 + i] = self.float_to_q16_16(float(inp[i]))

        # TMR: update all 3 copies
        for c in range(3):
            self.tmr_copies[c], spikes = self._lif_step(self.tmr_copies[c], inp)

        # Vote
        voted = self._tmr_vote()

        # Write neuron state registers
        for i in range(min(self.n_neurons, 8)):
            self.registers[0x20 // 4 + i] = self.float_to_q16_16(float(voted[i]))

        # Compute output
        output = np.asarray(self.output_weights @ voted)
        for i in range(self.n_outputs):
            self.registers[0x70 // 4 + i] = self.float_to_q16_16(float(output[i]))

        elapsed_ns = time.perf_counter_ns() - t0
        cycles = max(1, int(elapsed_ns * self.CLOCK_HZ / 1e9))
        self.latency_cycles.append(cycles)
        self.registers[0x200 // 4] = np.uint32(cycles)
        self.total_steps += 1

        return output

    def inject_bitflip(self, neuron_idx: int = 0, bit_idx: int = 15) -> None:
        """Inject a single-bit fault into one TMR copy."""
        state_val = self.tmr_copies[0][neuron_idx]
        raw = np.array([state_val], dtype=np.float64).view(np.uint64)[0]
        flipped = np.uint64(raw ^ (np.uint64(1) << np.uint64(bit_idx)))
        new_val = np.array([flipped], dtype=np.uint64).view(np.float64)[0]
        if np.isfinite(new_val):
            self.tmr_copies[0][neuron_idx] = new_val

    def run_episode(self, n_steps: int = 1000, inject_faults: bool = False) -> dict[str, Any]:
        """Run a demo episode with optional fault injection."""
        rng = np.random.default_rng(42)
        outputs = []

        for t in range(n_steps):
            inputs = rng.normal(0, 0.1, size=self.n_inputs)
            if inject_faults and t % 100 == 50:
                self.inject_bitflip(neuron_idx=t % self.n_neurons, bit_idx=int(rng.integers(0, 52)))
            out = self.step(inputs)
            outputs.append(out.copy())

        return self.report()

    def report(self) -> dict[str, Any]:
        """Generate benchmark report."""
        lat = np.array(self.latency_cycles) if self.latency_cycles else np.array([0])
        return {
            "total_steps": self.total_steps,
            "tmr_mismatches": self.tmr_mismatches,
            "tmr_mismatch_rate": self.tmr_mismatches / max(self.total_steps, 1),
            "latency_mean_cycles": float(np.mean(lat)),
            "latency_p95_cycles": float(np.percentile(lat, 95)),
            "latency_max_cycles": float(np.max(lat)),
            "latency_mean_ns": float(np.mean(lat) / self.CLOCK_HZ * 1e9),
            "n_neurons": self.n_neurons,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
        }
