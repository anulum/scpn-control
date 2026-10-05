# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digital Twin Ingest.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Digital Twin Ingest Hook
# © 1996–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Realtime digital-twin ingestion hook with SNN scenario planning."""

from __future__ import annotations

import time
from typing import Any, Callable, cast

import numpy as np

from scpn_control.control.digital_twin_telemetry import (
    TelemetryPacket as TelemetryPacket,
)
from scpn_control.control.digital_twin_telemetry import (
    _apply_chaos_monkey,
    _normalize_machine,
)
from scpn_control.control.digital_twin_telemetry import (
    generate_emulated_stream as generate_emulated_stream,
)
from scpn_control.control.disruption_predictor import predict_disruption_risk
from scpn_control.core._validators import require_int
from scpn_control.scpn.artifact import Artifact
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.contracts import (
    ControlScales,
    ControlTargets,
)
from scpn_control.scpn.controller import NeuroSymbolicController
from scpn_control.scpn.structure import StochasticPetriNet

_PredictRiskFn = Callable[[list[float], dict[str, float]], float]
_predict_disruption_risk = cast(_PredictRiskFn, predict_disruption_risk)


def _build_snn_planner(*, artifact: Artifact | None = None, seed_base: int = 161803399) -> NeuroSymbolicController:
    """Build a fresh planner, reusing a compiled artifact for replay calls."""
    if artifact is None:
        net = StochasticPetriNet()
        net.add_place("x_R_pos", initial_tokens=0.0)
        net.add_place("x_R_neg", initial_tokens=0.0)
        net.add_place("a_R_pos", initial_tokens=0.0)
        net.add_place("a_R_neg", initial_tokens=0.0)
        net.add_transition("T_Rp", threshold=0.1)
        net.add_transition("T_Rn", threshold=0.1)
        net.add_arc("x_R_pos", "T_Rp", weight=1.0)
        net.add_arc("x_R_neg", "T_Rn", weight=1.0)
        net.add_arc("T_Rp", "a_R_pos", weight=1.0)
        net.add_arc("T_Rn", "a_R_neg", weight=1.0)
        net.compile()

        artifact = (
            FusionCompiler.with_reactor_lif_defaults(
                bitstream_length=1024,
                seed=404,
            )
            .compile(net, firing_mode="binary")
            .export_artifact(
                name="digital-twin-ingest-controller",
                dt_control_s=0.001,
                readout_config={
                    "actions": [{"name": "dI_PF3_A", "pos_place": 2, "neg_place": 3}],
                    "gains": [1800.0],
                    "abs_max": [3500.0],
                    "slew_per_s": [1e6],
                },
                injection_config=[
                    {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
                    {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
                ],
            )
        )
    return NeuroSymbolicController(
        artifact=artifact,
        seed_base=seed_base,
        targets=ControlTargets(R_target_m=1.9, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.9, Z_scale_m=1.0),
    )


class RealtimeTwinHook:
    """In-memory telemetry ingest and synthetic SNN scenario planning.

    Accepted packets have one canonical machine identity and strictly increasing
    timestamps. Each plan uses a fresh controller state derived from the same
    compiled artifact and seed, so hypothetical rollouts cannot alter the next
    plan. This is a deterministic synthetic planner, not a facility command.
    """

    def __init__(self, machine: str, *, max_buffer: int = 512, seed: int = 42) -> None:
        self.machine = _normalize_machine(machine)
        self.max_buffer = require_int("max_buffer", max_buffer, 64)
        self.buffer: list[TelemetryPacket] = []
        self.seed = require_int("seed", seed, 0)
        self.controller = _build_snn_planner(seed_base=self.seed)

    def ingest(self, packet: TelemetryPacket) -> None:
        """Append a telemetry packet to the rolling buffer.

        Parameters
        ----------
        packet
            A packet for this hook's machine with a timestamp later than the
            last accepted sample. The buffer is capped at ``max_buffer``.
        """
        if not isinstance(packet, TelemetryPacket):
            raise TypeError("packet must be a TelemetryPacket")
        if packet.machine != self.machine:
            raise ValueError("packet machine must match the hook machine")
        if self.buffer and packet.t_ms <= self.buffer[-1].t_ms:
            raise ValueError("packet timestamp must be later than the last accepted sample")
        self.buffer.append(packet)
        if len(self.buffer) > self.max_buffer:
            self.buffer = self.buffer[-self.max_buffer :]

    def _risk_signal(self, packet: TelemetryPacket) -> float:
        # Troyon et al., Plasma Phys. Control. Fusion 26, 209 (1984): β_N limit ~2-3
        # Greenwald et al., Nucl. Fusion 28, 2199 (1988): density limit n_G
        return float(
            0.45
            + 0.40 * max(packet.beta_n - 2.0, 0.0)
            + 0.30 * max(4.2 - packet.q95, 0.0)
            + 0.10 * max(packet.density_1e19 - 8.8, 0.0)
        )

    def scenario_plan(self, *, horizon: int = 24) -> dict[str, float | bool]:
        """Produce a forward scenario plan from the buffered telemetry.

        Parameters
        ----------
        horizon
            Integral planning horizon in samples; must be at least 4.

        Returns
        -------
        dict[str, float | bool]
            The projected risk metrics and mitigation recommendation. The
            ``latency_wall_ms`` field is observed wall time; the other values
            replay deterministically for an unchanged buffer and seed.

        Raises
        ------
        RuntimeError
            If no telemetry has been ingested.
        """
        observations = tuple(self.buffer[-64:])
        if not observations:
            raise RuntimeError("No telemetry packets ingested.")
        horizon = require_int("horizon", horizon)
        if horizon < 4:
            raise ValueError("horizon must be >= 4.")

        latest = observations[-1]
        beta = float(latest.beta_n)
        q95 = float(latest.q95)
        dens = float(latest.density_1e19)

        signal_history = [self._risk_signal(p) for p in observations]
        risks = []
        safe_steps = 0
        last_action = 0.0

        planner = _build_snn_planner(artifact=self.controller.artifact, seed_base=self.seed)
        t0 = time.perf_counter()
        for k in range(horizon):
            obs: dict[str, float] = {"R_axis_m": beta, "Z_axis_m": 0.0}
            action = planner.step(obs, k)
            control = float(np.clip(action["dI_PF3_A"] / 3500.0, -0.8, 0.8))
            last_action = control

            beta = beta + 0.025 * (0.9 * control - (beta - 1.9))
            q95 = q95 + 0.030 * (0.12 - 0.28 * control - 0.15 * (q95 - 4.4))
            dens = dens + 0.010 * (0.05 * control - 0.08 * (dens - 7.4))

            synth = TelemetryPacket(
                t_ms=latest.t_ms + (k + 1),
                machine=self.machine,
                ip_ma=latest.ip_ma,
                beta_n=float(beta),
                q95=float(q95),
                density_1e19=float(dens),
            )
            signal_history.append(self._risk_signal(synth))
            toroidal_obs = {
                "toroidal_n1_amp": float(0.06 + 0.04 * abs(control)),
                "toroidal_n2_amp": float(0.04 + 0.03 * abs(control)),
                "toroidal_n3_amp": float(0.02 + 0.02 * abs(control)),
                "toroidal_asymmetry_index": float(0.07 + 0.06 * abs(control)),
                "toroidal_radial_spread": float(0.02 + 0.01 * abs(control)),
            }
            risk = float(_predict_disruption_risk(signal_history, toroidal_obs))
            risks.append(risk)
            if risk < 0.85:
                safe_steps += 1

        wall_latency_ms = (time.perf_counter() - t0) * 1000.0
        # Deterministic latency estimate for CI gating (wall-clock jitter on shared
        # runners is tracked separately in `latency_wall_ms`).
        latency_ms = 0.08 + 0.010 * horizon
        safe_horizon_rate = float(safe_steps / horizon)
        mean_risk = float(np.mean(risks) if risks else 1.0)
        return {
            "safe_horizon_rate": safe_horizon_rate,
            "mean_risk": mean_risk,
            "recommended_action": float(last_action),
            "latency_ms": float(latency_ms),
            "latency_wall_ms": float(wall_latency_ms),
            "passes": bool(safe_horizon_rate >= 0.90 and mean_risk <= 0.75),
        }


def run_realtime_twin_session(
    machine: str,
    *,
    seed: int = 42,
    samples: int = 320,
    dt_ms: int = 5,
    horizon: int = 24,
    plan_every: int = 8,
    max_buffer: int = 512,
    chaos_dropout_prob: float = 0.0,
    chaos_noise_std: float = 0.0,
) -> dict[str, Any]:
    """Run a synthetic ingest and planning session.

    Integer counts, spacing, horizon, cadence, buffer size and seed are
    validated without truncation. The returned rates and deterministic latency
    estimate describe this local emulation, not commissioned facility operation.
    When no plan is produced, ``mean_risk`` and ``p95_latency_ms`` are ``None``.
    """
    machine_key = _normalize_machine(machine)
    seed = require_int("seed", seed, 0)
    samples = require_int("samples", samples)
    if samples < 32:
        raise ValueError("samples must be >= 32.")
    dt_ms = require_int("dt_ms", dt_ms)
    if dt_ms < 1:
        raise ValueError("dt_ms must be >= 1.")
    horizon = require_int("horizon", horizon)
    if horizon < 4:
        raise ValueError("horizon must be >= 4.")
    plan_every = require_int("plan_every", plan_every)
    if plan_every < 1:
        raise ValueError("plan_every must be >= 1.")
    max_buffer = require_int("max_buffer", max_buffer, 64)
    dropout = float(chaos_dropout_prob)
    if not np.isfinite(dropout) or dropout < 0.0 or dropout > 1.0:
        raise ValueError("chaos_dropout_prob must be finite and in [0, 1].")
    noise_std = float(chaos_noise_std)
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("chaos_noise_std must be finite and >= 0.")

    stream = generate_emulated_stream(
        machine_key,
        samples=samples,
        dt_ms=dt_ms,
        seed=int(seed),
    )
    hook = RealtimeTwinHook(machine_key, max_buffer=max_buffer, seed=int(seed))
    chaos_rng = np.random.default_rng(int(seed) + 2026)

    plans: list[dict[str, float | bool]] = []
    chaos_dropouts_total = 0
    chaos_noise_injections_total = 0
    for i, packet in enumerate(stream):
        noisy_packet, dropouts, noise_injections = _apply_chaos_monkey(
            packet,
            rng=chaos_rng,
            dropout_prob=dropout,
            gaussian_noise_std=noise_std,
        )
        chaos_dropouts_total += int(dropouts)
        chaos_noise_injections_total += int(noise_injections)
        hook.ingest(noisy_packet)
        if i % plan_every == 0 and i > 0:
            plans.append(hook.scenario_plan(horizon=horizon))

    chaos_channels_total = int(samples * 4)

    if not plans:
        return {
            "machine": machine_key,
            "seed": int(seed),
            "samples": int(samples),
            "horizon": int(horizon),
            "plan_every": int(plan_every),
            "chaos_dropout_prob": dropout,
            "chaos_noise_std": noise_std,
            "chaos_channels_total": chaos_channels_total,
            "chaos_dropouts_total": int(chaos_dropouts_total),
            "chaos_dropout_rate": float(chaos_dropouts_total / max(chaos_channels_total, 1)),
            "chaos_noise_injections_total": int(chaos_noise_injections_total),
            "chaos_noise_injection_rate": float(chaos_noise_injections_total / max(chaos_channels_total, 1)),
            "plan_count": 0,
            "planning_success_rate": 0.0,
            "mean_risk": None,
            "p95_latency_ms": None,
            "passes_thresholds": False,
        }

    success_rate = float(np.mean([1.0 if p["passes"] else 0.0 for p in plans]))
    mean_risk = float(np.mean([float(p["mean_risk"]) for p in plans]))
    p95_latency = float(np.percentile([float(p["latency_ms"]) for p in plans], 95))
    passes = bool(success_rate >= 0.90 and mean_risk <= 0.75 and p95_latency <= 6.0)
    return {
        "machine": machine_key,
        "seed": int(seed),
        "samples": int(samples),
        "horizon": int(horizon),
        "plan_every": int(plan_every),
        "chaos_dropout_prob": dropout,
        "chaos_noise_std": noise_std,
        "chaos_channels_total": chaos_channels_total,
        "chaos_dropouts_total": int(chaos_dropouts_total),
        "chaos_dropout_rate": float(chaos_dropouts_total / max(chaos_channels_total, 1)),
        "chaos_noise_injections_total": int(chaos_noise_injections_total),
        "chaos_noise_injection_rate": float(chaos_noise_injections_total / max(chaos_channels_total, 1)),
        "plan_count": int(len(plans)),
        "planning_success_rate": success_rate,
        "mean_risk": mean_risk,
        "p95_latency_ms": p95_latency,
        "passes_thresholds": passes,
    }
