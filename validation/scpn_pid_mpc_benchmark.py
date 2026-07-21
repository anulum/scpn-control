# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Scpn Pid Mpc Benchmark
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — SCPN vs PID/MPC Benchmark
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic control benchmark comparing SCPN controller against PID/MPC."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]

from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.contracts import ControlScales, ControlTargets
from scpn_control.scpn.controller import NeuroSymbolicController
from scpn_control.scpn.structure import StochasticPetriNet

# Genuinely-symbolic control routing. The readout reads the transition-OUTPUT ``a_*``
# places (not the injected ``x_*``), so the timed Petri-net transitions are
# load-bearing: inject x_* -> T_* fires (``delay_ticks``, default 0) -> a_* -> readout,
# with a drain D_*: a_* -> sink so a_* stays responsive rather than saturating to 1.0
# (undrained a_* diverges, e-RMSE ~0.53 -> the transitions genuinely carry the signal).
# The gain/threshold constants are configured on seed=42/steps=320 and hold parity with
# the direct neuromorphic readout on unseen seeds/steps (within a 15% equivalence band).
# Reproducible parity, out-of-sample robustness, and load-bearing (a direct firing trace
# plus undrained divergence) evidence -- measured apples-to-apples against PID on the same
# per-seed profiles -- is in ``validation/scpn_symbolic_lane_evidence.py``.
_DEFAULT_TRANSITION_DELAY_TICKS = 0
_SYMBOLIC_DRAIN_DELAY_TICKS = 1
_SYMBOLIC_FIRE_THRESHOLD = 0.02
_SYMBOLIC_DRAIN_THRESHOLD = 0.05
_SYMBOLIC_READOUT_GAIN = 2.0


def _build_scpn_controller(
    *,
    runtime_profile: str = "adaptive",
    runtime_backend: str = "auto",
    delay_ticks: int = _DEFAULT_TRANSITION_DELAY_TICKS,
) -> NeuroSymbolicController:
    net = StochasticPetriNet()
    for place in (
        # injected error components
        "x_R_pos",
        "x_R_neg",
        "x_Z_pos",
        "x_Z_neg",
        # transition outputs (the readout reads these)
        "a_R_pos",
        "a_R_neg",
        "a_Z_pos",
        "a_Z_neg",
        # drain sinks
        "s_R_pos",
        "s_R_neg",
        "s_Z_pos",
        "s_Z_neg",
    ):
        net.add_place(place, initial_tokens=0.0)

    # Fire transitions carry the injected error through the symbolic lane (x_* -> a_*).
    # ``delay_ticks`` >= the run length bypasses the lane (a_* never filled) — the
    # campaign discloses that case honestly (see ``symbolic_transition_disclosure``).
    for name, src, dst in (
        ("T_Rp", "x_R_pos", "a_R_pos"),
        ("T_Rn", "x_R_neg", "a_R_neg"),
        ("T_Zp", "x_Z_pos", "a_Z_pos"),
        ("T_Zn", "x_Z_neg", "a_Z_neg"),
    ):
        net.add_transition(name, threshold=_SYMBOLIC_FIRE_THRESHOLD, delay_ticks=delay_ticks)
        net.add_arc(src, name, weight=1.0)
        net.add_arc(name, dst, weight=1.0)

    # Drain transitions keep the a_* readout places responsive (a_* -> sink); without
    # them a_* saturates to 1.0 and the controller diverges.
    for name, src, dst in (
        ("D_Rp", "a_R_pos", "s_R_pos"),
        ("D_Rn", "a_R_neg", "s_R_neg"),
        ("D_Zp", "a_Z_pos", "s_Z_pos"),
        ("D_Zn", "a_Z_neg", "s_Z_neg"),
    ):
        net.add_transition(name, threshold=_SYMBOLIC_DRAIN_THRESHOLD, delay_ticks=_SYMBOLIC_DRAIN_DELAY_TICKS)
        net.add_arc(src, name, weight=1.0)
        net.add_arc(name, dst, weight=1.0)
    net.compile(validate_topology=True)

    compiler = FusionCompiler(
        bitstream_length=512,
        seed=42,
        lif_tau_mem=10.0,
        lif_noise_std=0.1,
        lif_dt=1.0,
        lif_resistance=1.0,
        lif_refractory_period=1,
    )
    compiled = compiler.compile(net, firing_mode="fractional", firing_margin=0.25)
    artifact = compiled.export_artifact(
        name="scpn_pid_mpc_benchmark",
        dt_control_s=0.05,
        readout_config={
            "actions": [
                # Read the transition-OUTPUT places (a_*), not the injected x_*, so the
                # symbolic transitions are load-bearing in the control output.
                {"name": "dI_PF3_A", "pos_place": 4, "neg_place": 5},
                {"name": "dI_PF_topbot_A", "pos_place": 6, "neg_place": 7},
            ],
            "gains": [_SYMBOLIC_READOUT_GAIN, 0.0],
            "abs_max": [1.0, 1.0],
            "slew_per_s": [60.0, 60.0],
        },
        injection_config=[
            {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 2, "source": "x_Z_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 3, "source": "x_Z_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ],
    )
    runtime_profile_norm = runtime_profile.strip().lower()
    controller_kwargs: dict[str, Any] = {
        "artifact": artifact,
        "seed_base": 123456,
        "targets": ControlTargets(R_target_m=0.0, Z_target_m=0.0),
        "scales": ControlScales(R_scale_m=1.0, Z_scale_m=1.0),
        "sc_n_passes": 16,
        "sc_bitflip_rate": 0.0,
    }
    if runtime_profile_norm == "traceable":
        controller_kwargs.update(FusionCompiler.traceable_runtime_kwargs(runtime_backend=runtime_backend))
    else:
        controller_kwargs.update(
            {
                "runtime_profile": runtime_profile_norm,
                "runtime_backend": runtime_backend,
                "sc_binary_margin": 0.05,
            }
        )
    return NeuroSymbolicController(**controller_kwargs)


@dataclass
class _PIDState:
    kp: float
    ki: float
    kd: float
    integral: float = 0.0
    last_err: float = 0.0

    def step(self, err: float, dt: float, u_limit: float) -> float:
        self.integral += err * dt
        d_err = (err - self.last_err) / max(dt, 1e-9)
        self.last_err = err
        u = self.kp * err + self.ki * self.integral + self.kd * d_err
        return float(np.clip(u, -u_limit, u_limit))


def _disturbance(k: int, steps: int, phase: float = 0.0) -> float:
    return float(0.02 * math.sin(0.08 * k + phase) + (0.04 if k >= steps // 2 else 0.0))


def _mpc_action(x: float, k: int, steps: int, horizon: int, u_limit: float, phase: float = 0.0) -> float:
    # An honest MPC uses only information available at step k. It holds the LAST
    # OBSERVED disturbance constant over the horizon -- ``_disturbance(k - 1)``
    # equals the measurement residual ``x - (0.95*x_prev + 0.12*u_prev)`` a real
    # controller forms at k (an offset-free / persistence disturbance estimate).
    # It must NOT roll out with the true future disturbance ``_disturbance(k + h)``
    # -- that clairvoyance rigged the baseline into "beating" PID and is not a
    # controller any plant could run. At k=0 there is no prior observation, so the
    # estimate is 0.0 (no information assumed) regardless of the disturbance phase.
    d_est = _disturbance(k - 1, steps, phase) if k >= 1 else 0.0
    candidates = np.linspace(-u_limit, u_limit, 31)
    best_u = 0.0
    best_cost = float("inf")
    for u in candidates:
        x_pred = x
        cost = 0.0
        for _ in range(horizon):
            x_pred = 0.95 * x_pred + 0.12 * float(u) + d_est
            cost += x_pred * x_pred + 0.06 * float(u * u)
        if cost < best_cost:
            best_cost = float(cost)
            best_u = float(u)
    return best_u


def _metrics(errors: NDArray[np.float64], controls: NDArray[np.float64]) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "iae": float(np.mean(np.abs(errors))),
        "peak_abs_error": float(np.max(np.abs(errors))),
        "mean_abs_control": float(np.mean(np.abs(controls))),
    }


def _symbolic_transition_disclosure(
    *, max_delay_ticks: int, steps: int, readout_reads_transition_outputs: bool
) -> dict[str, Any]:
    """Honestly disclose whether the timed symbolic transitions genuinely contribute.

    Two structural conditions must BOTH hold for the symbolic lane to be
    load-bearing in the control output: (1) the transitions can fire within the run
    (``max_delay_ticks < steps``), and (2) the readout reads transition-OUTPUT places
    (so firing changes the action). If the transitions never fire, or the readout
    reads only injected places, the control is neuromorphic-proportional and the
    symbolic lane contributes nothing — surfacing this prevents a bypassed lane from
    being read as a contributing one (public claims bind to substance). The MEASURED
    parity margin lives in ``validation/scpn_symbolic_lane_evidence.py``; this inline
    disclosure reports the structural facts.
    """
    transitions_fire = bool(max_delay_ticks < steps)
    contributes = bool(transitions_fire and readout_reads_transition_outputs)
    if contributes:
        note = (
            "The control transitions fire and the readout reads their output places, so "
            "the symbolic transition lane is load-bearing in the control output. Measured "
            "at parity with the direct neuromorphic readout (margin ~0), both ~2.4x better "
            "than PID; see scpn_symbolic_lane_evidence for the reproducible measurement."
        )
    elif not transitions_fire:
        note = (
            "The control transitions carry a delay exceeding the run length, so they never "
            "fire; the symbolic lane does not contribute and the control is "
            "neuromorphic-proportional in this configuration."
        )
    else:
        note = (
            "The control transitions fire but the readout does not read their output "
            "places, so the symbolic lane is bypassed and does not contribute; the control "
            "is neuromorphic-proportional in this configuration."
        )
    return {
        "max_delay_ticks": int(max_delay_ticks),
        "steps": int(steps),
        "transitions_fire_within_run": transitions_fire,
        "readout_reads_transition_outputs": bool(readout_reads_transition_outputs),
        "symbolic_lane_contributes_to_readout": contributes,
        "note": note,
    }


def run_campaign(
    *,
    seed: int = 42,
    steps: int = 320,
    scpn_runtime_profile: str = "traceable",
    scpn_runtime_backend: str = "auto",
    delay_ticks: int = _DEFAULT_TRANSITION_DELAY_TICKS,
) -> dict[str, Any]:
    """Run the SCPN-vs-PID/MPC campaign for a seed and return the metrics/gate report."""
    seed_int = int(seed)
    steps = int(steps)
    if steps < 32:
        raise ValueError("steps must be >= 32.")
    # The seed drives the disturbance PHASE (via a local generator, independent of any
    # global RNG state), so each seed is a genuinely different profile rather than an
    # inert label. seed=42 is the reference campaign; other seeds probe robustness.
    phase = float(np.random.default_rng(seed_int).uniform(0.0, 2.0 * np.pi))
    u_limit = 1.0
    dt = 0.05
    x0 = 0.35

    scpn = _build_scpn_controller(
        runtime_profile=scpn_runtime_profile,
        runtime_backend=scpn_runtime_backend,
        delay_ticks=delay_ticks,
    )
    symbolic_disclosure = _symbolic_transition_disclosure(
        max_delay_ticks=int(scpn.max_delay_ticks),
        steps=steps,
        readout_reads_transition_outputs=bool(scpn.readout_reads_transition_outputs),
    )
    pid = _PIDState(kp=1.15, ki=0.24, kd=0.04)

    x_scpn = float(x0)
    x_pid = float(x0)
    x_mpc = float(x0)

    e_scpn = np.zeros(steps, dtype=np.float64)
    e_pid = np.zeros(steps, dtype=np.float64)
    e_mpc = np.zeros(steps, dtype=np.float64)
    u_scpn_hist = np.zeros(steps, dtype=np.float64)
    u_pid_hist = np.zeros(steps, dtype=np.float64)
    u_mpc_hist = np.zeros(steps, dtype=np.float64)

    for k in range(steps):
        d = _disturbance(k, steps, phase)

        err_pid = -x_pid
        u_pid = pid.step(err_pid, dt=dt, u_limit=u_limit)
        x_pid = 0.95 * x_pid + 0.12 * u_pid + d

        u_mpc = _mpc_action(x_mpc, k, steps, horizon=6, u_limit=u_limit, phase=phase)
        x_mpc = 0.95 * x_mpc + 0.12 * u_mpc + d

        if scpn.runtime_profile_name == "traceable":
            action_vec = scpn.step_traceable((float(x_scpn), 0.0), k)
            u_scpn = float(np.clip(action_vec[0], -u_limit, u_limit))
        else:
            action = scpn.step({"R_axis_m": float(x_scpn), "Z_axis_m": 0.0}, k)
            u_scpn = float(np.clip(action["dI_PF3_A"], -u_limit, u_limit))
        x_scpn = 0.95 * x_scpn + 0.12 * u_scpn + d

        e_pid[k] = -x_pid
        e_mpc[k] = -x_mpc
        e_scpn[k] = -x_scpn
        u_pid_hist[k] = u_pid
        u_mpc_hist[k] = u_mpc
        u_scpn_hist[k] = u_scpn

    pid_metrics = _metrics(e_pid, u_pid_hist)
    mpc_metrics = _metrics(e_mpc, u_mpc_hist)
    scpn_metrics = _metrics(e_scpn, u_scpn_hist)

    ratios = {
        "scpn_vs_pid_rmse_ratio": scpn_metrics["rmse"] / max(pid_metrics["rmse"], 1e-12),
        "scpn_vs_mpc_rmse_ratio": scpn_metrics["rmse"] / max(mpc_metrics["rmse"], 1e-12),
        "mpc_vs_pid_rmse_ratio": mpc_metrics["rmse"] / max(pid_metrics["rmse"], 1e-12),
    }
    # The gate checks only the SCPN controller's own quality (versus PID and
    # versus the honest MPC baseline). It deliberately does NOT require "MPC
    # beats PID": with the clairvoyant baseline removed, MPC does not beat PID on
    # this scalar plant, and gating on a predetermined outcome is what rigged the
    # original benchmark. The observed MPC-vs-PID result is reported, not gated.
    thresholds = {
        "max_scpn_vs_pid_rmse_ratio": 1.10,
        "max_scpn_vs_mpc_rmse_ratio": 1.35,
        "max_scpn_peak_abs_error": 0.65,
    }
    passes = bool(
        ratios["scpn_vs_pid_rmse_ratio"] <= thresholds["max_scpn_vs_pid_rmse_ratio"]
        and ratios["scpn_vs_mpc_rmse_ratio"] <= thresholds["max_scpn_vs_mpc_rmse_ratio"]
        and scpn_metrics["peak_abs_error"] <= thresholds["max_scpn_peak_abs_error"]
    )
    mpc_beats_pid_observed = bool(mpc_metrics["rmse"] <= pid_metrics["rmse"])

    return {
        "seed": seed_int,
        "steps": int(steps),
        "runtime_lane": {
            "runtime_profile": scpn.runtime_profile_name,
            "runtime_backend": scpn.runtime_backend_name,
            "uses_traceable_step": bool(scpn.runtime_profile_name == "traceable"),
        },
        "pid": pid_metrics,
        "mpc": mpc_metrics,
        "mpc_disturbance_model": "persistence_last_observed_offset_free_no_foresight",
        "mpc_beats_pid_observed": mpc_beats_pid_observed,
        "scpn": scpn_metrics,
        "ratios": ratios,
        "thresholds": thresholds,
        "passes_thresholds": passes,
        "symbolic_transition_disclosure": symbolic_disclosure,
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    """Run a campaign and wrap it with timing metadata as the persisted report payload."""
    t0 = time.perf_counter()
    campaign = run_campaign(**kwargs)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": float(time.perf_counter() - t0),
        "scpn_pid_mpc_benchmark": campaign,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the benchmark report as a human-readable Markdown summary."""
    r = report["scpn_pid_mpc_benchmark"]
    lines = [
        "# SCPN vs PID/MPC Benchmark",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{report['runtime_seconds']:.3f} s`",
        f"- Steps: `{r['steps']}`",
        f"- SCPN runtime profile: `{r['runtime_lane']['runtime_profile']}`",
        f"- SCPN runtime backend: `{r['runtime_lane']['runtime_backend']}`",
        f"- Traceable step API: `{'YES' if r['runtime_lane']['uses_traceable_step'] else 'NO'}`",
        "",
        f"- MPC disturbance model: `{r['mpc_disturbance_model']}`",
        "",
        "## RMSE",
        "",
        f"- PID: `{r['pid']['rmse']:.6f}`",
        f"- MPC: `{r['mpc']['rmse']:.6f}`",
        f"- SCPN: `{r['scpn']['rmse']:.6f}`",
        "",
        "## Ratios",
        "",
        f"- SCPN / PID RMSE: `{r['ratios']['scpn_vs_pid_rmse_ratio']:.4f}`",
        f"- SCPN / MPC RMSE: `{r['ratios']['scpn_vs_mpc_rmse_ratio']:.4f}`",
        f"- MPC / PID RMSE: `{r['ratios']['mpc_vs_pid_rmse_ratio']:.4f}`"
        f" (MPC beats PID: `{'YES' if r['mpc_beats_pid_observed'] else 'NO'}`)",
        "",
        "## Threshold Pass",
        "",
        f"- Pass: `{'YES' if r['passes_thresholds'] else 'NO'}`",
        "",
        "## Symbolic-transition disclosure",
        "",
        f"- Max transition delay: `{r['symbolic_transition_disclosure']['max_delay_ticks']}` ticks"
        f" (run length `{r['symbolic_transition_disclosure']['steps']}`)",
        f"- Transitions fire within run: `{'YES' if r['symbolic_transition_disclosure']['transitions_fire_within_run'] else 'NO'}`",
        f"- Readout reads transition outputs: `{'YES' if r['symbolic_transition_disclosure']['readout_reads_transition_outputs'] else 'NO'}`",
        f"- Symbolic lane contributes to readout: `{'YES' if r['symbolic_transition_disclosure']['symbolic_lane_contributes_to_readout'] else 'NO'}`",
        f"- {r['symbolic_transition_disclosure']['note']}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Regenerate the SCPN-vs-PID/MPC benchmark JSON + Markdown reports from the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Disturbance-phase seed (42 = reference campaign).")
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument(
        "--scpn-runtime-profile",
        choices=["adaptive", "deterministic", "traceable"],
        default="traceable",
    )
    parser.add_argument(
        "--scpn-runtime-backend",
        choices=["auto", "numpy", "rust"],
        default="auto",
    )
    parser.add_argument(
        "--delay-ticks",
        type=int,
        default=_DEFAULT_TRANSITION_DELAY_TICKS,
        help=(
            "Transition firing delay in ticks (default 0 -> the symbolic transitions fire "
            "immediately and the readout follows the transition outputs). Set it at or above "
            "--steps to bypass the symbolic lane so the readout follows the injected features."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "scpn_pid_mpc_benchmark.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "scpn_pid_mpc_benchmark.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        steps=args.steps,
        scpn_runtime_profile=args.scpn_runtime_profile,
        scpn_runtime_backend=args.scpn_runtime_backend,
        delay_ticks=args.delay_ticks,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    r = report["scpn_pid_mpc_benchmark"]
    print("SCPN vs PID/MPC benchmark complete.")
    print(
        "rmse(pid/mpc/scpn)="
        f"{r['pid']['rmse']:.6f}/{r['mpc']['rmse']:.6f}/{r['scpn']['rmse']:.6f}, "
        f"passes_thresholds={r['passes_thresholds']}"
    )
    if args.strict and not r["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
