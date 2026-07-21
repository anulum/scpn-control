# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Symbolic-transition control-lane evidence
"""Reproducible evidence that the SCPN symbolic-transition lane is load-bearing.

The ``scpn_pid_mpc_benchmark`` SCPN controller routes control THROUGH the timed
Petri-net transitions: the readout reads the transition-output ``a_*`` places, which
are filled by transitions firing (``x_*`` -> ``a_*``) and kept responsive by a drain
(``a_*`` -> sink). This module measures, reproducibly, three things a public
neuro-symbolic claim must bind to:

1. **Parity** — the genuinely-symbolic controller tracks at parity with the direct
   neuromorphic-proportional readout (readout reads the injected ``x_*`` places),
   both far better than PID. Margin ~0; NOT superiority.
2. **Robustness** — the parity holds on UNSEEN disturbance seeds and run lengths
   (the tuned constants are not overfit to the benchmark's seed=42/steps=320).
3. **Load-bearing** — removing the drain (so ``a_*`` saturates) makes the symbolic
   controller DIVERGE, proving the transitions genuinely carry the control signal
   rather than being a bypassed branch.

Run as a script to (re)generate ``validation/reports/scpn_symbolic_lane_evidence.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.contracts import ControlScales, ControlTargets
from scpn_control.scpn.controller import NeuroSymbolicController
from scpn_control.scpn.structure import StochasticPetriNet
from validation.scpn_pid_mpc_benchmark import (
    _SYMBOLIC_FIRE_THRESHOLD,
    _SYMBOLIC_READOUT_GAIN,
    _build_scpn_controller,
    _PIDState,
)

_SCHEMA = "scpn-control.symbolic-lane-evidence.v1"
# The plant + disturbance mirror scpn_pid_mpc_benchmark (a linear scalar plant with a
# sinusoid + mid-run step disturbance); the seed shifts the disturbance PHASE so a seed
# is a genuinely different profile, not just a different noise draw. seed=42 is the
# TUNING seed (in-sample); 7/123/999 are UNSEEN (out-of-sample). PID is measured on the
# SAME per-seed profile as symbolic/neuromorphic so every ratio is apples-to-apples.
_TUNING_SEED = 42
_UNSEEN_SEEDS = (7, 123, 999)
_PARITY_BAND = 0.15  # |symbolic/neuromorphic - 1| must stay within this on unseen seeds
_PID_GAINS = {"kp": 1.15, "ki": 0.24, "kd": 0.04}  # matches scpn_pid_mpc_benchmark


def _disturbance(k: int, steps: int, phase: float) -> float:
    return float(0.02 * np.sin(0.08 * k + phase) + (0.04 if k >= steps // 2 else 0.0))


# The neuromorphic baseline is the pre-symbolic shipped config: readout reads the
# injected x_* places, transitions bypassed, at ITS own tuned gain (5.0). The honest
# comparison is each controller at its own best: symbolic (gain 2.0) vs neuromorphic
# (gain 5.0) -> parity. Using a shared gain would handicap one and is not the claim.
_NEUROMORPHIC_READOUT_GAIN = 5.0


def _build_variant(variant: str) -> NeuroSymbolicController:
    """Build a controller variant: 'symbolic', 'neuromorphic', or 'symbolic_undrained'.

    'symbolic' is the ACTUAL production benchmark controller (imported), so the parity
    claim binds to the shipped configuration. 'neuromorphic' reproduces the pre-symbolic
    baseline: readout reads the injected ``x_*`` places, transitions bypassed, gain 5.0.
    'symbolic_undrained' reads ``a_*`` with firing transitions but no drain (a_* saturates).
    """
    if variant == "symbolic":
        return _build_scpn_controller(runtime_profile="traceable")

    net = StochasticPetriNet()
    # Both non-production variants use x_* (injected) + a_* (transition outputs); neither
    # drains, so no sink places are needed (neuromorphic never fires; undrained saturates).
    for place in ("x_R_pos", "x_R_neg", "x_Z_pos", "x_Z_neg", "a_R_pos", "a_R_neg", "a_Z_pos", "a_Z_neg"):
        net.add_place(place, initial_tokens=0.0)
    # neuromorphic: bypass the transitions (delay exceeds any run); undrained: fire (delay 0).
    fire_delay = 10_000 if variant == "neuromorphic" else 0
    for name, src, dst in (
        ("T_Rp", "x_R_pos", "a_R_pos"),
        ("T_Rn", "x_R_neg", "a_R_neg"),
        ("T_Zp", "x_Z_pos", "a_Z_pos"),
        ("T_Zn", "x_Z_neg", "a_Z_neg"),
    ):
        net.add_transition(name, threshold=_SYMBOLIC_FIRE_THRESHOLD, delay_ticks=fire_delay)
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
    # neuromorphic reads the injected x_* (0-3) at gain 5.0; undrained reads a_* (4-7) at gain 2.0.
    if variant == "neuromorphic":
        pos, neg, gain = (0, 2), (1, 3), _NEUROMORPHIC_READOUT_GAIN
    else:
        pos, neg, gain = (4, 6), (5, 7), _SYMBOLIC_READOUT_GAIN
    artifact = compiled.export_artifact(
        name="symbolic_lane_evidence",
        dt_control_s=0.05,
        readout_config={
            "actions": [
                {"name": "dI_PF3_A", "pos_place": pos[0], "neg_place": neg[0]},
                {"name": "dI_PF_topbot_A", "pos_place": pos[1], "neg_place": neg[1]},
            ],
            "gains": [gain, 0.0],
            "abs_max": [1.0, 1.0],
            "slew_per_s": [60.0, 60.0],
        },
        injection_config=[
            {"place_id": i, "source": s, "scale": 1.0, "offset": 0.0, "clamp_0_1": True}
            for i, s in enumerate(("x_R_pos", "x_R_neg", "x_Z_pos", "x_Z_neg"))
        ],
    )
    return NeuroSymbolicController(
        artifact=artifact,
        seed_base=123456,
        targets=ControlTargets(R_target_m=0.0, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=1.0, Z_scale_m=1.0),
        sc_n_passes=16,
        sc_bitflip_rate=0.0,
        **FusionCompiler.traceable_runtime_kwargs(runtime_backend="auto"),
    )


def _tracking_rmse(variant: str, seed: int, steps: int) -> float:
    """Closed-loop tracking-error RMSE for a controller variant on a seeded disturbance."""
    scpn = _build_variant(variant)
    phase = float(np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi))
    x = 0.35
    e = np.zeros(steps, dtype=np.float64)
    for k in range(steps):
        u = float(np.clip(scpn.step_traceable((float(x), 0.0), k)[0], -1.0, 1.0))
        x = 0.95 * x + 0.12 * u + _disturbance(k, steps, phase)
        e[k] = -x
    return float(np.sqrt(np.mean(e * e)))


def _pid_tracking_rmse(seed: int, steps: int) -> float:
    """PID closed-loop RMSE on the SAME seeded disturbance profile (apples-to-apples)."""
    pid = _PIDState(**_PID_GAINS)
    phase = float(np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi))
    x = 0.35
    e = np.zeros(steps, dtype=np.float64)
    for k in range(steps):
        u = pid.step(-x, dt=0.05, u_limit=1.0)
        x = 0.95 * x + 0.12 * u + _disturbance(k, steps, phase)
        e[k] = -x
    return float(np.sqrt(np.mean(e * e)))


def _measure_cohort(seeds: tuple[int, ...], step_counts: tuple[int, ...]) -> list[dict[str, Any]]:
    """Measure symbolic, neuromorphic, and PID RMSE per (seed, steps) on the same profile."""
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for steps in step_counts:
            sym = _tracking_rmse("symbolic", seed, steps)
            neuro = _tracking_rmse("neuromorphic", seed, steps)
            pid = _pid_tracking_rmse(seed, steps)
            rows.append(
                {
                    "seed": seed,
                    "steps": steps,
                    "symbolic_rmse": sym,
                    "neuromorphic_rmse": neuro,
                    "pid_rmse": pid,
                    "ratio_symbolic_over_neuromorphic": sym / neuro,
                    "ratio_symbolic_over_pid": sym / pid,
                    "ratio_neuromorphic_over_pid": neuro / pid,
                }
            )
    return rows


def evaluate_evidence(
    *,
    tuning_seed: int = _TUNING_SEED,
    unseen_seeds: tuple[int, ...] = _UNSEEN_SEEDS,
    step_counts: tuple[int, ...] = (240, 320),
) -> dict[str, Any]:
    """Measure parity, out-of-sample robustness, and load-bearing evidence.

    The tuned constants were fit on ``tuning_seed`` (in-sample); ``unseen_seeds`` are
    out-of-sample. The parity claim is judged on the UNSEEN cohort so it is not
    contaminated by the tuning point. PID is measured on the SAME per-seed profile as
    symbolic/neuromorphic, so ``ratio_symbolic_over_pid`` is apples-to-apples (unlike a
    fixed PID constant from another campaign).
    """
    in_sample = _measure_cohort((tuning_seed,), step_counts)
    unseen = _measure_cohort(unseen_seeds, step_counts)

    unseen_ratios = np.asarray([r["ratio_symbolic_over_neuromorphic"] for r in unseen], dtype=np.float64)
    sym_over_pid = np.asarray(
        [r["ratio_symbolic_over_pid"] for r in in_sample + unseen], dtype=np.float64
    )

    undrained = _tracking_rmse("symbolic_undrained", tuning_seed, 320)
    neuro_ref = _tracking_rmse("neuromorphic", tuning_seed, 320)
    return {
        "schema": _SCHEMA,
        "plant": "linear scalar x' = 0.95 x + 0.12 u + d(k); sinusoid + mid-run step disturbance",
        "metric": "closed-loop tracking-error RMSE (e = -x)",
        "readout_contract": "symbolic reads transition-output a_* (4-7); neuromorphic reads injected x_* (0-3)",
        "tuning_seed": int(tuning_seed),
        "unseen_seeds": list(unseen_seeds),
        "in_sample": in_sample,
        "robustness_unseen": unseen,
        "ratio_symbolic_over_neuromorphic_unseen": {
            "min": float(unseen_ratios.min()),
            "max": float(unseen_ratios.max()),
            "mean": float(unseen_ratios.mean()),
        },
        "parity_holds": bool(np.all(np.abs(unseen_ratios - 1.0) <= _PARITY_BAND)),
        "symbolic_vs_pid": {
            "ratio_symbolic_over_pid_mean": float(sym_over_pid.mean()),
            "symbolic_better_than_pid": bool(np.all(sym_over_pid < 1.0)),
            "note": "PID measured on the SAME per-seed disturbance profile (apples-to-apples).",
        },
        "load_bearing": {
            "symbolic_undrained_rmse": undrained,
            "neuromorphic_reference_rmse": neuro_ref,
            "undrained_over_reference": float(undrained / neuro_ref),
            "diverges_without_drain": bool(undrained > 3.0 * neuro_ref),
            "note": (
                "Removing the a_* drain saturates the readout places and the symbolic "
                "controller diverges, proving the transitions carry the control signal."
            ),
        },
        "claim": (
            "The symbolic transition lane produces control AT PARITY with the direct "
            "neuromorphic readout on OUT-OF-SAMPLE seeds (margin ~0), is load-bearing "
            "(diverges without its drain), and beats PID on the same profiles. NOT a "
            "superiority claim over the neuromorphic readout."
        ),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    """Return the evidence report with a self-consistent ``payload_sha256``."""
    report = evaluate_evidence(**kwargs)
    payload = dict(report)
    payload["payload_sha256"] = None
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    report["payload_sha256"] = digest
    return report


def main(argv: list[str] | None = None) -> int:
    """Regenerate the symbolic-lane evidence report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json", default=str(ROOT / "validation" / "reports" / "scpn_symbolic_lane_evidence.json")
    )
    args = parser.parse_args(argv)
    report = generate_report()
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    r = report["ratio_symbolic_over_neuromorphic_unseen"]
    print("Symbolic-lane evidence complete.")
    print(
        f"  parity (UNSEEN) symbolic/neuromorphic: min={r['min']:.3f} mean={r['mean']:.3f} max={r['max']:.3f}"
        f" -> parity_holds={report['parity_holds']}"
    )
    sp = report["symbolic_vs_pid"]
    print(
        f"  symbolic/PID (same profiles): mean={sp['ratio_symbolic_over_pid_mean']:.3f}"
        f" -> symbolic_better_than_pid={sp['symbolic_better_than_pid']}"
    )
    lb = report["load_bearing"]
    print(
        f"  load-bearing: undrained/reference={lb['undrained_over_reference']:.2f}x"
        f" -> diverges_without_drain={lb['diverges_without_drain']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
