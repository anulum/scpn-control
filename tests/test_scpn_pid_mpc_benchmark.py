# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Scpn Pid Mpc Benchmark
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — SCPN vs PID/MPC Benchmark Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for deterministic SCPN-vs-PID-vs-MPC benchmark lane."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "scpn_pid_mpc_benchmark.py"
SPEC = importlib.util.spec_from_file_location("scpn_pid_mpc_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
scpn_pid_mpc_benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scpn_pid_mpc_benchmark
SPEC.loader.exec_module(scpn_pid_mpc_benchmark)


def test_campaign_is_deterministic_for_seed() -> None:
    """The campaign is deterministic: the same seed yields identical metrics across runs."""
    a = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=180)
    b = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=180)
    assert a["pid"]["rmse"] == b["pid"]["rmse"]
    assert a["mpc"]["rmse"] == b["mpc"]["rmse"]
    assert a["scpn"]["rmse"] == b["scpn"]["rmse"]
    assert a["ratios"]["scpn_vs_pid_rmse_ratio"] == b["ratios"]["scpn_vs_pid_rmse_ratio"]
    assert a["ratios"]["scpn_vs_mpc_rmse_ratio"] == b["ratios"]["scpn_vs_mpc_rmse_ratio"]


def test_campaign_meets_thresholds_smoke() -> None:
    """The reference campaign passes its quality gate and both ratios stay within threshold."""
    out = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=240)
    assert out["passes_thresholds"] is True
    assert out["ratios"]["scpn_vs_pid_rmse_ratio"] <= out["thresholds"]["max_scpn_vs_pid_rmse_ratio"]
    assert out["ratios"]["scpn_vs_mpc_rmse_ratio"] <= out["thresholds"]["max_scpn_vs_mpc_rmse_ratio"]


def test_mpc_baseline_uses_no_future_disturbance_foresight() -> None:
    """The MPC baseline must not roll out with the true future disturbance.

    The original benchmark rigged the MPC by predicting with ``_disturbance(k + h)``
    (perfect foresight). Pin the honest, no-foresight model and record the MPC-vs-PID
    outcome as an observation rather than a gated requirement.
    """
    out = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=240)
    assert out["mpc_disturbance_model"] == "persistence_last_observed_offset_free_no_foresight"
    assert "mpc_beats_pid_observed" in out
    assert "require_mpc_beats_pid" not in out["thresholds"]


def test_campaign_controller_uses_nonzero_binary_margin() -> None:
    """The compiled controller carries a nonzero stochastic-computing binary margin."""
    controller = scpn_pid_mpc_benchmark._build_scpn_controller()
    assert getattr(controller, "_sc_binary_margin", 0.0) > 0.0


def test_traceable_runtime_lane_is_exposed_and_deterministic() -> None:
    """The traceable runtime lane is surfaced in the report and is deterministic per seed."""
    a = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=160, scpn_runtime_profile="traceable")
    b = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=160, scpn_runtime_profile="traceable")
    assert a["runtime_lane"]["runtime_profile"] == "traceable"
    assert a["runtime_lane"]["uses_traceable_step"] is True
    assert a["scpn"]["rmse"] == b["scpn"]["rmse"]


def test_render_markdown_contains_sections() -> None:
    """The rendered Markdown report contains the expected headline sections."""
    report = scpn_pid_mpc_benchmark.generate_report(seed=11, steps=120)
    text = scpn_pid_mpc_benchmark.render_markdown(report)
    assert "# SCPN vs PID/MPC Benchmark" in text
    assert "SCPN runtime profile" in text
    assert "RMSE" in text
    assert "Ratios" in text
    assert "Threshold Pass" in text


def test_campaign_does_not_mutate_global_numpy_rng_state() -> None:
    """The campaign seeds a local generator and leaves the global numpy RNG state untouched."""
    np.random.seed(531)
    state = np.random.get_state()

    scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=120)

    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected


@pytest.mark.parametrize("steps", [0, 16, 31])
def test_campaign_rejects_invalid_steps(steps: int) -> None:
    """A campaign shorter than the minimum horizon is rejected."""
    with pytest.raises(ValueError, match="steps"):
        scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=steps)


def test_symbolic_lane_is_genuinely_load_bearing_at_shipped_config() -> None:
    """The shipped config routes control THROUGH the transitions and discloses it honestly.

    The readout reads the transition-output a_* places and the fire transitions fire
    (fire delay 0; max_delay_ticks = the drain delay 1 < steps), so the symbolic lane is
    load-bearing. Measured at parity with the neuromorphic readout (see
    scpn_symbolic_lane_evidence); the disclosure states parity + load-bearing, not superiority.
    """
    out = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=320)
    disc = out["symbolic_transition_disclosure"]
    assert disc["max_delay_ticks"] == 1  # drain delay dominates (fire delay 0)
    assert disc["steps"] == 320
    assert disc["transitions_fire_within_run"] is True
    assert disc["readout_reads_transition_outputs"] is True
    assert disc["symbolic_lane_contributes_to_readout"] is True
    assert "parity" in disc["note"] and "load-bearing" in disc["note"]
    assert "superior" not in disc["note"].lower()


def test_disclosure_reports_bypass_when_transitions_cannot_fire() -> None:
    """Pushing the fire delay beyond the run length bypasses the symbolic lane; disclosed."""
    out = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=180, delay_ticks=1024)
    disc = out["symbolic_transition_disclosure"]
    assert disc["max_delay_ticks"] == 1024
    assert disc["transitions_fire_within_run"] is False
    assert disc["symbolic_lane_contributes_to_readout"] is False
    assert "neuromorphic-proportional" in disc["note"]


@pytest.mark.parametrize("delay_ticks", [0, 2, 179, 180, 1024])
def test_disclosure_honesty_gate_contributes_iff_fires_and_reads_outputs(delay_ticks: int) -> None:
    """Honesty gate: contributes IFF the transitions fire AND the readout reads their outputs.

    Forbids presenting a non-firing or bypassed symbolic lane as contributing, and — by
    reporting structural facts rather than a numeric margin — cannot smuggle a superiority
    claim through the report.
    """
    out = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=180, delay_ticks=delay_ticks)
    disc = out["symbolic_transition_disclosure"]
    expected_fire = disc["max_delay_ticks"] < disc["steps"]
    assert disc["transitions_fire_within_run"] is expected_fire
    expected_contributes = bool(expected_fire and disc["readout_reads_transition_outputs"])
    assert disc["symbolic_lane_contributes_to_readout"] is expected_contributes
    assert "superior" not in disc["note"].lower()


def test_symbolic_disclosure_covers_all_three_honest_states() -> None:
    """The disclosure distinguishes contributing / never-firing / firing-but-bypassed."""
    disc = scpn_pid_mpc_benchmark._symbolic_transition_disclosure
    contributing = disc(max_delay_ticks=1, steps=180, readout_reads_transition_outputs=True)
    assert contributing["symbolic_lane_contributes_to_readout"] is True
    assert "parity" in contributing["note"] and "load-bearing" in contributing["note"]

    never_fires = disc(max_delay_ticks=1024, steps=180, readout_reads_transition_outputs=True)
    assert never_fires["symbolic_lane_contributes_to_readout"] is False
    assert "never" in never_fires["note"]

    # Fires but the readout reads only injected places -> bypassed (the pre-redesign shape).
    bypassed = disc(max_delay_ticks=1, steps=180, readout_reads_transition_outputs=False)
    assert bypassed["transitions_fire_within_run"] is True
    assert bypassed["symbolic_lane_contributes_to_readout"] is False
    assert "bypassed" in bypassed["note"]


def test_render_markdown_contains_symbolic_disclosure() -> None:
    """The rendered Markdown surfaces the symbolic-transition disclosure fields."""
    report = scpn_pid_mpc_benchmark.generate_report(seed=11, steps=120)
    text = scpn_pid_mpc_benchmark.render_markdown(report)
    assert "Symbolic-transition disclosure" in text
    assert "Transitions fire within run" in text
    assert "Readout reads transition outputs" in text
    assert "Symbolic lane contributes to readout" in text


def test_seed_selects_a_distinct_disturbance_profile() -> None:
    """The seed drives the disturbance phase, so different seeds give different profiles.

    (Regression: the seed used to be an inert label that never influenced any number.)
    """
    a = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=200)
    b = scpn_pid_mpc_benchmark.run_campaign(seed=7, steps=200)
    assert a["scpn"]["rmse"] != b["scpn"]["rmse"]
    assert a["pid"]["rmse"] != b["pid"]["rmse"]
    assert a["seed"] == 42 and b["seed"] == 7


def test_non_traceable_runtime_profile_runs_the_step_lane() -> None:
    """A non-traceable profile exercises the dict-based ``scpn.step`` control lane."""
    out = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=120, scpn_runtime_profile="deterministic")
    assert out["runtime_lane"]["runtime_profile"] == "deterministic"
    assert out["runtime_lane"]["uses_traceable_step"] is False
    assert out["scpn"]["rmse"] >= 0.0


def test_main_regenerates_benchmark_reports(tmp_path: Path) -> None:
    """``main`` writes JSON + Markdown reports and reports the gate outcome (exit 0)."""
    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"
    rc = scpn_pid_mpc_benchmark.main(
        ["--seed", "42", "--steps", "120", "--output-json", str(out_json), "--output-md", str(out_md)]
    )
    assert rc == 0
    assert out_json.exists() and out_md.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["scpn_pid_mpc_benchmark"]["seed"] == 42
    assert "# SCPN vs PID/MPC Benchmark" in out_md.read_text(encoding="utf-8")


def test_main_strict_flag_fails_when_gate_does_not_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--strict`` turns a failing gate into a nonzero exit code."""
    baseline = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=120)
    failing = {**baseline, "passes_thresholds": False}
    monkeypatch.setattr(scpn_pid_mpc_benchmark, "run_campaign", lambda **_kw: failing)
    rc = scpn_pid_mpc_benchmark.main(
        ["--strict", "--steps", "120", "--output-json", str(tmp_path / "b.json"), "--output-md", str(tmp_path / "b.md")]
    )
    assert rc == 2
