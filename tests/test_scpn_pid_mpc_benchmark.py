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
from pathlib import Path
import sys

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
    a = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=180)
    b = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=180)
    assert a["pid"]["rmse"] == b["pid"]["rmse"]
    assert a["mpc"]["rmse"] == b["mpc"]["rmse"]
    assert a["scpn"]["rmse"] == b["scpn"]["rmse"]
    assert a["ratios"]["scpn_vs_pid_rmse_ratio"] == b["ratios"]["scpn_vs_pid_rmse_ratio"]
    assert a["ratios"]["scpn_vs_mpc_rmse_ratio"] == b["ratios"]["scpn_vs_mpc_rmse_ratio"]


def test_campaign_meets_thresholds_smoke() -> None:
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
    controller = scpn_pid_mpc_benchmark._build_scpn_controller()
    assert getattr(controller, "_sc_binary_margin", 0.0) > 0.0


def test_traceable_runtime_lane_is_exposed_and_deterministic() -> None:
    a = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=160, scpn_runtime_profile="traceable")
    b = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=160, scpn_runtime_profile="traceable")
    assert a["runtime_lane"]["runtime_profile"] == "traceable"
    assert a["runtime_lane"]["uses_traceable_step"] is True
    assert a["scpn"]["rmse"] == b["scpn"]["rmse"]


def test_render_markdown_contains_sections() -> None:
    report = scpn_pid_mpc_benchmark.generate_report(seed=11, steps=120)
    text = scpn_pid_mpc_benchmark.render_markdown(report)
    assert "# SCPN vs PID/MPC Benchmark" in text
    assert "SCPN runtime profile" in text
    assert "RMSE" in text
    assert "Ratios" in text
    assert "Threshold Pass" in text


def test_campaign_does_not_mutate_global_numpy_rng_state() -> None:
    np.random.seed(531)
    state = np.random.get_state()

    scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=120)

    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected


@pytest.mark.parametrize("steps", [0, 16, 31])
def test_campaign_rejects_invalid_steps(steps: int) -> None:
    with pytest.raises(ValueError, match="steps"):
        scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=steps)


def test_symbolic_transition_disclosure_is_honest_at_shipped_config() -> None:
    """At the shipped config the timed transitions never fire; the report says so.

    delay_ticks defaults to 1024 while a campaign runs far fewer steps, so the four
    control transitions cannot fire and the readout is neuromorphic-proportional on
    the injected features. The disclosure must surface this so the symbolic lane is
    not read as contributing (public claims bind to substance).
    """
    out = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=320)
    disc = out["symbolic_transition_disclosure"]
    assert disc["max_delay_ticks"] == 1024
    assert disc["steps"] == 320
    assert disc["transitions_fire_within_run"] is False
    assert disc["symbolic_lane_contributes_to_readout"] is False
    assert "neuromorphic-proportional" in disc["note"]


def test_symbolic_transition_disclosure_reflects_firing_when_delay_below_run() -> None:
    """With delay below the run length the transitions fire; the disclosure flips."""
    out = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=180, delay_ticks=2)
    disc = out["symbolic_transition_disclosure"]
    assert disc["max_delay_ticks"] == 2
    assert disc["transitions_fire_within_run"] is True
    assert disc["symbolic_lane_contributes_to_readout"] is True
    assert "participates" in disc["note"]


@pytest.mark.parametrize("delay_ticks", [0, 2, 179, 180, 1024])
def test_disclosure_matches_reality_gate(delay_ticks: int) -> None:
    """Honesty gate: the disclosure must equal the true fire/no-fire condition.

    The report may only claim a symbolic contribution when the transitions can
    actually fire (delay < run length). This forbids a static (never-firing)
    symbolic lane from being presented as a contributing one.
    """
    out = scpn_pid_mpc_benchmark.run_campaign(seed=42, steps=180, delay_ticks=delay_ticks)
    disc = out["symbolic_transition_disclosure"]
    expected_fire = disc["max_delay_ticks"] < disc["steps"]
    assert disc["transitions_fire_within_run"] is expected_fire
    assert disc["symbolic_lane_contributes_to_readout"] is expected_fire


def test_render_markdown_contains_symbolic_disclosure() -> None:
    report = scpn_pid_mpc_benchmark.generate_report(seed=11, steps=120)
    text = scpn_pid_mpc_benchmark.render_markdown(report)
    assert "Symbolic-transition disclosure" in text
    assert "Transitions fire within run" in text
    assert "Symbolic lane contributes to readout" in text
