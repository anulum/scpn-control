# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller log_path + passthrough feature dict tests
"""Regression tests for _build_feature_dict passthrough loop (lines 542-545),
triggered only when log_path is not None."""

from __future__ import annotations

import json

import pytest

from scpn_control.scpn.artifact import load_artifact, save_artifact
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.controller import (
    ControlScales,
    ControlTargets,
    NeuroSymbolicController,
)


def _artifact_with_passthrough(tmp_path, petri_net_std):
    """Create artifact with a passthrough injection source 'extra_sensor'."""
    compiler = FusionCompiler(bitstream_length=64, seed=0)
    compiled = compiler.compile(petri_net_std)
    injection_config = [
        {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 2, "source": "x_Z_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 3, "source": "x_Z_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 4, "source": "extra_sensor", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
    ]
    readout_config = {
        "actions": [
            {"name": "dI_PF3_A", "pos_place": 4, "neg_place": 5},
            {"name": "dI_topbot", "pos_place": 6, "neg_place": 7},
        ],
        "gains": [1000.0, 1000.0],
        "abs_max": [5000.0, 5000.0],
        "slew_per_s": [1e6, 1e6],
    }
    art = compiled.export_artifact(
        name="test-passthrough",
        readout_config=readout_config,
        injection_config=injection_config,
    )
    path = tmp_path / "passthrough.scpnctl.json"
    save_artifact(art, str(path))
    return str(path)


class TestLogPathPassthrough:
    def test_step_with_log_path_covers_passthrough(self, tmp_path, petri_net_std):
        """step(log_path=...) triggers _build_feature_dict passthrough (542-545)."""
        art_path = _artifact_with_passthrough(tmp_path, petri_net_std)
        art = load_artifact(art_path)
        ctrl = NeuroSymbolicController(
            artifact=art,
            seed_base=42,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            sc_n_passes=1,
            runtime_backend="numpy",
        )
        log_file = tmp_path / "log.jsonl"
        obs = {"R_axis_m": 6.2, "Z_axis_m": 0.0, "extra_sensor": 0.5}
        actions = ctrl.step(obs, k=0, log_path=str(log_file), log_root=tmp_path)
        assert isinstance(actions, dict)
        assert "dI_PF3_A" in actions
        assert log_file.exists()
        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert "features" in entry
        assert "extra_sensor" in entry["features"]

    def test_step_without_log_path_produces_actions(self, tmp_path, petri_net_std):
        """step() without log_path still produces valid actions."""
        art_path = _artifact_with_passthrough(tmp_path, petri_net_std)
        art = load_artifact(art_path)
        ctrl = NeuroSymbolicController(
            artifact=art,
            seed_base=42,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            sc_n_passes=1,
            runtime_backend="numpy",
        )
        obs = {"R_axis_m": 6.2, "Z_axis_m": 0.0, "extra_sensor": 0.5}
        actions = ctrl.step(obs, k=0)
        assert isinstance(actions, dict)
        assert "dI_PF3_A" in actions

    def test_log_path_multiple_steps_appends(self, tmp_path, petri_net_std):
        """Multiple steps append to the same log file."""
        art_path = _artifact_with_passthrough(tmp_path, petri_net_std)
        art = load_artifact(art_path)
        ctrl = NeuroSymbolicController(
            artifact=art,
            seed_base=42,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            sc_n_passes=1,
            runtime_backend="numpy",
        )
        log_file = tmp_path / "log_multi.jsonl"
        obs = {"R_axis_m": 6.25, "Z_axis_m": 0.01, "extra_sensor": 0.3}
        for k in range(3):
            ctrl.step(obs, k=k, log_path=str(log_file), log_root=tmp_path)
        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) == 3
        for line in lines:
            entry = json.loads(line)
            assert "features" in entry

    def test_log_path_rejects_parent_traversal(self, tmp_path, petri_net_std):
        art_path = _artifact_with_passthrough(tmp_path, petri_net_std)
        art = load_artifact(art_path)
        ctrl = NeuroSymbolicController(
            artifact=art,
            seed_base=42,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            sc_n_passes=1,
            runtime_backend="numpy",
        )
        obs = {"R_axis_m": 6.2, "Z_axis_m": 0.0, "extra_sensor": 0.5}
        escape = tmp_path / ".." / "escape.jsonl"

        try:
            with pytest.raises(ValueError, match="log_path"):
                ctrl.step(obs, k=0, log_path=str(escape), log_root=tmp_path)
            assert not escape.resolve().exists()
        finally:
            escape.resolve().unlink(missing_ok=True)
