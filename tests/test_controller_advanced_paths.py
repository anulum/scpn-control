# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Controller Advanced Path Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Coverage for controller validation errors, passthrough sources, bitflip,
antithetic sampling with odd passes, delayed transitions, and marking setter."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.scpn.artifact import load_artifact, save_artifact
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.controller import (
    ControlScales,
    ControlTargets,
    NeuroSymbolicController,
)


def _artifact_custom(tmp_path, petri_net_std, injection_config=None):
    compiler = FusionCompiler(bitstream_length=64, seed=0)
    compiled = compiler.compile(petri_net_std)
    readout_config = {
        "actions": [
            {"name": "dI_PF3_A", "pos_place": 4, "neg_place": 5},
            {"name": "dI_PF_topbot_A", "pos_place": 6, "neg_place": 7},
        ],
        "gains": [1000.0, 1000.0],
        "abs_max": [5000.0, 5000.0],
        "slew_per_s": [1e6, 1e6],
    }
    if injection_config is None:
        injection_config = [
            {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 2, "source": "x_Z_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 3, "source": "x_Z_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ]
    art = compiled.export_artifact(
        name="test-advanced",
        readout_config=readout_config,
        injection_config=injection_config,
    )
    path = tmp_path / "adv.scpnctl.json"
    save_artifact(art, str(path))
    return str(path)


def _ctrl(art_path, **kwargs):
    art = load_artifact(art_path)
    defaults = dict(
        artifact=art,
        seed_base=42,
        targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        sc_n_passes=1,
        sc_bitflip_rate=0.0,
        runtime_backend="numpy",
    )
    defaults.update(kwargs)
    return NeuroSymbolicController(**defaults)


class TestControllerValidation:
    def test_invalid_runtime_profile(self, tmp_path, petri_net_std):
        with pytest.raises(ValueError, match="runtime_profile"):
            _ctrl(_artifact_custom(tmp_path, petri_net_std), runtime_profile="bogus")

    def test_invalid_runtime_backend(self, tmp_path, petri_net_std):
        with pytest.raises(ValueError, match="runtime_backend"):
            _ctrl(_artifact_custom(tmp_path, petri_net_std), runtime_backend="cuda")

    def test_invalid_bitflip_rate(self, tmp_path, petri_net_std):
        with pytest.raises(ValueError, match="sc_bitflip_rate"):
            _ctrl(_artifact_custom(tmp_path, petri_net_std), sc_bitflip_rate=1.5)


class TestBitflipFaults:
    def test_step_with_bitflip(self, tmp_path, petri_net_std):
        art_path = _artifact_custom(tmp_path, petri_net_std)
        ctrl = _ctrl(art_path, sc_bitflip_rate=0.3)
        obs = {"R_axis_m": 6.29, "Z_axis_m": -0.02}
        actions = ctrl.step(obs, k=0)
        assert "dI_PF3_A" in actions
        assert np.isfinite(actions["dI_PF3_A"])


class TestAntithetic:
    def test_antithetic_odd_passes(self, tmp_path, petri_net_std):
        art_path = _artifact_custom(tmp_path, petri_net_std)
        ctrl = _ctrl(art_path, sc_n_passes=3, sc_antithetic=True)
        obs = {"R_axis_m": 6.19, "Z_axis_m": 0.01}
        for k in range(5):
            actions = ctrl.step(obs, k=k)
        assert "dI_PF3_A" in actions

    def test_antithetic_even_passes(self, tmp_path, petri_net_std):
        art_path = _artifact_custom(tmp_path, petri_net_std)
        ctrl = _ctrl(art_path, sc_n_passes=4, sc_antithetic=True)
        obs = {"R_axis_m": 6.22, "Z_axis_m": -0.01}
        actions = ctrl.step(obs, k=0)
        assert "dI_PF3_A" in actions


class TestMarkingSetter:
    def test_marking_set_and_get(self, controller_instance):
        m = controller_instance.marking
        assert len(m) == 8
        new_m = [0.5] * 8
        controller_instance.marking = new_m
        assert controller_instance.marking == pytest.approx([0.5] * 8, abs=1e-10)

    def test_marking_wrong_length_raises(self, controller_instance):
        with pytest.raises(ValueError, match="length"):
            controller_instance.marking = [0.5, 0.5]


class TestPassthroughSources:
    def test_step_with_passthrough(self, tmp_path, petri_net_std):
        injection_config = [
            {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 2, "source": "x_Z_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 3, "source": "x_Z_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 4, "source": "ext_sensor", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ]
        art_path = _artifact_custom(tmp_path, petri_net_std, injection_config=injection_config)
        ctrl = _ctrl(art_path)
        obs = {"R_axis_m": 6.2, "Z_axis_m": 0.0, "ext_sensor": 0.7}
        actions = ctrl.step(obs, k=0)
        assert "dI_PF3_A" in actions

    def test_passthrough_missing_key_raises(self, tmp_path, petri_net_std):
        injection_config = [
            {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 2, "source": "x_Z_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 3, "source": "x_Z_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 4, "source": "ext_sensor", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ]
        art_path = _artifact_custom(tmp_path, petri_net_std, injection_config=injection_config)
        ctrl = _ctrl(art_path)
        with pytest.raises(KeyError, match="passthrough"):
            ctrl.step({"R_axis_m": 6.2, "Z_axis_m": 0.0}, k=0)
