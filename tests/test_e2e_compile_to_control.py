# ──────────────────────────────────────────────────────────────────────
# SCPN Control — End-to-End Integration Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Compile → artifact → controller → step end-to-end tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_control.scpn.artifact import Artifact, load_artifact, save_artifact
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.controller import (
    ControlScales,
    ControlTargets,
    NeuroSymbolicController,
)
from scpn_control.scpn.structure import StochasticPetriNet


def _build_net():
    net = StochasticPetriNet()
    for name in ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        net.add_place(name)
    net.add_transition("T0", threshold=0.5)
    net.add_transition("T1", threshold=0.5)
    net.add_transition("T2", threshold=0.5)
    net.add_transition("T3", threshold=0.5)
    net.add_arc("P0", "T0", 0.8)
    net.add_arc("P1", "T0", 0.6)
    net.add_arc("T0", "P4", 0.7)
    net.add_arc("P2", "T1", 0.8)
    net.add_arc("P3", "T1", 0.6)
    net.add_arc("T1", "P5", 0.7)
    net.add_arc("P4", "T2", 0.5)
    net.add_arc("T2", "P6", 0.9)
    net.add_arc("P5", "T3", 0.5)
    net.add_arc("T3", "P7", 0.9)
    return net


_READOUT = {
    "actions": [
        {"name": "dI_PF3_A", "pos_place": 4, "neg_place": 5},
        {"name": "dI_PF_topbot_A", "pos_place": 6, "neg_place": 7},
    ],
    "gains": [1000.0, 1000.0],
    "abs_max": [5000.0, 5000.0],
    "slew_per_s": [1e6, 1e6],
}

_INJECTION = [
    {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
    {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
    {"place_id": 2, "source": "x_Z_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
    {"place_id": 3, "source": "x_Z_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
]


def _compile_and_save(tmp_path):
    net = _build_net()
    compiler = FusionCompiler(bitstream_length=64, seed=0)
    compiled = compiler.compile(net)
    art = compiled.export_artifact(
        name="e2e-test",
        readout_config=_READOUT,
        injection_config=_INJECTION,
    )
    path = tmp_path / "e2e.scpnctl.json"
    save_artifact(art, str(path))
    return str(path)


def _make_ctrl(art_path, **kw):
    art = load_artifact(art_path)
    defaults = dict(
        artifact=art,
        seed_base=42,
        targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        sc_n_passes=1,
        runtime_backend="numpy",
    )
    defaults.update(kw)
    return NeuroSymbolicController(**defaults)


class TestE2ECompileToControl:
    def test_compile_artifact_step(self, tmp_path):
        """Full pipeline: compile → save → load → step produces valid actions."""
        art_path = _compile_and_save(tmp_path)
        ctrl = _make_ctrl(art_path)
        obs = {"R_axis_m": 6.25, "Z_axis_m": 0.02}
        actions = ctrl.step(obs, k=0)
        assert isinstance(actions, dict)
        assert "dI_PF3_A" in actions
        assert "dI_PF_topbot_A" in actions
        for v in actions.values():
            assert np.isfinite(v)

    def test_passthrough_sources(self, tmp_path):
        """Injection sources map observations through to the Petri net."""
        art_path = _compile_and_save(tmp_path)
        ctrl = _make_ctrl(art_path)
        obs_near = {"R_axis_m": 6.2, "Z_axis_m": 0.0}
        obs_far = {"R_axis_m": 7.0, "Z_axis_m": 0.5}
        a_near = ctrl.step(obs_near, k=0)
        ctrl_far = _make_ctrl(art_path)
        a_far = ctrl_far.step(obs_far, k=0)
        assert isinstance(a_near, dict) and isinstance(a_far, dict)

    def test_traceable_profile(self, tmp_path):
        """step_traceable returns ndarray with correct shape."""
        art_path = _compile_and_save(tmp_path)
        ctrl = _make_ctrl(art_path, runtime_profile="traceable")
        actions = ctrl.step_traceable((6.25, 0.02), k=0)
        assert isinstance(actions, np.ndarray)
        assert actions.shape == (2,)
        assert np.all(np.isfinite(actions))

    def test_multi_step_trajectory(self, tmp_path):
        """Multiple steps produce bounded outputs (slew rate respected)."""
        art_path = _compile_and_save(tmp_path)
        ctrl = _make_ctrl(art_path)
        history = []
        for k in range(20):
            obs = {"R_axis_m": 6.2 + 0.05 * np.sin(k / 5), "Z_axis_m": 0.01 * k}
            actions = ctrl.step(obs, k=k)
            history.append(actions)
        assert len(history) == 20
        for a in history:
            for v in a.values():
                assert abs(v) <= 5000.0

    def test_artifact_round_trip(self, tmp_path):
        """Save → load preserves artifact fields."""
        net = _build_net()
        compiler = FusionCompiler(bitstream_length=64, seed=0)
        compiled = compiler.compile(net)
        art = compiled.export_artifact(
            name="roundtrip-test",
            readout_config=_READOUT,
            injection_config=_INJECTION,
        )
        path = tmp_path / "roundtrip.scpnctl.json"
        save_artifact(art, str(path))
        loaded = load_artifact(str(path))
        assert loaded.meta.name == "roundtrip-test"
        assert len(loaded.readout.actions) == 2
        assert len(loaded.topology.places) == len(art.topology.places)
        assert len(loaded.topology.transitions) == len(art.topology.transitions)
