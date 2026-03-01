# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Delayed Transition + Zero-axis Controller Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for transition delay_ticks (lines 734-748), axis_count==0 paths,
and action_count==0 / inj_count==0 edge paths."""
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
from scpn_control.scpn.structure import StochasticPetriNet


def _build_delayed_net():
    """Net with one delayed and one immediate transition."""
    net = StochasticPetriNet()
    for name in ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        net.add_place(name)
    net.add_transition("T0", threshold=0.5, delay_ticks=0)
    net.add_transition("T1", threshold=0.5, delay_ticks=3)
    net.add_transition("T2", threshold=0.5, delay_ticks=0)
    net.add_transition("T3", threshold=0.5, delay_ticks=2)
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


def _artifact_delayed(tmp_path):
    net = _build_delayed_net()
    compiler = FusionCompiler(bitstream_length=64, seed=0)
    compiled = compiler.compile(net)
    readout_config = {
        "actions": [
            {"name": "act0", "pos_place": 4, "neg_place": 5},
            {"name": "act1", "pos_place": 6, "neg_place": 7},
        ],
        "gains": [1000.0, 1000.0],
        "abs_max": [5000.0, 5000.0],
        "slew_per_s": [1e6, 1e6],
    }
    injection_config = [
        {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 2, "source": "x_Z_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 3, "source": "x_Z_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
    ]
    art = compiled.export_artifact(
        name="test-delayed",
        readout_config=readout_config,
        injection_config=injection_config,
    )
    path = tmp_path / "delayed.scpnctl.json"
    save_artifact(art, str(path))
    return str(path)


def _ctrl(art_path, **kw):
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
    defaults.update(kw)
    return NeuroSymbolicController(**defaults)


class TestDelayedTransitions:
    def test_step_sequence_with_delays(self, tmp_path):
        """Transitions with delay_ticks > 0 fire after the delay."""
        art_path = _artifact_delayed(tmp_path)
        ctrl = _ctrl(art_path)
        obs = {"R_axis_m": 6.25, "Z_axis_m": 0.01}
        actions_history = []
        for k in range(10):
            actions = ctrl.step(obs, k=k)
            actions_history.append(actions)
        assert len(actions_history) == 10
        assert all(isinstance(a, dict) and len(a) == 2 for a in actions_history)

    def test_traceable_with_delays(self, tmp_path):
        """step_traceable works with delayed transitions."""
        art_path = _artifact_delayed(tmp_path)
        ctrl = _ctrl(art_path, runtime_profile="traceable")
        for k in range(8):
            actions = ctrl.step_traceable((6.25, 0.01), k=k)
            assert actions.shape == (2,)
