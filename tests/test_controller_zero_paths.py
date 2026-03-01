# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Controller Zero-Count & Chunked Antithetic Path Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for axis_count==0 (lines 504,519), action_count==0 (778,785),
inj_count==0 (557), bitflip on empty array (804), and chunked antithetic
odd-pass path (688)."""
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.scpn.artifact import load_artifact, save_artifact
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.controller import (
    ControlScales,
    ControlTargets,
    FeatureAxisSpec,
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


def _artifact(tmp_path, *, readout_config=None, injection_config=None):
    net = _build_net()
    compiler = FusionCompiler(bitstream_length=64, seed=0)
    compiled = compiler.compile(net)
    if readout_config is None:
        readout_config = {
            "actions": [
                {"name": "dI_PF3_A", "pos_place": 4, "neg_place": 5},
                {"name": "dI_topbot", "pos_place": 6, "neg_place": 7},
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
        name="test-zero",
        readout_config=readout_config,
        injection_config=injection_config,
    )
    path = tmp_path / "zero.scpnctl.json"
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


class TestZeroAxes:
    def test_step_with_zero_feature_axes(self, tmp_path):
        """axis_count==0 returns empty arrays (lines 504, 519)."""
        art_path = _artifact(tmp_path, injection_config=[])
        ctrl = _ctrl(art_path, feature_axes=[])
        obs = {"R_axis_m": 6.2, "Z_axis_m": 0.0}
        actions = ctrl.step(obs, k=0)
        assert isinstance(actions, dict)
        assert len(actions) == 2

    def test_traceable_with_zero_axes_and_no_passthrough(self, tmp_path):
        """axis_count==0 + traceable with no passthrough injections."""
        art_path = _artifact(tmp_path, injection_config=[])
        ctrl = _ctrl(art_path, feature_axes=[], runtime_profile="traceable")
        actions = ctrl.step_traceable((), k=0)
        assert actions.shape == (2,)


class TestZeroActions:
    def test_step_no_actions_returns_defaults(self, tmp_path):
        """action_count==0 hits _decode_actions line 778 (returns {}),
        but step() wraps with hardcoded dI keys → both 0.0."""
        readout_config = {
            "actions": [],
            "gains": [],
            "abs_max": [],
            "slew_per_s": [],
        }
        art_path = _artifact(tmp_path, readout_config=readout_config)
        ctrl = _ctrl(art_path)
        assert ctrl._action_count == 0
        obs = {"R_axis_m": 6.2, "Z_axis_m": 0.0}
        actions = ctrl.step(obs, k=0)
        assert actions["dI_PF3_A"] == 0.0
        assert actions["dI_PF_topbot_A"] == 0.0

    def test_decode_actions_vector_empty(self, tmp_path):
        """_decode_actions_vector returns empty array (line 785)."""
        readout_config = {
            "actions": [],
            "gains": [],
            "abs_max": [],
            "slew_per_s": [],
        }
        art_path = _artifact(tmp_path, readout_config=readout_config)
        ctrl = _ctrl(art_path)
        marking = np.zeros(ctrl._nP)
        result = ctrl._decode_actions_vector(marking)
        assert result.shape == (0,)

    def test_traceable_no_actions(self, tmp_path):
        """Traceable path returns empty actions array."""
        readout_config = {
            "actions": [],
            "gains": [],
            "abs_max": [],
            "slew_per_s": [],
        }
        art_path = _artifact(tmp_path, readout_config=readout_config)
        ctrl = _ctrl(art_path, runtime_profile="traceable")
        actions = ctrl.step_traceable((6.2, 0.0), k=0)
        assert actions.shape == (0,)


class TestZeroInjections:
    def test_step_no_injections(self, tmp_path):
        """inj_count==0 skips injection (line 557)."""
        art_path = _artifact(tmp_path, injection_config=[])
        ctrl = _ctrl(art_path)
        obs = {"R_axis_m": 6.25, "Z_axis_m": 0.01}
        actions = ctrl.step(obs, k=0)
        assert isinstance(actions, dict)
        assert len(actions) == 2


class TestBitflipEmpty:
    def test_bitflip_on_empty_array(self, tmp_path):
        """Bitflip with out.size==0 returns early (line 804)."""
        readout_config = {
            "actions": [],
            "gains": [],
            "abs_max": [],
            "slew_per_s": [],
        }
        art_path = _artifact(tmp_path, readout_config=readout_config)
        ctrl = _ctrl(art_path, sc_bitflip_rate=0.5)
        rng = np.random.default_rng(0)
        result = ctrl._apply_bit_flip_faults(np.array([], dtype=np.float64), rng)
        assert result.shape == (0,)


class TestChunkedAntitheticOdd:
    def test_antithetic_odd_chunked(self, tmp_path):
        """Antithetic with odd passes + nT > chunk_size → line 688."""
        art_path = _artifact(tmp_path)
        ctrl = _ctrl(
            art_path,
            sc_n_passes=3,
            sc_antithetic=True,
            sc_antithetic_chunk_size=1,
        )
        obs = {"R_axis_m": 6.22, "Z_axis_m": -0.01}
        actions = ctrl.step(obs, k=0)
        assert "dI_PF3_A" in actions
        assert np.isfinite(actions["dI_PF3_A"])

    def test_antithetic_even_chunked(self, tmp_path):
        """Antithetic with even passes + nT > chunk_size."""
        art_path = _artifact(tmp_path)
        ctrl = _ctrl(
            art_path,
            sc_n_passes=4,
            sc_antithetic=True,
            sc_antithetic_chunk_size=1,
        )
        obs = {"R_axis_m": 6.22, "Z_axis_m": -0.01}
        actions = ctrl.step(obs, k=0)
        assert "dI_PF3_A" in actions
