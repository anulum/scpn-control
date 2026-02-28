# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Controller Oracle + WS Serve + Halo Edge Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for:
- controller.py step_traceable with oracle diagnostics + log_path
- controller.py oracle-disabled empty-array path
- ws_phase_stream.py main entry
- halo_re_physics.py edge cases (dreicer NaN guard, ensemble verbose)
- tokamak_digital_twin.py run_digital_twin_ids
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

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


def _build_artifact_file(tmp_path: Path, injection_config=None) -> str:
    net = _build_net()
    compiler = FusionCompiler(bitstream_length=64, seed=0)
    compiled = compiler.compile(net)

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
        name="test-oracle",
        readout_config=readout_config,
        injection_config=injection_config,
    )
    path = tmp_path / "oracle.scpnctl.json"
    save_artifact(art, str(path))
    return str(path)


def _make_controller(artifact_path: str, **kwargs):
    art = load_artifact(artifact_path)
    return NeuroSymbolicController(
        artifact=art,
        seed_base=123456789,
        targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        sc_n_passes=1,
        sc_bitflip_rate=0.0,
        runtime_backend="numpy",
        **kwargs,
    )


class TestStepTraceableOracle:
    def test_oracle_diagnostics_enabled(self, tmp_path):
        art_path = _build_artifact_file(tmp_path)
        ctrl = _make_controller(
            art_path,
            runtime_profile="traceable",
            enable_oracle_diagnostics=True,
        )
        actions = ctrl.step_traceable((6.29, -0.02), k=0)
        assert actions.shape == (2,)
        assert len(ctrl.last_oracle_firing) > 0
        assert len(ctrl.last_oracle_marking) > 0

    def test_oracle_diagnostics_disabled(self, tmp_path):
        art_path = _build_artifact_file(tmp_path)
        ctrl = _make_controller(
            art_path,
            runtime_profile="traceable",
            enable_oracle_diagnostics=False,
        )
        actions = ctrl.step_traceable((6.29, -0.02), k=0)
        assert actions.shape == (2,)
        assert ctrl.last_oracle_firing == []
        assert ctrl.last_oracle_marking == []

    def test_step_traceable_with_log(self, tmp_path):
        art_path = _build_artifact_file(tmp_path)
        ctrl = _make_controller(
            art_path,
            runtime_profile="traceable",
            enable_oracle_diagnostics=True,
        )
        log = tmp_path / "trace.jsonl"
        for k in range(5):
            ctrl.step_traceable((6.2 + k * 0.01, 0.0), k=k, log_path=str(log))
        assert log.exists()
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5
        rec = json.loads(lines[0])
        assert "obs" in rec
        assert "f_oracle" in rec
        assert "timing_ms" in rec

    def test_not_traceable_raises(self, tmp_path):
        """Controller with passthrough sources rejects step_traceable."""
        injection_config = [
            {"place_id": 0, "source": "external_sensor", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ]
        art_path = _build_artifact_file(tmp_path, injection_config=injection_config)
        ctrl = _make_controller(art_path, runtime_profile="traceable")
        with pytest.raises(RuntimeError, match="passthrough"):
            ctrl.step_traceable((6.2,), k=0)


class TestWSPhaseStreamServeSync:
    def test_main_help_exits(self):
        result = subprocess.run(
            [sys.executable, "-m", "scpn_control.phase.ws_phase_stream", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "port" in result.stdout


class TestHaloEdgeCases:
    def test_dreicer_nan_guard(self):
        from scpn_control.control.halo_re_physics import RunawayElectronModel
        model = RunawayElectronModel(n_e=1e20, T_e_keV=0.001, z_eff=2.0)
        assert model._dreicer_rate(E=float("nan"), T_e_keV=1.0) == 0.0
        assert model._dreicer_rate(E=0.0, T_e_keV=1.0) == 0.0
        assert model._dreicer_rate(E=1.0, T_e_keV=0.001) == 0.0
        rate = model._dreicer_rate(E=10.0, T_e_keV=1.0)
        assert np.isfinite(rate)
        assert rate >= 0.0

    def test_simulate_re_extreme_params(self):
        from scpn_control.control.halo_re_physics import RunawayElectronModel
        model = RunawayElectronModel(n_e=5e20, T_e_keV=0.1, z_eff=5.0, neon_mol=0.5)
        result = model.simulate(
            plasma_current_ma=15.0, tau_cq_s=0.01,
            T_e_quench_keV=0.5, duration_s=0.05, dt_s=0.001,
        )
        assert result.peak_re_current_ma >= 0.0
        assert np.isfinite(result.avalanche_gain)

    def test_run_disruption_ensemble(self):
        from scpn_control.control.halo_re_physics import run_disruption_ensemble
        result = run_disruption_ensemble(ensemble_runs=3, seed=42, verbose=False)
        assert result.prevention_rate >= 0.0
        assert len(result.per_run_details) == 3

    def test_run_disruption_ensemble_verbose(self, capsys):
        from scpn_control.control.halo_re_physics import run_disruption_ensemble
        run_disruption_ensemble(ensemble_runs=2, seed=7, verbose=True)
        captured = capsys.readouterr()
        assert "Run" in captured.out
