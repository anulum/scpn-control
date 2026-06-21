# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Controller Advanced Paths
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Controller Advanced Path Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Regression tests for controller validation errors, passthrough sources, bitflip,
antithetic sampling with odd passes, delayed transitions, and marking setter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scpn_control.scpn.controller as controller_mod
from scpn_control.scpn.artifact import load_artifact, save_artifact
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.controller import (
    ControlScales,
    ControlTargets,
    NeuroSymbolicController,
)


def _artifact_custom(tmp_path: Path, petri_net_std: Any, injection_config: list[dict[str, Any]] | None = None) -> str:
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


def _ctrl(art_path: str, **kwargs: Any) -> NeuroSymbolicController:
    art = load_artifact(art_path)
    defaults: dict[str, Any] = dict(
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

    def test_bitflip_rate_requires_fault_injection_opt_in(self, tmp_path, petri_net_std):
        with pytest.raises(ValueError, match="allow_fault_injection"):
            _ctrl(_artifact_custom(tmp_path, petri_net_std), sc_bitflip_rate=0.3)

    def test_bitflip_rate_requires_environment_gate(self, tmp_path, petri_net_std, monkeypatch):
        monkeypatch.delenv("SCPN_ALLOW_CONTROLLER_FAULT_INJECTION", raising=False)
        with pytest.raises(ValueError, match="SCPN_ALLOW_CONTROLLER_FAULT_INJECTION"):
            _ctrl(_artifact_custom(tmp_path, petri_net_std), sc_bitflip_rate=0.3, allow_fault_injection=True)


class TestBitflipFaults:
    def test_step_with_bitflip(self, tmp_path, petri_net_std, monkeypatch):
        monkeypatch.setenv("SCPN_ALLOW_CONTROLLER_FAULT_INJECTION", "1")
        art_path = _artifact_custom(tmp_path, petri_net_std)
        ctrl = _ctrl(art_path, sc_bitflip_rate=0.3, allow_fault_injection=True)
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

    def test_feature_dict_missing_key_raises_when_logging(self, tmp_path: Path, petri_net_std: Any) -> None:
        injection_config = [
            {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 2, "source": "x_Z_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 3, "source": "x_Z_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 4, "source": "ext_sensor", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ]
        art_path = _artifact_custom(tmp_path, petri_net_std, injection_config=injection_config)
        ctrl = _ctrl(art_path)
        log_file = tmp_path / "feat.jsonl"
        # The feature dict is built (and the passthrough key checked) before place
        # injection whenever logging is requested.
        with pytest.raises(KeyError, match="passthrough"):
            ctrl.step({"R_axis_m": 6.2, "Z_axis_m": 0.0}, k=0, log_path=str(log_file), log_root=tmp_path)


class TestRustBackendRequestWithoutRuntime:
    def test_rust_request_falls_back_to_numpy_when_allowed(
        self, tmp_path: Path, petri_net_std: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(controller_mod, "_HAS_RUST_SCPN_RUNTIME", False)
        ctrl = _ctrl(
            _artifact_custom(tmp_path, petri_net_std),
            runtime_backend="rust",
            allow_runtime_backend_fallback=True,
            allow_legacy_runtime_backend_fallback=True,
        )
        assert ctrl._runtime_backend == "numpy"

    def test_rust_request_raises_without_fallback(
        self, tmp_path: Path, petri_net_std: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(controller_mod, "_HAS_RUST_SCPN_RUNTIME", False)
        with pytest.raises(RuntimeError, match="Rust SCPN runtime is unavailable"):
            _ctrl(_artifact_custom(tmp_path, petri_net_std), runtime_backend="rust")


class TestNonAntitheticSampling:
    def test_numpy_binomial_path_without_antithetic(self, tmp_path: Path, petri_net_std: Any) -> None:
        ctrl = _ctrl(
            _artifact_custom(tmp_path, petri_net_std),
            runtime_backend="numpy",
            sc_n_passes=4,
            sc_antithetic=False,
            sc_binary_margin=0.05,
        )
        obs = {"R_axis_m": 6.27, "Z_axis_m": -0.03}
        actions = ctrl.step(obs, k=0)
        assert "dI_PF3_A" in actions
        assert np.isfinite(actions["dI_PF3_A"])
