# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller contract-kernel parity tests.
"""Controller integration tests for shared feature and action kernels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from scpn_control.scpn.artifact import Artifact, load_artifact, save_artifact
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.contracts import (
    ActionSpec,
    ControlScales,
    ControlTargets,
    decode_actions,
    extract_features,
)
from scpn_control.scpn.controller import NeuroSymbolicController
from scpn_control.scpn.structure import StochasticPetriNet


def _build_controller_net() -> StochasticPetriNet:
    """Build a controller net with both radial and vertical readouts."""
    net = StochasticPetriNet()
    for name in ("x_R_pos", "x_R_neg", "x_Z_pos", "x_Z_neg", "a_R_pos", "a_R_neg", "a_Z_pos", "a_Z_neg"):
        net.add_place(name, initial_tokens=0.0)
    for name in ("T_Rp", "T_Rn", "T_Zp", "T_Zn"):
        net.add_transition(name, threshold=0.1)
    net.add_arc("x_R_pos", "T_Rp", weight=1.0)
    net.add_arc("x_R_neg", "T_Rn", weight=1.0)
    net.add_arc("x_Z_pos", "T_Zp", weight=1.0)
    net.add_arc("x_Z_neg", "T_Zn", weight=1.0)
    net.add_arc("T_Rp", "a_R_pos", weight=1.0)
    net.add_arc("T_Rn", "a_R_neg", weight=1.0)
    net.add_arc("T_Zp", "a_Z_pos", weight=1.0)
    net.add_arc("T_Zn", "a_Z_neg", weight=1.0)
    net.compile()
    return net


def _compile_artifact(path: Path) -> Artifact:
    """Compile and persist the parity-test controller artifact."""
    compiled = FusionCompiler(bitstream_length=64, seed=11).compile(_build_controller_net())
    artifact = compiled.export_artifact(
        name="contract-kernel-parity",
        dt_control_s=0.001,
        readout_config={
            "actions": [
                {"name": "dI_PF3_A", "pos_place": 4, "neg_place": 5},
                {"name": "dI_PF_topbot_A", "pos_place": 6, "neg_place": 7},
            ],
            "gains": [1000.0, 1000.0],
            "abs_max": [5000.0, 5000.0],
            "slew_per_s": [1_000_000.0, 1_000_000.0],
        },
        injection_config=[
            {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 2, "source": "x_Z_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 3, "source": "x_Z_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ],
    )
    save_artifact(artifact, str(path))
    return load_artifact(str(path))


def _controller(artifact: Artifact) -> NeuroSymbolicController:
    """Return the deterministic controller used for contract parity."""
    return NeuroSymbolicController(
        artifact=artifact,
        seed_base=42,
        targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        sc_n_passes=1,
        runtime_backend="numpy",
    )


def test_controller_step_uses_shared_feature_and_action_contracts(tmp_path: Path) -> None:
    """Controller JSONL evidence matches the public contract helpers."""
    artifact = _compile_artifact(tmp_path / "contract-parity.scpnctl.json")
    controller = _controller(artifact)
    observation = {"R_axis_m": 6.0, "Z_axis_m": 0.25}
    log_path = tmp_path / "controller.jsonl"

    action = controller.step(observation, k=0, log_path=log_path.name, log_root=tmp_path)

    record = json.loads(log_path.read_text(encoding="utf-8"))
    expected_features = extract_features(
        observation,
        ControlTargets(R_target_m=6.2, Z_target_m=0.0),
        ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
    )
    expected_actions = decode_actions(
        marking=cast(list[float], record["marking"]),
        actions_spec=[
            ActionSpec(name=item.name, pos_place=item.pos_place, neg_place=item.neg_place)
            for item in artifact.readout.actions
        ],
        gains=artifact.readout.gains,
        abs_max=artifact.readout.abs_max,
        slew_per_s=artifact.readout.slew_per_s,
        dt=artifact.meta.dt_control_s,
        prev=[0.0 for _ in artifact.readout.actions],
    )

    assert record["features"] == pytest.approx(expected_features)
    assert record["actions"] == pytest.approx(expected_actions)
    assert action == pytest.approx(expected_actions)
