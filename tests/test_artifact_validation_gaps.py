# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Artifact Validation Gap Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for _validate error branches not exercised by test_controller.py.

Target lines: 279, 289, 293, 306, 313, 318, 323, 338, 355, 366, 371,
378, 383, 388, 394, 410, 418, 422, 424, 426, 692.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from scpn_control.scpn.artifact import (
    ArtifactValidationError,
    load_artifact,
    save_artifact,
)
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.structure import StochasticPetriNet


def _build_valid_artifact(tmp_path: Path) -> str:
    net = StochasticPetriNet()
    for name in ("P0", "P1", "P2", "P3"):
        net.add_place(name)
    net.add_transition("T0", threshold=0.5)
    net.add_transition("T1", threshold=0.5)
    net.add_arc("P0", "T0", 0.8)
    net.add_arc("P1", "T0", 0.6)
    net.add_arc("T0", "P2", 0.7)
    net.add_arc("P2", "T1", 0.5)
    net.add_arc("T1", "P3", 0.9)

    compiler = FusionCompiler(bitstream_length=64, seed=0)
    compiled = compiler.compile(net)

    readout_config = {
        "actions": [
            {"name": "act0", "pos_place": 2, "neg_place": 3},
        ],
        "gains": [1.0],
        "abs_max": [10.0],
        "slew_per_s": [100.0],
    }
    injection_config = [
        {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
    ]

    art = compiled.export_artifact(
        name="test-valid",
        readout_config=readout_config,
        injection_config=injection_config,
    )
    path = tmp_path / "valid.scpnctl.json"
    save_artifact(art, str(path))
    return str(path)


@pytest.fixture()
def artifact_path(tmp_path):
    return _build_valid_artifact(tmp_path)


def _load_modified(artifact_path: str, mutations: dict) -> None:
    """Load artifact JSON, apply nested mutations, write to temp, then load_artifact."""
    obj = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    for dotpath, value in mutations.items():
        keys = dotpath.split(".")
        target = obj
        for k in keys[:-1]:
            if k.isdigit():
                target = target[int(k)]
            else:
                target = target[k]
        last = keys[-1]
        if last.isdigit():
            target[int(last)] = value
        else:
            target[last] = value

    fd, bad_path = tempfile.mkstemp(suffix=".scpnctl.json")
    os.close(fd)
    try:
        Path(bad_path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
        load_artifact(bad_path)
    finally:
        os.unlink(bad_path)


def _load_modified_raw(artifact_path: str, mutator) -> None:
    """Load artifact JSON, apply arbitrary mutator function, write + load."""
    obj = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    mutator(obj)
    fd, bad_path = tempfile.mkstemp(suffix=".scpnctl.json")
    os.close(fd)
    try:
        Path(bad_path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
        load_artifact(bad_path)
    finally:
        os.unlink(bad_path)


class TestFiringModeValidation:
    def test_invalid_firing_mode(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="firing_mode"):
            _load_modified(artifact_path, {"meta.firing_mode": "invalid"})


class TestFixedPointRangeValidation:
    def test_data_width_zero(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="data_width"):
            _load_modified(artifact_path, {"meta.fixed_point.data_width": 0})

    def test_fraction_bits_negative(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="fraction_bits"):
            _load_modified(artifact_path, {"meta.fixed_point.fraction_bits": -1})


class TestStreamLengthRange:
    def test_stream_length_zero(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="stream_length"):
            _load_modified(artifact_path, {"meta.stream_length": 0})


class TestDtControlRange:
    def test_dt_control_zero(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="dt_control_s"):
            _load_modified(artifact_path, {"meta.dt_control_s": 0.0})

    def test_dt_control_negative(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="dt_control_s"):
            _load_modified(artifact_path, {"meta.dt_control_s": -0.001})


class TestWeightRangeValidation:
    def test_w_in_out_of_range(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="w_in"):
            _load_modified(artifact_path, {"weights.w_in.data.0": 1.5})

    def test_w_out_out_of_range(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="w_out"):
            _load_modified(artifact_path, {"weights.w_out.data.0": -0.1})


class TestTransitionRangeValidation:
    def test_threshold_out_of_range(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="threshold"):
            _load_modified(artifact_path, {"topology.transitions.0.threshold": 1.5})

    def test_delay_ticks_negative(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="delay_ticks"):
            _load_modified(artifact_path, {"topology.transitions.0.delay_ticks": -1})


class TestWeightShapeValidation:
    def test_w_in_data_length_mismatch(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="w_in data length"):
            _load_modified(artifact_path, {"weights.w_in.data": [0.5]})

    def test_w_out_data_length_mismatch(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="w_out data length"):
            _load_modified(artifact_path, {"weights.w_out.data": [0.5]})


class TestMarkingValidation:
    def test_marking_wrong_length(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="marking"):
            _load_modified(artifact_path, {"initial_state.marking": [0.5]})

    def test_marking_value_out_of_range(self, artifact_path):
        def _mutate(obj):
            obj["initial_state"]["marking"][0] = 1.5
        with pytest.raises(ArtifactValidationError, match="marking"):
            _load_modified_raw(artifact_path, _mutate)


class TestPlaceInjectionValidation:
    def test_place_id_bool_type(self, artifact_path):
        def _mutate(obj):
            obj["initial_state"]["place_injections"][0]["place_id"] = True
        with pytest.raises(ArtifactValidationError, match="place_id"):
            _load_modified_raw(artifact_path, _mutate)

    def test_source_empty_string(self, artifact_path):
        def _mutate(obj):
            obj["initial_state"]["place_injections"][0]["source"] = ""
        with pytest.raises(ArtifactValidationError, match="source"):
            _load_modified_raw(artifact_path, _mutate)


class TestReadoutValidation:
    def test_action_pos_place_bool(self, artifact_path):
        def _mutate(obj):
            obj["readout"]["actions"][0]["pos_place"] = True
        with pytest.raises(ArtifactValidationError, match="pos_place"):
            _load_modified_raw(artifact_path, _mutate)

    def test_action_neg_place_out_of_bounds(self, artifact_path):
        def _mutate(obj):
            obj["readout"]["actions"][0]["neg_place"] = 999
        with pytest.raises(ArtifactValidationError, match="neg_place"):
            _load_modified_raw(artifact_path, _mutate)

    def test_gains_length_mismatch(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="gains"):
            _load_modified(artifact_path, {"readout.gains.per_action": [1.0, 2.0, 3.0]})

    def test_abs_max_length_mismatch(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="abs_max"):
            _load_modified(artifact_path, {"readout.limits.per_action_abs_max": []})

    def test_slew_per_s_length_mismatch(self, artifact_path):
        with pytest.raises(ArtifactValidationError, match="slew_per_s"):
            _load_modified(artifact_path, {"readout.limits.slew_per_s": [1.0, 2.0]})


class TestNotesSerialisation:
    def test_save_with_notes(self, artifact_path, tmp_path):
        art = load_artifact(artifact_path)
        art.meta.notes = "test-memo"
        out = tmp_path / "noted.scpnctl.json"
        save_artifact(art, str(out))
        obj = json.loads(out.read_text(encoding="utf-8"))
        assert obj["meta"]["notes"] == "test-memo"

    def test_save_non_compact_packed(self, artifact_path, tmp_path):
        art = load_artifact(artifact_path)
        if art.weights.packed is None:
            pytest.skip("No packed weights in test artifact")
        out = tmp_path / "non_compact.scpnctl.json"
        save_artifact(art, str(out), compact_packed=False)
        obj = json.loads(out.read_text(encoding="utf-8"))
        assert "data_u64" in obj["weights"]["packed"]["w_in_packed"]
