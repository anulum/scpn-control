# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Inhibitor artifact contract tests.
"""Regression tests for fail-closed inhibitor handling in controller artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from scpn_control.scpn.artifact import (
    Artifact,
    ArtifactValidationError,
    get_artifact_json_schema,
    load_artifact,
    save_artifact,
)
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.contracts import ControlScales, ControlTargets
from scpn_control.scpn.controller import NeuroSymbolicController
from scpn_control.scpn.structure import StochasticPetriNet


def _inhibitor_net() -> StochasticPetriNet:
    """Return a net whose guard place inhibits the output transition."""

    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("guard", initial_tokens=1.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("move", threshold=0.5)
    net.add_arc("source", "move", weight=1.0)
    net.add_arc("guard", "move", weight=1.0, inhibitor=True)
    net.add_arc("move", "sink", weight=1.0)
    return net


def _controller_net() -> StochasticPetriNet:
    """Return a non-inhibitor net that can be exported as a controller artifact."""

    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("move", threshold=0.5)
    net.add_arc("source", "move", weight=1.0)
    net.add_arc("move", "sink", weight=1.0)
    return net


def _artifact() -> Artifact:
    """Compile and export a minimal non-inhibitor controller artifact."""

    compiled = FusionCompiler(bitstream_length=64, seed=5).compile(_controller_net())
    return cast(
        Artifact,
        compiled.export_artifact(
            name="inhibitor-contract",
            readout_config={"actions": [], "gains": [], "abs_max": [], "slew_per_s": []},
            injection_config=[],
        ),
    )


def _write_artifact_payload(path: Path, artifact: Artifact) -> dict[str, Any]:
    """Save an artifact and return its JSON payload for mutation."""

    save_artifact(artifact, path)
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _set_payload_value(payload: dict[str, Any], dotted_path: str, value: object) -> None:
    """Set one dotted JSON payload path in place."""

    target: Any = payload
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if isinstance(target, list):
            target = target[int(part)]
        else:
            target = cast(dict[str, Any], target)[part]
    if isinstance(target, list):
        target[int(parts[-1])] = value
    else:
        cast(dict[str, Any], target)[parts[-1]] = value


def test_structure_inhibitors_compile_but_artifact_export_fails_closed() -> None:
    """Inhibitor nets stay usable for structure analysis but not artifacts."""

    compiled = FusionCompiler(bitstream_length=64, seed=5).compile(_inhibitor_net(), allow_inhibitor=True)

    np.testing.assert_array_equal(compiled.W_in, np.array([[1.0, -1.0, 0.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="artifact export.*inhibitor arcs"):
        compiled.export_artifact(
            name="inhibitor-runtime",
            readout_config={"actions": [], "gains": [], "abs_max": [], "slew_per_s": []},
            injection_config=[],
        )


def test_artifact_loader_rejects_negative_input_weights(tmp_path: Path) -> None:
    """Artifact admission rejects negative weights before controller runtime."""

    artifact_path = tmp_path / "negative-w-in.scpnctl.json"
    payload = _write_artifact_payload(artifact_path, _artifact())
    weights = cast(dict[str, Any], payload["weights"])
    w_in = cast(dict[str, Any], weights["w_in"])
    data = cast(list[float], w_in["data"])
    data[0] = -1.0
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="inhibitor arcs"):
        load_artifact(artifact_path)


@pytest.mark.parametrize(
    ("dotted_path", "value", "match"),
    [
        ("weights.w_in.data", True, "w_in.data"),
        ("weights.w_in.data.0", True, "w_in weights"),
        ("weights.w_out.data.0", float("nan"), "w_out weights"),
    ],
)
def test_artifact_loader_rejects_non_numeric_dense_weights(
    tmp_path: Path, dotted_path: str, value: object, match: str
) -> None:
    """Artifact admission rejects non-numeric dense weights before runtime."""

    artifact_path = tmp_path / "invalid-weight.scpnctl.json"
    payload = _write_artifact_payload(artifact_path, _artifact())
    _set_payload_value(payload, dotted_path, value)
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match=match):
        load_artifact(artifact_path)


@pytest.mark.parametrize(
    ("matrix_name", "match"),
    [
        ("w_in", "w_in weights"),
        ("w_out", "w_out weights"),
    ],
)
def test_save_artifact_rejects_direct_non_numeric_dense_weights(tmp_path: Path, matrix_name: str, match: str) -> None:
    """Save admission rejects direct bool weights in constructed artifacts."""

    artifact = _artifact()
    if matrix_name == "w_in":
        artifact.weights.w_in.data[0] = cast(float, True)
    else:
        artifact.weights.w_out.data[0] = cast(float, True)

    with pytest.raises(ArtifactValidationError, match=match):
        save_artifact(artifact, tmp_path / "direct-invalid.scpnctl.json")


def test_controller_rejects_direct_negative_input_weight_artifact() -> None:
    """Directly constructed artifacts cannot bypass the inhibitor guard."""

    artifact = _artifact()
    artifact.weights.w_in.data[0] = -1.0

    with pytest.raises(ValueError, match="inhibitor arcs"):
        NeuroSymbolicController(
            artifact=artifact,
            seed_base=9,
            targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
            scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
            feature_axes=[],
        )


def test_artifact_json_schema_declares_non_negative_weight_contract() -> None:
    """The public artifact schema rejects negative dense controller weights."""

    schema = get_artifact_json_schema()
    weight_item = cast(dict[str, Any], schema["definitions"]["weight_matrix"]["properties"]["data"]["items"])

    assert weight_item["minimum"] == 0
    assert weight_item["maximum"] == 1
