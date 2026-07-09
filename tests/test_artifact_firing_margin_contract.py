# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Artifact firing-margin contract tests.
"""Regression tests for artifact-level firing-margin metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from scpn_control.scpn.artifact import ArtifactValidationError, load_artifact, save_artifact
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.contracts import ControlScales, ControlTargets
from scpn_control.scpn.controller import NeuroSymbolicController
from scpn_control.scpn.structure import StochasticPetriNet


def _minimal_net() -> StochasticPetriNet:
    """Return a single-transition Petri net for artifact contract tests."""

    net = StochasticPetriNet()
    net.add_place("source", initial_tokens=1.0)
    net.add_place("sink", initial_tokens=0.0)
    net.add_transition("move", threshold=0.5)
    net.add_arc("source", "move", weight=1.0)
    net.add_arc("move", "sink", weight=1.0)
    return net


def _artifact_path(tmp_path: Path, *, firing_margin: float = 0.2, omit_transition_margin: bool = True) -> Path:
    """Compile and save a fractional artifact with a metadata firing margin."""

    compiled = FusionCompiler(bitstream_length=64, seed=1).compile(
        _minimal_net(),
        firing_mode="fractional",
        firing_margin=firing_margin,
    )
    artifact = compiled.export_artifact(
        name="firing-margin-contract",
        readout_config={"actions": [], "gains": [], "abs_max": [], "slew_per_s": []},
        injection_config=[],
    )
    if omit_transition_margin:
        for transition in artifact.topology.transitions:
            transition.margin = None
    path = tmp_path / "margin.scpnctl.json"
    save_artifact(artifact, path)
    return path


def _mutate_payload(path: Path, tmp_path: Path, mutation: dict[str, Any]) -> Path:
    """Write a mutated copy of an artifact JSON payload."""

    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    for dotted_path, value in mutation.items():
        target = payload
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            target = cast(dict[str, Any], target[part])
        target[parts[-1]] = value
    out = tmp_path / "mutated.scpnctl.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def test_firing_margin_round_trips_and_drives_controller_defaults(tmp_path: Path) -> None:
    """Artifact metadata carries the default margin into controller runtime."""

    path = _artifact_path(tmp_path, firing_margin=0.2)
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    assert cast(dict[str, Any], payload["meta"])["firing_margin"] == 0.2

    artifact = load_artifact(path)
    assert artifact.meta.firing_margin == 0.2
    assert artifact.topology.transitions[0].margin is None

    controller = NeuroSymbolicController(
        artifact=artifact,
        seed_base=7,
        targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        feature_axes=[],
    )

    np.testing.assert_allclose(controller._margins, np.array([0.2], dtype=np.float64))


def test_legacy_artifact_without_firing_margin_loads_with_default(tmp_path: Path) -> None:
    """Older artifacts without metadata firing margins retain legacy behavior."""

    path = _artifact_path(tmp_path)
    legacy_path = _mutate_payload(path, tmp_path, {"meta.firing_margin": None})
    payload = cast(dict[str, Any], json.loads(legacy_path.read_text(encoding="utf-8")))
    cast(dict[str, Any], payload["meta"]).pop("firing_margin")
    legacy_path.write_text(json.dumps(payload), encoding="utf-8")

    artifact = load_artifact(legacy_path)

    assert artifact.meta.firing_margin == 0.05


@pytest.mark.parametrize("bad_margin", [-0.1, float("nan"), True, "0.1"])
def test_artifact_rejects_invalid_firing_margin(tmp_path: Path, bad_margin: object) -> None:
    """Artifact admission rejects non-finite or non-numeric firing margins."""

    path = _artifact_path(tmp_path)
    bad_path = _mutate_payload(path, tmp_path, {"meta.firing_margin": bad_margin})

    with pytest.raises(ArtifactValidationError, match="firing_margin"):
        load_artifact(bad_path)


@pytest.mark.parametrize("bad_margin", [-0.1, float("nan"), True])
def test_compiler_rejects_invalid_firing_margin(bad_margin: float) -> None:
    """Compiler rejects invalid margins before artifact export."""

    with pytest.raises(ValueError, match="firing_margin"):
        FusionCompiler(bitstream_length=64, seed=1).compile(
            _minimal_net(),
            firing_mode="fractional",
            firing_margin=bad_margin,
        )
