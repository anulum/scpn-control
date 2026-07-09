# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller AER observation adapter tests.
"""Controller integration tests for typed AER control observations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_control.scpn.artifact import Artifact, load_artifact, save_artifact
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.contracts import ControlScales, ControlTargets
from scpn_control.scpn.controller import NeuroSymbolicController
from scpn_control.scpn.observation import AERControlObservation, SpikeBuffer, SpikeEvent
from scpn_control.scpn.structure import StochasticPetriNet


def _build_aer_net() -> StochasticPetriNet:
    """Build the controller net used by typed AER adapter tests."""
    net = StochasticPetriNet()
    net.add_place("aer_input", initial_tokens=0.0)
    net.add_place("aer_action_pos", initial_tokens=0.0)
    net.add_place("aer_action_neg", initial_tokens=0.0)
    net.add_transition("T_aer", threshold=0.1)
    net.add_arc("aer_input", "T_aer", weight=1.0)
    net.add_arc("T_aer", "aer_action_pos", weight=1.0)
    net.compile()
    return net


def _compile_aer_artifact(path: Path) -> Artifact:
    """Compile and persist an artifact with a direct ``aer_0`` injection."""
    compiled = FusionCompiler(bitstream_length=64, seed=7).compile(_build_aer_net())
    artifact = compiled.export_artifact(
        name="aer-controller-adapter",
        dt_control_s=0.001,
        readout_config={
            "actions": [{"name": "dI_PF3_A", "pos_place": 1, "neg_place": 2}],
            "gains": [1000.0],
            "abs_max": [5000.0],
            "slew_per_s": [1_000_000.0],
        },
        injection_config=[{"place_id": 0, "source": "aer_0", "scale": 1.0, "offset": 0.0, "clamp_0_1": True}],
    )
    save_artifact(artifact, str(path))
    return load_artifact(str(path))


def _controller(artifact: Artifact) -> NeuroSymbolicController:
    """Return a deterministic controller configured for direct AER features."""
    return NeuroSymbolicController(
        artifact=artifact,
        seed_base=42,
        targets=ControlTargets(),
        scales=ControlScales(),
        feature_axes=(),
        sc_n_passes=1,
        runtime_backend="numpy",
    )


def _observation(*, require_monotonic: bool = False) -> AERControlObservation:
    """Return a typed AER observation with a nonzero ``aer_0`` feature."""
    buffer = SpikeBuffer(capacity=8)
    buffer.extend(
        [
            SpikeEvent(neuron_id=0, timestamp_ns=10),
            SpikeEvent(neuron_id=1, timestamp_ns=20),
            SpikeEvent(neuron_id=9, timestamp_ns=30),
        ]
    )
    return AERControlObservation(
        timestamp_ns=40,
        spike_stream=buffer,
        decode_window_ns=40,
        n_features=2,
        require_monotonic=require_monotonic,
    )


def test_controller_step_accepts_typed_aer_observation(tmp_path: Path) -> None:
    """Typed AER observations drive the same production path as mappings."""
    artifact = _compile_aer_artifact(tmp_path / "aer.scpnctl.json")

    typed_action = _controller(artifact).step(_observation(), k=0)
    mapping_action = _controller(artifact).step({"aer_0": 1.0 / 3.0, "aer_1": 1.0 / 3.0}, k=0)

    assert typed_action == mapping_action
    assert typed_action["dI_PF3_A"] == pytest.approx(1000.0)


def test_controller_step_logs_aer_admission_metadata(tmp_path: Path) -> None:
    """Controller JSONL logs retain decoded AER admission evidence."""
    artifact = _compile_aer_artifact(tmp_path / "aer.scpnctl.json")
    log_path = tmp_path / "controller.jsonl"

    action = _controller(artifact).step(
        _observation(),
        k=3,
        log_path=log_path.name,
        log_root=tmp_path,
    )

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert action["dI_PF3_A"] == pytest.approx(1000.0)
    assert record["obs"] == {"aer_0": pytest.approx(1.0 / 3.0), "aer_1": pytest.approx(1.0 / 3.0)}
    assert record["aer_admission"] == {
        "capacity": 8,
        "retained_events": 0,
        "overflowed": False,
        "out_of_order_event_count": 0,
        "monotonic_input": True,
    }


def test_controller_step_propagates_aer_monotonic_rejection(tmp_path: Path) -> None:
    """Strict monotonic AER admission fails before controller injection."""
    artifact = _compile_aer_artifact(tmp_path / "aer.scpnctl.json")
    buffer = SpikeBuffer(capacity=8)
    buffer.push(SpikeEvent(neuron_id=0, timestamp_ns=20))
    buffer.push(SpikeEvent(neuron_id=0, timestamp_ns=10))
    observation = AERControlObservation(
        timestamp_ns=30,
        spike_stream=buffer,
        decode_window_ns=30,
        n_features=2,
        require_monotonic=True,
    )

    with pytest.raises(ValueError, match="monotonic timestamp admission"):
        _controller(artifact).step(observation, k=0)

    assert buffer.snapshot() == [SpikeEvent(neuron_id=0, timestamp_ns=20), SpikeEvent(neuron_id=0, timestamp_ns=10)]
