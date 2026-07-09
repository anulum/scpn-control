# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Artifact JSON schema contract tests.
"""Regression tests for the public ``.scpnctl.json`` schema contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

import scpn_control.scpn.artifact as artifact_module
from scpn_control.scpn.artifact import (
    ActionReadout,
    Artifact,
    ArtifactMeta,
    CompilerInfo,
    FixedPoint,
    InitialState,
    PackedWeights,
    PackedWeightsGroup,
    PlaceInjection,
    PlaceSpec,
    Readout,
    SeedPolicy,
    Topology,
    TransitionSpec,
    WeightMatrix,
    Weights,
    get_artifact_json_schema,
    load_artifact,
    save_artifact,
)
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.structure import StochasticPetriNet

JsonObject = dict[str, Any]


def _object(value: object) -> JsonObject:
    """Return ``value`` as a JSON object after an assertion narrow."""

    assert isinstance(value, dict)
    return cast(JsonObject, value)


def _array(value: object) -> list[object]:
    """Return ``value`` as a JSON array after an assertion narrow."""

    assert isinstance(value, list)
    return cast(list[object], value)


def _payload(path: Path) -> JsonObject:
    """Load a JSON object payload from ``path``."""

    return _object(json.loads(path.read_text(encoding="utf-8")))


def _compiled_artifact() -> Artifact:
    """Compile a controller artifact through the production compiler path."""

    net = StochasticPetriNet()
    for name in ("source", "guard", "sink"):
        net.add_place(name)
    net.add_transition("move", threshold=0.5)
    net.add_arc("source", "move", 0.8)
    net.add_arc("guard", "move", 0.2)
    net.add_arc("move", "sink", 1.0)

    compiled = FusionCompiler(bitstream_length=64, seed=11).compile(net, firing_mode="fractional")
    artifact = compiled.export_artifact(
        name="schema-contract",
        readout_config={
            "actions": [{"name": "act0", "pos_place": 2, "neg_place": 0}],
            "gains": [1.0],
            "abs_max": [10.0],
            "slew_per_s": [100.0],
        },
        injection_config=[
            {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ],
    )
    return cast(Artifact, artifact)


def _packed_artifact() -> Artifact:
    """Return a packed artifact fixture that exercises both packed matrices."""

    return Artifact(
        meta=ArtifactMeta(
            artifact_version="1.0.0",
            name="packed-schema-contract",
            dt_control_s=0.001,
            stream_length=64,
            fixed_point=FixedPoint(data_width=16, fraction_bits=10, signed=False),
            firing_mode="binary",
            seed_policy=SeedPolicy(id="default", hash_fn="splitmix64", rng_family="xoshiro256++"),
            created_utc="2026-07-09T00:00:00Z",
            compiler=CompilerInfo(name="FusionCompiler", version="1.0.0", git_sha="abc1234"),
            firing_margin=0.05,
        ),
        topology=Topology(
            places=[PlaceSpec(id=0, name="P0"), PlaceSpec(id=1, name="P1")],
            transitions=[TransitionSpec(id=0, name="T0", threshold=0.5, delay_ticks=1)],
        ),
        weights=Weights(
            w_in=WeightMatrix(shape=[1, 2], data=[1.0, 0.0]),
            w_out=WeightMatrix(shape=[2, 1], data=[0.0, 1.0]),
            packed=PackedWeightsGroup(
                words_per_stream=1,
                w_in_packed=PackedWeights(shape=[1, 2, 1], data_u64=[2**64 - 1, 0]),
                w_out_packed=PackedWeights(shape=[2, 1, 1], data_u64=[0, 2**64 - 1]),
            ),
        ),
        readout=Readout(
            actions=[ActionReadout(id=0, name="act0", pos_place=1, neg_place=0)],
            gains=[1.0],
            abs_max=[10.0],
            slew_per_s=[100.0],
        ),
        initial_state=InitialState(
            marking=[1.0, 0.0],
            place_injections=[PlaceInjection(place_id=0, source="x_R_pos", scale=1.0, offset=0.0, clamp_0_1=True)],
        ),
    )


def test_artifact_json_schema_matches_saved_payload_contract(tmp_path: Path) -> None:
    """Schema required fields and closed objects match current saved payloads."""

    schema = get_artifact_json_schema()
    artifact = _compiled_artifact()
    artifact.meta.notes = "schema contract note"
    out = tmp_path / "schema-current.scpnctl.json"
    save_artifact(artifact, out)
    payload = _payload(out)

    assert schema["additionalProperties"] is False
    assert set(_array(schema["required"])) == {"meta", "topology", "weights", "readout", "initial_state"}
    assert set(_object(schema["properties"])) == set(payload) | {"formal_verification"}

    meta_schema = _object(_object(schema["properties"])["meta"])
    meta_payload = _object(payload["meta"])
    assert set(_array(meta_schema["required"])) == set(meta_payload) - {"notes"}
    assert _object(_object(meta_schema["properties"])["artifact_version"])["const"] == "1.0.0"
    assert _object(_object(meta_schema["properties"])["firing_mode"])["enum"] == ["binary", "fractional"]
    assert _object(_object(meta_schema["properties"])["fixed_point"])["additionalProperties"] is False
    assert _object(_object(meta_schema["properties"])["seed_policy"])["additionalProperties"] is False
    assert _object(_object(meta_schema["properties"])["compiler"])["additionalProperties"] is False

    topology_schema = _object(_object(schema["properties"])["topology"])
    transition_schema = _object(_object(_object(topology_schema["properties"])["transitions"])["items"])
    transition_payload = _object(_array(_object(payload["topology"])["transitions"])[0])
    assert set(_array(transition_schema["required"])) == set(transition_payload) - {"margin"}
    assert _object(_object(transition_schema["properties"])["delay_ticks"])["minimum"] == 0

    readout_schema = _object(_object(schema["properties"])["readout"])
    assert set(_array(readout_schema["required"])) == {"actions", "gains", "limits"}
    assert _object(_object(readout_schema["properties"])["gains"])["required"] == ["per_action"]
    assert _object(_object(readout_schema["properties"])["limits"])["required"] == [
        "per_action_abs_max",
        "slew_per_s",
    ]


def test_artifact_json_schema_matches_raw_and_compact_packed_payloads(tmp_path: Path) -> None:
    """Schema packed variants match raw and compact outputs from ``save_artifact``."""

    schema = get_artifact_json_schema()
    properties = _object(schema["properties"])
    packed_group = _object(_object(_object(properties["weights"])["properties"])["packed"])
    packed_weight_variants = _array(_object(_object(schema["definitions"])["packed_weight"])["oneOf"])
    raw_schema = _object(packed_weight_variants[0])
    compact_schema = _object(packed_weight_variants[1])
    artifact = _packed_artifact()
    raw_path = tmp_path / "raw.scpnctl.json"
    compact_path = tmp_path / "compact.scpnctl.json"

    save_artifact(artifact, raw_path, compact_packed=False)
    save_artifact(artifact, compact_path, compact_packed=True)
    raw_payload = _payload(raw_path)
    compact_payload = _payload(compact_path)
    raw_w_in = _object(_object(_object(raw_payload["weights"])["packed"])["w_in_packed"])
    compact_w_in = _object(_object(_object(compact_payload["weights"])["packed"])["w_in_packed"])

    assert packed_group["additionalProperties"] is False
    assert packed_group["required"] == ["words_per_stream", "w_in_packed"]
    assert set(_array(raw_schema["required"])) == set(raw_w_in)
    assert set(_array(compact_schema["required"])) == set(compact_w_in)
    assert _object(_object(_object(raw_schema["properties"])["data_u64"])["items"])["maximum"] == 2**64 - 1
    assert _object(_object(compact_schema["properties"])["encoding"])["const"] == "u64-le-zlib-base64"
    assert "data_b64" not in _object(packed_group["properties"])

    assert load_artifact(raw_path).weights.packed is not None
    assert load_artifact(compact_path).weights.packed is not None


def test_formal_schema_drift_guard_rejects_unmapped_evidence_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """Formal evidence fields cannot drift away from the emitted schema."""

    monkeypatch.setattr(
        artifact_module,
        "FORMAL_VERIFICATION_ALLOWED_FIELDS",
        artifact_module.FORMAL_VERIFICATION_ALLOWED_FIELDS | {"foreign_field"},
    )

    with pytest.raises(RuntimeError, match="formal_verification schema drift"):
        artifact_module._formal_verification_schema()
