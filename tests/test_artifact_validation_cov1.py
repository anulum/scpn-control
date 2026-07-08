# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — focused artifact validation COV-1 tests.
"""Focused coverage for artifact validator and compact-codec edge paths."""

from __future__ import annotations

import base64
import zlib
from collections.abc import Callable
from typing import cast

import pytest

import scpn_control.scpn.artifact as artifact_module
from scpn_control.scpn.artifact import (
    ActionReadout,
    Artifact,
    ArtifactMeta,
    ArtifactValidationError,
    CompilerInfo,
    FixedPoint,
    InitialState,
    PlaceInjection,
    PlaceSpec,
    Readout,
    SeedPolicy,
    Topology,
    TransitionSpec,
    WeightMatrix,
    Weights,
    _decode_u64_compact,
    _validate,
)


def _valid_artifact() -> Artifact:
    """Build the smallest artifact that passes local structural validation."""
    return Artifact(
        meta=ArtifactMeta(
            artifact_version="1.0.0",
            name="cov1-validator",
            dt_control_s=0.01,
            stream_length=64,
            fixed_point=FixedPoint(data_width=16, fraction_bits=8, signed=True),
            firing_mode="binary",
            seed_policy=SeedPolicy(id="deterministic", hash_fn="sha256", rng_family="splitmix64"),
            created_utc="2026-07-09T00:00:00Z",
            compiler=CompilerInfo(name="test", version="1.0", git_sha="0" * 40),
        ),
        topology=Topology(
            places=[PlaceSpec(id=0, name="p0"), PlaceSpec(id=1, name="p1")],
            transitions=[TransitionSpec(id=0, name="t0", threshold=0.5, margin=0.1, delay_ticks=0)],
        ),
        weights=Weights(
            w_in=WeightMatrix(shape=[1, 2], data=[0.0, 1.0]),
            w_out=WeightMatrix(shape=[2, 1], data=[0.0, 1.0]),
        ),
        readout=Readout(
            actions=[ActionReadout(id=0, name="coil", pos_place=0, neg_place=1)],
            gains=[1.0],
            abs_max=[10.0],
            slew_per_s=[100.0],
        ),
        initial_state=InitialState(
            marking=[0.0, 1.0],
            place_injections=[PlaceInjection(place_id=0, source="x_R_pos", scale=1.0, offset=0.0, clamp_0_1=True)],
        ),
    )


def _expect_validation_error(mutator: Callable[[Artifact], None], match: str) -> None:
    """Apply one invalid local mutation and assert the validator rejects it."""
    artifact = _valid_artifact()
    mutator(artifact)
    with pytest.raises(ArtifactValidationError, match=match):
        _validate(artifact)


def _compact_payload(raw: bytes) -> dict[str, object]:
    """Return a compact payload dictionary for arbitrary compressed bytes."""
    return {
        "encoding": "u64-le-zlib-base64",
        "count": 1,
        "data_u64_b64_zlib": base64.b64encode(raw).decode("ascii"),
    }


def test_decode_rejects_compressed_payload_over_configured_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Oversized compressed data fails before decompression."""
    monkeypatch.setattr(artifact_module, "MAX_COMPRESSED_BYTES", 1)

    with pytest.raises(ArtifactValidationError, match="Compressed payload too large"):
        _decode_u64_compact(_compact_payload(zlib.compress(b"12345678")))


def test_decode_rejects_payload_exceeding_decompressed_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The streaming decompressor limit is enforced via ``unconsumed_tail``."""
    monkeypatch.setattr(artifact_module, "MAX_DECOMPRESSED_BYTES", 7)

    with pytest.raises(ArtifactValidationError, match="exceeds configured limit"):
        _decode_u64_compact(_compact_payload(zlib.compress(b"1234567890123456")))


def test_decode_rejects_invalid_zlib_payload() -> None:
    """Base64-valid but non-zlib data is rejected as an invalid compact payload."""
    with pytest.raises(ArtifactValidationError, match="Invalid compact packed payload"):
        _decode_u64_compact(_compact_payload(b"not-zlib"))


def test_validate_rejects_fixed_point_type_and_range_errors() -> None:
    """Fixed-point metadata must use real integer widths and boolean signedness."""
    _expect_validation_error(
        lambda artifact: setattr(artifact.meta.fixed_point, "data_width", cast(int, True)),
        "data_width",
    )
    _expect_validation_error(
        lambda artifact: setattr(artifact.meta.fixed_point, "fraction_bits", 16),
        "fraction_bits must be <",
    )
    _expect_validation_error(
        lambda artifact: setattr(artifact.meta.fixed_point, "signed", cast(bool, "yes")),
        "signed must be a boolean",
    )


def test_validate_rejects_stream_length_and_dt_type_errors() -> None:
    """Control cadence metadata rejects boolean and non-finite values."""
    _expect_validation_error(lambda artifact: setattr(artifact.meta, "stream_length", cast(int, True)), "stream_length")
    _expect_validation_error(lambda artifact: setattr(artifact.meta, "dt_control_s", cast(float, True)), "dt_control_s")
    _expect_validation_error(lambda artifact: setattr(artifact.meta, "dt_control_s", float("nan")), "dt_control_s")


def test_validate_rejects_transition_numeric_type_errors() -> None:
    """Transition thresholds, margins, and delays must be finite numeric values."""
    _expect_validation_error(
        lambda artifact: setattr(artifact.topology.transitions[0], "threshold", cast(float, True)),
        "threshold",
    )
    _expect_validation_error(
        lambda artifact: setattr(artifact.topology.transitions[0], "threshold", float("nan")),
        "threshold",
    )
    _expect_validation_error(
        lambda artifact: setattr(artifact.topology.transitions[0], "margin", cast(float, True)),
        "margin",
    )
    _expect_validation_error(
        lambda artifact: setattr(artifact.topology.transitions[0], "margin", -0.1),
        "margin",
    )
    _expect_validation_error(
        lambda artifact: setattr(artifact.topology.transitions[0], "delay_ticks", cast(int, True)),
        "delay_ticks",
    )


def test_validate_rejects_place_injection_local_shape_errors() -> None:
    """Place injections are bounded to real places and finite affine terms."""
    _expect_validation_error(
        lambda artifact: setattr(artifact.initial_state.place_injections[0], "place_id", 2),
        "place_id",
    )
    _expect_validation_error(
        lambda artifact: setattr(artifact.initial_state.place_injections[0], "scale", float("inf")),
        "scale",
    )
    _expect_validation_error(
        lambda artifact: setattr(artifact.initial_state.place_injections[0], "offset", float("nan")),
        "offset",
    )
    _expect_validation_error(
        lambda artifact: setattr(artifact.initial_state.place_injections[0], "clamp_0_1", cast(bool, 1)),
        "clamp_0_1",
    )


def test_validate_rejects_readout_action_shape_errors() -> None:
    """Readout actions must use integer IDs, names, and in-bounds place IDs."""
    _expect_validation_error(
        lambda artifact: setattr(artifact.readout.actions[0], "id", cast(int, True)),
        "actions.id",
    )
    _expect_validation_error(lambda artifact: setattr(artifact.readout.actions[0], "name", ""), "actions.name")
    _expect_validation_error(
        lambda artifact: setattr(artifact.readout.actions[0], "neg_place", cast(int, True)),
        "neg_place",
    )
    _expect_validation_error(lambda artifact: setattr(artifact.readout.actions[0], "pos_place", 2), "pos_place")


def test_validate_rejects_readout_numeric_limit_errors() -> None:
    """Readout gains and actuator limits must be finite non-negative values."""
    _expect_validation_error(lambda artifact: artifact.readout.gains.__setitem__(0, cast(float, True)), "gains")
    _expect_validation_error(lambda artifact: artifact.readout.abs_max.__setitem__(0, -1.0), "abs_max")
    _expect_validation_error(lambda artifact: artifact.readout.slew_per_s.__setitem__(0, float("nan")), "slew_per_s")


def test_validate_accepts_minimal_cov1_artifact() -> None:
    """The local fixture remains a valid baseline for mutation tests."""
    _validate(_valid_artifact())
