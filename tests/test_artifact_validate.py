# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for artifact validate leaf

"""Drive production artifact validation leaf and owner re-exports."""

from __future__ import annotations

import pytest

import scpn_control.scpn.artifact as owner
import scpn_control.scpn.artifact_model as model
import scpn_control.scpn.artifact_validate as validate_leaf


def _minimal_meta(*, firing_mode: str = "binary") -> model.ArtifactMeta:
    """Build a minimal ArtifactMeta with a selectable firing mode."""
    return model.ArtifactMeta(
        artifact_version=model.ARTIFACT_SCHEMA_VERSION,
        name="probe",
        dt_control_s=0.01,
        stream_length=1,
        fixed_point=model.FixedPoint(data_width=16, fraction_bits=8, signed=True),
        firing_mode=firing_mode,
        seed_policy=model.SeedPolicy(id="s", hash_fn="blake3", rng_family="pcg64"),
        created_utc="2020-01-01T00:00:00Z",
        compiler=model.CompilerInfo(name="test", version="0", git_sha="0" * 40),
    )


def _minimal_artifact(*, meta: model.ArtifactMeta | None = None) -> model.Artifact:
    """Build a contract-valid two-place / one-transition artifact."""
    places = [model.PlaceSpec(id=0, name="p0"), model.PlaceSpec(id=1, name="p1")]
    transitions = [model.TransitionSpec(id=0, name="t0", threshold=0.5)]
    return model.Artifact(
        meta=meta if meta is not None else _minimal_meta(),
        topology=model.Topology(places=places, transitions=transitions),
        weights=model.Weights(
            w_in=model.WeightMatrix(shape=[1, 2], data=[0.0, 0.0]),
            w_out=model.WeightMatrix(shape=[2, 1], data=[0.0, 0.0]),
        ),
        readout=model.Readout(
            actions=[model.ActionReadout(id=0, name="a0", pos_place=0, neg_place=1)],
            gains=[1.0],
            abs_max=[1.0],
            slew_per_s=[1.0],
        ),
        initial_state=model.InitialState(marking=[0.0, 0.0], place_injections=[]),
        validate_on_init=False,
    )


def test_owner_reexports_validate_surface_by_identity() -> None:
    """Owner product surface re-exports the validate leaf by identity."""
    assert owner.ArtifactValidationError is validate_leaf.ArtifactValidationError
    assert owner.validate_artifact is validate_leaf.validate_artifact
    assert owner.validate_safety_critical_artifact is validate_leaf.validate_safety_critical_artifact


def test_validate_artifact_accepts_minimal_contract() -> None:
    """Structural contract accepts a well-formed minimal controller artifact."""
    art = _minimal_artifact()
    validate_leaf.validate_artifact(art)


def test_validate_artifact_rejects_bad_firing_mode() -> None:
    """Invalid firing_mode is rejected with ArtifactValidationError."""
    art = _minimal_artifact(meta=_minimal_meta(firing_mode="stochastic"))
    with pytest.raises(validate_leaf.ArtifactValidationError, match="firing_mode"):
        validate_leaf.validate_artifact(art)


def test_validate_safety_critical_requires_formal_evidence() -> None:
    """Safety-critical admit fails closed without formal_verification evidence."""
    art = _minimal_artifact()
    with pytest.raises(validate_leaf.ArtifactValidationError, match="formal_verification"):
        validate_leaf.validate_safety_critical_artifact(art)


def test_model_init_validation_uses_validate_leaf() -> None:
    """Direct Artifact construction with validate_on_init routes through the leaf."""
    with pytest.raises(validate_leaf.ArtifactValidationError, match="firing_mode"):
        model.Artifact(
            meta=_minimal_meta(firing_mode="stochastic"),
            topology=model.Topology(
                places=[model.PlaceSpec(id=0, name="p0"), model.PlaceSpec(id=1, name="p1")],
                transitions=[model.TransitionSpec(id=0, name="t0", threshold=0.5)],
            ),
            weights=model.Weights(
                w_in=model.WeightMatrix(shape=[1, 2], data=[0.0, 0.0]),
                w_out=model.WeightMatrix(shape=[2, 1], data=[0.0, 0.0]),
            ),
            readout=model.Readout(
                actions=[model.ActionReadout(id=0, name="a0", pos_place=0, neg_place=1)],
                gains=[1.0],
                abs_max=[1.0],
                slew_per_s=[1.0],
            ),
            initial_state=model.InitialState(marking=[0.0, 0.0], place_injections=[]),
            validate_on_init=True,
        )
