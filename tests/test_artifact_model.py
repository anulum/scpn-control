# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for artifact model leaf

"""Drive production artifact model dataclasses and owner re-exports."""

from __future__ import annotations

import scpn_control.scpn.artifact as owner
import scpn_control.scpn.artifact_model as model


def test_owner_reexports_model_types_and_constants() -> None:
    """Owner product surface re-exports the model leaf types by identity."""
    assert owner.Artifact is model.Artifact
    assert owner.ArtifactMeta is model.ArtifactMeta
    assert owner.Topology is model.Topology
    assert owner.FormalVerificationEvidence is model.FormalVerificationEvidence
    assert owner.ARTIFACT_SCHEMA_VERSION is model.ARTIFACT_SCHEMA_VERSION
    assert owner.FORMAL_VERIFICATION_BACKENDS is model.FORMAL_VERIFICATION_BACKENDS


def test_artifact_model_topology_sizes() -> None:
    """Artifact place/transition counts follow topology lists."""
    places = [model.PlaceSpec(id=0, name="p0"), model.PlaceSpec(id=1, name="p1")]
    transitions = [model.TransitionSpec(id=0, name="t0", threshold=0.5)]
    art = model.Artifact(
        meta=model.ArtifactMeta(
            artifact_version=model.ARTIFACT_SCHEMA_VERSION,
            name="probe",
            dt_control_s=0.01,
            stream_length=1,
            fixed_point=model.FixedPoint(data_width=16, fraction_bits=8, signed=True),
            firing_mode="stochastic",
            seed_policy=model.SeedPolicy(id="s", hash_fn="blake3", rng_family="pcg64"),
            created_utc="2020-01-01T00:00:00Z",
            compiler=model.CompilerInfo(name="test", version="0", git_sha="0" * 40),
        ),
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
    assert art.nP == 2
    assert art.nT == 1
