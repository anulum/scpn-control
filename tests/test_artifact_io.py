# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for artifact IO leaf

"""Drive production artifact load/save IO leaf and owner re-exports."""

from __future__ import annotations

from pathlib import Path

import scpn_control.scpn.artifact as owner
import scpn_control.scpn.artifact_io as io
import scpn_control.scpn.artifact_model as model


def _minimal_artifact() -> model.Artifact:
    """Build a contract-valid two-place / one-transition artifact."""
    places = [model.PlaceSpec(id=0, name="p0"), model.PlaceSpec(id=1, name="p1")]
    transitions = [model.TransitionSpec(id=0, name="t0", threshold=0.5)]
    return model.Artifact(
        meta=model.ArtifactMeta(
            artifact_version=model.ARTIFACT_SCHEMA_VERSION,
            name="probe",
            dt_control_s=0.01,
            stream_length=1,
            fixed_point=model.FixedPoint(data_width=16, fraction_bits=8, signed=True),
            firing_mode="binary",
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


def test_owner_reexports_io_surface_by_identity() -> None:
    """Owner product surface re-exports the IO leaf by identity."""
    assert owner.load_artifact is io.load_artifact
    assert owner.save_artifact is io.save_artifact
    assert owner.compute_artifact_payload_sha256 is io.compute_artifact_payload_sha256


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Save then load preserves payload hash and topology sizes."""
    art = _minimal_artifact()
    path = tmp_path / "probe.scpnctl.json"
    io.save_artifact(art, path)
    loaded = io.load_artifact(path)
    assert loaded.nP == 2
    assert loaded.nT == 1
    assert io.compute_artifact_payload_sha256(loaded) == io.compute_artifact_payload_sha256(art)


def test_payload_hash_is_stable_hex() -> None:
    """Canonical payload hash is a 64-character lowercase hex digest."""
    digest = io.compute_artifact_payload_sha256(_minimal_artifact())
    assert len(digest) == 64
    int(digest, 16)
