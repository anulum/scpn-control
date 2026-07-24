# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Artifact runtime-validation contract tests.
"""Runtime validation contracts for controller artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import scpn_control.scpn.artifact as artifact_module
import scpn_control.scpn.controller as controller_module
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
    load_artifact,
    save_artifact,
    validate_artifact,
)
from scpn_control.scpn.contracts import ControlScales, ControlTargets
from scpn_control.scpn.controller import NeuroSymbolicController, _artifact_topology_digest, _matrix_entries
from scpn_control.scpn.runtime_safety_certificate import (
    CertificateReplayResult,
    ControllerRuntimeBinding,
    RuntimeTarget,
    TimingEnvelope,
)


def _valid_artifact() -> Artifact:
    """Return the smallest valid artifact accepted by controller init."""
    return Artifact(
        meta=ArtifactMeta(
            artifact_version="1.0.0",
            name="runtime-validation-contract",
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


def _invalid_readout_artifact(*, validate_on_init: bool = True) -> Artifact:
    """Build an artifact with a readout length mismatch."""
    valid = _valid_artifact()
    return Artifact(
        meta=valid.meta,
        topology=valid.topology,
        weights=valid.weights,
        readout=Readout(
            actions=valid.readout.actions,
            gains=[],
            abs_max=valid.readout.abs_max,
            slew_per_s=valid.readout.slew_per_s,
        ),
        initial_state=valid.initial_state,
        validate_on_init=validate_on_init,
    )


def _controller_kwargs(artifact: Artifact) -> dict[str, Any]:
    """Return common controller constructor arguments."""
    return {
        "artifact": artifact,
        "seed_base": 9,
        "targets": ControlTargets(R_target_m=6.2, Z_target_m=0.0),
        "scales": ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        "feature_axes": [],
        "runtime_backend": "numpy",
    }


def _runtime_target() -> RuntimeTarget:
    """Return a runtime target for controller admission tests."""
    return RuntimeTarget(name="pytest-target", architecture="x86_64", runtime="numpy", toolchain="pytest")


def _timing_envelope() -> TimingEnvelope:
    """Return a schedulable timing envelope for admission tests."""
    return TimingEnvelope(
        control_period_us=1000.0,
        worst_case_response_us=100.0,
        deadline_us=500.0,
        proof_firing_depth=4,
    )


def _runtime_binding(artifact: Artifact, topology_sha256: str | None = None) -> ControllerRuntimeBinding:
    """Return a runtime binding tied to an artifact topology digest."""
    return ControllerRuntimeBinding(
        controller_id="runtime-validation-contract",
        controller_config={"runtime_backend": "numpy", "sc_n_passes": 1},
        petri_topology_sha256=_artifact_topology_digest(artifact) if topology_sha256 is None else topology_sha256,
        snn_parameters={"bitstream_length": 64, "seed": 0},
        solver_mode="scpn-numpy",
        runtime_target=_runtime_target(),
        timing_envelope=_timing_envelope(),
    )


def test_artifact_post_init_rejects_invalid_direct_artifact() -> None:
    """Direct artifact construction validates before a controller can use it."""
    with pytest.raises(ArtifactValidationError, match="readout.gains length"):
        _invalid_readout_artifact()


def test_artifact_post_init_bypass_still_requires_explicit_validation() -> None:
    """The branch-test bypass never makes an invalid artifact admissible."""
    artifact = _invalid_readout_artifact(validate_on_init=False)

    with pytest.raises(ArtifactValidationError, match="readout.gains length"):
        validate_artifact(artifact)


def test_load_artifact_calls_public_runtime_validator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """File loading routes parsed artifacts through the public validator."""
    import scpn_control.scpn.artifact_io as artifact_io_module

    artifact_path = tmp_path / "valid.scpnctl.json"
    save_artifact(_valid_artifact(), artifact_path)
    validated: list[Artifact] = []
    original_validate = artifact_io_module.validate_artifact

    def spy(artifact: Artifact) -> None:
        validated.append(artifact)
        original_validate(artifact)

    # Load/save live on the IO leaf (CTL-G07 R4-S4); spy the leaf binding.
    monkeypatch.setattr(artifact_io_module, "validate_artifact", spy)

    loaded = load_artifact(artifact_path)

    assert validated == [loaded]


def test_controller_init_calls_public_runtime_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Controller construction validates direct artifacts before runtime setup."""
    artifact = _valid_artifact()
    validated: list[Artifact] = []

    def spy(candidate: Artifact) -> None:
        validated.append(candidate)
        validate_artifact(candidate)

    monkeypatch.setattr(controller_module, "validate_artifact", spy)

    controller = NeuroSymbolicController(**_controller_kwargs(artifact))

    assert controller.artifact is artifact
    assert validated == [artifact]


def test_controller_rejects_bypassed_invalid_artifact_before_runtime_arrays() -> None:
    """Controller init rejects artifacts that bypassed construction validation."""
    artifact = _invalid_readout_artifact(validate_on_init=False)

    with pytest.raises(ArtifactValidationError, match="readout.gains length"):
        NeuroSymbolicController(**_controller_kwargs(artifact))


def test_controller_topology_digest_rejects_malformed_weight_shapes() -> None:
    """Runtime-certificate topology digests validate dense matrix metadata."""
    with pytest.raises(ValueError, match="two dimensions"):
        _matrix_entries([1.0], [1])
    with pytest.raises(ValueError, match="data length"):
        _matrix_entries([1.0], [1, 2])


def test_controller_runtime_safety_requires_complete_evidence_bundle() -> None:
    """Controller init rejects partial runtime-certificate evidence."""
    artifact = _valid_artifact()

    with pytest.raises(ValueError, match="requires certificate, binding, target, and replay inputs"):
        NeuroSymbolicController(**_controller_kwargs(artifact), runtime_safety_certificate={"status": "pass"})


def test_controller_runtime_safety_rejects_binding_topology_mismatch() -> None:
    """Controller init compares the live artifact digest before admission."""
    artifact = _valid_artifact()
    target = _runtime_target()
    binding = _runtime_binding(artifact, topology_sha256="b" * 64)

    with pytest.raises(ValueError, match="topology does not match"):
        NeuroSymbolicController(
            **_controller_kwargs(artifact),
            runtime_safety_certificate={"status": "pass"},
            runtime_safety_binding=binding,
            runtime_safety_target=target,
            runtime_safety_replay=CertificateReplayResult(True, True, True, True),
        )


def test_controller_runtime_safety_admission_sets_certificate_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Controller init stores the admitted runtime-certificate digest."""
    artifact = _valid_artifact()
    target = _runtime_target()
    binding = _runtime_binding(artifact)
    replay = CertificateReplayResult(True, True, True, True)
    certificate = {"status": "pass"}
    admitted_payload = {"payload_sha256": "a" * 64}
    seen: dict[str, object] = {}

    def admit(
        candidate: dict[str, Any],
        *,
        live_binding: ControllerRuntimeBinding,
        live_runtime_target: RuntimeTarget,
        replay: CertificateReplayResult,
    ) -> dict[str, Any]:
        seen["certificate"] = candidate
        seen["binding"] = live_binding
        seen["target"] = live_runtime_target
        seen["replay"] = replay
        return admitted_payload

    monkeypatch.setattr(controller_module, "assert_runtime_certificate_admissible", admit)

    controller = NeuroSymbolicController(
        **_controller_kwargs(artifact),
        runtime_safety_certificate=certificate,
        runtime_safety_binding=binding,
        runtime_safety_target=target,
        runtime_safety_replay=replay,
    )

    assert seen == {
        "certificate": certificate,
        "binding": binding,
        "target": target,
        "replay": replay,
    }
    assert controller.runtime_safety_admitted is True
    assert controller.runtime_safety_certificate_payload == admitted_payload
    assert controller.runtime_safety_certificate_sha256 == "a" * 64
