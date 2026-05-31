# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller Safety Case Tests
"""Workflow contract tests for controller safety-case evidence chaining."""

from __future__ import annotations

import json
import numpy as np
import pytest

from scpn_control.control.digital_twin_online_update import (
    BayesianUpdateResult,
    TwinObservation,
    TwinParameterPrior,
    digital_twin_update_evidence,
    validate_external_simulator_artifact,
)
from scpn_control.control.safety_case import (
    assert_controller_safety_case_admissible,
    controller_safety_case_evidence,
    load_controller_safety_case_evidence,
    save_controller_safety_case_evidence,
)
from scpn_control.core.differentiable_transport import (
    TransportRolloutGradientAudit,
    transport_campaign_metadata,
    transport_differentiability_evidence,
)
from scpn_control.scpn.artifact import (
    ActionReadout,
    Artifact,
    ArtifactMeta,
    CompilerInfo,
    FixedPoint,
    FormalVerificationEvidence,
    InitialState,
    PlaceSpec,
    Readout,
    SeedPolicy,
    Topology,
    TransitionSpec,
    WeightMatrix,
    Weights,
    compute_artifact_payload_sha256,
)


def _controller_artifact() -> Artifact:
    artifact = Artifact(
        meta=ArtifactMeta(
            artifact_version="1.0.0",
            name="safety-case-controller",
            dt_control_s=1.0e-3,
            stream_length=32,
            fixed_point=FixedPoint(data_width=16, fraction_bits=8, signed=True),
            firing_mode="binary",
            seed_policy=SeedPolicy(id="fixed", hash_fn="sha256", rng_family="pcg64"),
            created_utc="2026-05-31T00:00:00Z",
            compiler=CompilerInfo(name="test-compiler", version="1.0", git_sha="0" * 40),
        ),
        topology=Topology(
            places=[PlaceSpec(id=0, name="P0"), PlaceSpec(id=1, name="P1")],
            transitions=[TransitionSpec(id=0, name="T0", threshold=0.5)],
        ),
        weights=Weights(
            w_in=WeightMatrix(shape=[1, 2], data=[0.5, 0.0]),
            w_out=WeightMatrix(shape=[2, 1], data=[0.0, 0.5]),
        ),
        readout=Readout(
            actions=[ActionReadout(id=0, name="act0", pos_place=1, neg_place=0)],
            gains=[1.0],
            abs_max=[10.0],
            slew_per_s=[100.0],
        ),
        initial_state=InitialState(marking=[0.25, 0.0], place_injections=[]),
    )
    artifact.formal_verification = FormalVerificationEvidence(
        required=True,
        status="pass",
        backend="z3",
        solver="z3-solver 4.16.0",
        max_depth=8,
        checked_specs=["always_bounded_marking", "never_comarked"],
        artifact_sha256=compute_artifact_payload_sha256(artifact),
        report_sha256="a" * 64,
        claim_boundary="bounded SMT proof through depth 8 over compiled transition relation",
        report_uri="validation/reports/scpn_z3_formal.json",
    )
    return artifact


def _transport_evidence(controller_sha256: str):
    rho = np.linspace(0.05, 1.0, 16)
    profiles = np.vstack(
        [
            4.0 + 0.2 * (1.0 - rho),
            3.0 + 0.1 * (1.0 - rho),
            4.0 + 0.05 * (1.0 - rho),
            0.03 + 0.005 * rho,
        ]
    )
    chi = 0.04 * np.ones_like(profiles)
    sources = np.zeros_like(profiles)
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    metadata = transport_campaign_metadata(
        profiles,
        chi,
        sources,
        rho,
        1.0e-3,
        edge_values,
        backend="jax",
        gradient_tolerance=1.0e-6,
        equilibrium_psi=np.tile(np.linspace(0.2, 1.0, rho.size), (5, 1)),
    )
    audit = TransportRolloutGradientAudit(
        loss=0.125,
        epsilon=1.0e-5,
        tolerance=1.0e-6,
        checked_indices=((0, 0, 1),),
        source_max_abs_error=2.0e-7,
        passed=True,
    )
    return transport_differentiability_evidence(
        metadata,
        audit,
        controller_formal_artifact_sha256=controller_sha256,
    )


def _simulator_payload(code: str) -> dict[str, object]:
    digest = "a" * 64 if code == "TRANSP" else "b" * 64
    return {
        "schema_version": "1.0",
        "simulator_code": code,
        "artifact_uri": f"file:///validation/reports/digital_twin/{code.lower()}_case.nc",
        "artifact_sha256": digest,
        "case_id": f"{code}-shot-001",
        "time_base_s": [0.0, 0.01, 0.02],
        "signal_units": {"final_avg_temp": "keV"},
    }


def _digital_twin_evidence(controller_sha256: str):
    artifacts = tuple(validate_external_simulator_artifact(_simulator_payload(code)) for code in ("TRANSP", "TSC"))
    observation = TwinObservation(
        targets={"final_avg_temp": 2.0},
        tolerances={"final_avg_temp": 0.05},
        source="paired_transp_tsc_reference",
    )
    priors = (TwinParameterPrior("n_e", 0.8e20, 1.5e20, 1.0e20),)
    result = BayesianUpdateResult(
        best_parameters={"n_e": 1.1e20},
        best_loss=0.2,
        baseline_loss=0.8,
        evaluated_points=4,
        loss_history=(0.8, 0.5, 0.2),
        source=observation.source,
        evidence_kind="bounded_online_update",
    )
    return digital_twin_update_evidence(
        observation,
        priors,
        result,
        artifacts,
        controller_formal_artifact_sha256=controller_sha256,
    )


def test_controller_safety_case_binds_formal_transport_and_twin_evidence():
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence(controller_sha256)

    evidence = controller_safety_case_evidence(artifact, transport, digital_twin)

    assert evidence.controller_artifact_sha256 == controller_sha256
    assert evidence.formal_backend == "z3"
    assert evidence.transport_evidence_sha256
    assert evidence.digital_twin_evidence_sha256
    assert_controller_safety_case_admissible(evidence, artifact, transport, digital_twin)


def test_controller_safety_case_manifest_round_trips_with_integrity_digest(tmp_path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence(controller_sha256)
    evidence = controller_safety_case_evidence(artifact, transport, digital_twin)
    path = tmp_path / "controller_safety_case.json"

    save_controller_safety_case_evidence(evidence, path)
    loaded = load_controller_safety_case_evidence(path)

    assert loaded == evidence
    assert_controller_safety_case_admissible(loaded, artifact, transport, digital_twin)


def test_controller_safety_case_manifest_rejects_tampering(tmp_path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    path = tmp_path / "controller_safety_case.json"
    save_controller_safety_case_evidence(evidence, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evidence"]["formal_max_depth"] = 99
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity"):
        load_controller_safety_case_evidence(path)


def test_controller_safety_case_manifest_rejects_malformed_schema(tmp_path):
    path = tmp_path / "bad_controller_safety_case.json"
    path.write_text(json.dumps({"schema_version": 99, "evidence": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_controller_safety_case_evidence(path)


def test_controller_safety_case_rejects_mismatched_evidence_chain():
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence("b" * 64)

    with pytest.raises(ValueError, match="digital twin"):
        controller_safety_case_evidence(artifact, transport, digital_twin)


def test_controller_safety_case_rejects_non_passing_formal_proof():
    artifact = _controller_artifact()
    assert artifact.formal_verification is not None
    artifact.formal_verification.status = "blocked"
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence(controller_sha256)

    with pytest.raises(ValueError, match="safety-critical"):
        controller_safety_case_evidence(artifact, transport, digital_twin)
