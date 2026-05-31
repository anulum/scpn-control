# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller Safety Case Tests
"""Workflow contract tests for controller safety-case evidence chaining."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from scpn_control.control.codac_interface import CODACConfig, CODACInterface, codac_runtime_evidence
from scpn_control.control.digital_twin_online_update import (
    BayesianUpdateResult,
    TwinObservation,
    TwinParameterPrior,
    digital_twin_update_evidence,
    validate_external_simulator_artifact,
)
from scpn_control.control.hil_harness import ControlLoopMetrics, hil_replay_evidence
from scpn_control.control.safety_case import (
    ControllerSafetyCaseEvidence,
    ReadinessArtifactEvidence,
    SafetyCaseReadinessEvidence,
    assert_controller_safety_case_admissible,
    assert_controller_safety_case_readiness_admissible,
    controller_safety_case_evidence,
    evaluate_controller_safety_case_readiness,
    evaluate_controller_safety_case_readiness_from_artifacts,
    load_controller_safety_case_evidence,
    load_controller_safety_case_readiness,
    save_controller_safety_case_evidence,
    save_controller_safety_case_readiness,
)
from scpn_control.core.differentiable_transport import (
    TransportRolloutGradientAudit,
    transport_campaign_metadata,
    transport_differentiability_evidence,
)
from scpn_control.phase.realtime_monitor import RealtimeMonitor
from scpn_control.phase.ws_phase_stream import PhaseStreamServer, websocket_runtime_evidence
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
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.fpga_export import FPGAConfig, export_bitstream_project, hdl_export_evidence
from scpn_control.scpn.structure import StochasticPetriNet
from validation.validate_e2e_latency_evidence import build_e2e_latency_evidence_payload


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
        evaluated_points=3,
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


def _write_readiness_file(root: Path, uri: str, payload: dict[str, object]) -> str:
    path = root / uri
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_hardware_latency_payload() -> dict[str, object]:
    return build_e2e_latency_evidence_payload(
        {
            "iterations": 1000,
            "warmup": 100,
            "grid": "16x16",
            "target_hardware": {
                "id": "jetson-orin-nx-lab-unit-03",
                "class": "jetson",
                "machine": "aarch64",
                "processor": "arm",
                "platform": "Linux-PREEMPT_RT",
                "python": "3.12.0",
                "numpy": "2.0.0",
                "rt_kernel": "PREEMPT_RT-6.8-lab",
            },
            "kernel_only_us": {"p50": 45.0, "p95": 60.0, "p99": 80.0},
            "e2e_us": {"p50": 450.0, "p95": 700.0, "p99": 850.0},
            "e2e_overhead_factor": 10.0,
        }
    )


def _hil_replay_metrics() -> ControlLoopMetrics:
    return ControlLoopMetrics(
        iterations=1000,
        target_dt_us=1000.0,
        measured_dt_us=[450.0, 500.0, 550.0],
        p50_latency_us=500.0,
        p95_latency_us=700.0,
        p99_latency_us=850.0,
        max_latency_us=900.0,
        min_latency_us=450.0,
        mean_latency_us=600.0,
        jitter_std_us=25.0,
        overrun_count=0,
        overrun_fraction=0.0,
        sub_ms_achieved=True,
    )


def _target_hardware_hil_replay_payload() -> dict[str, object]:
    return hil_replay_evidence(
        _hil_replay_metrics(),
        controller_id="safety-case-controller",
        target_hardware_id="jetson-orin-nx-lab-unit-03",
        target_hardware_class="jetson-orin-preempt-rt",
        rt_kernel="linux-rt-6.8.0-lab",
        deployment_claim_allowed=True,
        generated_at="2026-05-31T00:00:00Z",
    )


def _codac_runtime_payload(*, facility_claim_allowed: bool = True) -> dict[str, object]:
    evidence = codac_runtime_evidence(
        CODACInterface(CODACConfig(), controller=object()),
        controller_id="safety-case-controller",
        observed_cycle_us=[450.0, 500.0, 550.0],
        interlock_checks=3,
        interlock_blocks=1,
        backpressure_events=0,
        generated_utc="2026-05-31T00:00:00Z",
        facility_claim_allowed=facility_claim_allowed,
    )
    return asdict(evidence)


def _websocket_runtime_payload(*, facility_claim_allowed: bool = True) -> dict[str, object]:
    monitor = RealtimeMonitor.from_paper27(L=4, N_per=10, zeta_uniform=0.5, psi_driver=0.0)
    server = PhaseStreamServer(
        monitor=monitor,
        api_key="secret-token-123456",
        require_tls=facility_claim_allowed,
    )
    counters = {
        "auth_successes": 1,
        "command_frames": 2,
        "broadcast_frames": 3,
        "peak_connected_clients": 1,
        "backpressure_disconnects": 0,
    }
    evidence = websocket_runtime_evidence(
        server,
        deployment_id="safety-case-phase-stream",
        bind_host="ops.phase.internal" if facility_claim_allowed else "127.0.0.1",
        uses_tls=facility_claim_allowed,
        counters=counters,
        generated_utc="2026-05-31T00:00:00Z",
        facility_claim_allowed=facility_claim_allowed,
    )
    return asdict(evidence)


def _hdl_export_payload(
    root: Path,
    controller_sha256: str,
    *,
    facility_claim_allowed: bool = True,
) -> dict[str, object]:
    net = StochasticPetriNet()
    net.add_place("P0", initial_tokens=0.8)
    net.add_place("P1", initial_tokens=0.0)
    net.add_transition("T0", threshold=0.5)
    net.add_arc("P0", "T0", weight=0.6)
    net.add_arc("T0", "P1", weight=0.9)
    compiled = FusionCompiler(bitstream_length=128, seed=7).compile(net)
    cfg = FPGAConfig(target="xilinx", clock_mhz=100.0)
    project_dir = root / "validation" / "reports" / "hardware" / "fpga_project"
    export_bitstream_project(compiled, cfg, project_dir)
    report_uri = "validation/reports/hardware/fpga_synth.rpt"
    report_path = root / report_uri
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("timing met\nslack 1.250 ns\n", encoding="utf-8")
    report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    evidence = hdl_export_evidence(
        compiled,
        cfg,
        project_dir,
        controller_artifact_sha256=controller_sha256,
        target_part="xc7a35tcpg236-1",
        synthesis_toolchain="vivado" if facility_claim_allowed else None,
        synthesis_report_sha256=report_digest if facility_claim_allowed else None,
        synthesis_report_uri=report_uri if facility_claim_allowed else None,
        timing_slack_ns=1.25 if facility_claim_allowed else None,
        generated_utc="2026-05-31T00:00:00Z",
        facility_claim_allowed=facility_claim_allowed,
    )
    return asdict(evidence)


def _local_hil_replay_payload() -> dict[str, object]:
    return hil_replay_evidence(
        _hil_replay_metrics(),
        controller_id="safety-case-controller",
        generated_at="2026-05-31T00:00:00Z",
    )


def _readiness_artifacts(root: Path, controller_sha256: str) -> tuple[ReadinessArtifactEvidence, ...]:
    external_uri = "validation/reports/external/physics_validation.json"
    timing_uri = "validation/reports/hardware/target_timing.json"
    hil_uri = "validation/reports/hardware/hil_replay.json"
    hdl_uri = "validation/reports/hardware/hdl_export.json"
    codac_uri = "validation/reports/hardware/codac_runtime.json"
    websocket_uri = "validation/reports/hardware/websocket_runtime.json"
    review_uri = "validation/reports/review/safety_review.json"
    external_digest = _write_readiness_file(root, external_uri, {"status": "pass", "source": "external"})
    timing_digest = _write_readiness_file(root, timing_uri, _target_hardware_latency_payload())
    hil_digest = _write_readiness_file(root, hil_uri, _target_hardware_hil_replay_payload())
    hdl_digest = _write_readiness_file(root, hdl_uri, _hdl_export_payload(root, controller_sha256))
    codac_digest = _write_readiness_file(root, codac_uri, _codac_runtime_payload())
    websocket_digest = _write_readiness_file(root, websocket_uri, _websocket_runtime_payload())
    review_digest = _write_readiness_file(root, review_uri, {"status": "pass", "source": "independent-review"})
    return (
        ReadinessArtifactEvidence(
            kind="external_physics_validation",
            artifact_sha256=external_digest,
            artifact_uri=external_uri,
            producer="independent-validation-campaign",
            generated_utc="2026-05-31T00:00:00Z",
        ),
        ReadinessArtifactEvidence(
            kind="target_hardware_timing",
            artifact_sha256=timing_digest,
            artifact_uri=timing_uri,
            producer="target-hardware-latency-bench",
            generated_utc="2026-05-31T00:00:00Z",
        ),
        ReadinessArtifactEvidence(
            kind="hil_replay_evidence",
            artifact_sha256=hil_digest,
            artifact_uri=hil_uri,
            producer="target-hardware-hil-replay",
            generated_utc="2026-05-31T00:00:00Z",
        ),
        ReadinessArtifactEvidence(
            kind="hdl_export_evidence",
            artifact_sha256=hdl_digest,
            artifact_uri=hdl_uri,
            producer="target-hardware-hdl-export",
            generated_utc="2026-05-31T00:00:00Z",
        ),
        ReadinessArtifactEvidence(
            kind="codac_runtime_evidence",
            artifact_sha256=codac_digest,
            artifact_uri=codac_uri,
            producer="target-hardware-codac-runtime",
            generated_utc="2026-05-31T00:00:00Z",
        ),
        ReadinessArtifactEvidence(
            kind="websocket_runtime_evidence",
            artifact_sha256=websocket_digest,
            artifact_uri=websocket_uri,
            producer="target-hardware-websocket-runtime",
            generated_utc="2026-05-31T00:00:00Z",
        ),
        ReadinessArtifactEvidence(
            kind="independent_safety_review",
            artifact_sha256=review_digest,
            artifact_uri=review_uri,
            producer="independent-safety-review",
            generated_utc="2026-05-31T00:00:00Z",
        ),
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


def test_controller_safety_case_manifest_rejects_unreadable_and_malformed_payloads(tmp_path: Path):
    unreadable_json = tmp_path / "not_json.json"
    unreadable_json.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="readable JSON"):
        load_controller_safety_case_evidence(unreadable_json)

    non_object = tmp_path / "non_object.json"
    non_object.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_controller_safety_case_evidence(non_object)

    missing_payload = tmp_path / "missing_payload.json"
    missing_payload.write_text(json.dumps({"schema_version": 1, "evidence": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence payload"):
        load_controller_safety_case_evidence(missing_payload)


def test_controller_safety_case_manifest_rejects_invalid_evidence_fields(tmp_path: Path):
    evidence = ControllerSafetyCaseEvidence(
        schema_version=1,
        controller_artifact_sha256="1" * 64,
        formal_report_sha256="2" * 64,
        formal_backend="z3",
        formal_max_depth=4,
        transport_evidence_sha256="3" * 64,
        digital_twin_evidence_sha256="4" * 64,
        claim_status="bounded safety-case evidence only",
    )
    path = tmp_path / "controller_safety_case.json"
    save_controller_safety_case_evidence(evidence, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evidence"]["formal_backend"] = "unchecked"
    payload["integrity_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="formal_backend|integrity"):
        load_controller_safety_case_evidence(path)


def test_controller_safety_case_readiness_blocks_without_external_evidence():
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )

    readiness = evaluate_controller_safety_case_readiness(evidence)

    assert isinstance(readiness, SafetyCaseReadinessEvidence)
    assert readiness.status == "blocked"
    assert readiness.safety_case_sha256
    assert "external_physics_validation_sha256" in readiness.blocking_reasons
    assert "target_hardware_timing_sha256" in readiness.blocking_reasons
    assert "hil_replay_evidence_sha256" in readiness.blocking_reasons
    assert "hdl_export_evidence_sha256" in readiness.blocking_reasons
    assert "codac_runtime_evidence_sha256" in readiness.blocking_reasons
    assert "websocket_runtime_evidence_sha256" in readiness.blocking_reasons
    assert "independent_safety_review_sha256" in readiness.blocking_reasons
    with pytest.raises(ValueError, match="blocked"):
        assert_controller_safety_case_readiness_admissible(readiness, evidence)


def test_controller_safety_case_readiness_accepts_complete_promotion_evidence():
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )

    readiness = evaluate_controller_safety_case_readiness(
        evidence,
        external_physics_validation_sha256="1" * 64,
        target_hardware_timing_sha256="2" * 64,
        hil_replay_evidence_sha256="4" * 64,
        hdl_export_evidence_sha256="6" * 64,
        codac_runtime_evidence_sha256="5" * 64,
        websocket_runtime_evidence_sha256="7" * 64,
        independent_safety_review_sha256="3" * 64,
    )

    assert readiness.status == "promotion_ready"
    assert readiness.blocking_reasons == ()
    assert readiness.external_physics_validation_sha256 == "1" * 64
    assert_controller_safety_case_readiness_admissible(readiness, evidence)


def test_controller_safety_case_readiness_accepts_typed_artifact_evidence(tmp_path: Path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = _readiness_artifacts(tmp_path, controller_sha256)

    readiness = evaluate_controller_safety_case_readiness_from_artifacts(
        evidence,
        artifacts,
        artifact_root=tmp_path,
    )

    assert readiness.status == "promotion_ready"
    assert readiness.external_physics_validation_sha256 == artifacts[0].artifact_sha256
    assert readiness.hdl_export_evidence_sha256 == artifacts[3].artifact_sha256
    assert readiness.codac_runtime_evidence_sha256 == artifacts[4].artifact_sha256
    assert readiness.websocket_runtime_evidence_sha256 == artifacts[5].artifact_sha256
    assert_controller_safety_case_readiness_admissible(readiness, evidence)


def test_controller_safety_case_readiness_rejects_unqualified_timing_artifact(tmp_path: Path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    timing_uri = artifacts[1].artifact_uri
    local_payload = _target_hardware_latency_payload()
    local_payload["target_hardware"]["id"] = "local-host-unqualified"
    local_payload["target_hardware"]["class"] = "unspecified-local"
    local_payload["target_hardware"]["rt_kernel"] = "unknown"
    timing_digest = _write_readiness_file(tmp_path, timing_uri, build_e2e_latency_evidence_payload(local_payload))
    artifacts[1] = ReadinessArtifactEvidence(
        kind="target_hardware_timing",
        artifact_sha256=timing_digest,
        artifact_uri=timing_uri,
        producer="target-hardware-latency-bench",
        generated_utc="2026-05-31T00:00:00Z",
    )

    with pytest.raises(ValueError, match="target hardware timing artifact is not admissible"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            tuple(artifacts),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_rejects_timing_artifact_digest_mismatch(tmp_path: Path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    artifacts[1] = ReadinessArtifactEvidence(
        kind="target_hardware_timing",
        artifact_sha256="f" * 64,
        artifact_uri=artifacts[1].artifact_uri,
        producer="target-hardware-latency-bench",
        generated_utc="2026-05-31T00:00:00Z",
    )

    with pytest.raises(ValueError, match="artifact_sha256"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            tuple(artifacts),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_rejects_local_hil_replay_artifact(tmp_path: Path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    hil_uri = artifacts[2].artifact_uri
    hil_digest = _write_readiness_file(tmp_path, hil_uri, _local_hil_replay_payload())
    artifacts[2] = ReadinessArtifactEvidence(
        kind="hil_replay_evidence",
        artifact_sha256=hil_digest,
        artifact_uri=hil_uri,
        producer="target-hardware-hil-replay",
        generated_utc="2026-05-31T00:00:00Z",
    )

    with pytest.raises(ValueError, match="HIL replay artifact is not admissible"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            tuple(artifacts),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_rejects_local_hdl_export_artifact(tmp_path: Path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    hdl_uri = artifacts[3].artifact_uri
    hdl_digest = _write_readiness_file(
        tmp_path,
        hdl_uri,
        _hdl_export_payload(tmp_path, controller_sha256, facility_claim_allowed=False),
    )
    artifacts[3] = ReadinessArtifactEvidence(
        kind="hdl_export_evidence",
        artifact_sha256=hdl_digest,
        artifact_uri=hdl_uri,
        producer="target-hardware-hdl-export",
        generated_utc="2026-05-31T00:00:00Z",
    )

    with pytest.raises(ValueError, match="HDL export artifact is not admissible"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            tuple(artifacts),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_rejects_hdl_export_controller_mismatch(tmp_path: Path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    hdl_uri = artifacts[3].artifact_uri
    hdl_digest = _write_readiness_file(
        tmp_path,
        hdl_uri,
        _hdl_export_payload(tmp_path, "b" * 64),
    )
    artifacts[3] = ReadinessArtifactEvidence(
        kind="hdl_export_evidence",
        artifact_sha256=hdl_digest,
        artifact_uri=hdl_uri,
        producer="target-hardware-hdl-export",
        generated_utc="2026-05-31T00:00:00Z",
    )

    with pytest.raises(ValueError, match="not bound"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            tuple(artifacts),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_rejects_local_codac_runtime_artifact(tmp_path: Path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    codac_uri = artifacts[4].artifact_uri
    codac_digest = _write_readiness_file(
        tmp_path,
        codac_uri,
        _codac_runtime_payload(facility_claim_allowed=False),
    )
    artifacts[4] = ReadinessArtifactEvidence(
        kind="codac_runtime_evidence",
        artifact_sha256=codac_digest,
        artifact_uri=codac_uri,
        producer="target-hardware-codac-runtime",
        generated_utc="2026-05-31T00:00:00Z",
    )

    with pytest.raises(ValueError, match="CODAC runtime artifact is not admissible"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            tuple(artifacts),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_rejects_local_websocket_runtime_artifact(tmp_path: Path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    websocket_uri = artifacts[5].artifact_uri
    websocket_digest = _write_readiness_file(
        tmp_path,
        websocket_uri,
        _websocket_runtime_payload(facility_claim_allowed=False),
    )
    artifacts[5] = ReadinessArtifactEvidence(
        kind="websocket_runtime_evidence",
        artifact_sha256=websocket_digest,
        artifact_uri=websocket_uri,
        producer="target-hardware-websocket-runtime",
        generated_utc="2026-05-31T00:00:00Z",
    )

    with pytest.raises(ValueError, match="WebSocket runtime artifact is not admissible"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            tuple(artifacts),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_artifacts_reject_wrong_kind_and_unsafe_uri(tmp_path: Path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    valid_artifacts = _readiness_artifacts(tmp_path, controller_sha256)

    with pytest.raises(ValueError, match="kind"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            (
                ReadinessArtifactEvidence(
                    kind="external_physics_validation",
                    artifact_sha256="1" * 64,
                    artifact_uri="validation/reports/external/physics_validation.json",
                    producer="independent-validation-campaign",
                    generated_utc="2026-05-31T00:00:00Z",
                ),
            ),
            artifact_root=tmp_path,
        )
    with pytest.raises(ValueError, match="artifact_uri"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            (
                ReadinessArtifactEvidence(
                    kind="external_physics_validation",
                    artifact_sha256="1" * 64,
                    artifact_uri="../outside.json",
                    producer="independent-validation-campaign",
                    generated_utc="2026-05-31T00:00:00Z",
                ),
                ReadinessArtifactEvidence(
                    kind="target_hardware_timing",
                    artifact_sha256="2" * 64,
                    artifact_uri="validation/reports/hardware/target_timing.json",
                    producer="target-hardware-latency-bench",
                    generated_utc="2026-05-31T00:00:00Z",
                ),
                valid_artifacts[2],
                valid_artifacts[3],
                valid_artifacts[4],
            ),
            artifact_root=tmp_path,
        )

    for unsafe_uri in ("", "file:validation/report.json", "validation//report.json", "validation\\report.json"):
        with pytest.raises(ValueError, match="artifact_uri"):
            evaluate_controller_safety_case_readiness_from_artifacts(
                evidence,
                (
                    ReadinessArtifactEvidence(
                        kind="external_physics_validation",
                        artifact_sha256="1" * 64,
                        artifact_uri=unsafe_uri,
                        producer="independent-validation-campaign",
                        generated_utc="2026-05-31T00:00:00Z",
                    ),
                    ReadinessArtifactEvidence(
                        kind="target_hardware_timing",
                        artifact_sha256="2" * 64,
                        artifact_uri="validation/reports/hardware/target_timing.json",
                        producer="target-hardware-latency-bench",
                        generated_utc="2026-05-31T00:00:00Z",
                    ),
                    valid_artifacts[2],
                    valid_artifacts[3],
                    valid_artifacts[4],
                ),
                artifact_root=tmp_path,
            )


def test_controller_safety_case_readiness_artifacts_reject_invalid_envelope_contracts(tmp_path: Path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    valid_artifacts = _readiness_artifacts(tmp_path, controller_sha256)

    with pytest.raises(ValueError, match="non-empty tuple"):
        evaluate_controller_safety_case_readiness_from_artifacts(evidence, (), artifact_root=tmp_path)
    with pytest.raises(ValueError, match="non-empty tuple"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence, list(valid_artifacts), artifact_root=tmp_path
        )
    with pytest.raises(ValueError, match="ReadinessArtifactEvidence"):
        evaluate_controller_safety_case_readiness_from_artifacts(evidence, (object(),), artifact_root=tmp_path)

    invalid_digest = ReadinessArtifactEvidence(
        kind="external_physics_validation",
        artifact_sha256="g" * 64,
        artifact_uri="validation/reports/external/physics_validation.json",
        producer="independent-validation-campaign",
        generated_utc="2026-05-31T00:00:00Z",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            (
                invalid_digest,
                valid_artifacts[1],
                valid_artifacts[2],
                valid_artifacts[3],
                valid_artifacts[4],
                valid_artifacts[5],
            ),
            artifact_root=tmp_path,
        )

    empty_producer = ReadinessArtifactEvidence(
        kind="external_physics_validation",
        artifact_sha256="1" * 64,
        artifact_uri="validation/reports/external/physics_validation.json",
        producer="",
        generated_utc="2026-05-31T00:00:00Z",
    )
    with pytest.raises(ValueError, match="producer"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            (
                empty_producer,
                valid_artifacts[1],
                valid_artifacts[2],
                valid_artifacts[3],
                valid_artifacts[4],
                valid_artifacts[5],
            ),
            artifact_root=tmp_path,
        )

    empty_generated = ReadinessArtifactEvidence(
        kind="external_physics_validation",
        artifact_sha256="1" * 64,
        artifact_uri="validation/reports/external/physics_validation.json",
        producer="independent-validation-campaign",
        generated_utc="",
    )
    with pytest.raises(ValueError, match="generated_utc"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            (
                empty_generated,
                valid_artifacts[1],
                valid_artifacts[2],
                valid_artifacts[3],
                valid_artifacts[4],
                valid_artifacts[5],
            ),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_artifacts_reject_missing_files(tmp_path: Path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = list(_readiness_artifacts(tmp_path, controller_sha256))
    artifacts[0] = ReadinessArtifactEvidence(
        kind="external_physics_validation",
        artifact_sha256="1" * 64,
        artifact_uri="validation/reports/external/missing_physics_validation.json",
        producer="independent-validation-campaign",
        generated_utc="2026-05-31T00:00:00Z",
    )

    with pytest.raises(ValueError, match="does not resolve"):
        evaluate_controller_safety_case_readiness_from_artifacts(evidence, tuple(artifacts), artifact_root=tmp_path)


def test_controller_safety_case_readiness_artifacts_reject_duplicate_kind(tmp_path: Path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    artifacts = _readiness_artifacts(tmp_path, controller_sha256)

    with pytest.raises(ValueError, match="duplicate"):
        evaluate_controller_safety_case_readiness_from_artifacts(
            evidence,
            (artifacts[0], artifacts[0], artifacts[1], artifacts[3]),
            artifact_root=tmp_path,
        )


def test_controller_safety_case_readiness_manifest_round_trips(tmp_path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    readiness = evaluate_controller_safety_case_readiness(
        evidence,
        external_physics_validation_sha256="1" * 64,
        target_hardware_timing_sha256="2" * 64,
        hil_replay_evidence_sha256="4" * 64,
        hdl_export_evidence_sha256="6" * 64,
        codac_runtime_evidence_sha256="5" * 64,
        websocket_runtime_evidence_sha256="7" * 64,
        independent_safety_review_sha256="3" * 64,
    )
    path = tmp_path / "controller_safety_case_readiness.json"

    save_controller_safety_case_readiness(readiness, path)
    loaded = load_controller_safety_case_readiness(path)

    assert loaded == readiness
    assert_controller_safety_case_readiness_admissible(loaded, evidence)


def test_controller_safety_case_readiness_manifest_rejects_tampering(tmp_path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    readiness = evaluate_controller_safety_case_readiness(
        evidence,
        external_physics_validation_sha256="1" * 64,
        target_hardware_timing_sha256="2" * 64,
        hil_replay_evidence_sha256="4" * 64,
        hdl_export_evidence_sha256="6" * 64,
        codac_runtime_evidence_sha256="5" * 64,
        websocket_runtime_evidence_sha256="7" * 64,
        independent_safety_review_sha256="3" * 64,
    )
    path = tmp_path / "controller_safety_case_readiness.json"
    save_controller_safety_case_readiness(readiness, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["readiness"]["status"] = "blocked"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity"):
        load_controller_safety_case_readiness(path)


def test_controller_safety_case_readiness_manifest_rejects_malformed_schema(tmp_path):
    path = tmp_path / "bad_controller_safety_case_readiness.json"
    path.write_text(json.dumps({"schema_version": 99, "readiness": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_controller_safety_case_readiness(path)


def test_controller_safety_case_readiness_manifest_rejects_unreadable_and_malformed_payloads(tmp_path: Path):
    unreadable_json = tmp_path / "not_json_readiness.json"
    unreadable_json.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="readable JSON"):
        load_controller_safety_case_readiness(unreadable_json)

    non_object = tmp_path / "non_object_readiness.json"
    non_object.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_controller_safety_case_readiness(non_object)

    missing_payload = tmp_path / "missing_readiness.json"
    missing_payload.write_text(json.dumps({"schema_version": 1, "readiness": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="payload"):
        load_controller_safety_case_readiness(missing_payload)


def test_controller_safety_case_readiness_rejects_drift_and_bad_digest():
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    readiness = evaluate_controller_safety_case_readiness(
        evidence,
        external_physics_validation_sha256="1" * 64,
        target_hardware_timing_sha256="2" * 64,
        hil_replay_evidence_sha256="4" * 64,
        hdl_export_evidence_sha256="6" * 64,
        codac_runtime_evidence_sha256="5" * 64,
        websocket_runtime_evidence_sha256="7" * 64,
        independent_safety_review_sha256="3" * 64,
    )
    drifted = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    object.__setattr__(drifted, "formal_max_depth", drifted.formal_max_depth + 1)

    with pytest.raises(ValueError, match="safety_case_sha256"):
        assert_controller_safety_case_readiness_admissible(readiness, drifted)
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_controller_safety_case_readiness(
            evidence,
            external_physics_validation_sha256="not-a-digest",
            target_hardware_timing_sha256="2" * 64,
            hil_replay_evidence_sha256="4" * 64,
            hdl_export_evidence_sha256="6" * 64,
            codac_runtime_evidence_sha256="5" * 64,
            websocket_runtime_evidence_sha256="7" * 64,
            independent_safety_review_sha256="3" * 64,
        )


def test_controller_safety_case_readiness_admission_rejects_type_and_state_drift():
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    evidence = controller_safety_case_evidence(
        artifact,
        _transport_evidence(controller_sha256),
        _digital_twin_evidence(controller_sha256),
    )
    readiness = evaluate_controller_safety_case_readiness(
        evidence,
        external_physics_validation_sha256="1" * 64,
        target_hardware_timing_sha256="2" * 64,
        hil_replay_evidence_sha256="4" * 64,
        hdl_export_evidence_sha256="6" * 64,
        codac_runtime_evidence_sha256="5" * 64,
        websocket_runtime_evidence_sha256="7" * 64,
        independent_safety_review_sha256="3" * 64,
    )

    with pytest.raises(ValueError, match="readiness must"):
        assert_controller_safety_case_readiness_admissible(object(), evidence)
    with pytest.raises(ValueError, match="safety_case must"):
        assert_controller_safety_case_readiness_admissible(readiness, object())

    stale_schema = SafetyCaseReadinessEvidence(**{**readiness.__dict__, "schema_version": 1})
    with pytest.raises(ValueError, match="schema_version"):
        assert_controller_safety_case_readiness_admissible(stale_schema, evidence)

    bad_status = SafetyCaseReadinessEvidence(**{**readiness.__dict__, "status": "unchecked"})
    with pytest.raises(ValueError, match="status"):
        assert_controller_safety_case_readiness_admissible(bad_status, evidence)

    drifted = SafetyCaseReadinessEvidence(**{**readiness.__dict__, "claim_status": "bounded stale readiness"})
    with pytest.raises(ValueError, match="evidence mismatch"):
        assert_controller_safety_case_readiness_admissible(drifted, evidence)


def test_controller_safety_case_rejects_mismatched_evidence_chain():
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence("b" * 64)

    with pytest.raises(ValueError, match="digital twin"):
        controller_safety_case_evidence(artifact, transport, digital_twin)

    with pytest.raises(ValueError, match="transport evidence"):
        controller_safety_case_evidence(
            artifact, _transport_evidence("c" * 64), _digital_twin_evidence(controller_sha256)
        )


def test_controller_safety_case_rejects_invalid_public_input_types(tmp_path: Path):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence(controller_sha256)
    evidence = controller_safety_case_evidence(artifact, transport, digital_twin)

    with pytest.raises(ValueError, match="safety_case must"):
        evaluate_controller_safety_case_readiness(object())
    with pytest.raises(ValueError, match="controller_artifact must"):
        controller_safety_case_evidence(object(), transport, digital_twin)
    with pytest.raises(ValueError, match="transport_evidence must"):
        controller_safety_case_evidence(artifact, object(), digital_twin)
    with pytest.raises(ValueError, match="digital_twin_evidence must"):
        controller_safety_case_evidence(artifact, transport, object())
    with pytest.raises(ValueError, match="readiness must"):
        save_controller_safety_case_readiness(object(), tmp_path / "readiness.json")
    with pytest.raises(ValueError, match="evidence must"):
        save_controller_safety_case_evidence(object(), tmp_path / "evidence.json")
    with pytest.raises(ValueError, match="evidence must"):
        assert_controller_safety_case_admissible(object(), artifact, transport, digital_twin)

    stale_schema = ControllerSafetyCaseEvidence(**{**evidence.__dict__, "schema_version": 2})
    with pytest.raises(ValueError, match="schema_version"):
        assert_controller_safety_case_admissible(stale_schema, artifact, transport, digital_twin)


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("controller_artifact_sha256", "9" * 64, "controller_artifact_sha256"),
        ("formal_report_sha256", "9" * 64, "formal_report_sha256"),
        ("formal_backend", "explicit-state", "formal_backend"),
        ("formal_max_depth", 99, "formal_max_depth"),
        ("transport_evidence_sha256", "9" * 64, "transport_evidence_sha256"),
        ("digital_twin_evidence_sha256", "9" * 64, "digital_twin_evidence_sha256"),
        ("claim_status", "bounded but stale", "claim_status"),
    ],
)
def test_controller_safety_case_admission_rejects_each_evidence_field_drift(field_name, replacement, message):
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence(controller_sha256)
    evidence = controller_safety_case_evidence(artifact, transport, digital_twin)
    drifted = ControllerSafetyCaseEvidence(**{**evidence.__dict__, field_name: replacement})

    with pytest.raises(ValueError, match=message):
        assert_controller_safety_case_admissible(drifted, artifact, transport, digital_twin)


def test_controller_safety_case_rejects_bad_transport_and_twin_claims():
    artifact = _controller_artifact()
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence(controller_sha256)

    failed_transport = _transport_evidence(controller_sha256)
    object.__setattr__(failed_transport, "audit_passed", False)
    with pytest.raises(ValueError, match="passed gradient audit"):
        controller_safety_case_evidence(artifact, failed_transport, digital_twin)

    non_jax_transport = _transport_evidence(controller_sha256)
    object.__setattr__(non_jax_transport, "backend", "numpy")
    with pytest.raises(ValueError, match="JAX backend"):
        controller_safety_case_evidence(artifact, non_jax_transport, digital_twin)

    non_improving_twin = _digital_twin_evidence(controller_sha256)
    object.__setattr__(non_improving_twin, "improved_over_baseline", False)
    with pytest.raises(ValueError, match="improve over baseline"):
        controller_safety_case_evidence(artifact, transport, non_improving_twin)

    missing_tsc_twin = _digital_twin_evidence(controller_sha256)
    object.__setattr__(missing_tsc_twin, "simulator_codes", ("TRANSP",))
    with pytest.raises(ValueError, match="TRANSP and TSC"):
        controller_safety_case_evidence(artifact, transport, missing_tsc_twin)


def test_controller_safety_case_rejects_non_passing_formal_proof():
    artifact = _controller_artifact()
    assert artifact.formal_verification is not None
    artifact.formal_verification.status = "blocked"
    controller_sha256 = compute_artifact_payload_sha256(artifact)
    transport = _transport_evidence(controller_sha256)
    digital_twin = _digital_twin_evidence(controller_sha256)

    with pytest.raises(ValueError, match="safety-critical"):
        controller_safety_case_evidence(artifact, transport, digital_twin)
