# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller Safety Case Tests
"""Fixtures and artifact builders for controller safety-case tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scpn_control.control import hil_evidence
from scpn_control.control.codac_interface import CODACConfig, CODACInterface, codac_runtime_evidence
from scpn_control.control.digital_twin_online_update import (
    BayesianUpdateResult,
    DigitalTwinUpdateEvidence,
    TwinObservation,
    TwinParameterPrior,
    digital_twin_update_evidence,
    validate_external_simulator_artifact,
)
from scpn_control.control.hil_harness import ControlLoopMetrics, hil_replay_evidence
from scpn_control.control.safety_case import (
    ReadinessArtifactEvidence,
)
from scpn_control.core.differentiable_transport import (
    TransportDifferentiabilityEvidence,
    TransportRolloutGradientAudit,
    transport_campaign_metadata,
    transport_differentiability_evidence,
)
from scpn_control.phase.realtime_monitor import RealtimeMonitor
from scpn_control.phase.ws_phase_stream import (
    PhaseStreamServer,
    websocket_runtime_evidence,
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


def _transport_evidence(controller_sha256: str) -> TransportDifferentiabilityEvidence:
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


def _digital_twin_evidence(controller_sha256: str) -> DigitalTwinUpdateEvidence:
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
            "generated_utc": "2026-05-31T00:00:00Z",
            "command": (
                "env PYTHONPATH=src python benchmarks/e2e_control_latency.py "
                "--iterations 1000 --warmup 100 "
                "--target-hardware-id jetson-orin-nx-lab-unit-03 "
                "--target-hardware-class jetson --rt-kernel PREEMPT_RT-6.8-lab "
                "--output-json validation/reports/hardware/target_timing.json"
            ),
            "evidence_class": "local_regression",
            "production_claim_allowed": False,
            "context": {
                "cpu_affinity": [2, 3],
                "isolation_method": "reserved-core-taskset",
                "loadavg_start": [0.12, 0.10, 0.08],
                "loadavg_end": [0.13, 0.10, 0.08],
                "governor": "performance",
                "heavy_jobs_running": "not-observed",
            },
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
        deployment_claim_allowed=False,
        generated_at="2026-05-31T00:00:00Z",
    )


def _qualified_hil_replay_payload() -> dict[str, object]:
    """Return a HIL replay payload as an external qualifier would issue it.

    The producer in this package never qualifies its own replay. The payload is
    therefore the target-hardware replay with the admission block set to the
    qualified status and both digests recomputed over the changed content, the
    way ``tests/test_hil_evidence.py`` builds its qualified specimen.
    """
    payload = _target_hardware_hil_replay_payload()
    admission = payload["admission"]
    assert isinstance(admission, dict)
    admission["deployment_claim_allowed"] = True
    admission["claim_status"] = "qualified_target_hardware_deployment_evidence"
    payload["replay_digest"] = hil_evidence._sha256_json(
        {key: payload[key] for key in ("controller_id", "target_hardware", "timing", "safety_events", "admission")}
    )
    payload["payload_sha256"] = hil_evidence._sha256_json(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )
    return payload


def _codac_runtime_payload(*, facility_claim_allowed: bool = False) -> dict[str, object]:
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
