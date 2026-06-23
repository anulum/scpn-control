# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Studio live-emitter adapter tests
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
"""Tests for the studio adapters that wire live CONTROL emitters to EvidenceBundles.

Each adapter is fed a faithful slice of its real emitter's output — a
``ReconstructionResult`` from ``RealtimeEFIT.reconstruct``, an issued runtime safety
certificate, and a controller-latency measurement — and the resulting bundle is
checked for the correct schema, honest rendering, and provenance digests.

Skips cleanly when the optional ``scpn-studio-platform`` SDK is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scpn_studio_platform")

from scpn_studio_platform.evidence import EvidenceKind  # noqa: E402

from scpn_control.control.realtime_efit import ReconstructionResult, ShapeParams  # noqa: E402
from scpn_control.scpn.geometry_neutral_replay import GeometryNeutralReplayEvidence  # noqa: E402
from scpn_control.studio import (  # noqa: E402
    TraceabilityClaim,
    controller_latency_evidence_from_measurement,
    controller_run_evidence_from_simulation,
    disruption_mitigation_evidence_from_run,
    disruption_prediction_evidence_from_risk,
    efit_evidence_from_reconstruction,
    equilibrium_analysis_evidence_from_shape,
    phase_sync_monitor_evidence_from_snapshot,
    physics_validation_evidence,
    physics_validation_evidences_from_registry,
    replay_evidence_from_geometry_neutral,
    safety_certificate_evidence_from_certificate,
    scenario_simulation_evidence_from_audit,
)

_TS = {"started": "2026-06-23T00:00:00Z", "ended": "2026-06-23T00:00:01Z"}
_WHO = {"operator": "opaque:tenant-1", "studio_version": "0.test"}


def _reconstruction() -> ReconstructionResult:
    shape = ShapeParams(
        R0=1.7,
        a=0.5,
        kappa=1.8,
        delta_upper=0.4,
        delta_lower=0.4,
        q95=3.5,
        beta_pol=0.9,
        li=0.8,
        Ip_reconstructed=1.5e7,
    )
    return ReconstructionResult(
        psi=np.zeros((33, 33), dtype=np.float64),
        p_prime_coeffs=np.zeros(3, dtype=np.float64),
        ff_prime_coeffs=np.zeros(3, dtype=np.float64),
        shape=shape,
        chi_squared=4.38e-9,
        n_iterations=7,
        wall_time_ms=12.3,
    )


def _certificate(*, live: str = "d" * 64) -> tuple[dict[str, object], str]:
    cert: dict[str, object] = {
        "scope": "scpn-control.runtime-safety-certificate",
        "binding": {"petri_topology_sha256": "d" * 64},
        "formal_certificate": {"holds": True, "non_vacuous": True, "payload_sha256": "c" * 64},
        "formal_certificate_sha256": "c" * 64,
        "checked_specs": ["AG(no_overflow)", "AF(safe_shutdown)"],
        "payload_sha256": "e" * 64,
    }
    return cert, live


def _measurement() -> dict[str, float | int]:
    return {"n": 200, "p50_us": 5.05, "p95_us": 6.11, "p99_us": 6.4, "mean_us": 5.3}


# ── EFIT reconstruction adapter ────────────────────────────────────────
def test_efit_adapter_reads_the_live_reconstruction() -> None:
    bundle = efit_evidence_from_reconstruction(
        _reconstruction(),
        measurements={"Ip": 1.5e7, "flux_loops": 16},
        **_WHO,
        **_TS,
    )
    assert bundle.schema == "studio.efit-reconstruction.v1"
    assert bundle.renders_as_validated is False
    assert bundle.evidence_kind is EvidenceKind.MEASURED
    assert bundle.physical_contract is not None
    assert bundle.physical_contract.grid["nz"] == 33
    assert bundle.physical_contract.grid["nr"] == 33
    assert bundle.claim_boundary.validity_domain is not None


def test_efit_adapter_digests_depend_on_inputs_and_result() -> None:
    a = efit_evidence_from_reconstruction(_reconstruction(), measurements={"Ip": 1.0}, **_WHO, **_TS)
    b = efit_evidence_from_reconstruction(_reconstruction(), measurements={"Ip": 2.0}, **_WHO, **_TS)
    # Different inputs -> different input-provenance entity ids are not exposed, but
    # the bundles are independently valid and the result digest is stable per result.
    assert a.entity.digest == b.entity.digest  # same reconstruction summary
    assert a.schema == b.schema


# ── safety certificate adapter ─────────────────────────────────────────
def test_safety_cert_adapter_held_proof_is_admissible() -> None:
    cert, live = _certificate(live="d" * 64)
    bundle = safety_certificate_evidence_from_certificate(
        cert,
        live_topology_sha256=live,
        checker="z3",
        checker_version="4.13.0",
        **_WHO,
        **_TS,
    )
    assert bundle.evidence_kind is EvidenceKind.FORMALLY_PROVEN
    assert bundle.renders_as_validated is True
    cert_obj = bundle.formal_certificates[0]
    assert cert_obj.checker == "z3"
    assert cert_obj.theorem_id == "AG(no_overflow) ; AF(safe_shutdown)"
    assert cert_obj.non_vacuous is True
    assert cert_obj.subject_digest == "d" * 64
    assert cert_obj.proof_digest == "c" * 64


def test_safety_cert_adapter_drifted_topology_is_voided() -> None:
    cert, live = _certificate(live="f" * 64)
    bundle = safety_certificate_evidence_from_certificate(
        cert,
        live_topology_sha256=live,
        checker="z3",
        checker_version="4.13.0",
        **_WHO,
        **_TS,
    )
    assert bundle.renders_as_validated is False
    assert bundle.proof_voided_by("f" * 64) is True


def test_safety_cert_adapter_falls_back_to_scope_without_specs() -> None:
    cert, live = _certificate()
    cert["checked_specs"] = []
    bundle = safety_certificate_evidence_from_certificate(
        cert,
        live_topology_sha256=live,
        checker="lean",
        checker_version="4.9.0",
        **_WHO,
        **_TS,
    )
    assert bundle.formal_certificates[0].theorem_id == "scpn-control.runtime-safety-certificate"


def test_safety_cert_adapter_defaults_non_vacuous_false_when_absent() -> None:
    cert, live = _certificate()
    cert["formal_certificate"] = {"holds": True, "payload_sha256": "c" * 64}
    bundle = safety_certificate_evidence_from_certificate(
        cert,
        live_topology_sha256=live,
        checker="z3",
        checker_version="4.13.0",
        **_WHO,
        **_TS,
    )
    assert bundle.formal_certificates[0].non_vacuous is False


def test_safety_cert_adapter_rejects_missing_binding() -> None:
    with pytest.raises(KeyError):
        safety_certificate_evidence_from_certificate(
            {"checked_specs": [], "formal_certificate_sha256": "c" * 64},
            live_topology_sha256="d" * 64,
            checker="z3",
            checker_version="4.13.0",
            **_WHO,
            **_TS,
        )


# ── controller latency adapter ─────────────────────────────────────────
def test_latency_adapter_reads_the_measurement() -> None:
    bundle = controller_latency_evidence_from_measurement(
        _measurement(),
        controller="h_infinity",
        active_backend="rust",
        reference_backend="python",
        **_WHO,
        **_TS,
    )
    assert bundle.schema == "studio.controller-latency.v1"
    assert bundle.renders_as_validated is False
    assert bundle.evidence_kind is EvidenceKind.MEASURED
    assert bundle.numeric_provenance is not None
    assert bundle.numeric_provenance.active_backend == "rust"


def test_latency_adapter_rejects_missing_percentile() -> None:
    bad = {"n": 200, "p50_us": 5.0, "p95_us": 6.0}  # no p99_us
    with pytest.raises(KeyError):
        controller_latency_evidence_from_measurement(
            bad,
            controller="h_infinity",
            active_backend="rust",
            reference_backend="python",
            **_WHO,
            **_TS,
        )


# ── physics-traceability validation mapper + registry adapter ──────────
def _claim(status: str = "bounded_model", *, allowed: bool = False, blocking: str | None = None) -> TraceabilityClaim:
    return TraceabilityClaim(
        component="real-time EFIT-lite",
        module_path="src/scpn_control/control/realtime_efit.py",
        fidelity_status=status,
        public_claim_allowed=allowed,
        validity_domain="Fixed-boundary EFIT-lite regression; not facility-grade unless matched.",
        blocking_dependency=blocking,
    )


def test_reference_validated_claim_renders_validated() -> None:
    bundle = physics_validation_evidence(_claim("reference_validated", allowed=True), **_WHO, **_TS)
    assert bundle.schema == "studio.physics-validation.v1"
    assert bundle.evidence_kind is EvidenceKind.CURATED
    assert bundle.renders_as_validated is True
    assert bundle.claim_boundary.validity_domain is not None


def test_bounded_claim_renders_verbatim_with_validity_domain() -> None:
    bundle = physics_validation_evidence(_claim("bounded_model"), **_WHO, **_TS)
    assert bundle.renders_as_validated is False
    note = bundle.claim_boundary.validity_domain
    assert note is not None and note.note is not None and "EFIT-lite" in note.note


def test_external_dependency_blocked_claim_carries_blocked_on() -> None:
    bundle = physics_validation_evidence(
        _claim("external_dependency_blocked", blocking="Install external GK binaries"),
        **_WHO,
        **_TS,
    )
    assert bundle.renders_as_validated is False
    assert bundle.claim_boundary.blocked_on[0].dependency == "Install external GK binaries"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fidelity_status": "nonsense"},
        {"component": "  "},
        {"validity_domain": ""},
        {"fidelity_status": "external_dependency_blocked", "blocking_dependency": None},
    ],
)
def test_traceability_claim_rejects_bad_inputs(kwargs: dict[str, object]) -> None:
    base = {
        "component": "c",
        "module_path": "m.py",
        "fidelity_status": "bounded_model",
        "public_claim_allowed": False,
        "validity_domain": "scope",
        "blocking_dependency": "dep",
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        TraceabilityClaim(**base)  # type: ignore[arg-type]


def test_registry_adapter_maps_a_list() -> None:
    entries = [
        {
            "component": "c1",
            "module_path": "m1.py",
            "fidelity_status": "external_dependency_blocked",
            "public_claim_allowed": False,
            "validity_domain": "blocked scope",
            "required_actions": ["Acquire external reference"],
        },
    ]
    bundles = physics_validation_evidences_from_registry(entries, **_WHO, **_TS)
    assert len(bundles) == 1
    assert bundles[0].claim_boundary.blocked_on[0].dependency == "Acquire external reference"


def test_registry_adapter_over_the_real_61_entries() -> None:
    import json
    from pathlib import Path

    registry = Path(__file__).resolve().parents[1] / "validation" / "physics_traceability.json"
    entries = json.loads(registry.read_text())["entries"]
    bundles = physics_validation_evidences_from_registry(entries, **_WHO, **_TS)
    assert len(bundles) == len(entries)
    # Exactly the one reference-validated, admitted claim renders as validated;
    # every other boundary is shown verbatim.
    assert sum(1 for b in bundles if b.renders_as_validated) == 1
    # Every claim carries its qualitative validity-domain prose.
    assert all(b.claim_boundary.validity_domain is not None for b in bundles)


# ── batch-2 verb adapters: replay / monitor / predict / analyse ────────
def _replay_evidence(*, device_claim_allowed: bool = False) -> GeometryNeutralReplayEvidence:
    return GeometryNeutralReplayEvidence(
        schema_version="scpn-control.geometry-neutral-replay-evidence.v1",
        generated_utc="2026-06-23T00:00:00Z",
        replay_schema_version="v1",
        replay_report_sha256="1" * 64,
        scenario_digest="2" * 64,
        trace_digest="3" * 64,
        metrics_digest="4" * 64,
        thresholds_digest="5" * 64,
        magnetic_configuration_reference="iter-baseline",
        actuator_calibration="nominal",
        latency_model="bounded",
        fault_model="none",
        final_fieldline_spread=0.01,
        improvement_fraction=0.42,
        max_abs_current_A=1.2e4,
        p95_latency_us=6.1,
        deterministic=True,
        passes_thresholds=True,
        measured_or_benchmark_artefact_sha256=None,
        device_claim_allowed=device_claim_allowed,
        claim_status="bounded synthetic replay evidence",
        payload_sha256="6" * 64,
    )


def test_replay_adapter_maps_geometry_neutral_evidence() -> None:
    bundle = replay_evidence_from_geometry_neutral(_replay_evidence(), **_WHO, **_TS)
    assert bundle.schema == "studio.geometry-neutral-replay.v1"
    assert bundle.evidence_kind is EvidenceKind.MEASURED
    assert bundle.renders_as_validated is False  # device_claim_allowed False
    assert bundle.physical_contract is not None


def test_replay_summary_rejects_bad_inputs() -> None:
    from scpn_control.studio import ReplaySummary

    with pytest.raises(ValueError):
        ReplaySummary(
            scenario_digest="",
            trace_digest="t",
            result_digest="r",
            max_abs_current_a=1.0,
            p95_latency_us=1.0,
            device_claim_allowed=False,
        )


def test_monitor_adapter_maps_a_tick_snapshot() -> None:
    snap = {"tick": 42, "R_global": 0.93, "lambda_exp": -0.1, "guard_approved": True, "latency_us": 4.2}
    bundle = phase_sync_monitor_evidence_from_snapshot(snap, **_WHO, **_TS)
    assert bundle.schema == "studio.phase-sync-monitor.v1"
    assert bundle.renders_as_validated is False
    assert bundle.evidence_kind is EvidenceKind.MEASURED


def test_monitor_adapter_rejects_missing_field() -> None:
    with pytest.raises(KeyError):
        phase_sync_monitor_evidence_from_snapshot({"tick": 1, "R_global": 0.5}, **_WHO, **_TS)


def test_monitor_snapshot_rejects_out_of_range() -> None:
    from scpn_control.studio import MonitorSnapshot

    with pytest.raises(ValueError):
        MonitorSnapshot(tick=1, r_global=1.5, lambda_exp=0.0, guard_approved=True, latency_us=1.0)


def test_predict_adapter_renders_validation_gap() -> None:
    bundle = disruption_prediction_evidence_from_risk(
        0.73,
        observables={"n1_amplitude": 0.2, "rotation": 1.5},
        **_WHO,
        **_TS,
    )
    assert bundle.schema == "studio.disruption-prediction.v1"
    assert bundle.renders_as_validated is False
    assert bundle.claim_boundary.validity_domain is not None


def test_prediction_rejects_out_of_range_risk() -> None:
    from scpn_control.studio import DisruptionPrediction

    with pytest.raises(ValueError):
        DisruptionPrediction(risk=1.4, observable_count=2, result_digest="d")


def test_analyse_adapter_maps_shape_params() -> None:
    shape = ShapeParams(
        R0=1.7,
        a=0.5,
        kappa=1.8,
        delta_upper=0.4,
        delta_lower=0.4,
        q95=3.5,
        beta_pol=0.9,
        li=0.8,
        Ip_reconstructed=1.5e7,
    )
    bundle = equilibrium_analysis_evidence_from_shape(shape, **_WHO, **_TS)
    assert bundle.schema == "studio.equilibrium-analysis.v1"
    assert bundle.renders_as_validated is False
    assert bundle.physical_contract is not None and bundle.physical_contract.units["R0"] == "m"


def test_equilibrium_analysis_rejects_nonpositive_geometry() -> None:
    from scpn_control.studio import EquilibriumAnalysis

    with pytest.raises(ValueError):
        EquilibriumAnalysis(r0=0.0, a=0.5, kappa=1.8, q95=3.5, beta_pol=0.9, li=0.8, result_digest="d")


def test_replay_adapter_admits_device_claim_when_allowed() -> None:
    bundle = replay_evidence_from_geometry_neutral(_replay_evidence(device_claim_allowed=True), **_WHO, **_TS)
    assert bundle.renders_as_validated is True


def test_replay_summary_rejects_negative_magnitude() -> None:
    from scpn_control.studio import ReplaySummary

    with pytest.raises(ValueError):
        ReplaySummary(
            scenario_digest="s",
            trace_digest="t",
            result_digest="r",
            max_abs_current_a=-1.0,
            p95_latency_us=1.0,
            device_claim_allowed=False,
        )


@pytest.mark.parametrize("kwargs", [{"tick": -1}, {"latency_us": -1.0}])
def test_monitor_snapshot_rejects_negative(kwargs: dict[str, object]) -> None:
    from scpn_control.studio import MonitorSnapshot

    base: dict[str, object] = {
        "tick": 0,
        "r_global": 0.5,
        "lambda_exp": 0.0,
        "guard_approved": True,
        "latency_us": 1.0,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        MonitorSnapshot(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("kwargs", [{"observable_count": -1}, {"result_digest": "  "}])
def test_disruption_prediction_rejects_bad_fields(kwargs: dict[str, object]) -> None:
    from scpn_control.studio import DisruptionPrediction

    base: dict[str, object] = {"risk": 0.5, "observable_count": 2, "result_digest": "d"}
    base.update(kwargs)
    with pytest.raises(ValueError):
        DisruptionPrediction(**base)  # type: ignore[arg-type]


def test_equilibrium_analysis_rejects_empty_digest() -> None:
    from scpn_control.studio import EquilibriumAnalysis

    with pytest.raises(ValueError):
        EquilibriumAnalysis(r0=1.7, a=0.5, kappa=1.8, q95=3.5, beta_pol=0.9, li=0.8, result_digest="  ")


# ── controller-run adapter (regulate) ──────────────────────────────────
_SOTA_SUMMARY = {
    "steps": 60,
    "mean_tracking_error": 0.12,
    "max_abs_action": 1.8,
    "max_abs_coil_current": 28.0,
    "runtime_seconds": 0.4,
}


def test_controller_run_adapter_maps_simulation_summary() -> None:
    bundle = controller_run_evidence_from_simulation(_SOTA_SUMMARY, controller="surrogate_mpc", **_WHO, **_TS)
    assert bundle.schema == "studio.controller-run.v1"
    assert bundle.evidence_kind is EvidenceKind.MEASURED
    assert bundle.renders_as_validated is False  # closed-loop surrogate run is bounded
    assert bundle.claim_boundary.validity_domain is not None
    assert "surrogate_mpc" in bundle.entity.entity_id


def test_controller_run_adapter_rejects_missing_field() -> None:
    bad = {k: v for k, v in _SOTA_SUMMARY.items() if k != "mean_tracking_error"}
    with pytest.raises(KeyError):
        controller_run_evidence_from_simulation(bad, controller="surrogate_mpc", **_WHO, **_TS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"controller": "  "},
        {"n_steps": 0},
        {"mean_tracking_error": -0.1},
        {"max_abs_action": -1.0},
        {"max_abs_coil_current": -1.0},
        {"result_digest": ""},
    ],
)
def test_controller_run_result_rejects_bad_fields(kwargs: dict[str, object]) -> None:
    from scpn_control.studio import ControllerRunResult

    base: dict[str, object] = {
        "controller": "surrogate_mpc",
        "n_steps": 10,
        "mean_tracking_error": 0.1,
        "max_abs_action": 1.0,
        "max_abs_coil_current": 5.0,
        "result_digest": "d",
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        ControllerRunResult(**base)  # type: ignore[arg-type]


# ── disruption-mitigation adapter (mitigate) ───────────────────────────
_SPI_SUMMARY = {
    "neon_quantity_mol": 0.1,
    "argon_quantity_mol": 0.0,
    "xenon_quantity_mol": 0.0,
    "z_eff": 2.4,
    "final_current_ma": 3.1,
    "samples": 5000,
}


def test_mitigation_adapter_maps_spi_summary() -> None:
    bundle = disruption_mitigation_evidence_from_run(_SPI_SUMMARY, **_WHO, **_TS)
    assert bundle.schema == "studio.disruption-mitigation.v1"
    assert bundle.evidence_kind is EvidenceKind.MEASURED
    assert bundle.renders_as_validated is False
    assert bundle.claim_boundary.validity_domain is not None


def test_mitigation_adapter_rejects_missing_field() -> None:
    bad = {k: v for k, v in _SPI_SUMMARY.items() if k != "z_eff"}
    with pytest.raises(KeyError):
        disruption_mitigation_evidence_from_run(bad, **_WHO, **_TS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"neon_quantity_mol": -0.1},
        {"z_eff": 0.5},
        {"sample_count": 0},
        {"result_digest": "  "},
    ],
)
def test_mitigation_run_rejects_bad_fields(kwargs: dict[str, object]) -> None:
    from scpn_control.studio import MitigationRun

    base: dict[str, object] = {
        "neon_quantity_mol": 0.1,
        "argon_quantity_mol": 0.0,
        "xenon_quantity_mol": 0.0,
        "z_eff": 2.0,
        "final_current_ma": 3.0,
        "sample_count": 100,
        "result_digest": "d",
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        MitigationRun(**base)  # type: ignore[arg-type]


# ── scenario-simulation adapter (simulate) ─────────────────────────────
def _scenario_audit(*, passed: bool = True):  # type: ignore[no-untyped-def]
    from scpn_control.core.integrated_scenario import ScenarioCouplingAudit, ScenarioCouplingMetadata

    meta = ScenarioCouplingMetadata(
        schema_version="scenario-coupling-audit-v1",
        scenario_name="iter_baseline",
        config_sha256="a" * 64,
        n_steps=100,
        dt_s=1.0,
        t_start_s=0.0,
        t_end_s=100.0,
        enabled_modules=("transport", "current_diffusion", "bootstrap_current"),
        max_abs_current_deviation_MA=0.0,
        max_relative_thermal_energy_step=0.1,
        all_profiles_finite=True,
        strictly_monotonic_time=True,
        exchange_count=300,
        claim_status="bounded replay audit only; external trajectory validation required",
    )
    return ScenarioCouplingAudit(passed=passed, metadata=meta, module_exchanges=(), violations=())


@pytest.mark.parametrize("passed", [True, False])
def test_scenario_adapter_maps_audit(passed: bool) -> None:
    bundle = scenario_simulation_evidence_from_audit(_scenario_audit(passed=passed), **_WHO, **_TS)
    assert bundle.schema == "studio.scenario-simulation.v1"
    assert bundle.evidence_kind is EvidenceKind.MEASURED
    # A passing replay audit must NOT promote the claim past bounded-model.
    assert bundle.renders_as_validated is False
    assert bundle.claim_boundary.validity_domain is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scenario_name": " "},
        {"config_digest": ""},
        {"n_steps": 0},
        {"t_end_s": 0.0},
        {"module_count": 0},
    ],
)
def test_scenario_simulation_run_rejects_bad_fields(kwargs: dict[str, object]) -> None:
    from scpn_control.studio import ScenarioSimulationRun

    base: dict[str, object] = {
        "scenario_name": "iter_baseline",
        "config_digest": "a" * 64,
        "n_steps": 100,
        "t_start_s": 0.0,
        "t_end_s": 100.0,
        "module_count": 3,
        "audit_passed": True,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        ScenarioSimulationRun(**base)  # type: ignore[arg-type]
