# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Physics Traceability Gate Tests

"""Exercise the physics traceability registry and generated report gates."""

from __future__ import annotations

import doctest
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from validation import validate_physics_traceability as trace_module
from validation.validate_physics_traceability import main, validate_physics_traceability

ROOT = Path(__file__).resolve().parents[1]


def _registry_with_header(entries: list[dict[str, object]]) -> dict[str, object]:
    """Wrap original schema fixtures with required headers and disabled marker enforcement.

    Declared test-only domains and references establish schema policy, not new
    scientific, facility or measured evidence.
    """
    return {
        "spdx_license_id": "AGPL-3.0-or-later",
        "commercial_license": "available",
        "concepts_copyright": "Concepts 1996-2026 Miroslav Sotek. All rights reserved.",
        "code_copyright": "Code 2020-2026 Miroslav Sotek. All rights reserved.",
        "orcid": "0009-0009-3560-0851",
        "contact": "www.anulum.li | protoscience@anulum.li",
        "file": "SCPN Control - Physics Traceability Test Registry",
        "enforce_source_marker_coverage": False,
        "schema_version": "1.1",
        "entries": entries,
    }


def test_repository_physics_traceability_records_open_fidelity_gaps() -> None:
    """Validate the live registry, including every approximation-marked source."""
    report = validate_physics_traceability(ROOT / "validation" / "physics_traceability.json")

    assert report["status"] == "pass"
    assert report["total"] >= 6
    assert report["open_fidelity_gaps"] >= 5
    assert report["public_claim_blocked"] >= 5
    assert report["resolved_module_paths"] == report["total"]
    assert report["resolved_evidence_paths"] >= report["total"]
    assert report["external_validation_tracker_count"] == 8
    assert all(
        isinstance(entry["external_validation_tracker_issue"], int)
        for entry in report["entries"]
        if entry["fidelity_status"] in {"bounded_model", "validation_gap", "external_dependency_blocked"}
    )
    assert report["source_marker_coverage"]["total"] >= 30
    assert report["source_marker_coverage"]["covered"] == report["source_marker_coverage"]["total"]
    assert report["source_marker_coverage"]["missing"] == []
    assert all(entry["fidelity_status"] != "synthetic_only" for entry in report["entries"])
    components = {entry["component"] for entry in report["entries"]}
    assert "nonlinear gyrokinetic heat-flux saturation" in components
    assert "DIII-D experimental replay" in components
    nonlinear_gk_entry = next(
        entry for entry in report["entries"] if entry["component"] == "nonlinear gyrokinetic heat-flux saturation"
    )
    assert nonlinear_gk_entry["covered_source_paths"] == ["src/scpn_control/core/gk_nonlinear.py"]
    assert nonlinear_gk_entry["external_validation_tracker_issue"] == 47
    linear_gk_entry = next(
        entry for entry in report["entries"] if entry["component"] == "linear gyrokinetic cross-code agreement"
    )
    assert linear_gk_entry["covered_source_paths"] == ["src/scpn_control/core/gk_eigenvalue.py"]
    assert linear_gk_entry["external_validation_tracker_issue"] == 47
    geometry_entry = next(
        entry
        for entry in report["entries"]
        if entry["component"] == "Miller local-equilibrium geometry and field-pitch contract"
    )
    assert geometry_entry["covered_source_paths"] == ["src/scpn_control/core/gk_geometry.py"]
    species_entry = next(
        entry
        for entry in report["entries"]
        if entry["component"] == "gyrokinetic species and collision bounded-operator contract"
    )
    assert species_entry["covered_source_paths"] == ["src/scpn_control/core/gk_species.py"]
    jax_gk_entry = next(
        entry for entry in report["entries"] if entry["component"] == "JAX gyrokinetic numerical parity guard"
    )
    assert jax_gk_entry["covered_source_paths"] == ["src/scpn_control/core/jax_gk_solver.py"]
    ood_entry = next(
        entry
        for entry in report["entries"]
        if entry["component"] == "gyrokinetic OOD detector distribution-bound contract"
    )
    assert ood_entry["covered_source_paths"] == ["src/scpn_control/core/gk_ood_detector.py"]
    dedicated_core_paths = {
        "reduced gyrokinetic transport closure contract": "src/scpn_control/core/gyrokinetic_transport.py",
        "momentum transport and torque-balance approximation contract": "src/scpn_control/core/momentum_transport.py",
        "EPED pedestal and peeling-ballooning approximation contract": "src/scpn_control/core/eped_pedestal.py",
        "ELM crash and RMP suppression approximation contract": "src/scpn_control/core/elm_model.py",
        "MARFE radiation-condensation density-limit contract": "src/scpn_control/core/marfe.py",
        "NTM island evolution and control approximation contract": "src/scpn_control/core/ntm_dynamics.py",
        "auxiliary current-drive deposition and efficiency model": "src/scpn_control/core/current_drive.py",
        "ideal-MHD stability metric approximation contract": "src/scpn_control/core/stability_mhd.py",
        "sawtooth-to-NTM seeding approximation contract": "src/scpn_control/core/tearing_mode_coupling.py",
        "blob transport and scrape-off-layer approximation contract": "src/scpn_control/core/blob_transport.py",
        "checkpoint state serialisation boundary contract": "src/scpn_control/core/checkpoint.py",
        "disruption sequence phase-ordering contract": "src/scpn_control/core/disruption_sequence.py",
        "Grad-Shafranov fusion-kernel numerical contract": "src/scpn_control/core/fusion_kernel.py",
        "gyrokinetic online learner stability contract": "src/scpn_control/core/gk_online_learner.py",
        "integrated scenario coupling contract": "src/scpn_control/core/integrated_scenario.py",
        "neural equilibrium cross-validation": "src/scpn_control/core/neural_equilibrium.py",
        "neural transport surrogate validation contract": "src/scpn_control/core/neural_transport.py",
        "neural turbulence surrogate validation contract": "src/scpn_control/core/neural_turbulence.py",
        "orbit-following guiding-centre approximation contract": "src/scpn_control/core/orbit_following.py",
        "full-chain uncertainty quantification contract": "src/scpn_control/core/uncertainty.py",
        "VMEC-lite stellarator equilibrium approximation contract": "src/scpn_control/core/vmec_lite.py",
    }
    for component, source_path in dedicated_core_paths.items():
        entry = next(item for item in report["entries"] if item["component"] == component)
        assert entry["covered_source_paths"] == [source_path]
    transport_entry = next(
        item
        for item in report["entries"]
        if item["component"] == "transport solver neoclassical and source-term approximation contract"
    )
    assert transport_entry["covered_source_paths"] == sorted(
        [
            "src/scpn_control/core/integrated_transport_solver.py",
            "src/scpn_control/core/transport_model_selection.py",
            "src/scpn_control/core/transport_orchestration.py",
        ]
    )
    core_entry = next(
        entry
        for entry in report["entries"]
        if entry["component"] == "bounded analytical approximations in physics modules"
    )
    assert "src/scpn_control/core/gk_eigenvalue.py" not in core_entry["covered_source_paths"]
    assert "src/scpn_control/core/gk_geometry.py" not in core_entry["covered_source_paths"]
    assert "src/scpn_control/core/gk_nonlinear.py" not in core_entry["covered_source_paths"]
    assert "src/scpn_control/core/gk_ood_detector.py" not in core_entry["covered_source_paths"]
    assert "src/scpn_control/core/gk_species.py" not in core_entry["covered_source_paths"]
    assert "src/scpn_control/core/jax_gk_solver.py" not in core_entry["covered_source_paths"]
    for source_path in dedicated_core_paths.values():
        assert source_path not in core_entry["covered_source_paths"]
    assert core_entry["covered_source_paths"] == []
    gym_entry = next(
        entry for entry in report["entries"] if entry["component"] == "reduced-order plant and control environments"
    )
    assert gym_entry["covered_source_paths"] == ["src/scpn_control/control/gym_tokamak_env.py"]
    rzip_entry = next(
        entry for entry in report["entries"] if entry["component"] == "RZIP rigid vertical stability model"
    )
    assert rzip_entry["covered_source_paths"] == ["src/scpn_control/control/rzip_model.py"]
    realtime_efit_entry = next(
        entry for entry in report["entries"] if entry["component"] == "real-time EFIT-lite equilibrium reconstruction"
    )
    assert realtime_efit_entry["covered_source_paths"] == sorted(
        [
            "src/scpn_control/control/realtime_efit.py",
            "src/scpn_control/control/realtime_efit_contracts.py",
            "src/scpn_control/control/realtime_efit_claims.py",
            "src/scpn_control/control/realtime_efit_diagnostics.py",
            "src/scpn_control/control/realtime_efit_solver.py",
            "src/scpn_control/control/realtime_efit_runtime.py",
            "src/scpn_control/control/realtime_efit_topology.py",
        ]
    )
    kinetic_efit_entry = next(
        entry for entry in report["entries"] if entry["component"] == "kinetic EFIT pressure and q-profile coupling"
    )
    assert kinetic_efit_entry["covered_source_paths"] == ["src/scpn_control/core/kinetic_efit.py"]
    halo_entry = next(
        entry
        for entry in report["entries"]
        if entry["component"] == "halo current and runaway electron disruption model"
    )
    assert halo_entry["covered_source_paths"] == [
        "src/scpn_control/control/_disruption_claims.py",
        "src/scpn_control/control/_disruption_ensemble.py",
        "src/scpn_control/control/_halo_current_model.py",
        "src/scpn_control/control/_runaway_electron_model.py",
        "src/scpn_control/control/halo_re_physics.py",
    ]
    federated_entry = next(
        entry for entry in report["entries"] if entry["component"] == "federated disruption prediction"
    )
    assert federated_entry["covered_source_paths"] == sorted(
        [
            "src/scpn_control/control/federated_disruption.py",
            "src/scpn_control/control/_federated_model.py",
            "src/scpn_control/control/_federated_privacy.py",
            "src/scpn_control/control/_federated_clients.py",
            "src/scpn_control/control/_federated_server.py",
            "src/scpn_control/control/_federated_config.py",
            "src/scpn_control/control/_federated_benchmark.py",
            "src/scpn_control/control/_federated_state.py",
            "src/scpn_control/control/_federated_aggregation.py",
        ]
    )
    free_boundary_entry = next(
        entry for entry in report["entries"] if entry["component"] == "direct free-boundary tracking controller"
    )
    assert free_boundary_entry["covered_source_paths"] == ["src/scpn_control/control/free_boundary_tracking.py"]
    density_entry = next(
        entry for entry in report["entries"] if entry["component"] == "density control and particle-source model"
    )
    assert density_entry["covered_source_paths"] == ["src/scpn_control/control/density_controller.py"]
    disruption_contract_entry = next(
        entry for entry in report["entries"] if entry["component"] == "disruption mitigation contract layer"
    )
    assert disruption_contract_entry["covered_source_paths"] == [
        "src/scpn_control/control/_disruption_episode_physics.py",
        "src/scpn_control/control/_disruption_episode_runtime.py",
        "src/scpn_control/control/_disruption_shot_replay.py",
        "src/scpn_control/control/disruption_contracts.py",
    ]
    digital_twin_entry = next(
        entry
        for entry in report["entries"]
        if entry["component"] == "tokamak digital twin topology and diffusion model"
    )
    assert digital_twin_entry["covered_source_paths"] == ["src/scpn_control/control/tokamak_digital_twin.py"]
    soc_entry = next(
        entry for entry in report["entries"] if entry["component"] == "advanced SOC turbulence learning controller"
    )
    assert soc_entry["covered_source_paths"] == ["src/scpn_control/control/advanced_soc_fusion_learning.py"]
    burn_entry = next(
        entry for entry in report["entries"] if entry["component"] == "DT burn control and alpha-heating model"
    )
    assert burn_entry["covered_source_paths"] == ["src/scpn_control/control/burn_controller.py"]
    volt_second_entry = next(
        entry for entry in report["entries"] if entry["component"] == "volt-second budget and flux-consumption manager"
    )
    assert volt_second_entry["covered_source_paths"] == sorted(
        [
            "src/scpn_control/control/volt_second_manager.py",
            "src/scpn_control/control/volt_second_core.py",
            "src/scpn_control/control/volt_second_claims.py",
            "src/scpn_control/control/volt_second_profiles.py",
            "src/scpn_control/control/volt_second_runtime.py",
        ]
    )
    kuramoto_entry = next(
        entry for entry in report["entries"] if entry["component"] == "Kuramoto-Sakaguchi phase synchronisation runtime"
    )
    assert kuramoto_entry["covered_source_paths"] == ["src/scpn_control/phase/kuramoto.py"]
    fpga_entry = next(
        entry for entry in report["entries"] if entry["component"] == "SCPN FPGA fixed-point export boundary"
    )
    assert fpga_entry["covered_source_paths"] == ["src/scpn_control/scpn/fpga_export.py"]
    replay_entry = next(
        entry for entry in report["entries"] if entry["component"] == "geometry-neutral stellarator replay fixture"
    )
    assert replay_entry["covered_source_paths"] == ["src/scpn_control/scpn/geometry_neutral_replay.py"]
    control_entry = next(
        entry for entry in report["entries"] if entry["component"] == "bounded control plant approximations"
    )
    assert "src/scpn_control/control/gym_tokamak_env.py" not in control_entry["covered_source_paths"]
    assert "src/scpn_control/control/rzip_model.py" not in control_entry["covered_source_paths"]
    assert "src/scpn_control/control/realtime_efit.py" not in control_entry["covered_source_paths"]
    assert "src/scpn_control/control/halo_re_physics.py" not in control_entry["covered_source_paths"]
    assert "src/scpn_control/control/free_boundary_tracking.py" not in control_entry["covered_source_paths"]
    assert "src/scpn_control/control/density_controller.py" not in control_entry["covered_source_paths"]
    assert "src/scpn_control/control/disruption_contracts.py" not in control_entry["covered_source_paths"]
    assert "src/scpn_control/control/tokamak_digital_twin.py" not in control_entry["covered_source_paths"]
    assert "src/scpn_control/control/advanced_soc_fusion_learning.py" not in control_entry["covered_source_paths"]
    assert "src/scpn_control/control/burn_controller.py" not in control_entry["covered_source_paths"]
    assert "src/scpn_control/control/volt_second_manager.py" not in control_entry["covered_source_paths"]
    assert control_entry["covered_source_paths"] == []
    phase_runtime_entry = next(
        entry for entry in report["entries"] if entry["component"] == "bounded phase and spiking runtime approximations"
    )
    assert "src/scpn_control/phase/kuramoto.py" not in phase_runtime_entry["covered_source_paths"]
    assert "src/scpn_control/scpn/fpga_export.py" not in phase_runtime_entry["covered_source_paths"]
    assert "src/scpn_control/scpn/geometry_neutral_replay.py" not in phase_runtime_entry["covered_source_paths"]
    assert phase_runtime_entry["covered_source_paths"] == []
    aggregate_parent_components = {
        "bounded analytical approximations in physics modules",
        "bounded control plant approximations",
        "bounded phase and spiking runtime approximations",
    }
    for entry in report["entries"]:
        if entry["component"] not in aggregate_parent_components:
            continue
        assert entry["covered_source_paths"] == []
        claim_admission_requirements = " ".join(entry["claim_admission_requirements"])
        assert "Split " not in claim_admission_requirements
        assert "Replace aggregate" not in claim_admission_requirements
        assert "per-module traceability entries" not in claim_admission_requirements


def test_traceability_rejects_unbounded_gap_claim(tmp_path: Path) -> None:
    """Verify traceability rejects unbounded gap claim."""
    registry = _registry_with_header(
        [
            {
                "component": "bad physics claim",
                "module_path": "src/scpn_control/core/bad.py",
                "fidelity_status": "validation_gap",
                "public_claim_allowed": True,
                "model_references": ["Example 2026"],
                "equation_contract": "d x / d t = x",
                "unit_contract": "SI",
                "validity_domain": "unit test",
                "validation_evidence": ["tests/test_bad.py"],
                "evidence_paths": ["tests/test_physics_traceability.py"],
                "claim_admission_requirements": ["replace claim with evidence"],
            }
        ]
    )
    path = tmp_path / "physics_traceability.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_physics_traceability(path)

    assert report["status"] == "fail"
    fields = {error["field"] for error in report["errors"]}
    assert "public_claim_allowed" in fields


def test_traceability_rejects_synthetic_only_fidelity_status(tmp_path: Path) -> None:
    """Verify traceability rejects synthetic only fidelity status."""
    registry = _registry_with_header(
        [
            {
                "component": "synthetic-only replay",
                "module_path": "validation/reference_data/diiid",
                "fidelity_status": "synthetic_only",
                "public_claim_allowed": False,
                "model_references": ["Repository real-data manifest schema 1.0"],
                "equation_contract": "Replay evidence must use real or immutable reference artefacts.",
                "unit_contract": "SI or explicitly dimensionless units",
                "validity_domain": "synthetic fixture only",
                "validation_evidence": ["unit test evidence"],
                "evidence_paths": ["tests/test_physics_traceability.py"],
                "claim_admission_requirements": ["replace synthetic evidence with reference artefacts"],
            }
        ]
    )
    path = tmp_path / "physics_traceability.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_physics_traceability(path)

    assert report["status"] == "fail"
    errors = [error for error in report["errors"] if error["field"] == "fidelity_status"]
    assert any("synthetic_only" in str(error["error"]) for error in errors)


def test_traceability_rejects_missing_contract_fields(tmp_path: Path) -> None:
    """Verify traceability rejects missing contract fields."""
    registry = _registry_with_header(
        [
            {
                "component": "missing contracts",
                "module_path": "src/scpn_control/core/bad.py",
                "fidelity_status": "bounded_model",
                "public_claim_allowed": False,
                "model_references": [],
                "equation_contract": "",
                "unit_contract": "",
                "validity_domain": "",
                "validation_evidence": [],
                "evidence_paths": [],
                "claim_admission_requirements": [],
            }
        ]
    )
    path = tmp_path / "physics_traceability.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_physics_traceability(path)

    assert report["status"] == "fail"
    fields = {error["field"] for error in report["errors"]}
    assert {"model_references", "equation_contract", "unit_contract", "validity_domain", "evidence_paths"} <= fields


def test_traceability_rejects_missing_json_header_metadata(tmp_path: Path) -> None:
    """Verify traceability rejects missing json header metadata."""
    registry = {
        "schema_version": "1.1",
        "entries": [
            {
                "component": "otherwise valid",
                "module_path": "src/scpn_control/core/example.py",
                "fidelity_status": "reference_validated",
                "public_claim_allowed": True,
                "model_references": ["Example 2026"],
                "equation_contract": "d x / d t = x",
                "unit_contract": "SI",
                "validity_domain": "unit test",
                "validation_evidence": ["tests/test_example.py"],
                "evidence_paths": ["tests/test_physics_traceability.py"],
                "claim_admission_requirements": ["keep evidence current"],
            }
        ],
    }
    path = tmp_path / "physics_traceability.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_physics_traceability(path)

    assert report["status"] == "fail"
    fields = {error["field"] for error in report["errors"]}
    assert "spdx_license_id" in fields
    assert "file" in fields


def test_traceability_rejects_unresolved_module_or_evidence_paths(tmp_path: Path) -> None:
    """Verify traceability rejects unresolved module or evidence paths."""
    registry = _registry_with_header(
        [
            {
                "component": "missing path proof",
                "module_path": "src/scpn_control/core/not_a_real_module.py",
                "fidelity_status": "reference_validated",
                "public_claim_allowed": True,
                "model_references": ["Example 2026"],
                "equation_contract": "d x / d t = x",
                "unit_contract": "SI",
                "validity_domain": "unit test",
                "validation_evidence": ["unit test evidence"],
                "evidence_paths": ["validation/not_a_real_evidence_file.json"],
                "covered_source_paths": [],
                "claim_admission_requirements": ["keep evidence current"],
            }
        ]
    )
    path = tmp_path / "physics_traceability.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_physics_traceability(path)

    assert report["status"] == "fail"
    fields = {error["field"] for error in report["errors"]}
    assert "module_path" in fields
    assert "evidence_paths" in fields


def test_traceability_rejects_missing_source_marker_coverage(tmp_path: Path) -> None:
    """Verify traceability rejects missing source marker coverage."""
    registry = _registry_with_header(
        [
            {
                "component": "partial bounded model coverage",
                "module_path": "src/scpn_control/core",
                "fidelity_status": "bounded_model",
                "public_claim_allowed": False,
                "model_references": ["Example 2026"],
                "equation_contract": "declared approximation contract",
                "unit_contract": "SI",
                "validity_domain": "bounded model test",
                "validation_evidence": ["unit test evidence"],
                "evidence_paths": ["tests/test_physics_traceability.py"],
                "covered_source_paths": ["src/scpn_control/core/orbit_following.py"],
                "claim_admission_requirements": ["cover every approximation marker"],
            }
        ]
    )
    registry["enforce_source_marker_coverage"] = True
    path = tmp_path / "physics_traceability.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_physics_traceability(path)

    assert report["status"] == "fail"
    assert report["source_marker_coverage"]["total"] >= 30
    assert "src/scpn_control/core/integrated_transport_solver.py" in report["source_marker_coverage"]["missing"]


def test_traceability_rejects_source_coverage_outside_module_scope(tmp_path: Path) -> None:
    """Verify traceability rejects source coverage outside module scope."""
    registry = _registry_with_header(
        [
            {
                "component": "scope leakage",
                "module_path": "src/scpn_control/core",
                "fidelity_status": "bounded_model",
                "public_claim_allowed": False,
                "model_references": ["Example 2026"],
                "equation_contract": "declared bounded analytical contract",
                "unit_contract": "SI",
                "validity_domain": "bounded model test",
                "validation_evidence": ["unit test evidence"],
                "evidence_paths": ["tests/test_physics_traceability.py"],
                "covered_source_paths": ["src/scpn_control/control/gym_tokamak_env.py"],
                "claim_admission_requirements": ["keep source coverage inside module scope"],
            }
        ]
    )
    path = tmp_path / "physics_traceability.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_physics_traceability(path)

    assert report["status"] == "fail"
    errors = [error for error in report["errors"] if error["field"] == "covered_source_paths"]
    assert any("outside module_path scope" in str(error["error"]) for error in errors)


def test_traceability_main_writes_json_report(tmp_path: Path) -> None:
    """Verify traceability main writes json report."""
    output = tmp_path / "physics_traceability_report.json"

    exit_code = main(
        [
            "--registry",
            str(ROOT / "validation" / "physics_traceability.json"),
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["open_fidelity_gaps"] >= 5


def _minimal_tracker_registry() -> dict[str, object]:
    """Build the original ROADMAP-based metadata fixture for tracker refusal rules.

    Its reference_validated flag is a test declaration, not physical admission.
    """
    return {
        "schema_version": "1.1",
        "spdx_license_id": "AGPL-3.0-or-later",
        "commercial_license": "available",
        "concepts_copyright": "Concepts 1996-2026 Miroslav Sotek. All rights reserved.",
        "code_copyright": "Code 2020-2026 Miroslav Sotek. All rights reserved.",
        "orcid": "0009-0009-3560-0851",
        "contact": "www.anulum.li | protoscience@anulum.li",
        "file": "test registry",
        "entries": [
            {
                "component": "test component",
                "module_path": "ROADMAP.md",
                "equation_contract": "test contract",
                "fidelity_status": "reference_validated",
                "model_references": ["test reference"],
                "public_claim_allowed": True,
                "claim_admission_requirements": ["keep the reference current"],
                "unit_contract": "dimensionless test units",
                "validation_evidence": ["ROADMAP.md"],
                "evidence_paths": ["ROADMAP.md"],
                "validity_domain": "test-only validation fixture",
            }
        ],
    }


def _first_registry_entry(registry: dict[str, object]) -> dict[str, object]:
    """Return the mutable fixture entry with its container type checked."""
    entries = registry["entries"]
    assert isinstance(entries, list)
    entry = entries[0]
    assert isinstance(entry, dict)
    return cast(dict[str, object], entry)


def test_traceability_rejects_duplicate_external_validation_tracker_issue(tmp_path: Path) -> None:
    """Verify traceability rejects duplicate external validation tracker issue."""
    registry = _minimal_tracker_registry()
    registry["external_validation_trackers"] = [
        {
            "title": "first tracker",
            "issue": 46,
            "url": "https://github.com/anulum/scpn-control/issues/46",
            "scope": "first scope",
        },
        {
            "title": "duplicate tracker",
            "issue": 46,
            "url": "https://github.com/anulum/scpn-control/issues/46",
            "scope": "duplicate scope",
        },
    ]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_physics_traceability(path)

    assert report["status"] == "fail"
    assert any(error["field"] == "issue" and "unique" in error["error"] for error in report["errors"])


def test_traceability_rejects_external_validation_tracker_url_issue_mismatch(tmp_path: Path) -> None:
    """Verify traceability rejects external validation tracker url issue mismatch."""
    registry = _minimal_tracker_registry()
    registry["external_validation_trackers"] = [
        {
            "title": "mismatched tracker",
            "issue": 46,
            "url": "https://github.com/anulum/scpn-control/issues/47",
            "scope": "mismatch scope",
        }
    ]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_physics_traceability(path)

    assert report["status"] == "fail"
    assert any(error["field"] == "url" and "match issue number" in error["error"] for error in report["errors"])


def test_external_validation_trackers_are_linked_from_roadmap_and_report() -> None:
    """Verify external validation trackers are linked from roadmap and report."""
    report = validate_physics_traceability(ROOT / "validation" / "physics_traceability.json")
    roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    generated_report = (ROOT / "docs" / "physics_traceability.md").read_text(encoding="utf-8")

    trackers = report["external_validation_trackers"]
    assert len(trackers) == 8
    for tracker in trackers:
        assert tracker["url"] in roadmap
        assert tracker["url"] in generated_report
        assert f"#{tracker['issue']}" in generated_report


def test_roadmap_does_not_embed_live_traceability_inventory_counts() -> None:
    """ROADMAP is a public readiness narrative, not a live inventory mirror.

    Exact registry totals belong in ``docs/physics_traceability.md`` (generated
    from ``validation/physics_traceability.json``). Embedding them in ROADMAP
    forced residual/registry churn into a public product document.
    """
    roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    assert "Current generated status is" not in roadmap
    assert "open fidelity gaps" not in roadmap
    assert "blocked full-fidelity public claims" not in roadmap
    assert "docs/physics_traceability.md" in roadmap
    assert "validation/physics_traceability.json" in roadmap


def test_traceability_requires_trackers_when_fidelity_gaps_remain(tmp_path: Path) -> None:
    """Verify traceability requires trackers when fidelity gaps remain."""
    registry = _registry_with_header(
        [
            {
                "component": "open bounded model",
                "module_path": "ROADMAP.md",
                "fidelity_status": "bounded_model",
                "public_claim_allowed": False,
                "model_references": ["Example 2026"],
                "equation_contract": "bounded model contract",
                "unit_contract": "dimensionless test units",
                "validity_domain": "test-only bounded validation fixture",
                "validation_evidence": ["ROADMAP.md"],
                "evidence_paths": ["ROADMAP.md"],
                "claim_admission_requirements": ["link collaboration tracker before promotion"],
            }
        ]
    )
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_physics_traceability(path)

    assert report["status"] == "fail"
    assert any(error["field"] == "external_validation_trackers" for error in report["errors"])


def test_traceability_rejects_open_entry_without_tracker_issue(tmp_path: Path) -> None:
    """Verify traceability rejects open entry without tracker issue."""
    registry = _minimal_tracker_registry()
    entry = _first_registry_entry(registry)
    entry["fidelity_status"] = "validation_gap"
    entry["public_claim_allowed"] = False
    registry["external_validation_trackers"] = [
        {
            "title": "tracker",
            "issue": 47,
            "url": "https://github.com/anulum/scpn-control/issues/47",
            "scope": "test scope",
        }
    ]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_physics_traceability(path)

    assert report["status"] == "fail"
    assert any(error["field"] == "external_validation_tracker_issue" for error in report["errors"])


def test_traceability_rejects_unknown_entry_tracker_issue(tmp_path: Path) -> None:
    """Verify traceability rejects unknown entry tracker issue."""
    registry = _minimal_tracker_registry()
    entry = _first_registry_entry(registry)
    entry["fidelity_status"] = "validation_gap"
    entry["public_claim_allowed"] = False
    entry["external_validation_tracker_issue"] = 99
    registry["external_validation_trackers"] = [
        {
            "title": "tracker",
            "issue": 47,
            "url": "https://github.com/anulum/scpn-control/issues/47",
            "scope": "test scope",
        }
    ]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_physics_traceability(path)

    assert report["status"] == "fail"
    assert any(
        error["field"] == "external_validation_tracker_issue" and "must exist" in error["error"]
        for error in report["errors"]
    )


@pytest.fixture
def local_traceability_repository(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Copy real defining source/evidence bytes into an isolated declared registry root.

    The single historical bounded entry retains its no-public-claim boundary.
    The copied files establish local path/marker/schema behavior only; neither
    the copy nor later metadata mutations establish new scientific evidence.
    """
    base: dict[str, Any] = json.loads((ROOT / "validation/physics_traceability.json").read_text())
    entry = next(e for e in base["entries"] if e["component"] == "linear gyrokinetic cross-code agreement")
    base["entries"] = [entry]
    repo = tmp_path / "repository"
    for name in {entry["module_path"], *entry["evidence_paths"], *entry["covered_source_paths"]}:
        original = ROOT / name
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.is_dir():
            shutil.copytree(original, target, ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
        else:
            shutil.copy2(original, target)
    (repo / "validation").mkdir(exist_ok=True)
    path = repo / "validation/physics_traceability.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return path, base


def _write_local_registry(path: Path, payload: dict[str, Any]) -> None:
    """Persist independent modified declaration bytes without invoking private production helpers."""
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    "field,value",
    [
        ("spdx_license_id", []),
        ("spdx_license_id", "wrong"),
        ("file", None),
        ("schema_version", "old"),
        ("entries", []),
        ("entries", "bad"),
        ("enforce_source_marker_coverage", "false"),
        ("enforce_source_marker_coverage", None),
        ("external_validation_trackers", None),
    ],
)
def test_public_registry_header_and_container_refusals(
    local_traceability_repository: tuple[Path, dict[str, Any]], field: str, value: object
) -> None:
    """Refuse bad header/container/enforcement declarations through the public validator."""
    path, payload = local_traceability_repository
    payload[field] = value
    _write_local_registry(path, payload)
    report = validate_physics_traceability(path)
    assert report["status"] == "fail" and any(e["field"] == field for e in report["errors"])


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("fidelity_status", [], "fidelity_status"),
        ("fidelity_status", None, "fidelity_status"),
        ("module_path", None, "module_path"),
        ("equation_contract", 1, "equation_contract"),
        ("model_references", "bad", "model_references"),
        ("claim_admission_requirements", None, "claim_admission_requirements"),
        ("evidence_paths", [], "evidence_paths"),
        ("evidence_paths", [None], "evidence_paths"),
        ("covered_source_paths", "bad", "covered_source_paths"),
        ("covered_source_paths", ["missing.py"], "covered_source_paths"),
        ("public_claim_allowed", "true", "public_claim_allowed"),
        ("external_validation_tracker_issue", True, "external_validation_tracker_issue"),
        ("external_validation_tracker_issue", "47", "external_validation_tracker_issue"),
    ],
)
def test_public_entry_domain_refusals(
    local_traceability_repository: tuple[Path, dict[str, Any]], field: str, value: object, expected: str
) -> None:
    """Keep invalid entry diagnostics/counts visible without type-membership crashes."""
    path, payload = local_traceability_repository
    payload["entries"][0][field] = value
    _write_local_registry(path, payload)
    report = validate_physics_traceability(path)
    assert report["status"] == "fail" and report["total"] == 1 and len(report["entries"]) == 1
    assert any(e["field"] == expected for e in report["errors"])


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("title", None, "title"),
        ("scope", "", "scope"),
        ("issue", True, "issue"),
        ("issue", 0, "issue"),
        ("issue", "47", "issue"),
        ("issue", [], "issue"),
        ("url", None, "url"),
        ("url", "https://github.com/anulum/scpn-control/issues/46", "url"),
    ],
)
def test_invalid_tracker_is_not_returned_as_validated_metadata(
    local_traceability_repository: tuple[Path, dict[str, Any]], field: str, value: object, expected: str
) -> None:
    """Refuse invalid/boolean/mismatched tracker records from the returned valid list."""
    path, payload = local_traceability_repository
    tracker = next(t for t in payload["external_validation_trackers"] if t["issue"] == 47)
    tracker[field] = value
    payload["external_validation_trackers"] = [tracker]
    _write_local_registry(path, payload)
    report = validate_physics_traceability(path)
    assert report["status"] == "fail" and report["external_validation_trackers"] == []
    assert report["external_validation_tracker_count"] == 0 and any(e["field"] == expected for e in report["errors"])


def test_duplicate_and_nondictionary_tracker_and_entry_observations(
    local_traceability_repository: tuple[Path, dict[str, Any]],
) -> None:
    """Count raw entry declarations while retaining only the first valid tracker record."""
    path, payload = local_traceability_repository
    tracker = next(t for t in payload["external_validation_trackers"] if t["issue"] == 47)
    payload["external_validation_trackers"] = [None, tracker, dict(tracker)]
    payload["entries"].append(None)
    _write_local_registry(path, payload)
    report = validate_physics_traceability(path)
    assert report["status"] == "fail" and report["total"] == 2 and len(report["entries"]) == 1
    assert report["external_validation_trackers"] == [tracker]
    assert {"entry", "external_validation_trackers", "issue"} <= {e["field"] for e in report["errors"]}


@pytest.mark.parametrize(
    "contents",
    [
        "[]",
        "{",
        '{"extra":NaN}',
        '{"extra":Infinity}',
        '{"extra":-Infinity}',
        '{"extra":1e400}',
        '{"x":1,"x":2}',
        '{"extra":{"x":1,"x":2}}',
        "[" * 1200 + "0" + "]" * 1200,
    ],
)
def test_actual_registry_decode_refusals(tmp_path: Path, contents: str) -> None:
    """Reject malformed/nonfinite/duplicate/deep actual UTF-8 declarations without mocks."""
    path = tmp_path / "registry.json"
    path.write_text(contents)
    report = validate_physics_traceability(path)
    assert report["status"] == "fail" and report["entries"] == []
    assert report["errors"][0]["field"] in {"json", "root"}


def test_actual_registry_read_and_invalid_path_refusals(tmp_path: Path) -> None:
    """Refuse actual missing/directory/invalid-UTF8/null/loop registry inputs."""
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"\xff")
    loop = tmp_path / "loop.json"
    loop.symlink_to(loop.name)
    for path in [tmp_path / "missing", tmp_path, invalid, Path("bad\0registry"), loop]:
        report = validate_physics_traceability(path)
        assert report["status"] == "fail" and report["errors"][0]["field"] == "json"


@pytest.mark.parametrize("kind", ["absolute", "traversal", "symlink", "null", "loop"])
def test_repository_path_containment_refusals(
    local_traceability_repository: tuple[Path, dict[str, Any]], kind: str
) -> None:
    """Refuse real external/escaping/null/loop paths as module, evidence and covered source."""
    path, payload = local_traceability_repository
    repo = path.parent.parent
    external = repo.parent / "external.md"
    shutil.copy2(ROOT / "ROADMAP.md", external)
    link = repo / "link.md"
    if kind == "absolute":
        value = str(external)
    elif kind == "traversal":
        value = "../external.md"
    elif kind == "null":
        value = "bad\0path"
    elif kind == "loop":
        link.symlink_to(link.name)
        value = "link.md"
    else:
        link.symlink_to(external)
        value = "link.md"
    payload["entries"][0].update(module_path=value, evidence_paths=[value], covered_source_paths=[value])
    _write_local_registry(path, payload)
    report = validate_physics_traceability(path)
    assert report["status"] == "fail" and report["resolved_module_paths"] == 0
    assert {"module_path", "evidence_paths", "covered_source_paths"} <= {e["field"] for e in report["errors"]}


def test_absolute_contained_paths_and_real_module_scope_refusals(
    local_traceability_repository: tuple[Path, dict[str, Any]],
) -> None:
    """Admit canonical in-root absolute paths and refuse covered paths outside a file module."""
    path, payload = local_traceability_repository
    repo = path.parent.parent
    entry = payload["entries"][0]
    entry["module_path"] = str(repo / entry["module_path"])
    entry["evidence_paths"] = [str(repo / p) for p in entry["evidence_paths"]]
    entry["covered_source_paths"] = [entry["module_path"]]
    _write_local_registry(path, payload)
    assert validate_physics_traceability(path)["status"] == "pass"
    other = repo / "other.py"
    shutil.copy2(ROOT / "src/scpn_control/core/gk_species.py", other)
    entry["covered_source_paths"] = ["other.py"]
    _write_local_registry(path, payload)
    report = validate_physics_traceability(path)
    assert report["status"] == "fail" and any("outside module_path scope" in e["error"] for e in report["errors"])


@pytest.mark.parametrize("kind", ["invalid_utf8", "directory", "escaping_symlink"])
def test_actual_source_marker_scan_failures(
    local_traceability_repository: tuple[Path, dict[str, Any]], kind: str
) -> None:
    """Make actual scan read/decode/containment failures visible without ignored bytes."""
    path, _ = local_traceability_repository
    source = path.parent.parent / "src/scpn_control/broken.py"
    if kind == "invalid_utf8":
        source.write_bytes(b"\xff approximation")
    elif kind == "directory":
        source.mkdir()
    else:
        external = path.parent.parent.parent / "external.py"
        shutil.copy2(ROOT / "src/scpn_control/core/gk_species.py", external)
        source.symlink_to(external)
    report = validate_physics_traceability(path)
    assert report["status"] == "fail" and any(e["field"] == "source_marker_coverage" for e in report["errors"])


def test_optional_coverage_fields_and_empty_source_root(
    local_traceability_repository: tuple[Path, dict[str, Any]],
) -> None:
    """Observe absent optional coverage/enforcement and copied roots with no Python markers."""
    path, payload = local_traceability_repository
    payload.pop("enforce_source_marker_coverage")
    payload["entries"][0].pop("covered_source_paths")
    _write_local_registry(path, payload)
    report = validate_physics_traceability(path)
    assert report["status"] == "pass" and report["entries"][0]["covered_source_paths"] == []
    shutil.rmtree(path.parent.parent / "src")
    payload["entries"][0].update(module_path="validation", evidence_paths=["validation"], covered_source_paths=None)
    _write_local_registry(path, payload)
    report = validate_physics_traceability(path)
    assert report["status"] == "pass" and report["source_marker_coverage"] == {"total": 0, "covered": 0, "missing": []}


def test_standalone_registry_main_text_json_output_and_path_refusals(
    local_traceability_repository: tuple[Path, dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise actual persisted CLI report bytes and supported output IO/null failures."""
    path, _ = local_traceability_repository
    output = path.parent / "output/report.json"
    assert main(["--registry", str(path), "--output-json", str(output), "--json-out"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == report and output.read_bytes().endswith(b"\n")
    for value in [str(path.parent), str(path / "child"), "bad\0output"]:
        assert main(["--registry", str(path), "--output-json", value, "--json-out"]) == 1
        assert json.loads(capsys.readouterr().out)["errors"][-1]["field"] == "output_json"
    assert main(["--registry", str(path.parent / "missing")]) == 1
    text = capsys.readouterr()
    assert "total=0" in text.out and "ERROR" in text.err


def test_standard_library_registry_cli_from_other_cwd_without_site(
    local_traceability_repository: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    """Run the actual source CLI without installed dependencies or PYTHONPATH."""
    path, _ = local_traceability_repository
    c = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "validation/validate_physics_traceability.py"),
            "--registry",
            str(path),
            "--json-out",
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1"),
        text=True,
        capture_output=True,
        check=False,
    )
    assert c.returncode == 0 and c.stderr == "" and json.loads(c.stdout)["status"] == "pass"


def test_defining_registry_native_examples_execute() -> None:
    """Execute real native read-refusal examples without private calls or fabricated physics."""
    result = doctest.testmod(trace_module, raise_on_error=True)
    assert result.failed == 0 and result.attempted >= 2


def test_finite_extra_json_metadata_is_allowed(local_traceability_repository: tuple[Path, dict[str, Any]]) -> None:
    """Retain legal finite decimal metadata without implying measured scientific validation."""
    path, payload = local_traceability_repository
    payload["extra"] = {"decimal": 12.5, "nested": [-0.5, 1e-300]}
    _write_local_registry(path, payload)
    report = validate_physics_traceability(path)
    assert report["status"] == "pass" and report["public_claim_blocked"] == 1
