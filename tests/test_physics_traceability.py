# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Physics Traceability Gate Tests

from __future__ import annotations

import json
from pathlib import Path

from validation.validate_physics_traceability import main, validate_physics_traceability


ROOT = Path(__file__).resolve().parents[1]


def _registry_with_header(entries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "spdx_license_id": "AGPL-3.0-or-later",
        "commercial_license": "available",
        "concepts_copyright": "Concepts 1996-2026 Miroslav Sotek. All rights reserved.",
        "code_copyright": "Code 2020-2026 Miroslav Sotek. All rights reserved.",
        "orcid": "0009-0009-3560-0851",
        "contact": "www.anulum.li | protoscience@anulum.li",
        "file": "SCPN Control - Physics Traceability Test Registry",
        "enforce_source_marker_coverage": False,
        "schema_version": "1.0",
        "entries": entries,
    }


def test_repository_physics_traceability_records_open_fidelity_gaps() -> None:
    report = validate_physics_traceability(ROOT / "validation" / "physics_traceability.json")

    assert report["status"] == "pass"
    assert report["total"] >= 6
    assert report["open_fidelity_gaps"] >= 5
    assert report["public_claim_blocked"] >= 5
    assert report["resolved_module_paths"] == report["total"]
    assert report["resolved_evidence_paths"] >= report["total"]
    assert report["source_marker_coverage"]["total"] >= 30
    assert report["source_marker_coverage"]["covered"] == report["source_marker_coverage"]["total"]
    assert report["source_marker_coverage"]["missing"] == []
    components = {entry["component"] for entry in report["entries"]}
    assert "nonlinear gyrokinetic heat-flux saturation" in components
    assert "DIII-D experimental replay" in components
    nonlinear_gk_entry = next(
        entry for entry in report["entries"] if entry["component"] == "nonlinear gyrokinetic heat-flux saturation"
    )
    assert nonlinear_gk_entry["covered_source_paths"] == ["src/scpn_control/core/gk_nonlinear.py"]
    linear_gk_entry = next(entry for entry in report["entries"] if entry["component"] == "linear gyrokinetic cross-code agreement")
    assert linear_gk_entry["covered_source_paths"] == ["src/scpn_control/core/gk_eigenvalue.py"]
    geometry_entry = next(
        entry for entry in report["entries"] if entry["component"] == "Miller geometry and field approximation contract"
    )
    assert geometry_entry["covered_source_paths"] == ["src/scpn_control/core/gk_geometry.py"]
    species_entry = next(
        entry for entry in report["entries"] if entry["component"] == "gyrokinetic species and collision approximation contract"
    )
    assert species_entry["covered_source_paths"] == ["src/scpn_control/core/gk_species.py"]
    jax_gk_entry = next(entry for entry in report["entries"] if entry["component"] == "JAX gyrokinetic numerical parity guard")
    assert jax_gk_entry["covered_source_paths"] == ["src/scpn_control/core/jax_gk_solver.py"]
    ood_entry = next(
        entry for entry in report["entries"] if entry["component"] == "gyrokinetic OOD detector distribution-bound contract"
    )
    assert ood_entry["covered_source_paths"] == ["src/scpn_control/core/gk_ood_detector.py"]
    dedicated_core_paths = {
        "transport solver neoclassical and source-term approximation contract": "src/scpn_control/core/integrated_transport_solver.py",
        "reduced gyrokinetic transport closure contract": "src/scpn_control/core/gyrokinetic_transport.py",
        "momentum transport and torque-balance approximation contract": "src/scpn_control/core/momentum_transport.py",
        "EPED pedestal and peeling-ballooning approximation contract": "src/scpn_control/core/eped_pedestal.py",
        "ELM crash and RMP suppression approximation contract": "src/scpn_control/core/elm_model.py",
        "MARFE radiation-condensation density-limit contract": "src/scpn_control/core/marfe.py",
        "NTM island evolution and control approximation contract": "src/scpn_control/core/ntm_dynamics.py",
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
    core_entry = next(entry for entry in report["entries"] if entry["component"] == "bounded analytical approximations in physics modules")
    assert "src/scpn_control/core/gk_eigenvalue.py" not in core_entry["covered_source_paths"]
    assert "src/scpn_control/core/gk_geometry.py" not in core_entry["covered_source_paths"]
    assert "src/scpn_control/core/gk_nonlinear.py" not in core_entry["covered_source_paths"]
    assert "src/scpn_control/core/gk_ood_detector.py" not in core_entry["covered_source_paths"]
    assert "src/scpn_control/core/gk_species.py" not in core_entry["covered_source_paths"]
    assert "src/scpn_control/core/jax_gk_solver.py" not in core_entry["covered_source_paths"]
    for source_path in dedicated_core_paths.values():
        assert source_path not in core_entry["covered_source_paths"]
    assert core_entry["covered_source_paths"] == []
    gym_entry = next(entry for entry in report["entries"] if entry["component"] == "reduced-order plant and control environments")
    assert gym_entry["covered_source_paths"] == ["src/scpn_control/control/gym_tokamak_env.py"]
    rzip_entry = next(entry for entry in report["entries"] if entry["component"] == "RZIP rigid vertical stability model")
    assert rzip_entry["covered_source_paths"] == ["src/scpn_control/control/rzip_model.py"]
    realtime_efit_entry = next(entry for entry in report["entries"] if entry["component"] == "real-time EFIT-lite equilibrium reconstruction")
    assert realtime_efit_entry["covered_source_paths"] == ["src/scpn_control/control/realtime_efit.py"]
    halo_entry = next(entry for entry in report["entries"] if entry["component"] == "halo current and runaway electron disruption model")
    assert halo_entry["covered_source_paths"] == ["src/scpn_control/control/halo_re_physics.py"]
    free_boundary_entry = next(entry for entry in report["entries"] if entry["component"] == "direct free-boundary tracking controller")
    assert free_boundary_entry["covered_source_paths"] == ["src/scpn_control/control/free_boundary_tracking.py"]
    density_entry = next(entry for entry in report["entries"] if entry["component"] == "density control and particle-source model")
    assert density_entry["covered_source_paths"] == ["src/scpn_control/control/density_controller.py"]
    disruption_contract_entry = next(entry for entry in report["entries"] if entry["component"] == "disruption mitigation contract layer")
    assert disruption_contract_entry["covered_source_paths"] == ["src/scpn_control/control/disruption_contracts.py"]
    digital_twin_entry = next(entry for entry in report["entries"] if entry["component"] == "tokamak digital twin topology and diffusion model")
    assert digital_twin_entry["covered_source_paths"] == ["src/scpn_control/control/tokamak_digital_twin.py"]
    soc_entry = next(entry for entry in report["entries"] if entry["component"] == "advanced SOC turbulence learning controller")
    assert soc_entry["covered_source_paths"] == ["src/scpn_control/control/advanced_soc_fusion_learning.py"]
    burn_entry = next(entry for entry in report["entries"] if entry["component"] == "DT burn control and alpha-heating model")
    assert burn_entry["covered_source_paths"] == ["src/scpn_control/control/burn_controller.py"]
    volt_second_entry = next(entry for entry in report["entries"] if entry["component"] == "volt-second budget and flux-consumption manager")
    assert volt_second_entry["covered_source_paths"] == ["src/scpn_control/control/volt_second_manager.py"]
    control_entry = next(entry for entry in report["entries"] if entry["component"] == "bounded control plant approximations")
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


def test_traceability_rejects_unbounded_gap_claim(tmp_path: Path) -> None:
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
                "required_actions": ["replace claim with evidence"],
            }
        ]
    )
    path = tmp_path / "physics_traceability.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_physics_traceability(path)

    assert report["status"] == "fail"
    fields = {error["field"] for error in report["errors"]}
    assert "public_claim_allowed" in fields


def test_traceability_rejects_missing_contract_fields(tmp_path: Path) -> None:
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
                "required_actions": [],
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
    registry = {
        "schema_version": "1.0",
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
                "required_actions": ["keep evidence current"],
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
                "required_actions": ["keep evidence current"],
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
                "required_actions": ["cover every approximation marker"],
            }
        ]
    )
    registry["enforce_source_marker_coverage"] = True
    path = tmp_path / "physics_traceability.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_physics_traceability(path)

    assert report["status"] == "fail"
    assert report["source_marker_coverage"]["total"] >= 30
    assert "src/scpn_control/core/gk_geometry.py" in report["source_marker_coverage"]["missing"]


def test_traceability_rejects_source_coverage_outside_module_scope(tmp_path: Path) -> None:
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
                "required_actions": ["keep source coverage inside module scope"],
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
