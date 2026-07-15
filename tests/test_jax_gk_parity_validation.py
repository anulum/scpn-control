# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JAX GK parity validation tests

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_control.core.jax_gk_solver import _HAS_JAX, write_jax_gk_parity_artifact
from validation.benchmark_jax_gk_parity import build_benchmark_report, write_benchmark_report
from validation.validate_jax_gk_parity import validate_jax_gk_parity


def _valid_parity_report() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "scpn-control.jax-gk-parity.v1",
        "case": "cyclone_base_case",
        "backend": "cpu",
        "jax_version": "0.5.0",
        "jaxlib_version": "0.5.0",
        "platform": "linux-x86_64",
        "device_kind": "cpu",
        "dtype": "float32",
        "x64_enabled": False,
        "executed_at": "2026-05-18T06:30:00Z",
        "native_gamma_max_cs_over_a": 0.21,
        "jax_gamma_max_cs_over_a": 0.209,
        "native_omega_r_cs_over_a": -0.42,
        "jax_omega_r_cs_over_a": -0.419,
        "gamma_relative_tolerance": 0.05,
        "omega_absolute_tolerance": 0.02,
        "solver_contract": "native_linear_gk_local_dispersion",
        "normalisation": "c_s_over_a",
        "evidence_boundary": "backend_parity_only",
        "external_validation_required": True,
        "admitted_for_control": False,
        "solver_kwargs": {
            "B0": 2.0,
            "R0": 2.78,
            "a": 1.0,
            "n_ky_ion": 4,
            "n_theta": 16,
            "q": 1.4,
            "s_hat": 0.78,
        },
        "case_parameters": {
            "case": "cyclone_base_case",
            "solver_kwargs": {
                "B0": 2.0,
                "R0": 2.78,
                "a": 1.0,
                "n_ky_ion": 4,
                "n_theta": 16,
                "q": 1.4,
                "s_hat": 0.78,
            },
            "species": [
                {
                    "name": "deuterium",
                    "mass_kg": 3.343583719e-27,
                    "charge_e": 1,
                    "density_19": 5.0,
                    "temperature_keV": 1.0,
                    "R_L_n": 2.2,
                    "R_L_T": 6.9,
                    "adiabatic": False,
                },
                {
                    "name": "electron",
                    "mass_kg": 9.1093837015e-31,
                    "charge_e": -1,
                    "density_19": 5.0,
                    "temperature_keV": 1.0,
                    "R_L_n": 2.2,
                    "R_L_T": 6.9,
                    "adiabatic": True,
                },
            ],
            "electron_model": "adiabatic",
        },
        "case_acceptance": {
            "required_mode_types": ["ITG"],
            "max_gamma_max_cs_over_a": None,
            "description": "Cyclone Base Case ion-temperature-gradient parity guard",
        },
        "native_mode_types": ["ITG", "ITG", "stable", "stable"],
        "jax_mode_types": ["ITG", "ITG", "stable", "stable"],
        "native_dominant_mode_type": "ITG",
        "jax_dominant_mode_type": "ITG",
    }
    payload["solver_kwargs_sha256"] = _payload_sha256(payload["solver_kwargs"], include_payload_field=True)
    payload["case_parameters_sha256"] = _payload_sha256(payload["case_parameters"], include_payload_field=True)
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def _payload_sha256(payload: object, *, include_payload_field: bool = False) -> str:
    import hashlib

    digest_payload = payload
    if isinstance(payload, dict) and not include_payload_field:
        digest_payload = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_strict_jax_parity_gate_requires_persisted_artifacts(tmp_path: Path) -> None:
    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert report["status"] == "fail"
    assert report["parity_artifacts"] == 0
    assert report["errors"][0]["error"] == "no JAX GK parity artifacts found"


def test_jax_parity_gate_accepts_backend_metadata_and_tolerances(tmp_path: Path) -> None:
    artifact = tmp_path / "cbc_cpu.json"
    artifact.write_text(json.dumps(_valid_parity_report()), encoding="utf-8")

    report = validate_jax_gk_parity(
        tmp_path,
        require_parity_artifacts=True,
        require_cases=("cyclone_base_case",),
        require_backends=("cpu",),
    )

    assert report["status"] == "pass"
    assert report["parity_artifacts"] == 1
    assert report["backend_counts"] == {"cpu": 1}
    assert report["case_counts"] == {"cyclone_base_case": 1}
    assert report["observed_case_backend_pairs"] == ["cyclone_base_case/cpu"]
    assert report["complete_required_case_backend_coverage"] is True
    assert report["max_gamma_relative_error"] < 0.01
    assert len(report["entries_payload_sha256"]) == 64
    assert len(report["report_payload_sha256"]) == 64
    assert report["entries"][0]["case"] == "cyclone_base_case"
    assert report["entries"][0]["backend"] == "cpu"
    assert report["entries"][0]["native_dominant_mode_type"] == "ITG"


def test_repository_jax_parity_evidence_covers_release_cpu_gpu_campaign() -> None:
    artifact_root = Path(__file__).resolve().parents[1] / "validation" / "reports" / "jax_gk_parity"

    report = validate_jax_gk_parity(
        artifact_root,
        require_parity_artifacts=True,
        require_cases=("cyclone_base_case", "tem_kinetic_electron", "stable_mode"),
        require_backends=("cpu", "gpu"),
    )

    observed_pairs = {(entry["case"], entry["backend"]) for entry in report["entries"]}
    assert report["status"] == "pass"
    assert observed_pairs == {
        ("cyclone_base_case", "cpu"),
        ("cyclone_base_case", "gpu"),
        ("tem_kinetic_electron", "cpu"),
        ("tem_kinetic_electron", "gpu"),
        ("stable_mode", "cpu"),
        ("stable_mode", "gpu"),
    }
    assert all(entry["evidence_boundary"] == "backend_parity_only" for entry in report["entries"])


def test_jax_parity_gate_rejects_missing_required_case_backend_pair(tmp_path: Path) -> None:
    artifact = tmp_path / "cbc_cpu.json"
    artifact.write_text(json.dumps(_valid_parity_report()), encoding="utf-8")

    report = validate_jax_gk_parity(
        tmp_path,
        require_parity_artifacts=True,
        require_cases=("cyclone_base_case", "tem_kinetic_electron"),
        require_backends=("cpu", "gpu"),
    )

    assert report["status"] == "fail"
    assert report["complete_required_case_backend_coverage"] is False
    assert any(error["field"] == "required_case_backend" for error in report["errors"])


def test_jax_parity_gate_rejects_missing_backend_metadata(tmp_path: Path) -> None:
    payload = _valid_parity_report()
    payload["jaxlib_version"] = ""
    artifact = tmp_path / "cbc_cpu.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "jaxlib_version"


def test_jax_parity_gate_rejects_out_of_tolerance_artifact(tmp_path: Path) -> None:
    payload = _valid_parity_report()
    payload["jax_gamma_max_cs_over_a"] = 0.1
    payload["payload_sha256"] = _payload_sha256(payload)
    artifact = tmp_path / "cbc_cpu.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "gamma_max_cs_over_a"


def test_jax_parity_gate_rejects_payload_digest_replay(tmp_path: Path) -> None:
    payload = _valid_parity_report()
    payload["payload_sha256"] = "0" * 64
    artifact = tmp_path / "cbc_cpu.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "payload_sha256"


def test_jax_parity_gate_rejects_case_parameter_digest_replay(tmp_path: Path) -> None:
    payload = _valid_parity_report()
    case_parameters = payload["case_parameters"]
    assert isinstance(case_parameters, dict)
    species = case_parameters["species"]
    assert isinstance(species, list)
    species[0]["R_L_T"] = 99.0
    payload["payload_sha256"] = _payload_sha256(payload)
    artifact = tmp_path / "cbc_cpu.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "case_parameters_sha256"


def test_jax_parity_gate_rejects_mode_spectrum_replay(tmp_path: Path) -> None:
    payload = _valid_parity_report()
    payload["jax_mode_types"] = ["stable", "stable", "stable", "stable"]
    payload["jax_dominant_mode_type"] = "stable"
    payload["payload_sha256"] = _payload_sha256(payload)
    artifact = tmp_path / "cbc_cpu.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "mode_types"


def test_jax_parity_gate_rejects_control_admission_replay(tmp_path: Path) -> None:
    payload = _valid_parity_report()
    payload["admitted_for_control"] = True
    payload["payload_sha256"] = _payload_sha256(payload)
    artifact = tmp_path / "cbc_cpu.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "admitted_for_control"


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_jax_parity_writer_persists_valid_backend_artifact(tmp_path: Path) -> None:
    payload, artifact_path = write_jax_gk_parity_artifact(
        tmp_path,
        solver_kwargs={"n_ky_ion": 2, "n_theta": 8},
        gamma_relative_tolerance=1.0,
        omega_absolute_tolerance=1.0,
        executed_at="2026-05-31T00:00:00Z",
    )

    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert artifact_path.exists()
    assert payload["payload_sha256"] == report["entries"][0]["payload_sha256"]
    assert report["status"] == "pass"
    assert report["entries"][0]["evidence_boundary"] == "backend_parity_only"


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_jax_parity_writer_persists_kinetic_electron_mode_contract(tmp_path: Path) -> None:
    payload, artifact_path = write_jax_gk_parity_artifact(
        tmp_path,
        case="tem_kinetic_electron",
        solver_kwargs={"n_ky_ion": 4, "n_theta": 8},
        gamma_relative_tolerance=1.0,
        omega_absolute_tolerance=1.0,
        executed_at="2026-05-31T00:00:00Z",
    )

    report = validate_jax_gk_parity(
        tmp_path,
        require_parity_artifacts=True,
        require_cases=("tem_kinetic_electron",),
        require_backends=(str(payload["backend"]),),
    )

    assert artifact_path.exists()
    assert report["status"] == "pass"
    assert payload["case_parameters"]["electron_model"] == "kinetic"
    assert "TEM" in payload["native_mode_types"]
    assert "TEM" in payload["jax_mode_types"]


def test_jax_parity_benchmark_report_keeps_timing_separate_from_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "cbc_cpu.json"
    artifact.write_text(json.dumps(_valid_parity_report()), encoding="utf-8")
    validation_report = validate_jax_gk_parity(
        tmp_path,
        require_parity_artifacts=True,
        require_cases=("cyclone_base_case",),
        require_backends=("cpu",),
    )

    benchmark = build_benchmark_report(
        artifact_root=tmp_path,
        generated_artifacts=[
            {
                "case": "cyclone_base_case",
                "backend": "cpu",
                "device_kind": "cpu",
                "path": "cbc_cpu.json",
                "payload_sha256": validation_report["entries"][0]["payload_sha256"],
                "elapsed_s": 0.125,
            }
        ],
        validation_report=validation_report,
        total_elapsed_s=0.25,
        cases=("cyclone_base_case",),
    )
    write_benchmark_report(benchmark, tmp_path / "benchmark.json", tmp_path / "benchmark.md")

    assert benchmark["schema_version"] == "scpn-control.jax-gk-parity-benchmark.v1"
    assert benchmark["validation_report_payload_sha256"] == validation_report["report_payload_sha256"]
    assert benchmark["generated_artifact_count"] == 1
    assert "external GK validation remains required" in benchmark["claim_boundary"]
    assert "JAX GK Parity Benchmark Report" in (tmp_path / "benchmark.md").read_text(encoding="utf-8")


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_jax_parity_writer_keeps_explicit_json_path_with_default_solver_kwargs(tmp_path: Path) -> None:
    """An explicit .json path with default solver kwargs keeps the path and takes the no-kwargs build path.

    Exercises the no-solver-kwargs merge in build (658->660) and the .json-suffix short-circuit in the
    writer (726->729).
    """
    payload, artifact_path = write_jax_gk_parity_artifact(
        tmp_path / "parity.json",
        gamma_relative_tolerance=1.0,
        omega_absolute_tolerance=1.0,
        executed_at="2026-05-31T00:00:00Z",
    )
    assert artifact_path.name == "parity.json"
    assert payload["case"] == "cyclone_base_case"
