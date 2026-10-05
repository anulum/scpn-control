# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JAX GK parity validation tests

"""Exercise persisted parity declarations and the original real JAX writer chain.

Copied report mutations test schema/refusal policy and never certify a newly
measured backend or replace native/JAX physical execution.
"""

from __future__ import annotations

import doctest
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scpn_control.core.jax_gk_solver import _HAS_JAX, write_jax_gk_parity_artifact
from validation import validate_jax_gk_parity as parity_module
from validation.benchmark_jax_gk_parity import build_benchmark_report, write_benchmark_report
from validation.validate_jax_gk_parity import main, validate_jax_gk_parity


def _valid_parity_report() -> dict[str, object]:
    """Build the original schema fixture, without claiming measured backend evidence.

    Its invented scalars/metadata exercise declarations and canonical digest
    rules. The original real writer tests below exercise numerical execution.
    """
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
    """Hash the independent test canonicalisation, excluding the artifact self key.

    Include mode preserves nested metadata keys; no production helper is called.
    """
    import hashlib

    digest_payload = payload
    if isinstance(payload, dict) and not include_payload_field:
        digest_payload = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_strict_jax_parity_gate_requires_persisted_artifacts(tmp_path: Path) -> None:
    """Refuse an empty directory when persisted evidence is explicitly required."""
    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert report["status"] == "fail"
    assert report["parity_artifacts"] == 0
    assert report["errors"][0]["error"] == "no JAX GK parity artifacts found"


def test_jax_parity_gate_accepts_backend_metadata_and_tolerances(tmp_path: Path) -> None:
    """Admit the original schema fixture and expose its counts, drift and coverage."""
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
    """Read all six unchanged historical artifacts through the real release matrix gate."""
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
    """Keep an admitted entry while refusing incomplete named Cartesian coverage."""
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
    """Refuse missing JAXLIB metadata before admitting an artifact."""
    payload = _valid_parity_report()
    payload["jaxlib_version"] = ""
    artifact = tmp_path / "cbc_cpu.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "jaxlib_version"


def test_jax_parity_gate_rejects_out_of_tolerance_artifact(tmp_path: Path) -> None:
    """Refuse resealed declarations whose gamma drift exceeds their positive bound."""
    payload = _valid_parity_report()
    payload["jax_gamma_max_cs_over_a"] = 0.1
    payload["payload_sha256"] = _payload_sha256(payload)
    artifact = tmp_path / "cbc_cpu.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "gamma_max_cs_over_a"


def test_jax_parity_gate_rejects_payload_digest_replay(tmp_path: Path) -> None:
    """Refuse a syntactically valid self digest that does not bind the declaration."""
    payload = _valid_parity_report()
    payload["payload_sha256"] = "0" * 64
    artifact = tmp_path / "cbc_cpu.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "payload_sha256"


def test_jax_parity_gate_rejects_case_parameter_digest_replay(tmp_path: Path) -> None:
    """Refuse changed species metadata whose nested digest was not updated."""
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
    """Refuse a resealed declaration with differing native/JAX ordered mode spectra."""
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
    """Refuse parity artifacts claiming control authority despite a valid self digest."""
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
    """Run the real native/JAX writer and admit its persisted comparison through the reader."""
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
    """Run the real kinetic-electron writer and check its native/JAX TEM spectra."""
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
    """Exercise the real report formatter with declared test timing, not a measured campaign."""
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


@pytest.fixture
def historical_parity_payload() -> dict[str, Any]:
    """Decode one unchanged historical CPU artifact into independent mutable bytes.

    Mutations below establish declaration policy only, not new measured parity.
    """
    path = Path(__file__).resolve().parents[1] / "validation/reports/jax_gk_parity/cyclone_base_case_cpu_cpu.json"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _write_declaration(tmp_path: Path, payload: dict[str, Any]) -> Path:
    """Reseal a copied declaration with independent test canonicalisation.

    Nested digests are retained to expose mismatches when metadata is changed.
    """
    payload["payload_sha256"] = _payload_sha256(payload)
    path = tmp_path / "parity.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("schema_version", "wrong", "schema_version"),
        ("jax_version", None, "jax_version"),
        ("case", [], "case"),
        ("case", "unknown", "case"),
        ("backend", {}, "backend"),
        ("backend", "unknown", "backend"),
        ("x64_enabled", 1, "x64_enabled"),
        ("external_validation_required", False, "external_validation_required"),
        ("external_validation_required", "true", "external_validation_required"),
        ("admitted_for_control", None, "admitted_for_control"),
        ("solver_contract", "wrong", "solver_contract"),
        ("normalisation", "wrong", "normalisation"),
        ("evidence_boundary", "wrong", "evidence_boundary"),
        ("solver_kwargs_sha256", "x" * 64, "solver_kwargs_sha256"),
        ("case_parameters_sha256", "short", "case_parameters_sha256"),
        ("solver_kwargs", [], "solver_kwargs"),
        ("solver_kwargs", {}, "solver_kwargs"),
        ("solver_kwargs", {"R0": 999}, "solver_kwargs_sha256"),
        ("case_parameters", None, "case_parameters"),
        ("case_parameters", {}, "case_parameters"),
        ("case_acceptance", None, "case_acceptance"),
        ("case_acceptance", {}, "case_acceptance"),
        ("native_mode_types", "ITG", "native_mode_types"),
        ("native_mode_types", [], "native_mode_types"),
        ("native_mode_types", [" "], "native_mode_types"),
        ("jax_mode_types", [1], "jax_mode_types"),
        ("native_dominant_mode_type", 1, "native_dominant_mode_type"),
        ("jax_dominant_mode_type", " ", "jax_dominant_mode_type"),
        ("native_gamma_max_cs_over_a", -1, "gamma_max_cs_over_a"),
        ("jax_gamma_max_cs_over_a", -1, "gamma_max_cs_over_a"),
        ("gamma_relative_tolerance", 0, "gamma_relative_tolerance"),
        ("omega_absolute_tolerance", -1, "omega_absolute_tolerance"),
        ("jax_omega_r_cs_over_a", 999, "omega_r_cs_over_a"),
        ("jax_dominant_mode_type", "TEM", "dominant_mode_type"),
        ("case_acceptance", {"required_mode_types": []}, "case_acceptance.required_mode_types"),
        ("case_acceptance", {"required_mode_types": ["TEM"]}, "case_acceptance.required_mode_types"),
        (
            "case_acceptance",
            {"required_mode_types": ["ITG"], "max_gamma_max_cs_over_a": True},
            "case_acceptance.max_gamma_max_cs_over_a",
        ),
        (
            "case_acceptance",
            {"required_mode_types": ["ITG"], "max_gamma_max_cs_over_a": "bad"},
            "case_acceptance.max_gamma_max_cs_over_a",
        ),
        (
            "case_acceptance",
            {"required_mode_types": ["ITG"], "max_gamma_max_cs_over_a": 10**400},
            "case_acceptance.max_gamma_max_cs_over_a",
        ),
        (
            "case_acceptance",
            {"required_mode_types": ["ITG"], "max_gamma_max_cs_over_a": 0},
            "case_acceptance.max_gamma_max_cs_over_a",
        ),
    ],
)
def test_copied_public_declaration_domain_refusals(
    tmp_path: Path, historical_parity_payload: dict[str, Any], field: str, value: object, expected: str
) -> None:
    """Refuse malformed shape/metadata/spectrum/domain through the real public API."""
    historical_parity_payload[field] = value
    path = _write_declaration(tmp_path, historical_parity_payload)
    report = validate_jax_gk_parity(path, require_parity_artifacts=True)
    assert report["status"] == "fail" and report["entries"] == []
    assert any(error["field"] == expected for error in report["errors"])


@pytest.mark.parametrize(
    "field",
    [
        "native_gamma_max_cs_over_a",
        "jax_gamma_max_cs_over_a",
        "native_omega_r_cs_over_a",
        "jax_omega_r_cs_over_a",
        "gamma_relative_tolerance",
        "omega_absolute_tolerance",
    ],
)
@pytest.mark.parametrize("value", [True, "bad", 10**400])
def test_real_numeric_domain_refusals(
    tmp_path: Path, historical_parity_payload: dict[str, Any], field: str, value: object
) -> None:
    """Reject booleans, nonnumeric values and float-overflow integers for every scalar."""
    historical_parity_payload[field] = value
    report = validate_jax_gk_parity(_write_declaration(tmp_path, historical_parity_payload))
    assert report["status"] == "fail" and report["parity_artifacts"] == 0
    assert any(
        error["field"] == field and error["error"] == "field must be finite numeric" for error in report["errors"]
    )


@pytest.mark.parametrize(
    "contents",
    [
        '{"extra":{"x":NaN}}',
        '{"extra":[Infinity]}',
        '{"extra":-Infinity}',
        '{"extra":1e400}',
        '{"x":1,"x":2}',
        '{"extra":{"x":1,"x":2}}',
        "[1]",
        "{",
        "[" * 1200 + "0" + "]" * 1200,
    ],
)
def test_real_json_read_and_decode_refusals(tmp_path: Path, contents: str) -> None:
    """Reject malformed/nonfinite/duplicate/deep declarations without production helpers."""
    path = tmp_path / "broken.json"
    path.write_text(contents, encoding="utf-8")
    report = validate_jax_gk_parity(path, require_parity_artifacts=True)
    assert report["status"] == "fail" and report["entries"] == []
    assert report["errors"][0]["field"] in {"json", "root"}


def test_real_unreadable_child_and_invalid_utf8(tmp_path: Path) -> None:
    """Read actual directory-shaped JSON child and invalid UTF-8 bytes as findings."""
    (tmp_path / "directory.json").mkdir()
    (tmp_path / "bad.json").write_bytes(b"\xff")
    report = validate_jax_gk_parity(tmp_path)
    assert report["status"] == "fail" and len(report["errors"]) == 2
    assert all(error["field"] == "json" for error in report["errors"])


@pytest.mark.parametrize("value", [None, "false", 1])
def test_public_policy_boolean_refusal(tmp_path: Path, value: Any) -> None:
    """Normalize invalid policy declarations to false while retaining a FAIL finding."""
    report = validate_jax_gk_parity(tmp_path, require_parity_artifacts=value)
    assert report["status"] == "fail" and report["require_parity_artifacts"] is False
    assert report["errors"][0]["field"] == "require_parity_artifacts"


@pytest.mark.parametrize("required", ["case", "backend"])
def test_unsupported_requirement_configuration(tmp_path: Path, required: str) -> None:
    """Preserve the public ValueError contract for unsupported requested names."""
    with pytest.raises(ValueError, match=f"unsupported required {required}"):
        if required == "case":
            validate_jax_gk_parity(tmp_path, require_cases=["unknown"])
        else:
            validate_jax_gk_parity(tmp_path, require_backends=["unknown"])


def test_requirement_normalization_duplicates_and_single_axis(
    tmp_path: Path, historical_parity_payload: dict[str, Any]
) -> None:
    """Retain duplicate admitted files while normalizing blank/repeated required names."""
    path = _write_declaration(tmp_path, historical_parity_payload)
    (tmp_path / "duplicate.json").write_bytes(path.read_bytes())
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/ignored.json").write_text("broken")
    report = validate_jax_gk_parity(tmp_path, require_cases=[" ", " cyclone_base_case ", "cyclone_base_case"])
    assert report["status"] == "pass" and report["parity_artifacts"] == 2
    assert report["complete_required_case_backend_coverage"] is None
    assert report["case_counts"] == {"cyclone_base_case": 2}
    assert report["required_cases"] == ["cyclone_base_case"]
    report = validate_jax_gk_parity(path, require_backends={"cpu", " "})
    assert report["status"] == "pass" and report["required_backends"] == ["cpu"]


def test_artifact_digest_shape_case_and_extra_exclusions(
    tmp_path: Path, historical_parity_payload: dict[str, Any]
) -> None:
    """Exercise invalid and uppercase digests plus the explicit report-key exclusion."""
    path = _write_declaration(tmp_path, historical_parity_payload)
    for value in [None, "short", "g" * 64, historical_parity_payload["payload_sha256"].upper()]:
        historical_parity_payload["payload_sha256"] = value
        path.write_text(json.dumps(historical_parity_payload))
        report = validate_jax_gk_parity(path)
        assert report["status"] == "fail" and any(error["field"] == "payload_sha256" for error in report["errors"])
    historical_parity_payload.pop("payload_sha256")
    path = _write_declaration(tmp_path, historical_parity_payload)
    historical_parity_payload["report_payload_sha256"] = {"unverified": 1}
    path.write_text(json.dumps(historical_parity_payload))
    assert validate_jax_gk_parity(path)["status"] == "pass"


def test_equal_unknown_modes_and_finite_growth_bound(tmp_path: Path, historical_parity_payload: dict[str, Any]) -> None:
    """Document equality-only mode admission without inventing a physical mode classifier."""
    historical_parity_payload.update(
        native_mode_types=[" mystery "],
        jax_mode_types=["mystery"],
        native_dominant_mode_type="mystery",
        jax_dominant_mode_type="mystery",
        case_acceptance={"required_mode_types": ["mystery"], "max_gamma_max_cs_over_a": 100},
    )
    report = validate_jax_gk_parity(_write_declaration(tmp_path, historical_parity_payload))
    assert report["status"] == "pass" and report["entries"][0]["native_dominant_mode_type"] == "mystery"
    historical_parity_payload.update(native_dominant_mode_type="absent", jax_dominant_mode_type="absent")
    report = validate_jax_gk_parity(_write_declaration(tmp_path, historical_parity_payload))
    assert report["status"] == "fail" and any(
        error["error"] == "dominant mode missing from spectrum" for error in report["errors"]
    )


def test_standalone_main_json_text_and_write_refusal(
    tmp_path: Path, historical_parity_payload: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise actual output bytes, CSV requirements, text diagnostics and write/path refusals."""
    path = _write_declaration(tmp_path, historical_parity_payload)
    output = tmp_path / "out/report.json"
    assert (
        main(
            [
                "--artifact-root",
                str(path),
                "--require-cases",
                " ,cyclone_base_case,",
                "--require-backends",
                "cpu,",
                "--output-json",
                str(output),
                "--json-out",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == report
    assert report["status"] == "pass" and output.read_bytes().endswith(b"\n")
    for refused in [str(tmp_path), str(path / "child.json"), "bad\0path"]:
        assert main(["--artifact-root", str(path), "--output-json", refused, "--json-out"]) == 1
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "fail" and report["errors"][-1]["field"] == "output_json"
        expected = _payload_sha256({k: v for k, v in report.items() if k != "report_payload_sha256"})
        assert report["report_payload_sha256"] == expected
    assert main(["--artifact-root", str(tmp_path / "missing"), "--require-parity-artifacts"]) == 1
    text = capsys.readouterr()
    assert "parity_artifacts=0" in text.out and "ERROR" in text.err
    assert main(["--artifact-root", str(path)]) == 0
    assert "JAX GK parity: pass" in capsys.readouterr().out


@pytest.mark.parametrize("no_site", [False, True])
def test_actual_standalone_cli_needs_no_jax_import(
    tmp_path: Path, historical_parity_payload: dict[str, Any], no_site: bool
) -> None:
    """Run the actual script from another cwd with empty PYTHONPATH, optionally without site."""
    path = _write_declaration(tmp_path, historical_parity_payload)
    env = dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    command = [
        sys.executable,
        *(["-S"] if no_site else []),
        str(Path(parity_module.__file__)),
        "--artifact-root",
        str(path),
        "--require-parity-artifacts",
        "--json-out",
    ]
    completed = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True, check=False)
    assert completed.returncode == 0 and completed.stderr == ""
    report = json.loads(completed.stdout)
    assert report["status"] == "pass" and report["entries"][0]["evidence_boundary"] == "backend_parity_only"


def test_native_module_examples_are_executable() -> None:
    """Execute the native public empty-directory example without a mock or physical model."""
    result = doctest.testmod(parity_module, raise_on_error=True)
    assert result.failed == 0 and result.attempted >= 2
