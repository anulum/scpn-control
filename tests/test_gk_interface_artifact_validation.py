# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — External GK interface artifact validation tests

from __future__ import annotations

import json
from pathlib import Path

from validation.validate_gk_interface_artifacts import canonical_artifact_sha256, validate_gk_interface_artifacts


def _valid_real_executable_artifact() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "scpn-control.gk-interface-artifact.v1",
        "interface_code": "GENE",
        "source": "real_executable",
        "code_version": "3.1.0",
        "run_id": "gene-cbc-parser-2026-05-18",
        "executed_at": "2026-05-18T08:15:00Z",
        "binary_path": "/opt/gene/bin/gene",
        "input_deck_uri": "gene/cbc/input_deck",
        "output_artifact_uri": "gene/cbc/stdout.txt",
        "parsed_output_uri": "gene/cbc/parsed_output.json",
        "input_deck_sha256": "a" * 64,
        "output_artifact_sha256": "b" * 64,
        "parsed_output_sha256": "c" * 64,
        "parser_version": "scpn-control-gk-interface-1",
        "units": "m^2/s transport; c_s/a frequencies; k_y*rho_s wavenumber",
        "chi_i_m2_s": 1.8,
        "chi_e_m2_s": 1.1,
        "D_e_m2_s": 0.45,
        "gamma_max_cs_over_a": 0.19,
        "omega_r_cs_over_a": -0.34,
        "k_y_rho_s_at_max": 0.31,
    }
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    return payload


def test_strict_gk_interface_gate_requires_real_artifacts(tmp_path: Path) -> None:
    report = validate_gk_interface_artifacts(tmp_path, require_interface_artifacts=True)

    assert report["status"] == "fail"
    assert report["interface_artifacts"] == 0
    assert report["errors"][0]["error"] == "no external GK interface artifacts found"


def test_gk_interface_gate_accepts_real_executable_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "gene_cbc_parser.json"
    artifact.write_text(json.dumps(_valid_real_executable_artifact()), encoding="utf-8")

    report = validate_gk_interface_artifacts(tmp_path, require_interface_artifacts=True)

    assert report["status"] == "pass"
    assert report["interface_artifacts"] == 1
    assert report["entries"][0]["interface_code"] == "GENE"
    assert report["entries"][0]["source"] == "real_executable"


def test_gk_interface_gate_accepts_documented_public_reference_artifact(tmp_path: Path) -> None:
    payload = _valid_real_executable_artifact()
    payload["interface_code"] = "QuaLiKiz"
    payload["source"] = "documented_public_reference"
    payload.pop("binary_path")
    payload["reference_doi"] = "10.1016/j.cpc.2016.09.003"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "qualikiz_public_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_interface_artifacts(tmp_path, require_interface_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["interface_code"] == "QuaLiKiz"


def test_gk_interface_gate_rejects_mock_source(tmp_path: Path) -> None:
    payload = _valid_real_executable_artifact()
    payload["source"] = "mock_subprocess"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "mock_gene.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_interface_artifacts(tmp_path, require_interface_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_gk_interface_gate_rejects_missing_hash_provenance(tmp_path: Path) -> None:
    payload = _valid_real_executable_artifact()
    payload["output_artifact_sha256"] = "not-a-hash"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "bad_hash.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_interface_artifacts(tmp_path, require_interface_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "output_artifact_sha256"


def test_gk_interface_gate_rejects_relative_binary_path(tmp_path: Path) -> None:
    payload = _valid_real_executable_artifact()
    payload["binary_path"] = "gene"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "relative_binary.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_interface_artifacts(tmp_path, require_interface_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "binary_path"


def test_gk_interface_gate_rejects_mutable_binary_root(tmp_path: Path) -> None:
    payload = _valid_real_executable_artifact()
    payload["binary_path"] = "/tmp/gene"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "mutable_binary.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_interface_artifacts(tmp_path, require_interface_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "binary_path"


def test_gk_interface_gate_rejects_tampered_payload_digest(tmp_path: Path) -> None:
    payload = _valid_real_executable_artifact()
    payload["chi_i_m2_s"] = 2.4
    artifact = tmp_path / "tampered_transport.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_interface_artifacts(tmp_path, require_interface_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "payload_sha256"


def test_gk_interface_gate_rejects_missing_parsed_output_digest(tmp_path: Path) -> None:
    payload = _valid_real_executable_artifact()
    payload.pop("parsed_output_sha256")
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "missing_parsed_digest.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_interface_artifacts(tmp_path, require_interface_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "parsed_output_sha256"


def test_gk_interface_gate_rejects_traversing_artifact_uri(tmp_path: Path) -> None:
    payload = _valid_real_executable_artifact()
    payload["parsed_output_uri"] = "../parsed_output.json"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "traversing_artifact_uri.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_interface_artifacts(tmp_path, require_interface_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "parsed_output_uri"
