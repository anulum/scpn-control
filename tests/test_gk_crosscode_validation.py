# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK cross-code evidence validation tests

from __future__ import annotations

import json
from pathlib import Path

from validation.validate_gk_crosscode import validate_gk_crosscode_evidence


def _valid_gene_report() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "case": "cyclone_base",
        "external_code": "GENE",
        "source": "real_binary",
        "binary_path": "/opt/gene/bin/gene",
        "code_version": "2026.05",
        "run_id": "gene-cbc-2026-05-18",
        "executed_at": "2026-05-18T06:00:00Z",
        "units": "c_s/a",
        "gamma_max_cs_over_a": 0.18,
        "omega_r_cs_over_a": -0.42,
        "k_y_rho_s_at_max": 0.30,
        "native_gamma_max_cs_over_a": 0.19,
        "native_omega_r_cs_over_a": -0.40,
        "native_k_y_rho_s_at_max": 0.31,
    }


def test_strict_crosscode_gate_requires_real_external_run(tmp_path: Path) -> None:
    report = validate_gk_crosscode_evidence(tmp_path, require_external_runs=True)

    assert report["status"] == "fail"
    assert report["external_runs"] == 0
    assert report["errors"][0]["error"] == "no real external GK evidence reports found"


def test_crosscode_gate_accepts_real_binary_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "gene_cbc.json"
    evidence.write_text(json.dumps(_valid_gene_report()), encoding="utf-8")

    report = validate_gk_crosscode_evidence(tmp_path, require_external_runs=True)

    assert report["status"] == "pass"
    assert report["external_runs"] == 1
    assert report["entries"][0]["external_code"] == "GENE"
    assert report["entries"][0]["gamma_relative_error"] < 0.2


def test_crosscode_gate_rejects_missing_binary_provenance(tmp_path: Path) -> None:
    payload = _valid_gene_report()
    payload["binary_path"] = ""
    evidence = tmp_path / "gene_cbc.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_crosscode_evidence(tmp_path, require_external_runs=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "binary_path"


def test_crosscode_gate_rejects_out_of_tolerance_evidence(tmp_path: Path) -> None:
    payload = _valid_gene_report()
    payload["native_gamma_max_cs_over_a"] = 0.35
    evidence = tmp_path / "gene_cbc.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_crosscode_evidence(tmp_path, require_external_runs=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "gamma_max_cs_over_a"
