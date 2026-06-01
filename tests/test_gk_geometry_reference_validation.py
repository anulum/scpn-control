# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK geometry reference validation tests

from __future__ import annotations

import json
from pathlib import Path

from validation.validate_gk_geometry_reference import validate_gk_geometry_reference


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CASES = ROOT / "validation" / "reference_data" / "gk_geometry" / "miller_reference_cases.json"


def test_repository_geometry_reference_cases_pass() -> None:
    report = validate_gk_geometry_reference(REFERENCE_CASES)

    assert report["status"] == "pass"
    assert report["schema_version"] == "scpn-control.gk-geometry-reference.v2"
    assert len(report["reference_file_sha256"]) == 64
    assert len(report["payload_sha256"]) == 64
    assert report["tolerances"] == {"absolute": 1.0e-11, "relative": 1.0e-10}
    assert report["units"]["R"] == "m"
    assert report["units"]["theta"] == "rad"
    assert report["units"]["B_toroidal"] == "T"
    assert report["public_claims"] == {
        "bounded_local_miller_geometry_reference": True,
        "full_equilibrium_reconstruction": False,
        "full_equilibrium_blocked_reason": (
            "Requires independent Miller-geometry implementation or external equilibrium-code evidence."
        ),
    }
    assert report["cases"] == 3
    assert {entry["case"] for entry in report["entries"]} == {
        "circular_cyclone_limit",
        "shaped_positive_triangularity",
        "high_shear_local_equilibrium",
    }
    assert all(len(entry["case_sha256"]) == 64 for entry in report["entries"])


def test_geometry_reference_gate_rejects_missing_required_case(tmp_path: Path) -> None:
    payload = {
        "spdx_license_id": "AGPL-3.0-or-later",
        "commercial_license": "available",
        "concepts_copyright": "Concepts 1996-2026 Miroslav Sotek. All rights reserved.",
        "code_copyright": "Code 2020-2026 Miroslav Sotek. All rights reserved.",
        "orcid": "0009-0009-3560-0851",
        "contact": "www.anulum.li | protoscience@anulum.li",
        "file": "SCPN Control - Miller Geometry Reference Cases",
        "schema_version": "1.0",
        "cases": [
            {
                "case": "circular_cyclone_limit",
                "parameters": {
                    "R0": 2.78,
                    "a": 1.0,
                    "rho": 0.5,
                    "kappa": 1.0,
                    "delta": 0.0,
                    "dR_dr": 0.0,
                    "q": 1.4,
                    "B0": 2.0,
                },
                "sample_points": [
                    {
                        "theta": 0.0,
                        "R": 3.28,
                        "Z": 0.0,
                        "jacobian": 0.5,
                        "g_rr": 1.0,
                        "g_rt": 0.0,
                        "g_tt": 4.0,
                        "B_toroidal": 1.6951219512195121,
                        "b_dot_grad_theta": 0.21649046654086498,
                    }
                ],
            }
        ],
    }
    path = tmp_path / "miller_reference_cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_geometry_reference(path)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "case"


def test_geometry_reference_gate_rejects_metric_drift(tmp_path: Path) -> None:
    payload = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    payload["cases"][0]["sample_points"][0]["g_tt"] = 99.0
    path = tmp_path / "miller_reference_cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_geometry_reference(path)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "g_tt"


def test_geometry_reference_gate_rejects_duplicate_valid_case(tmp_path: Path) -> None:
    payload = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    payload["cases"].append(payload["cases"][0])
    path = tmp_path / "miller_reference_cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_geometry_reference(path)

    assert report["status"] == "fail"
    assert any(error["field"] == "case" and "duplicate" in error["error"] for error in report["errors"])
