# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — NTM reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_ntm_reference import canonical_artifact_sha256, validate_ntm_reference


def _valid_ntm_reference_artifact() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "scpn-control.ntm-reference.v1",
        "source": "measured_ntm_campaign",
        "machine": "DIII-D",
        "shot_id": "ntm-eccd-167221",
        "reference_dataset_id": "diiid-ntm-eccd-2026-05-23",
        "executed_at": "2026-05-23T14:45:00Z",
        "q_profile_uri": "ntm/diiid/q_profile.npz",
        "rational_surface_uri": "ntm/diiid/rational_surface.json",
        "island_width_trace_uri": "ntm/diiid/island_width_trace.npz",
        "eccd_alignment_uri": "ntm/diiid/eccd_alignment.json",
        "q_profile_sha256": "1" * 64,
        "rational_surface_sha256": "2" * 64,
        "island_width_trace_sha256": "3" * 64,
        "eccd_alignment_sha256": "4" * 64,
        "units": {
            "island_width": "m",
            "time": "s",
            "current": "A",
            "q": "1",
            "rho": "1",
            "power": "W",
            "deposition_width": "m",
            "island_growth_rate": "m/s",
        },
        "rho_grid": [0.2, 0.4, 0.6, 0.8, 0.95],
        "q_profile": [1.1, 1.45, 2.0, 2.7, 3.4],
        "rational_surface": {
            "rho": 0.6,
            "r_s_m": 1.2,
            "m": 2,
            "n": 1,
            "q": 2.0,
            "shear": 1.25,
            "a_m": 2.0,
            "R0_m": 6.2,
        },
        "seed_island_width_range_m": [0.004, 0.025],
        "eccd_alignment": {
            "power_W": 3.0e6,
            "current_A": 1.8e5,
            "deposition_width_m": 0.045,
            "alignment_error_m": 0.012,
        },
        "metrics": {
            "rational_surface_rho_error": 0.006,
            "island_growth_relative_error": 0.08,
            "saturated_width_relative_error": 0.10,
            "suppression_time_relative_error": 0.11,
            "eccd_alignment_error_m": 0.012,
        },
        "tolerances": {
            "rational_surface_rho_error": 0.01,
            "island_growth_relative_error": 0.12,
            "saturated_width_relative_error": 0.15,
            "suppression_time_relative_error": 0.15,
            "eccd_alignment_error_m": 0.02,
        },
    }
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    return payload


def test_strict_ntm_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_ntm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no NTM reference artifacts found"


def test_ntm_gate_accepts_measured_campaign(tmp_path: Path) -> None:
    artifact = tmp_path / "diiid_ntm_reference.json"
    artifact.write_text(json.dumps(_valid_ntm_reference_artifact()), encoding="utf-8")

    report = validate_ntm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "measured_ntm_campaign"


def test_ntm_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    payload = _valid_ntm_reference_artifact()
    payload["source"] = "documented_public_reference"
    payload.pop("machine")
    payload.pop("shot_id")
    payload["reference_doi"] = "10.1063/1.2184031"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "published_ntm_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_ntm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "documented_public_reference"


def test_ntm_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_ntm_reference_artifact()
    payload["source"] = "synthetic"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "synthetic_ntm_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_ntm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_ntm_gate_rejects_missing_measured_campaign_identity(tmp_path: Path) -> None:
    payload = _valid_ntm_reference_artifact()
    payload.pop("shot_id")
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "missing_campaign_identity.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_ntm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "campaign"


def test_ntm_gate_rejects_nonmonotone_rho_grid(tmp_path: Path) -> None:
    payload = _valid_ntm_reference_artifact()
    payload["rho_grid"] = [0.2, 0.6, 0.5]
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "bad_rho_grid.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_ntm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "rho_grid"


def test_ntm_gate_rejects_q_profile_not_length_matched_to_rho_grid(tmp_path: Path) -> None:
    payload = _valid_ntm_reference_artifact()
    payload["q_profile"] = [1.1, 1.45, 2.0]
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "bad_q_profile.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_ntm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "q_profile"


def test_ntm_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    payload = _valid_ntm_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["saturated_width_relative_error"] = 0.3
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "bad_saturated_width_error.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_ntm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "saturated_width_relative_error"


def test_ntm_gate_rejects_tampered_payload_digest(tmp_path: Path) -> None:
    payload = _valid_ntm_reference_artifact()
    eccd = cast(dict[str, object], payload["eccd_alignment"])
    eccd["alignment_error_m"] = 0.004
    artifact = tmp_path / "tampered_ntm_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_ntm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "payload_sha256"


def test_ntm_gate_rejects_traversing_artifact_uri(tmp_path: Path) -> None:
    payload = _valid_ntm_reference_artifact()
    payload["island_width_trace_uri"] = "../island_width_trace.npz"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "traversing_ntm_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_ntm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "island_width_trace_uri"
