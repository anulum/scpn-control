# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM feature-provenance audit tests

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from validation.audit_mast_efm_feature_provenance import AUDIT_SCHEMA, build_audit, write_report
from validation.build_mast_efm_neural_equilibrium_dataset import DATASET_SCHEMA


def _write_reference(path: Path, *, include_ip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time_s": np.array([0.1]),
        "psirz_Wb_per_rad": np.zeros((1, 2, 3)),
        "psirz_valid_mask": np.ones((1, 2, 3), dtype=bool),
        "pprime_Pa_per_Wb_rad": np.ones((1, 2)),
        "pprime_valid_mask": np.ones((1, 2), dtype=bool),
        "q_profile": np.ones((1, 2)),
        "q_profile_valid_mask": np.ones((1, 2), dtype=bool),
        "lcfs_r_m": np.ones((1, 4)),
        "lcfs_z_m": np.ones((1, 4)),
        "lcfs_valid_mask": np.ones((1, 4), dtype=bool),
        "magnetic_axis_r_m": np.array([0.7]),
        "magnetic_axis_z_m": np.array([0.0]),
    }
    if include_ip:
        payload["Ip_MA"] = np.array([1.0])
    np.savez_compressed(path, **payload)


def _write_dataset_report(path: Path, references: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": DATASET_SCHEMA,
                "reference_dataset_id": "mast-efm-test",
                "reference_paths": references,
            }
        ),
        encoding="utf-8",
    )


def test_feature_provenance_audit_blocks_unresolved_fallbacks(tmp_path: Path) -> None:
    sas_root = tmp_path / "sas"
    reference = sas_root / "converted/reference.npz"
    _write_reference(reference)
    report = tmp_path / "dataset.json"
    _write_dataset_report(report, ["converted/reference.npz"])

    audit = build_audit(report, sas_root)

    assert audit["schema_version"] == AUDIT_SCHEMA
    assert audit["status"] == "blocked"
    assert set(audit["blocked_features"]) == {"Ip_MA", "Bt_T", "ffprime_scale"}
    assert audit["feature_status"]["Ip_MA"]["present_keys"] == []


def test_feature_provenance_audit_records_resolved_direct_key(tmp_path: Path) -> None:
    sas_root = tmp_path / "sas"
    reference = sas_root / "converted/reference.npz"
    _write_reference(reference, include_ip=True)
    report = tmp_path / "dataset.json"
    _write_dataset_report(report, ["converted/reference.npz"])

    audit = build_audit(report, sas_root)

    assert audit["feature_status"]["Ip_MA"]["status"] == "resolved"
    assert audit["feature_status"]["Ip_MA"]["present_keys"] == ["Ip_MA"]
    assert "Bt_T" in audit["blocked_features"]


def test_write_report_lists_available_keys_and_next_steps(tmp_path: Path) -> None:
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "status": "blocked",
        "reference_dataset_id": "mast-efm-test",
        "reference_count": 1,
        "feature_status": {
            "Ip_MA": {
                "status": "blocked",
                "present_keys": [],
                "resolution": "not present in converted public EFM bundles",
            }
        },
        "all_reference_keys": ["psirz_Wb_per_rad"],
        "next_processing_steps": ["inspect original metadata"],
    }

    write_report(audit, tmp_path / "audit.json", tmp_path / "audit.md")

    markdown = (tmp_path / "audit.md").read_text(encoding="utf-8")
    assert "MAST EFM Feature-Provenance Audit" in markdown
    assert "`Ip_MA`" in markdown
    assert "inspect original metadata" in markdown
