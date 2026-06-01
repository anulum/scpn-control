# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural-equilibrium campaign-plan tests

from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.build_mast_efm_neural_equilibrium_dataset import DATASET_SCHEMA
from validation.plan_neural_equilibrium_training_campaign import CampaignInputs, REPORT_SCHEMA, build_plan, write_report
from validation.validate_public_data_acquisition import SCHEMA_VERSION as PUBLIC_DATA_SCHEMA


def _write_mast_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": DATASET_SCHEMA,
                "status": "blocked",
                "reference_dataset_id": "mast-efm-test",
                "equilibria_count": 12,
                "grid_shape": [65, 129],
                "split_counts": {"train": 8, "validation": 2, "test": 2},
                "fallback_features": [],
                "ragged_target_policy": {
                    "keys": ["lcfs_r_m", "lcfs_z_m", "lcfs_valid_mask"],
                    "padding": "NaN for coordinates and False for validity mask",
                    "point_count_key": "lcfs_point_count",
                    "max_lcfs_points": 157,
                },
                "candidate_report": "converted/neural_equilibrium_reference/candidate.json",
                "dataset_path": "processed/neural_equilibrium/mast_efm_supervised_dataset.npz",
                "dataset_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )


def _write_public_data_manifest(root: Path) -> None:
    manifest_dir = root / "zenodo_1"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "files_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": PUBLIC_DATA_SCHEMA,
                "source": "zenodo",
                "doi": "10.5281/zenodo.1",
                "title": "fixture",
                "license": "cc-by-4.0",
                "record_sha256": "b" * 64,
                "large_numeric_files_downloaded": False,
                "large_numeric_files_policy": "deferred: pull multi-GB arrays on the storage target",
                "files": [
                    {
                        "key": "large.nc",
                        "size_bytes": 1024,
                        "checksum": "md5:" + "c" * 32,
                        "download_url": "https://zenodo.org/api/records/1/files/large.nc/content",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_build_plan_prepares_mast_and_deferred_public_data_lanes(tmp_path: Path) -> None:
    mast_report = tmp_path / "mast.json"
    public_root = tmp_path / "public"
    _write_mast_report(mast_report)
    _write_public_data_manifest(public_root)

    plan = build_plan(CampaignInputs(mast_dataset_report=mast_report, sas_root=tmp_path, public_data_root=public_root))

    assert plan["schema_version"] == REPORT_SCHEMA
    assert plan["status"] == "prepared"
    assert plan["mast_efm_dataset"]["status"] == "prepared"
    assert plan["mast_efm_dataset"]["payload"]["exists_on_this_host"] is False
    assert plan["mast_efm_dataset"]["payload"]["verified_available"] is False
    assert "ML350 is storage-only" in plan["execution_host_policy"]
    assert plan["prepared_dataset_lanes"][0]["status"] == "prepared_on_sas"
    assert "dry-run trainer" in plan["prepared_dataset_lanes"][0]["next_action"]
    assert "workstation or external cloud" in plan["prepared_dataset_lanes"][0]["next_action"]
    assert all("fallback" not in item for item in plan["mast_efm_dataset"]["blocked_before_admission"])
    assert plan["prepared_dataset_lanes"][1]["status"] == "manifested_large_payloads_deferred"
    assert plan["prepared_dataset_lanes"][1]["public_data_summary"]["deferred_bytes"] == 1024
    assert {budget["scenario"] for budget in plan["gpu_budget_estimates"]} >= {
        "mast_efm_single_seed_full_output",
        "qlknn_qualikiz_payload_processing",
        "publication_grade_equilibrium_campaign",
    }
    assert len(plan["payload_sha256"]) == 64


def test_build_plan_can_require_sas_payload(tmp_path: Path) -> None:
    mast_report = tmp_path / "mast.json"
    public_root = tmp_path / "public"
    _write_mast_report(mast_report)
    _write_public_data_manifest(public_root)

    with pytest.raises(FileNotFoundError, match="SAS dataset payload is missing"):
        build_plan(
            CampaignInputs(
                mast_dataset_report=mast_report,
                sas_root=tmp_path,
                public_data_root=public_root,
                require_sas_payload=True,
            )
        )

    plan = build_plan(
        CampaignInputs(
            mast_dataset_report=mast_report,
            sas_root=tmp_path,
            public_data_root=public_root,
            require_sas_payload=True,
            verified_sas_payload=True,
        )
    )
    assert plan["mast_efm_dataset"]["payload"]["verified_available"] is True


def test_write_report_records_gpu_budget_table(tmp_path: Path) -> None:
    mast_report = tmp_path / "mast.json"
    public_root = tmp_path / "public"
    _write_mast_report(mast_report)
    _write_public_data_manifest(public_root)
    plan = build_plan(CampaignInputs(mast_dataset_report=mast_report, sas_root=tmp_path, public_data_root=public_root))

    write_report(plan, tmp_path / "plan.json", tmp_path / "plan.md")

    markdown = (tmp_path / "plan.md").read_text(encoding="utf-8")
    assert "Neural-Equilibrium Training Campaign Plan" in markdown
    assert "mast_efm_single_seed_full_output" in markdown
    assert "qlknn_qualikiz_payload_processing" in markdown
    assert "predictive EFIT/P-EFIT" in markdown
    assert "ML350 is storage-only" in markdown
