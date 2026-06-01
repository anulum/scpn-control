# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM original feature-source audit tests

from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.audit_mast_efm_original_feature_sources import (
    AUDIT_SCHEMA,
    build_original_feature_source_audit,
    classify_feature_sources,
    write_report,
)
from validation.build_mast_efm_neural_equilibrium_dataset import DATASET_SCHEMA


def _source(
    units: str,
    description: str,
    *,
    dims: list[str] | None = None,
    shape: list[int] | None = None,
) -> dict[str, object]:
    return {
        "attrs": {
            "_ARRAY_DIMENSIONS": dims or ["time"],
            "description": description,
            "mds_name": "\\TOP.ANALYSED.EFM:TEST",
            "quality": "Not Checked",
            "uda_name": "EFM_TEST",
            "units": units,
        },
        "chunks": shape or [4],
        "dtype": "<f4",
        "shape": shape or [4],
    }


def _write_dataset_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": DATASET_SCHEMA,
                "reference_dataset_id": "mast-efm-test",
                "reference_paths": ["converted/neural_equilibrium_reference/mast_efm_shot_30419_reference.npz"],
                "shots": [{"shot_id": 30419, "split": "train", "equilibria_count": 4}],
            }
        ),
        encoding="utf-8",
    )


def _write_zmetadata(zarr_path: Path, variables: dict[str, dict[str, object]]) -> None:
    metadata: dict[str, object] = {".zattrs": {}, ".zgroup": {"zarr_format": 2}}
    for name, variable in variables.items():
        metadata[f"{name}/.zarray"] = {
            "chunks": variable["chunks"],
            "compressor": None,
            "dtype": variable["dtype"],
            "fill_value": None,
            "filters": None,
            "order": "C",
            "shape": variable["shape"],
            "zarr_format": 2,
        }
        metadata[f"{name}/.zattrs"] = variable["attrs"]
    zarr_path.mkdir(parents=True, exist_ok=True)
    (zarr_path / ".zmetadata").write_text(
        json.dumps({"metadata": metadata, "zarr_consolidated_format": 1}),
        encoding="utf-8",
    )


def test_classify_feature_sources_admits_current_and_blocks_policy_choices() -> None:
    variables = {
        "plasma_current_x": _source("A", "Input experimental fitted total plasma current"),
        "bphi_rmag": _source("T", "Toroidal B field total at magnetic axis"),
        "ffprime": _source("T-rad", "ffprime profile", dims=["time", "psi_norm"], shape=[4, 65]),
    }

    status = classify_feature_sources(variables)

    assert status["Ip_MA"]["status"] == "source_found_requires_rebuild"
    assert status["Ip_MA"]["selected_source"] == "plasma_current_x"
    assert status["Ip_MA"]["required_transform"] == "A_to_MA"
    assert status["Bt_T"]["status"] == "source_found_requires_rebuild"
    assert status["Bt_T"]["selected_source"] == "bphi_rmag"
    assert status["ffprime_scale"]["status"] == "source_found_requires_policy"
    assert status["ffprime_scale"]["required_transform"] == "profile_to_training_scalar"


def test_original_feature_source_audit_reads_consolidated_zarr_metadata(tmp_path: Path) -> None:
    sas_root = tmp_path / "sas"
    dataset_report = tmp_path / "dataset.json"
    _write_dataset_report(dataset_report)
    zarr_path = sas_root / "mast/level1/shot_30419/efm.zarr"
    _write_zmetadata(
        zarr_path,
        {
            "plasma_current_x": _source("A", "Input experimental fitted total plasma current"),
            "bphi_rmag": _source("T", "Toroidal B field total at magnetic axis"),
            "ffprime": _source("T-rad", "ffprime profile", dims=["time", "psi_norm"], shape=[4, 65]),
        },
    )

    audit = build_original_feature_source_audit(dataset_report, sas_root)

    assert audit["schema_version"] == AUDIT_SCHEMA
    assert audit["status"] == "blocked"
    assert audit["can_rebuild_dataset_now"] is False
    assert audit["feature_status"]["Ip_MA"]["selected_source"] == "plasma_current_x"
    assert audit["feature_status"]["Bt_T"]["selected_source"] == "bphi_rmag"
    assert audit["feature_status"]["ffprime_scale"]["status"] == "source_found_requires_policy"
    assert audit["shots"][0]["zarr_path"] == "mast/level1/shot_30419/efm.zarr"


def test_original_feature_source_audit_rejects_missing_metadata(tmp_path: Path) -> None:
    sas_root = tmp_path / "sas"
    dataset_report = tmp_path / "dataset.json"
    _write_dataset_report(dataset_report)
    (sas_root / "mast/level1/shot_30419/efm.zarr").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="consolidated Zarr metadata is missing"):
        build_original_feature_source_audit(dataset_report, sas_root)


def test_write_report_lists_original_sources_and_blocker(tmp_path: Path) -> None:
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "status": "blocked",
        "reference_dataset_id": "mast-efm-test",
        "shot_count": 1,
        "can_rebuild_dataset_now": False,
        "feature_status": {
            "Ip_MA": {
                "status": "source_found_requires_rebuild",
                "selected_source": "plasma_current_x",
                "required_transform": "A_to_MA",
                "resolution": "measured total plasma current is available in original public EFM metadata",
            }
        },
        "next_processing_steps": ["define ffprime profile reduction before rebuilding the supervised dataset"],
    }

    write_report(audit, tmp_path / "audit.json", tmp_path / "audit.md")

    markdown = (tmp_path / "audit.md").read_text(encoding="utf-8")
    assert "MAST EFM Original Feature-Source Audit" in markdown
    assert "`plasma_current_x`" in markdown
    assert "define ffprime profile reduction" in markdown
