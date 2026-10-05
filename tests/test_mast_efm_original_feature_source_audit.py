# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM original feature-source audit tests

"""Exercise original-source admission on real engineering stores and public consumers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
from mast_efm_zarr_fixtures import sample_dataset, write_zarr

from validation.audit_mast_efm_original_feature_sources import (
    AUDIT_SCHEMA,
    build_original_feature_source_audit,
    classify_feature_sources,
    write_report,
)
from validation.build_mast_efm_neural_equilibrium_dataset import DatasetInput, build_dataset
from validation.build_mast_efm_neural_equilibrium_dataset import write_report as write_dataset_report
from validation.convert_mast_efm_neural_equilibrium_reference import convert_campaign
from validation.neural_equilibrium_campaign_inputs import canonical_campaign_digest


def _source(
    units: str,
    description: str,
    *,
    dims: list[str] | None = None,
    shape: list[int] | None = None,
) -> dict[str, Any]:
    """Retain the original four-time/65-column candidate metadata observations."""
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


def _write_dataset_report(path: Path, *, max_times: int | None = None) -> Path:
    """Produce a complete three-split engineering campaign with real four-time/65-column stores.

    Reference conversion and dataset declarations use their actual public entry
    points. Values and provenance remain explicit local engineering observations;
    this fixture makes no authentic MAST acquisition or predictive claim.
    """
    storage = path.parent / "storage"
    for shot_id in (30419, 30420, 30421):
        ds = sample_dataset().isel(time=[0, 1, 2, 2]).assign_coords(time=[0.1, 0.2, 0.3, 0.4]).drop_dims("psi_norm")
        ds["status"].values[:] = 1
        for name, value, units, description in (
            ("ffprime", 2, "T-rad", "ffprime profile"),
            ("pprime", 1, "Pa/Wb", "pressure derivative engineering profile"),
            ("qpsi_c", 2, "1", "engineering q profile"),
        ):
            descriptor = _source(units, description, dims=["time", "psi_norm"], shape=[4, 65])
            ds[name] = xr.DataArray(
                np.full((4, 65), value, dtype=np.float32), dims=("time", "psi_norm"), attrs=descriptor["attrs"]
            )
        for name, units, description in (
            ("plasma_current_x", "A", "Input experimental fitted total plasma current"),
            ("bphi_rmag", "T", "Toroidal B field total at magnetic axis"),
        ):
            ds[name] = ds[name].astype(np.float32)
            ds[name].attrs = _source(units, description)["attrs"]
        write_zarr(storage / f"mast/level1/shot_{shot_id}/efm.zarr", ds)
    campaign = path.parent / "campaign.json"
    campaign.write_text(
        json.dumps(
            {
                "shots": [
                    {"shot_id": s, "status": "acquired", "local_path": f"mast/level1/shot_{s}/efm.zarr"}
                    for s in (30419, 30420, 30421)
                ]
            }
        )
    )
    candidate = convert_campaign(
        dataset_root=storage,
        campaign_manifest=campaign,
        output_root=storage / "converted",
        max_times_per_shot=max_times,
    )
    assert candidate["status"] == "pass"
    candidate_path = storage / "converted/candidate.json"
    candidate_path.write_text(json.dumps(candidate))
    report = build_dataset(
        DatasetInput(candidate_path, storage, storage / "processed/dataset.npz", (30419,), (30420,), (30421,))
    )
    write_dataset_report(report, path, path.with_suffix(".md"))
    return storage


def _load_source(store: Path) -> xr.Dataset:
    """Load a genuine store before rewriting it through the real Zarr writer."""
    with xr.open_zarr(store, consolidated=True, chunks=None) as source:
        return cast(xr.Dataset, source.load())


def test_classify_feature_sources_admits_current_and_blocks_policy_choices() -> None:
    """Retain original candidate names, metadata shapes and preferred-transform assertions."""
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
    assert status["ffprime_scale"]["status"] == "source_found_requires_rebuild"
    assert status["ffprime_scale"]["required_transform"] == "profile_rms_to_campaign_median_normalised_scalar"


def test_original_feature_source_audit_reads_consolidated_zarr_metadata(tmp_path: Path) -> None:
    """Original descriptors grant readiness only alongside actual matching converted observations."""
    dataset_report = tmp_path / "dataset.json"
    storage_root = _write_dataset_report(dataset_report)

    audit = build_original_feature_source_audit(dataset_report, storage_root)

    assert audit["schema_version"] == AUDIT_SCHEMA
    assert audit["status"] == "source_ready"
    assert audit["can_rebuild_dataset_now"] is True
    assert audit["feature_status"]["Ip_MA"]["selected_source"] == "plasma_current_x"
    assert audit["feature_status"]["Bt_T"]["selected_source"] == "bphi_rmag"
    assert audit["feature_status"]["ffprime_scale"]["status"] == "source_found_requires_rebuild"
    assert audit["shots"][0]["zarr_path"] == "mast/level1/shot_30419/efm.zarr"
    assert all(shot["conversion_check"]["reference_arrays_match"] for shot in audit["shots"])
    assert audit["shots"][0]["source_variables"]["ffprime"]["shape"] == [4, 65]


def test_original_feature_source_audit_rejects_missing_metadata(tmp_path: Path) -> None:
    """Missing original metadata remains an explicit refusal even when converted references exist."""
    dataset_report = tmp_path / "dataset.json"
    storage_root = _write_dataset_report(dataset_report)
    (storage_root / "mast/level1/shot_30419/efm.zarr/.zmetadata").unlink()

    with pytest.raises(FileNotFoundError, match="consolidated Zarr metadata is missing"):
        build_original_feature_source_audit(dataset_report, storage_root)


def test_write_report_lists_original_sources_and_blocker(tmp_path: Path) -> None:
    """Retain original Markdown observations using a complete genuinely blocked source audit."""
    dataset_report = tmp_path / "dataset.json"
    storage = _write_dataset_report(dataset_report)
    store = storage / "mast/level1/shot_30419/efm.zarr"
    ds = _load_source(store)
    ds["ffprime"].attrs["units"] = "unsupported"
    write_zarr(store, ds)
    audit = build_original_feature_source_audit(dataset_report, storage)
    assert audit["status"] == "blocked"
    audit["next_processing_steps"] = ["define ffprime profile reduction before rebuilding the supervised dataset"]
    audit["payload_sha256"] = canonical_campaign_digest(audit)

    write_report(audit, tmp_path / "audit.json", tmp_path / "audit.md")

    markdown = (tmp_path / "audit.md").read_text(encoding="utf-8")
    assert "MAST EFM Original Feature-Source Audit" in markdown
    assert "`plasma_current_x`" in markdown
    assert "define ffprime profile reduction" in markdown
