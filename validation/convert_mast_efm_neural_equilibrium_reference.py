#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural equilibrium reference converter
"""Convert public MAST EFM equilibrium data into claim-gated reference bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.mast_efm_reference_arrays import DatasetLike as DatasetLike
from validation.mast_efm_reference_arrays import extract_reference_arrays as extract_reference_arrays
from validation.mast_efm_reference_contracts import (
    BLOCKED_REASON as BLOCKED_REASON,
)
from validation.mast_efm_reference_contracts import (
    CANDIDATE_SCHEMA as CANDIDATE_SCHEMA,
)
from validation.mast_efm_reference_contracts import (
    REFERENCE_ARRAY_KEYS as REFERENCE_ARRAY_KEYS,
)
from validation.mast_efm_reference_contracts import (
    REQUIRED_EFM_VARIABLES as REQUIRED_EFM_VARIABLES,
)
from validation.mast_efm_reference_contracts import (
    TIME_ALIGNED_ARRAY_KEYS as TIME_ALIGNED_ARRAY_KEYS,
)
from validation.mast_efm_reference_contracts import (
    json_sha256,
    positive_int,
    read_campaign,
    storage_path,
)


@dataclass(frozen=True)
class ConvertedShot:
    """Immutable summary of a converted MAST EFM shot."""

    shot_id: int
    source_path: str
    output_path: str
    sha256: str
    selected_time_count: int
    grid_shape: tuple[int, int]
    lcfs_points: int
    status: str


def convert_campaign(
    *,
    dataset_root: Path,
    campaign_manifest: Path,
    output_root: Path,
    max_times_per_shot: int | None = None,
    reference_url_template: str = "https://mastapp.site/json/shots/{shot_id}",
) -> dict[str, Any]:
    """Convert a MAST EFM campaign manifest into reference-array bundles."""
    shots = read_campaign(campaign_manifest)
    if max_times_per_shot is not None:
        positive_int(max_times_per_shot, field="max_times_per_shot")
    if output_root.resolve() == campaign_manifest.resolve():
        raise ValueError("output_root must not alias the campaign manifest")
    selected_paths = {
        shot["shot_id"]: storage_path(dataset_root, shot.get("local_path"))
        for shot in shots
        if shot.get("status") == "acquired"
    }
    if any(output_root.resolve().is_relative_to(path) for path in selected_paths.values()):
        raise ValueError("output_root must not overwrite an original Zarr source")
    converted: list[ConvertedShot] = []
    errors: list[dict[str, object]] = []
    for shot in shots:
        if shot.get("status") != "acquired":
            errors.append({"shot_id": shot.get("shot_id"), "field": "status", "error": "shot was not acquired"})
            continue
        shot_id = shot["shot_id"]
        zarr_path = selected_paths[shot_id]
        try:
            converted.append(
                convert_shot_zarr(
                    shot_id=shot_id,
                    zarr_path=zarr_path,
                    output_path=output_root / f"mast_efm_shot_{shot_id}_reference.npz",
                    max_times=max_times_per_shot,
                )
            )
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            errors.append({"shot_id": shot_id, "field": "conversion", "error": str(exc)})
    status = "pass" if converted and not errors else "fail"
    report: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA,
        "status": status,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset_root": dataset_root.as_posix(),
        "campaign_manifest": campaign_manifest.as_posix(),
        "output_root": output_root.as_posix(),
        "source": "documented_public_reference",
        "reference_url_template": reference_url_template,
        "reference_dataset_id": _reference_dataset_id(converted),
        "reference_equilibria_count": sum(item.selected_time_count for item in converted),
        "target_schema_status": "reference_only_no_prediction_metrics",
        "admission_ready": False,
        "blocked_reason": BLOCKED_REASON,
        "required_follow_up": [
            "derive or supply pressure profile arrays rather than treating pprime as pressure",
            "run the exact neural-equilibrium model on the same shot/time/grid cases",
            "persist prediction arrays outside git with SHA-256 digests",
            "compute psi, pressure, q-profile, LCFS boundary, and magnetic-axis metrics against declared tolerances",
            "emit scpn-control.neural-equilibrium-reference.v1 artefacts only after the strict evidence package exists",
        ],
        "array_keys": list(REFERENCE_ARRAY_KEYS),
        "shots": [item.__dict__ for item in converted],
        "errors": errors,
    }
    report["payload_sha256"] = json_sha256({**report, "payload_sha256": None})
    return report


def convert_shot_zarr(
    *,
    shot_id: int,
    zarr_path: Path,
    output_path: Path,
    max_times: int | None = None,
) -> ConvertedShot:
    """Open and convert one MAST EFM Zarr group to a compressed reference bundle."""
    positive_int(shot_id, field="shot_id")
    if output_path.suffix.lower() != ".npz":
        raise ValueError("output_path must have an explicit .npz suffix")
    if output_path.resolve().is_relative_to(zarr_path.resolve()):
        raise ValueError("output_path must not overwrite an original Zarr source")
    arrays = read_reference_zarr(shot_id=shot_id, zarr_path=zarr_path, max_times=max_times)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = arrays
    np.savez_compressed(output_path, **payload)
    return _converted_summary(shot_id=shot_id, source_path=zarr_path, output_path=output_path, arrays=arrays)


def read_reference_zarr(*, shot_id: int, zarr_path: Path, max_times: int | None = None) -> dict[str, NDArray[Any]]:
    """Open consolidated Zarr v2 without Dask and extract actual observed arrays.

    The captured store supplies the actual decoded bytes. Successful reading
    is not physical acquisition authentication; original-source audits must bind
    the selected source snapshots and converted output independently.
    """
    from validation.mast_efm_zarr_store import capture_zarr_store, read_reference_store

    if not zarr_path.is_dir():
        raise FileNotFoundError(f"MAST EFM Zarr path does not exist: {zarr_path}")
    return read_reference_store(capture_zarr_store(zarr_path), shot_id=shot_id, max_times=max_times)


def _converted_summary(
    *, shot_id: int, source_path: Path, output_path: Path, arrays: dict[str, NDArray[Any]]
) -> ConvertedShot:
    """Describe the actual persisted selected reference arrays."""
    psirz = arrays["psirz_Wb_per_rad"]
    lcfs = arrays["lcfs_r_m"]
    return ConvertedShot(
        shot_id=shot_id,
        source_path=source_path.as_posix(),
        output_path=output_path.as_posix(),
        sha256=_sha256(output_path),
        selected_time_count=int(arrays["time_s"].shape[0]),
        grid_shape=(int(psirz.shape[-2]), int(psirz.shape[-1])),
        lcfs_points=int(lcfs.shape[-1]),
        status="reference_candidate",
    )


def _reference_dataset_id(converted: list[ConvertedShot]) -> str:
    """Bind the retained candidate identity to shot order and output digests."""
    if not converted:
        return "mast-efm-empty"
    shots = "-".join(str(item.shot_id) for item in converted)
    digest = hashlib.sha256("|".join(item.sha256 for item in converted).encode("ascii")).hexdigest()[:16]
    return f"mast-efm-{shots}-{digest}"


def _sha256(path: Path) -> str:
    """Hash the persisted compressed reference bundle."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for MAST EFM reference-candidate conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/SCPN-CONTROL"))
    parser.add_argument(
        "--campaign-manifest",
        type=Path,
        default=Path("/data/SCPN-CONTROL/manifests/mast_level1_efm_campaign_30419_30424.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/SCPN-CONTROL/converted/neural_equilibrium_reference"),
    )
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--max-times-per-shot", type=int, default=None)
    parser.add_argument("--json-out", action="store_true")
    args = parser.parse_args(argv)

    report_out = args.report_out or args.output_root / "mast_efm_neural_equilibrium_reference_candidate.json"
    try:
        shots = read_campaign(args.campaign_manifest)
        original = [
            storage_path(args.dataset_root, shot.get("local_path"))
            for shot in shots
            if shot.get("status") == "acquired"
        ]
        converted_outputs = [args.output_root / f"mast_efm_shot_{shot['shot_id']}_reference.npz" for shot in shots]
        selected_report = report_out.resolve()
        if (
            selected_report == args.campaign_manifest.resolve()
            or any(selected_report.is_relative_to(path) for path in original)
            or any(selected_report == path.resolve() for path in converted_outputs)
        ):
            raise ValueError("report output must not overwrite campaign, original source or reference arrays")
        report = convert_campaign(
            dataset_root=args.dataset_root,
            campaign_manifest=args.campaign_manifest,
            output_root=args.output_root,
            max_times_per_shot=args.max_times_per_shot,
        )
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"MAST EFM conversion refused: {exc}", file=sys.stderr)
        return 1
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "MAST EFM neural-equilibrium reference conversion: "
            f"{report['status']} shots={len(report['shots'])} equilibria={report['reference_equilibria_count']} "
            f"admission_ready={report['admission_ready']}"
        )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
