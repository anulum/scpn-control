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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

CANDIDATE_SCHEMA = "scpn-control.mast-efm-neural-equilibrium-reference-candidate.v1"
REQUIRED_EFM_VARIABLES = (
    "psirz",
    "psi_axis",
    "psi_boundary",
    "plasma_current_x",
    "bphi_rmag",
    "ffprime",
    "pprime",
    "qpsi_c",
    "lcfs_r",
    "lcfs_z",
    "magnetic_axis_r",
    "magnetic_axis_z",
    "status",
    "cnvrgd_times",
)
REFERENCE_ARRAY_KEYS = (
    "time_s",
    "r_grid_m",
    "z_grid_m",
    "psirz_Wb_per_rad",
    "psirz_valid_mask",
    "psi_axis_Wb_per_rad",
    "psi_boundary_Wb_per_rad",
    "Ip_MA",
    "Bt_T",
    "ffprime_rms_T_rad",
    "pprime_Pa_per_Wb_rad",
    "pprime_valid_mask",
    "q_profile",
    "q_profile_valid_mask",
    "lcfs_r_m",
    "lcfs_z_m",
    "lcfs_valid_mask",
    "magnetic_axis_r_m",
    "magnetic_axis_z_m",
    "shot_id",
)
TIME_ALIGNED_ARRAY_KEYS = tuple(key for key in REFERENCE_ARRAY_KEYS if key not in {"r_grid_m", "z_grid_m"})
BLOCKED_REASON = (
    "Reference arrays were converted from public MAST EFM data, but predictive EFIT/P-EFIT claims remain blocked "
    "until exact-model predictions, pressure reconstruction, metrics, tolerances, and strict admission artefacts exist."
)


class DatasetLike(Protocol):
    """Minimal xarray-like interface used by the converter core."""

    variables: Any
    sizes: Any

    def __getitem__(self, key: str) -> Any: ...


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
    campaign = _read_json(campaign_manifest)
    shots = campaign.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("campaign manifest must contain a non-empty shots list")
    output_root.mkdir(parents=True, exist_ok=True)
    converted: list[ConvertedShot] = []
    errors: list[dict[str, object]] = []
    for shot in shots:
        if not isinstance(shot, dict):
            errors.append({"field": "shots", "error": "shot entry must be an object"})
            continue
        if shot.get("status") != "acquired":
            errors.append({"shot_id": shot.get("shot_id"), "field": "status", "error": "shot was not acquired"})
            continue
        shot_id = _positive_int(shot.get("shot_id"), field="shot_id")
        local_path = shot.get("local_path")
        if not isinstance(local_path, str) or not local_path.strip():
            errors.append({"shot_id": shot_id, "field": "local_path", "error": "shot local_path must be a string"})
            continue
        zarr_path = dataset_root / local_path
        try:
            converted.append(
                convert_shot_zarr(
                    shot_id=shot_id,
                    zarr_path=zarr_path,
                    output_path=output_root / f"mast_efm_shot_{shot_id}_reference.npz",
                    max_times=max_times_per_shot,
                )
            )
        except Exception as exc:
            errors.append({"shot_id": shot_id, "field": "conversion", "error": str(exc)})
    status = "pass" if converted and not errors else "fail"
    report: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA,
        "status": status,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
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
    report["payload_sha256"] = _json_sha256({**report, "payload_sha256": None})
    return report


def convert_shot_zarr(
    *,
    shot_id: int,
    zarr_path: Path,
    output_path: Path,
    max_times: int | None = None,
) -> ConvertedShot:
    """Open and convert one MAST EFM Zarr group to a compressed reference bundle."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("xarray is required to convert MAST EFM Zarr data") from exc
    if not zarr_path.exists():
        raise FileNotFoundError(f"MAST EFM Zarr path does not exist: {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=True)
    try:
        arrays = extract_reference_arrays(ds, shot_id=shot_id, max_times=max_times)
    finally:
        ds.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # numpy's savez_compressed stub types **kwds against the positional allow_pickle: bool;
    # the array keyword payload is the documented, correct call. See SYS-AUDIT-02-MYPY1.
    np.savez_compressed(output_path, **arrays)  # type: ignore[arg-type]
    return _converted_summary(shot_id=shot_id, source_path=zarr_path, output_path=output_path, arrays=arrays)


def extract_reference_arrays(
    ds: DatasetLike, *, shot_id: int, max_times: int | None = None
) -> dict[str, npt.NDArray[np.float64]]:
    """Extract full-order MAST EFM equilibrium arrays without prediction or metric fabrication."""
    missing = [name for name in REQUIRED_EFM_VARIABLES if name not in ds.variables]
    if missing:
        raise ValueError(f"MAST EFM dataset missing required variables: {', '.join(missing)}")
    time = _as_1d_array(ds, "time", fallback_size=_time_size(ds))
    indices = _converged_time_indices(ds, time.size)
    if indices.size == 0:
        raise ValueError("MAST EFM dataset has no converged time slices")
    psi_axis_all = _take_time(ds["psi_axis"], indices, name="psi_axis").astype(np.float64)
    psi_boundary_all = _take_time(ds["psi_boundary"], indices, name="psi_boundary").astype(np.float64)
    magnetic_axis_r_all = _take_time(ds["magnetic_axis_r"], indices, name="magnetic_axis_r").astype(np.float64)
    magnetic_axis_z_all = _take_time(ds["magnetic_axis_z"], indices, name="magnetic_axis_z").astype(np.float64)
    scalar_mask = (
        np.isfinite(psi_axis_all)
        & np.isfinite(psi_boundary_all)
        & np.isfinite(magnetic_axis_r_all)
        & np.isfinite(magnetic_axis_z_all)
        & (psi_axis_all != psi_boundary_all)
    )
    indices = indices[np.flatnonzero(scalar_mask)]
    if max_times is not None:
        if max_times <= 0:
            raise ValueError("max_times must be positive when provided")
        indices = indices[:max_times]
    if indices.size == 0:
        raise ValueError("MAST EFM dataset has no finite converged equilibrium slices")
    psirz = _take_time(ds["psirz"], indices, name="psirz")
    r_grid, z_grid = _psirz_coordinate_grids(ds, ds["psirz"], (int(psirz.shape[-2]), int(psirz.shape[-1])))
    plasma_current = _take_time(ds["plasma_current_x"], indices, name="plasma_current_x").astype(np.float64)
    toroidal_field = _take_time(ds["bphi_rmag"], indices, name="bphi_rmag").astype(np.float64)
    ffprime_profile = _take_time(ds["ffprime"], indices, name="ffprime").astype(np.float64)
    pprime = _take_time(ds["pprime"], indices, name="pprime")
    q_profile = _take_time(ds["qpsi_c"], indices, name="qpsi_c")
    lcfs_r = _take_time(ds["lcfs_r"], indices, name="lcfs_r")
    lcfs_z = _take_time(ds["lcfs_z"], indices, name="lcfs_z")
    arrays = {
        "time_s": time[indices].astype(np.float64),
        "r_grid_m": r_grid.astype(np.float64),
        "z_grid_m": z_grid.astype(np.float64),
        "psirz_Wb_per_rad": psirz.astype(np.float64),
        "psirz_valid_mask": np.isfinite(psirz),
        "psi_axis_Wb_per_rad": _take_time(ds["psi_axis"], indices, name="psi_axis").astype(np.float64),
        "psi_boundary_Wb_per_rad": _take_time(ds["psi_boundary"], indices, name="psi_boundary").astype(np.float64),
        "Ip_MA": (plasma_current / 1.0e6).astype(np.float64),
        "Bt_T": toroidal_field.astype(np.float64),
        "ffprime_rms_T_rad": _profile_rms(ffprime_profile),
        "pprime_Pa_per_Wb_rad": pprime.astype(np.float64),
        "pprime_valid_mask": np.isfinite(pprime),
        "q_profile": q_profile.astype(np.float64),
        "q_profile_valid_mask": np.isfinite(q_profile),
        "lcfs_r_m": lcfs_r.astype(np.float64),
        "lcfs_z_m": lcfs_z.astype(np.float64),
        "lcfs_valid_mask": np.isfinite(lcfs_r) & np.isfinite(lcfs_z),
        "magnetic_axis_r_m": _take_time(ds["magnetic_axis_r"], indices, name="magnetic_axis_r").astype(np.float64),
        "magnetic_axis_z_m": _take_time(ds["magnetic_axis_z"], indices, name="magnetic_axis_z").astype(np.float64),
        "shot_id": np.full(indices.shape, shot_id, dtype=np.int64),
    }
    _validate_reference_arrays(arrays)
    return arrays


def _converted_summary(
    *, shot_id: int, source_path: Path, output_path: Path, arrays: dict[str, npt.NDArray[np.float64]]
) -> ConvertedShot:
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


def _validate_reference_arrays(arrays: dict[str, npt.NDArray[np.float64]]) -> None:
    first = arrays["time_s"].shape[0]
    for key in REFERENCE_ARRAY_KEYS:
        if key not in arrays:
            raise ValueError(f"missing converted array {key}")
    for key in TIME_ALIGNED_ARRAY_KEYS:
        if arrays[key].shape[0] != first:
            raise ValueError(f"converted array {key} does not share the selected time dimension")
    r_grid = arrays["r_grid_m"]
    z_grid = arrays["z_grid_m"]
    if r_grid.ndim != 1 or z_grid.ndim != 1:
        raise ValueError("converted coordinate grids must be one-dimensional")
    if not np.all(np.isfinite(r_grid)) or not np.all(np.isfinite(z_grid)):
        raise ValueError("converted coordinate grids must be finite")
    if not (np.all(np.diff(r_grid) > 0.0) or np.all(np.diff(r_grid) < 0.0)):
        raise ValueError("converted r_grid_m must be strictly monotonic")
    if not (np.all(np.diff(z_grid) > 0.0) or np.all(np.diff(z_grid) < 0.0)):
        raise ValueError("converted z_grid_m must be strictly monotonic")
    finite_required = (
        "time_s",
        "psi_axis_Wb_per_rad",
        "psi_boundary_Wb_per_rad",
        "Ip_MA",
        "Bt_T",
        "ffprime_rms_T_rad",
        "magnetic_axis_r_m",
        "magnetic_axis_z_m",
        "shot_id",
    )
    for key in finite_required:
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"converted array {key} contains non-finite values")
    masked_pairs = (
        ("psirz_Wb_per_rad", "psirz_valid_mask"),
        ("pprime_Pa_per_Wb_rad", "pprime_valid_mask"),
        ("q_profile", "q_profile_valid_mask"),
        ("lcfs_r_m", "lcfs_valid_mask"),
        ("lcfs_z_m", "lcfs_valid_mask"),
    )
    for value_key, mask_key in masked_pairs:
        if arrays[value_key].shape != arrays[mask_key].shape:
            raise ValueError(f"converted mask {mask_key} does not match {value_key}")
        if not np.any(arrays[mask_key]):
            raise ValueError(f"converted array {value_key} has no finite valid points")
    if arrays["psirz_Wb_per_rad"].ndim != 3:
        raise ValueError("psirz must be a time, z, r array")
    if arrays["psirz_Wb_per_rad"].shape[-1] != r_grid.shape[0]:
        raise ValueError("r_grid_m length must match psirz radial dimension")
    if arrays["psirz_Wb_per_rad"].shape[-2] != z_grid.shape[0]:
        raise ValueError("z_grid_m length must match psirz vertical dimension")
    if arrays["lcfs_r_m"].shape != arrays["lcfs_z_m"].shape:
        raise ValueError("LCFS R/Z arrays must have matching shapes")
    if np.any(arrays["psi_axis_Wb_per_rad"] == arrays["psi_boundary_Wb_per_rad"]):
        raise ValueError("psi_axis and psi_boundary must not be equal for selected equilibria")
    if np.any(arrays["ffprime_rms_T_rad"] <= 0.0):
        raise ValueError("ffprime_rms_T_rad must be positive for selected equilibria")


def _converged_time_indices(ds: DatasetLike, n_time: int) -> npt.NDArray[np.intp]:
    status = np.asarray(ds["status"].values).reshape(-1)
    converged = np.asarray(ds["cnvrgd_times"].values).reshape(-1)
    if status.size != n_time:
        status = np.resize(status, n_time)
    if converged.size != n_time:
        converged = np.resize(converged, n_time)
    status_float = status.astype(float)
    converged_float = converged.astype(float)
    status_ok = np.isfinite(status_float) & ((status_float == 0.0) | (status_float == 1.0))
    mask = status_ok & np.isfinite(converged_float) & (converged_float > 0.0)
    if not np.any(mask):
        mask = status_ok
    return np.flatnonzero(mask)


def _take_time(data_array: Any, indices: npt.NDArray[np.intp], *, name: str) -> npt.NDArray[np.float64]:
    values = np.asarray(data_array.values)
    dims = tuple(getattr(data_array, "dims", ()))
    if "time" in dims:
        axis = dims.index("time")
    elif values.shape and values.shape[0] >= indices.max(initial=0) + 1:
        axis = 0
    else:
        raise ValueError(f"{name} does not expose a usable time dimension")
    return np.take(values, indices, axis=axis)


def _profile_rms(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim < 2:
        raise ValueError("profile RMS requires a time-aligned profile array")
    flattened = arr.reshape(arr.shape[0], -1)
    result = np.empty(flattened.shape[0], dtype=np.float64)
    for row in range(flattened.shape[0]):
        valid = flattened[row][np.isfinite(flattened[row])]
        if valid.size == 0:
            raise ValueError("profile RMS cannot be computed from an all-non-finite row")
        result[row] = float(np.sqrt(np.mean(np.square(valid))))
    return result


def _psirz_coordinate_grids(
    ds: DatasetLike, psirz_array: Any, grid_shape: tuple[int, int]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    dims = tuple(getattr(psirz_array, "dims", ()))
    if len(dims) < 3:
        raise ValueError("psirz must expose time, z, and r dimensions")
    z_dim = dims[-2]
    r_dim = dims[-1]
    return (
        _coordinate_grid_from_dimension(ds, r_dim, expected_size=grid_shape[-1]),
        _coordinate_grid_from_dimension(ds, z_dim, expected_size=grid_shape[-2]),
    )


def _coordinate_grid_from_dimension(ds: DatasetLike, dimension: str, *, expected_size: int) -> npt.NDArray[np.float64]:
    if dimension not in ds.variables:
        raise ValueError(f"MAST EFM dataset missing exact coordinate grid for {dimension}")
    grid = np.asarray(ds[dimension].values, dtype=np.float64).reshape(-1)
    if grid.size != expected_size:
        raise ValueError(f"coordinate grid {dimension} length does not match psirz dimension")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"coordinate grid {dimension} contains non-finite values")
    if not (np.all(np.diff(grid) > 0.0) or np.all(np.diff(grid) < 0.0)):
        raise ValueError(f"coordinate grid {dimension} must be strictly monotonic")
    return grid


def _as_1d_array(ds: DatasetLike, name: str, *, fallback_size: int) -> npt.NDArray[np.float64]:
    if name in ds.variables:
        arr = np.asarray(ds[name].values, dtype=np.float64).reshape(-1)
        if arr.size:
            return arr
    return np.arange(fallback_size, dtype=np.float64)


def _time_size(ds: DatasetLike) -> int:
    sizes = getattr(ds, "sizes", {})
    if isinstance(sizes, dict) and "time" in sizes:
        return int(sizes["time"])
    return int(np.asarray(ds["status"].values).size)


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _reference_dataset_id(converted: list[ConvertedShot]) -> str:
    if not converted:
        return "mast-efm-empty"
    shots = "-".join(str(item.shot_id) for item in converted)
    digest = hashlib.sha256("|".join(item.sha256 for item in converted).encode("ascii")).hexdigest()[:16]
    return f"mast-efm-{shots}-{digest}"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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

    report = convert_campaign(
        dataset_root=args.dataset_root,
        campaign_manifest=args.campaign_manifest,
        output_root=args.output_root,
        max_times_per_shot=args.max_times_per_shot,
    )
    report_out = args.report_out or args.output_root / "mast_efm_neural_equilibrium_reference_candidate.json"
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
