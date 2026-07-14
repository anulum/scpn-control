#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium dataset builder
"""Build a canonical MAST EFM supervised dataset with repo-published evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

DATASET_SCHEMA = "scpn-control.mast-efm-neural-equilibrium-supervised-dataset.v1"
FEATURE_NAMES = (
    "Ip_MA",
    "Bt_T",
    "R_axis_m",
    "Z_axis_m",
    "pprime_scale",
    "ffprime_scale",
    "simag_Wb",
    "sibry_Wb",
    "kappa",
    "delta_upper",
    "delta_lower",
    "q95",
)
FALLBACK_FEATURES = ("Ip_MA", "Bt_T", "ffprime_scale")
FEATURE_SOURCE_POLICY: dict[str, dict[str, Any]] = {
    "Ip_MA": {
        "source_key": "Ip_MA",
        "original_source": "plasma_current_x",
        "transform": "A_to_MA during reference conversion; identity_MA during dataset build",
        "units": "MA",
    },
    "Bt_T": {
        "source_key": "Bt_T",
        "original_source": "bphi_rmag",
        "transform": "identity_T",
        "units": "T",
    },
    "ffprime_scale": {
        "source_key": "ffprime_rms_T_rad",
        "original_source": "ffprime",
        "transform": "campaign_median_normalised_rms",
        "units": "dimensionless",
        "clip": [0.25, 4.0],
    },
}
TARGET_KEYS = (
    "psirz_Wb_per_rad",
    "psirz_valid_mask",
    "psi_axis_Wb_per_rad",
    "psi_boundary_Wb_per_rad",
    "pprime_Pa_per_Wb_rad",
    "pprime_valid_mask",
    "q_profile",
    "q_profile_valid_mask",
    "lcfs_r_m",
    "lcfs_z_m",
    "lcfs_valid_mask",
    "magnetic_axis_r_m",
    "magnetic_axis_z_m",
)
RAGGED_LCFS_KEYS = ("lcfs_r_m", "lcfs_z_m", "lcfs_valid_mask")
DEFAULT_TRAIN_SHOTS = (30419, 30420, 30421, 30422)
DEFAULT_VALIDATION_SHOTS = (30423,)
DEFAULT_TEST_SHOTS = (30424,)


@dataclass(frozen=True)
class DatasetInput:
    """Input paths and split declaration for the dataset builder."""

    candidate_report: Path
    storage_root: Path
    output_npz: Path
    train_shots: tuple[int, ...] = DEFAULT_TRAIN_SHOTS
    validation_shots: tuple[int, ...] = DEFAULT_VALIDATION_SHOTS
    test_shots: tuple[int, ...] = DEFAULT_TEST_SHOTS


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: dict[str, Any]) -> str:
    """Return a canonical JSON SHA-256 digest."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def safe_storage_reference(path_text: str, storage_root: Path) -> str:
    """Return a stable storage-relative reference for an internal path."""

    path = Path(path_text)
    try:
        return path.resolve().relative_to(storage_root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _per_equilibrium_array(data: dict[str, NDArray[Any]], key: str, count: int, fallback: float) -> NDArray[np.float64]:
    raw = data.get(key)
    if raw is None:
        return np.full(count, fallback, dtype=np.float64)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 0:
        values = np.full(count, float(arr), dtype=np.float64)
    else:
        flat = arr.reshape(-1)
        if flat.size != count:
            raise ValueError(f"{key} must contain {count} per-equilibrium values")
        values = flat.astype(np.float64)
    values[~np.isfinite(values)] = fallback
    return values


def _last_finite_profile_value(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
    valid = mask & np.isfinite(values)
    if not np.any(valid):
        return 4.0
    indices = np.flatnonzero(valid)
    return float(values[indices[-1]])


def _profile_last_values(
    data: dict[str, NDArray[Any]], value_key: str, mask_key: str, count: int, fallback: float
) -> NDArray[np.float64]:
    raw = data.get(value_key)
    if raw is None:
        return np.full(count, fallback, dtype=np.float64)
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim == 1:
        values = np.tile(values.reshape(1, -1), (count, 1))
    if values.shape[0] != count:
        raise ValueError(f"{value_key} row count must match psirz_Wb_per_rad")
    raw_mask = data.get(mask_key)
    if raw_mask is None:
        mask = np.ones(values.shape, dtype=bool)
    else:
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.ndim == 1:
            mask = np.tile(mask.reshape(1, -1), (count, 1))
        if mask.shape != values.shape:
            raise ValueError(f"{mask_key} shape must match {value_key}")
    return np.asarray([_last_finite_profile_value(values[row], mask[row]) for row in range(count)], dtype=np.float64)


def _pprime_scales(data: dict[str, NDArray[Any]], count: int) -> NDArray[np.float64]:
    raw = data.get("pprime_Pa_per_Wb_rad")
    if raw is None:
        return np.ones(count, dtype=np.float64)
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim == 1:
        values = np.tile(values.reshape(1, -1), (count, 1))
    if values.shape[0] != count:
        raise ValueError("pprime_Pa_per_Wb_rad row count must match psirz_Wb_per_rad")
    mask = np.asarray(data.get("pprime_valid_mask", np.ones(values.shape, dtype=bool)), dtype=bool)
    if mask.ndim == 1:
        mask = np.tile(mask.reshape(1, -1), (count, 1))
    if mask.shape != values.shape:
        raise ValueError("pprime_valid_mask shape must match pprime_Pa_per_Wb_rad")
    magnitudes = np.ones(count, dtype=np.float64)
    for row in range(count):
        valid = mask[row] & np.isfinite(values[row])
        if np.any(valid):
            magnitudes[row] = float(np.nanmean(np.abs(values[row, valid])))
    positive = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0.0)]
    reference = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(reference) or reference <= 0.0:
        reference = 1.0
    return np.clip(magnitudes / reference, 0.25, 4.0).astype(np.float64)


def _ffprime_rms_values(data: dict[str, NDArray[Any]], count: int) -> NDArray[np.float64] | None:
    raw = data.get("ffprime_rms_T_rad")
    if raw is None:
        return None
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size != count:
        raise ValueError("ffprime_rms_T_rad row count must match psirz_Wb_per_rad")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("ffprime_rms_T_rad must contain finite positive per-equilibrium values")
    return values


def _campaign_ffprime_reference(rows: list[dict[str, NDArray[Any]]]) -> float | None:
    values: list[NDArray[np.float64]] = []
    for row in rows:
        psi = np.asarray(row["psirz_Wb_per_rad"])
        count = int(psi.shape[0])
        rms = _ffprime_rms_values(row, count)
        if rms is not None:
            values.append(rms)
    if not values or len(values) != len(rows):
        return None
    concatenated = np.concatenate(values)
    positive = concatenated[np.isfinite(concatenated) & (concatenated > 0.0)]
    if positive.size == 0:
        return None
    reference = float(np.median(positive))
    if not np.isfinite(reference) or reference <= 0.0:
        return None
    return reference


def _public_feature_availability(
    rows: list[dict[str, NDArray[Any]]], ffprime_reference: float | None
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    fallback: list[str] = []
    policy: dict[str, dict[str, Any]] = {}
    for feature in ("Ip_MA", "Bt_T"):
        key = str(FEATURE_SOURCE_POLICY[feature]["source_key"])
        if all(key in row for row in rows):
            policy[feature] = dict(FEATURE_SOURCE_POLICY[feature])
        else:
            fallback.append(feature)
    if ffprime_reference is not None and all("ffprime_rms_T_rad" in row for row in rows):
        policy["ffprime_scale"] = {**FEATURE_SOURCE_POLICY["ffprime_scale"], "campaign_reference": ffprime_reference}
    else:
        fallback.append("ffprime_scale")
    return tuple(fallback), policy


def _sourced_scalar_feature(
    data: dict[str, NDArray[Any]], key: str, count: int, fallback: float
) -> NDArray[np.float64]:
    raw = data.get(key)
    if raw is None:
        return np.full(count, fallback, dtype=np.float64)
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size != count:
        raise ValueError(f"{key} row count must match psirz_Wb_per_rad")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key} contains non-finite values")
    return values


def _lcfs_geometry_features(
    data: dict[str, NDArray[Any]], axis_r: NDArray[np.float64], axis_z: NDArray[np.float64], count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    raw_r = data.get("lcfs_r_m")
    raw_z = data.get("lcfs_z_m")
    if raw_r is None or raw_z is None:
        return (
            np.full(count, 1.7, dtype=np.float64),
            np.zeros(count, dtype=np.float64),
            np.zeros(count, dtype=np.float64),
        )
    r_values = np.asarray(raw_r, dtype=np.float64)
    z_values = np.asarray(raw_z, dtype=np.float64)
    if r_values.ndim == 1:
        r_values = np.tile(r_values.reshape(1, -1), (count, 1))
    if z_values.ndim == 1:
        z_values = np.tile(z_values.reshape(1, -1), (count, 1))
    if r_values.shape[0] != count or z_values.shape != r_values.shape:
        raise ValueError("LCFS arrays must align with psirz_Wb_per_rad")
    mask = np.asarray(data.get("lcfs_valid_mask", np.ones(r_values.shape, dtype=bool)), dtype=bool)
    if mask.ndim == 1:
        mask = np.tile(mask.reshape(1, -1), (count, 1))
    if mask.shape != r_values.shape:
        raise ValueError("lcfs_valid_mask shape must match lcfs_r_m")
    kappa = np.full(count, 1.7, dtype=np.float64)
    delta_upper = np.zeros(count, dtype=np.float64)
    delta_lower = np.zeros(count, dtype=np.float64)
    for row in range(count):
        valid = mask[row] & np.isfinite(r_values[row]) & np.isfinite(z_values[row])
        if np.count_nonzero(valid) < 3:
            continue
        r_row = r_values[row, valid]
        z_row = z_values[row, valid]
        radial_minor = float(np.max(np.abs(r_row - axis_r[row])))
        if radial_minor <= 1e-9 or not np.isfinite(radial_minor):
            continue
        z_relative = z_row - axis_z[row]
        vertical_minor = float(max(np.max(z_relative), -np.min(z_relative)))
        if vertical_minor > 0.0 and np.isfinite(vertical_minor):
            kappa[row] = float(np.clip(vertical_minor / radial_minor, 0.5, 3.5))
        upper_index = int(np.argmax(z_relative))
        lower_index = int(np.argmin(z_relative))
        delta_upper[row] = float(np.clip((axis_r[row] - r_row[upper_index]) / radial_minor, -1.0, 1.0))
        delta_lower[row] = float(np.clip((axis_r[row] - r_row[lower_index]) / radial_minor, -1.0, 1.0))
    return kappa, delta_upper, delta_lower


def build_feature_matrix(
    data: dict[str, NDArray[Any]], *, ffprime_reference: float | None = None
) -> NDArray[np.float64]:
    """Build the current 12-column neural-equilibrium feature matrix."""

    psi = np.asarray(data["psirz_Wb_per_rad"])
    if psi.ndim != 3:
        raise ValueError("psirz_Wb_per_rad must have shape (n_equilibria, nz, nr)")
    count = int(psi.shape[0])
    axis_r = _per_equilibrium_array(data, "magnetic_axis_r_m", count, 1.0)
    axis_z = _per_equilibrium_array(data, "magnetic_axis_z_m", count, 0.0)
    psi_axis = _per_equilibrium_array(data, "psi_axis_Wb_per_rad", count, 0.0)
    psi_boundary = _per_equilibrium_array(data, "psi_boundary_Wb_per_rad", count, 1.0)
    pprime_scale = _pprime_scales(data, count)
    ffprime_rms = _ffprime_rms_values(data, count)
    if ffprime_rms is not None and ffprime_reference is not None:
        ffprime_scale = np.clip(ffprime_rms / ffprime_reference, 0.25, 4.0).astype(np.float64)
    else:
        ffprime_scale = np.ones(count, dtype=np.float64)
    q95 = _profile_last_values(data, "q_profile", "q_profile_valid_mask", count, 4.0)
    kappa, delta_upper, delta_lower = _lcfs_geometry_features(data, axis_r, axis_z, count)
    columns = {
        "Ip_MA": _sourced_scalar_feature(data, "Ip_MA", count, 8.0),
        "Bt_T": _sourced_scalar_feature(data, "Bt_T", count, 5.0),
        "R_axis_m": axis_r,
        "Z_axis_m": axis_z,
        "pprime_scale": pprime_scale,
        "ffprime_scale": ffprime_scale,
        "simag_Wb": psi_axis,
        "sibry_Wb": psi_boundary,
        "kappa": kappa,
        "delta_upper": delta_upper,
        "delta_lower": delta_lower,
        "q95": q95,
    }
    features = np.column_stack([columns[name] for name in FEATURE_NAMES]).astype(np.float64)
    if not np.all(np.isfinite(features)):
        raise ValueError("feature matrix contains non-finite values")
    return features


def _load_npz(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _split_label(shot_id: int, train: set[int], validation: set[int], test: set[int]) -> str:
    if shot_id in train:
        return "train"
    if shot_id in validation:
        return "validation"
    if shot_id in test:
        return "test"
    raise ValueError(f"shot {shot_id} is not assigned to train, validation, or test")


def _concat_arrays(rows: list[dict[str, NDArray[Any]]], key: str) -> NDArray[Any]:
    arrays = [np.asarray(row[key]) for row in rows]
    first_shape = arrays[0].shape[1:]
    for arr in arrays:
        if arr.ndim == 0 or arr.shape[1:] != first_shape:
            raise ValueError(f"{key} shape is not consistent across shots")
    return np.concatenate(arrays, axis=0)


def _concat_lcfs_arrays(rows: list[dict[str, NDArray[Any]]], key: str) -> tuple[NDArray[Any], NDArray[np.int64]]:
    arrays = [np.asarray(row[key]) for row in rows]
    if any(arr.ndim != 2 for arr in arrays):
        raise ValueError(f"{key} must be a two-dimensional per-slice LCFS array")
    max_points = max(int(arr.shape[1]) for arr in arrays)
    total_slices = sum(int(arr.shape[0]) for arr in arrays)
    if arrays[0].dtype == np.bool_:
        padded = np.zeros((total_slices, max_points), dtype=np.bool_)
    else:
        padded = np.full((total_slices, max_points), np.nan, dtype=np.float64)
    point_counts = np.empty(total_slices, dtype=np.int64)
    offset = 0
    for arr in arrays:
        count = int(arr.shape[0])
        points = int(arr.shape[1])
        padded[offset : offset + count, :points] = arr
        point_counts[offset : offset + count] = points
        offset += count
    return padded, point_counts


def build_dataset(inputs: DatasetInput) -> dict[str, Any]:
    """Build the storage-hosted supervised dataset and compact repository report."""

    candidate = load_json(inputs.candidate_report)
    shots = candidate.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("candidate report must contain a non-empty shots list")
    split_sets = {
        "train": set(inputs.train_shots),
        "validation": set(inputs.validation_shots),
        "test": set(inputs.test_shots),
    }
    if sum(len(values) for values in split_sets.values()) != len(set().union(*split_sets.values())):
        raise ValueError("train, validation, and test shots must not overlap")
    rows: list[dict[str, NDArray[Any]]] = []
    shot_reports: list[dict[str, Any]] = []
    features: list[NDArray[np.float64]] = []
    split_labels: list[NDArray[np.str_]] = []
    reference_paths: list[str] = []
    for shot in sorted(shots, key=lambda item: int(item["shot_id"])):
        shot_id = int(shot["shot_id"])
        label = _split_label(shot_id, split_sets["train"], split_sets["validation"], split_sets["test"])
        path = Path(str(shot["output_path"]))
        data = _load_npz(path)
        count = int(np.asarray(data["psirz_Wb_per_rad"]).shape[0])
        rows.append(data)
        features.append(np.empty((count, len(FEATURE_NAMES)), dtype=np.float64))
        split_labels.append(np.full(count, label, dtype="U10"))
        reference_paths.append(safe_storage_reference(str(path), inputs.storage_root))
        shot_reports.append(
            {
                "shot_id": shot_id,
                "split": label,
                "equilibria_count": count,
                "reference_path": safe_storage_reference(str(path), inputs.storage_root),
                "reference_sha256": sha256_file(path),
                "grid_shape": [int(feature) for feature in np.asarray(data["psirz_Wb_per_rad"]).shape[1:]],
                "time_start_s": float(np.asarray(data["time_s"], dtype=np.float64)[0]),
                "time_end_s": float(np.asarray(data["time_s"], dtype=np.float64)[-1]),
            }
        )
    ffprime_reference = _campaign_ffprime_reference(rows)
    fallback_features, feature_source_policy = _public_feature_availability(rows, ffprime_reference)
    features = [build_feature_matrix(data, ffprime_reference=ffprime_reference) for data in rows]
    target_payload: dict[str, NDArray[Any]] = {}
    lcfs_point_count: NDArray[np.int64] | None = None
    for key in TARGET_KEYS:
        if key in RAGGED_LCFS_KEYS:
            target_payload[key], counts = _concat_lcfs_arrays(rows, key)
            if lcfs_point_count is None:
                lcfs_point_count = counts
            elif not np.array_equal(lcfs_point_count, counts):
                raise ValueError("LCFS ragged point counts differ between R and Z/mask arrays")
        else:
            target_payload[key] = _concat_arrays(rows, key)
    if lcfs_point_count is None:
        raise ValueError("LCFS point-count metadata was not produced")
    feature_payload = np.concatenate(features, axis=0)
    labels = np.concatenate(split_labels, axis=0)
    shot_ids = _concat_arrays(rows, "shot_id").astype(np.int64)
    time_s = _concat_arrays(rows, "time_s").astype(np.float64)
    r_grid = np.asarray(rows[0]["r_grid_m"], dtype=np.float64)
    z_grid = np.asarray(rows[0]["z_grid_m"], dtype=np.float64)
    for row in rows[1:]:
        if not np.array_equal(r_grid, np.asarray(row["r_grid_m"], dtype=np.float64)):
            raise ValueError("r_grid_m differs across shots")
        if not np.array_equal(z_grid, np.asarray(row["z_grid_m"], dtype=np.float64)):
            raise ValueError("z_grid_m differs across shots")
    inputs.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        inputs.output_npz,
        features=feature_payload,
        feature_names=np.asarray(FEATURE_NAMES),
        split=labels,
        shot_id=shot_ids,
        time_s=time_s,
        r_grid_m=r_grid,
        z_grid_m=z_grid,
        lcfs_point_count=lcfs_point_count,
        # numpy savez_compressed stub types **kwds against allow_pickle: bool, so a
        # dict of arrays unpacked as keyword targets trips arg-type; runtime is correct.
        **target_payload,  # type: ignore[arg-type]
    )
    split_counts = {name: int(np.count_nonzero(labels == name)) for name in ("train", "validation", "test")}
    report: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA,
        "status": "blocked",
        "source": "documented_public_reference",
        "candidate_report": safe_storage_reference(str(inputs.candidate_report), inputs.storage_root),
        "candidate_payload_sha256": candidate.get("payload_sha256"),
        "reference_dataset_id": candidate.get("reference_dataset_id"),
        "dataset_path": safe_storage_reference(str(inputs.output_npz), inputs.storage_root),
        "dataset_sha256": sha256_file(inputs.output_npz),
        "feature_names": list(FEATURE_NAMES),
        "fallback_features": list(fallback_features),
        "feature_source_policy": feature_source_policy,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "target_keys": list(TARGET_KEYS),
        "ragged_target_policy": {
            "keys": list(RAGGED_LCFS_KEYS),
            "padding": "NaN for coordinates and False for validity mask",
            "point_count_key": "lcfs_point_count",
            "max_lcfs_points": int(lcfs_point_count.max()),
        },
        "shot_count": len(shot_reports),
        "equilibria_count": int(feature_payload.shape[0]),
        "split_counts": split_counts,
        "split_policy": {
            "train_shots": list(inputs.train_shots),
            "validation_shots": list(inputs.validation_shots),
            "test_shots": list(inputs.test_shots),
            "policy": "shot-held-out deterministic split; no random time-slice leakage across holdout shots",
        },
        "grid_shape": [
            int(target_payload["psirz_Wb_per_rad"].shape[1]),
            int(target_payload["psirz_Wb_per_rad"].shape[2]),
        ],
        "r_grid_m": {"count": int(r_grid.size), "min": float(r_grid[0]), "max": float(r_grid[-1])},
        "z_grid_m": {"count": int(z_grid.size), "min": float(z_grid[0]), "max": float(z_grid[-1])},
        "shots": shot_reports,
        "reference_paths": reference_paths,
        "admission_ready": False,
        "strict_artefact_emitted": False,
        "blocked_reason": (
            "This is a supervised public-MAST-EFM dataset for training and holdout evaluation. "
            + (
                "Predictive EFIT/P-EFIT admission remains blocked because no trained full-output pressure/q-profile/LCFS "
                "predictive artefact has passed tolerances."
                if not fallback_features
                else "Predictive EFIT/P-EFIT admission remains blocked because Ip_MA, Bt_T, and ffprime_scale are fallback "
                "features and no trained full-output pressure/q-profile/LCFS predictive artefact has passed tolerances."
            )
        ),
        "next_processing_steps": [
            "train a full-output model on the train split and evaluate only once on validation/test shot splits",
            "keep public-source feature policy fixed while training and holdout evaluation are performed",
            "emit compact holdout metrics and keep large weights/predictions on storage-host storage by SHA-256",
            "run validate_neural_equilibrium_reference.py only after full predictive artefacts and tolerances exist",
        ],
    }
    report["payload_sha256"] = sha256_json({**report, "payload_sha256": None})
    return report


def write_report(report: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    """Write compact dataset JSON and Markdown evidence."""

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# MAST EFM Neural-Equilibrium Supervised Dataset",
        "",
        f"Schema: `{report['schema_version']}`",
        f"Status: `{report['status']}`",
        f"Reference dataset: `{report['reference_dataset_id']}`",
        f"Dataset path: `{report['dataset_path']}`",
        f"Dataset SHA-256: `{report['dataset_sha256']}`",
        f"Equilibria: {report['equilibria_count']}",
        f"Grid shape: {report['grid_shape'][0]} x {report['grid_shape'][1]}",
        f"Maximum LCFS points: {report['ragged_target_policy']['max_lcfs_points']}",
        f"Split counts: train={report['split_counts']['train']}, validation={report['split_counts']['validation']}, test={report['split_counts']['test']}",
        "",
        "## Split policy",
        "",
        f"- Train shots: {', '.join(str(item) for item in report['split_policy']['train_shots'])}",
        f"- Validation shots: {', '.join(str(item) for item in report['split_policy']['validation_shots'])}",
        f"- Test shots: {', '.join(str(item) for item in report['split_policy']['test_shots'])}",
        f"- Policy: {report['split_policy']['policy']}",
        "",
        "## Targets",
        "",
    ]
    lines.extend(f"- `{key}`" for key in report["target_keys"])
    lines.extend(
        [
            "",
            "LCFS coordinates are padded with NaN values, LCFS validity masks are padded with False values, "
            f"and `{report['ragged_target_policy']['point_count_key']}` records the real point count per slice.",
            "",
            "## Admission boundary",
            "",
            report["blocked_reason"],
            "",
            "Fallback features: "
            + (
                ", ".join(f"`{item}`" for item in report["fallback_features"])
                if report["fallback_features"]
                else "none"
            ),
            "",
            "## Feature source policy",
            "",
        ]
    )
    for feature, policy in report.get("feature_source_policy", {}).items():
        lines.append(f"- `{feature}` from `{policy['source_key']}` using `{policy['transform']}`")
    lines.extend(
        [
            "",
            "## Next processing steps",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["next_processing_steps"])
    lines.append("")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text("\n".join(lines), encoding="utf-8")


def _parse_shots(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one shot id is required")
    return values


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--storage-root", default=Path("/data/SCPN-CONTROL"), type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--train-shots", type=_parse_shots, default=DEFAULT_TRAIN_SHOTS)
    parser.add_argument("--validation-shots", type=_parse_shots, default=DEFAULT_VALIDATION_SHOTS)
    parser.add_argument("--test-shots", type=_parse_shots, default=DEFAULT_TEST_SHOTS)
    return parser.parse_args()


def main() -> None:
    """Build the dataset and write compact evidence."""

    args = parse_args()
    report = build_dataset(
        DatasetInput(
            candidate_report=args.candidate_report,
            storage_root=args.storage_root,
            output_npz=args.output_npz,
            train_shots=args.train_shots,
            validation_shots=args.validation_shots,
            test_shots=args.test_shots,
        )
    )
    write_report(report, args.json_out, args.report_out)


if __name__ == "__main__":
    main()
