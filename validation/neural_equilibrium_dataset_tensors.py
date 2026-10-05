# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium dataset builder

"""Bind converted reference bytes, validate arrays and assemble canonical supervised targets."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray

from validation.neural_equilibrium_dataset_contracts import TARGET_KEYS


def _real(data: dict[str, NDArray[Any]], key: str) -> NDArray[np.float64]:
    """Require a real numeric array representable in float64; shape and masking are checked by the caller."""
    if key not in data:
        raise ValueError(f"reference is missing {key}")
    value = np.asarray(data[key])
    if value.dtype.kind not in "fiu":
        raise ValueError(f"{key} must be a real numeric array")
    return np.asarray(value, dtype=np.float64)


def _finite_shape(data: dict[str, NDArray[Any]], key: str, shape: tuple[int, ...]) -> NDArray[np.float64]:
    """Require exact real finite scalar/grid shape, without broadcasting or boolean/text coercion."""
    values = _real(data, key)
    if values.shape != shape or not np.all(np.isfinite(values)):
        raise ValueError(f"{key} must have finite values with shape {shape}")
    return values


def _target(data: dict[str, NDArray[Any]], key: str, mask_key: str, shape: tuple[int, ...]) -> None:
    """Require exact target/boolean-mask shapes and finite declared valid observations."""
    values = _real(data, key)
    mask = data.get(mask_key)
    if values.shape != shape or mask is None or mask.dtype != np.bool_ or mask.shape != shape:
        raise ValueError(f"{key} and {mask_key} must have matching target/boolean mask shapes")
    if not np.all(np.isfinite(values[mask])):
        raise ValueError(f"{key} must be finite on valid observations")


def load_verified_npz(
    path: Path,
    expected_sha256: str,
    *,
    mismatch_message: str = "selected NPZ SHA-256 does not match its declaration",
) -> dict[str, NDArray[Any]]:
    """Capture one NPZ file, verify those bytes and decode that exact capture without pickle.

    Reference producer and supervised trainer share this decoded-byte custody.
    A changing pathname cannot substitute the verified capture; this is not a
    coherent multi-file snapshot or physical-source authentication. One compressed
    bundle is temporarily retained alongside decoded arrays. The caller owns
    expected-digest declaration admission and its contextual mismatch diagnostic.

    >>> load_verified_npz(Path(__file__), "0" * 64)
    Traceback (most recent call last):
        ...
    ValueError: selected NPZ SHA-256 does not match its declaration
    """
    try:
        source = path.read_bytes()
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"cannot read selected NPZ snapshot: {exc}") from exc
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError(mismatch_message)
    try:
        with BytesIO(source) as snapshot, np.load(snapshot, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}
    except (OSError, ValueError, TypeError, EOFError, BadZipFile) as exc:
        raise ValueError(f"cannot decode selected NPZ snapshot: {exc}") from exc


def load_reference(shot: dict[str, Any], path: Path) -> dict[str, NDArray[Any]]:
    """Verify the selected candidate SHA and decode real NPZ arrays without pickle, then check producer shape/shot custody.

    One compressed-file byte snapshot is SHA-bound before the same bytes are
    decoded. Later pathname changes cannot replace the decoded input, and no
    before/after hash ABA gap remains. The snapshot temporarily retains one
    compressed reference bundle in memory, alongside its decoded arrays.
    This does not authenticate physical measurements or freeze the whole
    campaign's files. Nonfinite masked targets stay unobserved. Descending grids
    are flipped with their flux/masks to preserve coordinates in canonical
    increasing output. LCFS compaction is performed later during assembly.

    The canonical corpus is a storage-host selection; its reported presence
    is not assumed from its declaration. The standalone producer tests exercise
    this loader through build_dataset on actual converter-shaped NPZ bytes.
    >>> source = Path(__file__)
    >>> load_reference({"sha256": "0" * 64}, source)
    Traceback (most recent call last):
        ...
    ValueError: reference bundle SHA-256 does not match candidate
    """
    data = load_verified_npz(path, shot["sha256"], mismatch_message="reference bundle SHA-256 does not match candidate")
    n = shot["selected_time_count"]
    nz, nr = shot["grid_shape"]
    _target(data, "psirz_Wb_per_rad", "psirz_valid_mask", (n, nz, nr))
    shot_ids = data.get("shot_id")
    if (
        shot_ids is None
        or shot_ids.dtype.kind not in "iu"
        or shot_ids.shape != (n,)
        or np.any(shot_ids != shot["shot_id"])
    ):
        raise ValueError("reference shot_id must be integer and match the candidate shot on every row")
    times = _finite_shape(data, "time_s", (n,))
    if np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("reference time_s must be nonnegative and strictly increasing")
    for key in ("psi_axis_Wb_per_rad", "psi_boundary_Wb_per_rad", "magnetic_axis_r_m", "magnetic_axis_z_m"):
        _finite_shape(data, key, (n,))
    for key in ("Ip_MA", "Bt_T", "ffprime_rms_T_rad"):
        if key in data:
            values = _finite_shape(data, key, (n,))
            if key == "ffprime_rms_T_rad" and np.any(values <= 0):
                raise ValueError("ffprime_rms_T_rad must be strictly positive")
    for key, mask in (("pprime_Pa_per_Wb_rad", "pprime_valid_mask"), ("q_profile", "q_profile_valid_mask")):
        value = _real(data, key)
        if value.ndim != 2 or value.shape[0] != n or value.shape[1] < 1:
            raise ValueError(f"{key} must have nonempty per-row profile columns")
        _target(data, key, mask, value.shape)
    width = shot["lcfs_points"]
    _target(data, "lcfs_r_m", "lcfs_valid_mask", (n, width))
    _target(data, "lcfs_z_m", "lcfs_valid_mask", (n, width))
    if np.any(np.count_nonzero(data["lcfs_valid_mask"], axis=1) == 0):
        raise ValueError("each reference LCFS row must contain a valid point")
    for key, size, axis in (("r_grid_m", nr, 2), ("z_grid_m", nz, 1)):
        grid = _finite_shape(data, key, (size,))
        differences = np.diff(grid)
        if not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError(f"{key} must be strictly monotonic")
        if grid[0] > grid[-1]:
            data[key] = grid[::-1].copy()
            data["psirz_Wb_per_rad"] = np.flip(data["psirz_Wb_per_rad"], axis=axis).copy()
            data["psirz_valid_mask"] = np.flip(data["psirz_valid_mask"], axis=axis).copy()
    return data


def assemble_targets(rows: list[dict[str, NDArray[Any]]]) -> dict[str, NDArray[Any]]:
    """Concatenate compatible reference targets and compact valid LCFS points in their original order.

    Masked LCFS coordinates are omitted rather than declared real points.
    Counts record each valid prefix; output padding uses NaN/False. Profile/grid
    disagreement refuses instead of inventing interpolation or new observations.
    """
    first = rows[0]
    for row in rows[1:]:
        for key in ("r_grid_m", "z_grid_m"):
            if not np.array_equal(first[key], row[key]):
                raise ValueError(f"{key} differs across shots")
    output: dict[str, NDArray[Any]] = {}
    for key in TARGET_KEYS:
        if key.startswith("lcfs_"):
            continue
        try:
            output[key] = np.concatenate([row[key] for row in rows], axis=0)
        except ValueError as exc:
            raise ValueError(f"{key} shape is not consistent across shots") from exc
    counts = np.concatenate([np.count_nonzero(row["lcfs_valid_mask"], axis=1) for row in rows]).astype(np.int64)
    width = int(counts.max())
    count = counts.size
    r = np.full((count, width), np.nan, dtype=np.float64)
    z = np.full((count, width), np.nan, dtype=np.float64)
    mask = np.zeros((count, width), dtype=bool)
    offset = 0
    for row in rows:
        for index in range(row["lcfs_valid_mask"].shape[0]):
            valid = row["lcfs_valid_mask"][index]
            size = int(counts[offset])
            r[offset, :size] = row["lcfs_r_m"][index, valid]
            z[offset, :size] = row["lcfs_z_m"][index, valid]
            mask[offset, :size] = True
            offset += 1
    output.update(lcfs_r_m=r, lcfs_z_m=z, lcfs_valid_mask=mask, lcfs_point_count=counts)
    return output


def persist_dataset(path: Path, payload: dict[str, NDArray[Any]]) -> None:
    """Write explicit actual NPZ fields without overload ignores or untyped target keyword bypasses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=payload["features"],
        feature_names=payload["feature_names"],
        split=payload["split"],
        shot_id=payload["shot_id"],
        time_s=payload["time_s"],
        r_grid_m=payload["r_grid_m"],
        z_grid_m=payload["z_grid_m"],
        lcfs_point_count=payload["lcfs_point_count"],
        psirz_Wb_per_rad=payload["psirz_Wb_per_rad"],
        psirz_valid_mask=payload["psirz_valid_mask"],
        psi_axis_Wb_per_rad=payload["psi_axis_Wb_per_rad"],
        psi_boundary_Wb_per_rad=payload["psi_boundary_Wb_per_rad"],
        pprime_Pa_per_Wb_rad=payload["pprime_Pa_per_Wb_rad"],
        pprime_valid_mask=payload["pprime_valid_mask"],
        q_profile=payload["q_profile"],
        q_profile_valid_mask=payload["q_profile_valid_mask"],
        lcfs_r_m=payload["lcfs_r_m"],
        lcfs_z_m=payload["lcfs_z_m"],
        lcfs_valid_mask=payload["lcfs_valid_mask"],
        magnetic_axis_r_m=payload["magnetic_axis_r_m"],
        magnetic_axis_z_m=payload["magnetic_axis_z_m"],
    )
