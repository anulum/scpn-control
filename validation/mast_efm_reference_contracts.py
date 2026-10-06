# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural equilibrium reference converter
"""Declare converter arrays and validate campaign JSON, IDs and storage paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

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


def positive_int(value: object, *, field: str) -> int:
    """Require a genuine positive integer, including bounded conversion controls.

    >>> positive_int(True, field="shot_id")
    Traceback (most recent call last):
        ...
    ValueError: shot_id must be a positive integer
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    if field == "shot_id" and value > (1 << 63) - 1:
        raise ValueError("shot_id must be representable in int64")
    return value


def storage_path(root: Path, value: object) -> Path:
    """Resolve a nonempty relative local path inside the selected storage root."""
    if not isinstance(value, str) or not value.strip() or "\\" in value or "\x00" in value:
        raise ValueError("shot local_path must be a relative storage path")
    path = Path(value)
    # A rooted POSIX spelling is absolute in the document on every platform.
    if path.is_absolute() or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
        raise ValueError("shot local_path must remain inside dataset_root")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("shot local_path must remain inside dataset_root")
    return resolved


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys rather than silently picking one."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    """Reject the nonfinite extensions accepted by the default JSON decoder."""
    raise ValueError(f"nonfinite JSON constant: {value}")


def read_campaign(path: Path) -> list[dict[str, Any]]:
    """Read finite unique-key campaign JSON and validate all shot IDs before writes."""
    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    json.dumps(payload, allow_nan=False)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    shots = payload.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("campaign manifest must contain a non-empty shots list")
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for shot in shots:
        if not isinstance(shot, dict):
            raise ValueError("shot entry must be an object")
        shot_id = positive_int(shot.get("shot_id"), field="shot_id")
        if shot_id in seen:
            raise ValueError(f"duplicate campaign shot_id: {shot_id}")
        seen.add(shot_id)
        result.append(shot)
    return result


def json_sha256(payload: object) -> str:
    """Hash canonical finite JSON using the retained candidate digest convention."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
