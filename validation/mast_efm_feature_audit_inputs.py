# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM source-channel byte custody.
"""Inspect SHA-bound converted channels on every selected shot.

This audits converted feature sources, not the original acquisition process,
supervised tensors, target validity or predictive admission. Alternative names
remain inventory hints until the actual converter and feature producer support
them. JSON and each NPZ are captured separately; no multifile snapshot is claimed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from validation.neural_equilibrium_dataset_contracts import (
    FALLBACK_FEATURES,
    FEATURE_SOURCE_POLICY,
    candidate_reference,
)
from validation.neural_equilibrium_dataset_reporting import validate_dataset_report
from validation.neural_equilibrium_dataset_tensors import load_verified_npz

AUDIT_SCHEMA = "scpn-control.mast-efm-feature-provenance-audit.v1"
FEATURE_CANDIDATES = {
    "Ip_MA": ("Ip_MA", "plasma_current_MA", "plasma_current_A", "ip", "Ip", "current_A"),
    "Bt_T": ("Bt_T", "bcentr_T", "b_tor_T", "toroidal_field_T", "Bt", "bcentr"),
    "ffprime_scale": ("ffprime_scale", "ffprime_rms_T_rad", "ffprime", "ffprime_Wb_per_rad", "fpol", "fpol_profile"),
}


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuse duplicate keys at every JSON object depth before validation."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_dataset_declaration(path: Path) -> tuple[dict[str, Any], str]:
    """Validate the actual producer contract and hash the same captured UTF-8 JSON bytes.

    Finite JSON and producer self-digest validation are shared with the actual
    dataset writer. This does not inspect the declared supervised dataset bytes.

    >>> declared = Path(__file__).resolve().parent / "reports/mast_efm_neural_equilibrium_dataset.json"
    >>> report, captured_sha = read_dataset_declaration(declared)
    >>> report["status"], len(captured_sha)
    ('blocked', 64)
    """
    try:
        captured = path.read_bytes()
        report = json.loads(captured.decode("utf-8"), object_pairs_hook=_unique)
        validate_dataset_report(report)
    except (OSError, ValueError, TypeError, RecursionError, RuntimeError) as exc:
        raise ValueError(f"cannot read feature-audit dataset declaration: {exc}") from exc
    return report, hashlib.sha256(captured).hexdigest()


def reference_paths(report: dict[str, Any], storage_root: Path) -> list[Path]:
    """Resolve validated portable relative paths and refuse symlink escapes from storage.

    The producer report must have been validated before this path selection.
    """
    result: list[Path] = []
    for relative in report["reference_paths"]:
        result.append(candidate_reference(relative, storage_root))
    return result


def _real_vector(data: dict[str, NDArray[Any]], key: str, size: int) -> NDArray[np.float64]:
    """Require a real vector with exact rows and finite float64 representation."""
    value = data.get(key)
    if value is None or value.dtype.kind not in "fiu" or value.shape != (size,):
        raise ValueError(f"{key} must be a real per-equilibrium vector with shape {(size,)}")
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{key} must have finite float64 values")
    return result


def inspect_reference_sources(report: dict[str, Any], storage_root: Path) -> list[dict[str, Any]]:
    """Verify declared bytes, shot/time/grid bindings and supported source vectors on each shot.

    Missing canonical source channels remain blocked. Present malformed channels
    refuse, including booleans, complex/text/object arrays, wrong row counts,
    nonfinite float64 values and nonpositive FF-prime RMS. Signed finite current
    and field values retain the producer's existing semantics. Raw targets are
    inventoried; this does not revalidate targets or regenerate the dataset.
    """
    result: list[dict[str, Any]] = []
    grids: dict[str, NDArray[np.float64]] = {}
    for shot, path in zip(report["shots"], reference_paths(report, storage_root), strict=True):
        data = load_verified_npz(
            path, shot["reference_sha256"], mismatch_message="reference bundle SHA-256 does not match dataset report"
        )
        n = shot["equilibria_count"]
        ids = data.get("shot_id")
        if ids is None or ids.dtype.kind not in "iu" or ids.shape != (n,) or np.any(ids != shot["shot_id"]):
            raise ValueError("reference shot_id must match the dataset report on every row")
        times = _real_vector(data, "time_s", n)
        if (
            np.any(times < 0)
            or np.any(np.diff(times) <= 0)
            or float(times[0]) != shot["time_start_s"]
            or float(times[-1]) != shot["time_end_s"]
        ):
            raise ValueError("reference times must match the dataset report's ordered bounds")
        for key, size in (("z_grid_m", shot["grid_shape"][0]), ("r_grid_m", shot["grid_shape"][1])):
            grid = _real_vector(data, key, size)
            differences = np.diff(grid)
            if not (np.all(differences > 0) or np.all(differences < 0)):
                raise ValueError(f"{key} must be strictly monotonic")
            ordered = np.sort(grid)
            if float(ordered[0]) != report[key]["min"] or float(ordered[-1]) != report[key]["max"]:
                raise ValueError(f"{key} bounds must match the dataset report")
            if key in grids and not np.array_equal(grids[key], ordered):
                raise ValueError(f"{key} differs across selected references")
            grids[key] = ordered
        sourced: list[str] = []
        for feature in FALLBACK_FEATURES:
            key = FEATURE_SOURCE_POLICY[feature]["source_key"]
            if key not in data:
                continue
            values = _real_vector(data, key, n)
            if feature == "ffprime_scale" and np.any(values <= 0):
                raise ValueError("ffprime_rms_T_rad must be strictly positive")
            sourced.append(feature)
        keys = sorted(data)
        result.append(
            {
                "shot_id": shot["shot_id"],
                "reference_path": shot["reference_path"],
                "reference_sha256": shot["reference_sha256"],
                "equilibria_count": n,
                "key_count": len(keys),
                "keys": keys,
                "shapes": {key: list(data[key].shape) for key in keys},
                "sourced_features": sourced,
            }
        )
    return result


def feature_status_for_shots(shots: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate complete canonical-channel coverage; aliases alone never resolve a feature.

    Per-shot declarations must be validated before using this aggregate. Inventory
    union remains visible, while completeness requires every selected reference.
    """
    all_keys = {key for shot in shots for key in shot["keys"]}
    result: dict[str, dict[str, Any]] = {}
    for feature in FALLBACK_FEATURES:
        complete = sum(feature in shot["sourced_features"] for shot in shots)
        resolved = complete == len(shots)
        result[feature] = {
            "status": "resolved" if resolved else "blocked",
            "candidate_keys": list(FEATURE_CANDIDATES[feature]),
            "present_keys": sorted(set(FEATURE_CANDIDATES[feature]) & all_keys),
            "source_key": FEATURE_SOURCE_POLICY[feature]["source_key"],
            "complete_reference_count": complete,
            "reference_count": len(shots),
            "resolution": "canonical source channel validated on every selected reference"
            if resolved
            else "canonical source channel missing on at least one selected reference",
        }
    return result
