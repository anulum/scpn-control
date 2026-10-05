# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Original source/reference observations.
"""Match actual captured original observations to each SHA-bound converted reference.

Source stores and reference files are captured separately. This proves local
conversion equivalence and declaration consistency, not a campaign transaction,
source signature or measurement authenticity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from validation.mast_efm_original_source_policy import classify_feature_sources, variables_from_metadata
from validation.mast_efm_reference_contracts import REFERENCE_ARRAY_KEYS
from validation.mast_efm_zarr_store import capture_zarr_store, consolidated_metadata, read_reference_store
from validation.neural_equilibrium_dataset_contracts import candidate_reference
from validation.neural_equilibrium_dataset_tensors import load_verified_npz


def load_zarr_candidate_metadata(zarr_path: Path) -> dict[str, dict[str, Any]]:
    """Inspect candidate descriptors only; this public metadata API never grants readiness.

    >>> load_zarr_candidate_metadata(Path(__file__))
    Traceback (most recent call last):
        ...
    FileNotFoundError: consolidated Zarr metadata is missing: ...
    """
    path = zarr_path / ".zmetadata"
    if not path.is_file():
        raise FileNotFoundError(f"consolidated Zarr metadata is missing: {path}")
    return variables_from_metadata(consolidated_metadata(path.read_bytes()))


def _match_reference(source: dict[str, NDArray[Any]], reference: dict[str, NDArray[Any]], shot: dict[str, Any]) -> None:
    """Require every referenced time/grid/channel/target/mask to equal an observed conversion row."""
    n = shot["equilibria_count"]
    times = reference.get("time_s")
    if times is None or times.dtype.kind not in "fiu" or times.shape != (n,):
        raise ValueError("reference time_s must be a real vector matching the selected row count")
    converted_times = source["time_s"]
    indices = np.searchsorted(converted_times, times)
    if (
        not np.all(np.isfinite(times))
        or np.any(np.diff(np.asarray(times, dtype=np.float64)) <= 0)
        or np.any(indices >= converted_times.size)
        or not np.array_equal(converted_times[indices], times)
    ):
        raise ValueError("reference time_s must match ordered observed original conversion rows")
    for key in REFERENCE_ARRAY_KEYS:
        observed = reference.get(key)
        expected = source[key] if key in {"r_grid_m", "z_grid_m"} else source[key][indices]
        if observed is None:
            raise ValueError(f"reference is missing original conversion array {key}")
        if (
            (expected.dtype.kind == "b" and observed.dtype.kind != "b")
            or (key == "shot_id" and observed.dtype.kind not in "iu")
            or (expected.dtype.kind == "f" and observed.dtype.kind not in "fiu")
        ):
            raise ValueError(f"reference {key} has an unsupported observation dtype")
        if observed.shape != expected.shape or not np.array_equal(observed, expected, equal_nan=True):
            raise ValueError(f"reference {key} does not match observed original conversion")


def inspect_original_sources(dataset: dict[str, Any], storage_root: Path) -> list[dict[str, Any]]:
    """Capture/decode actual stores, classify preferred metadata and compare selected reference bytes.

    Missing consolidated metadata preserves FileNotFoundError. Present stores
    that cannot convert or match the selected reference produce blocked checks;
    they never gain readiness merely from descriptors. All source-file digests
    come from the immutable mapping actually decoded by xarray/Zarr.
    """
    result: list[dict[str, Any]] = []
    for shot in dataset["shots"]:
        relative = f"mast/level1/shot_{shot['shot_id']}/efm.zarr"
        snapshot = capture_zarr_store(candidate_reference(relative, storage_root))
        variables = variables_from_metadata(consolidated_metadata(snapshot.files[".zmetadata"]))
        observed_count: int | None = None
        errors: list[str] = []
        try:
            arrays = read_reference_store(snapshot, shot_id=shot["shot_id"])
            observed_count = arrays["time_s"].size
            reference = load_verified_npz(
                candidate_reference(shot["reference_path"], storage_root),
                shot["reference_sha256"],
                mismatch_message="original-source reference SHA does not match declaration",
            )
            _match_reference(arrays, reference, shot)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            errors.append(str(exc))
        result.append(
            {
                **{key: shot[key] for key in ("shot_id", "reference_path", "reference_sha256", "equilibria_count")},
                "zarr_path": relative,
                "source_variables": variables,
                "feature_status": classify_feature_sources(variables),
                "source_snapshot": snapshot.manifest(),
                "conversion_check": {
                    "status": "blocked" if errors else "pass",
                    "errors": errors,
                    "observed_time_count": observed_count,
                    "matched_reference_count": 0 if errors else shot["equilibria_count"],
                    "reference_arrays_match": not errors,
                },
            }
        )
    # The actual manifest/report is finite JSON, including arbitrary source attrs.
    json.dumps(result, allow_nan=False)
    return result
