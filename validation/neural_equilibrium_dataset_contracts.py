# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium dataset builder

"""Validate producer controls, candidate declarations and local storage custody."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from validation.convert_mast_efm_neural_equilibrium_reference import CANDIDATE_SCHEMA as CANDIDATE_SCHEMA

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
    """Select a candidate/storage/output and nonempty disjoint genuine shot tuples.

    Paths are local Path objects, output has an explicit NPZ suffix and selected
    paths must remain within storage at execution. This validates declarations,
    not authentic MAST measurements or licence/facility admission.

    >>> DatasetInput(Path("candidate.json"), Path("storage"), Path("output.npy"))
    Traceback (most recent call last):
        ...
    ValueError: output_npz must have an explicit .npz suffix
    """

    candidate_report: Path
    storage_root: Path
    output_npz: Path
    train_shots: tuple[int, ...] = DEFAULT_TRAIN_SHOTS
    validation_shots: tuple[int, ...] = DEFAULT_VALIDATION_SHOTS
    test_shots: tuple[int, ...] = DEFAULT_TEST_SHOTS

    def __post_init__(self) -> None:
        """Refuse coerced paths, unsupported output formats and ambiguous shot partitions before IO."""
        for name in ("candidate_report", "storage_root", "output_npz"):
            value = getattr(self, name)
            if (
                not isinstance(value, Path)
                or str(value) != str(value).strip()
                or any(ord(c) < 32 or ord(c) == 127 for c in str(value))
            ):
                raise ValueError(f"{name} must be a Path without controls")
        if self.output_npz.suffix != ".npz":
            raise ValueError("output_npz must have an explicit .npz suffix")
        all_ids: list[int] = []
        for name in ("train_shots", "validation_shots", "test_shots"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{name} must be a nonempty shot tuple")
            for value in values:
                integer(value, name)
                all_ids.append(value)
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("train, validation, and test shots must be distinct and must not overlap")


def integer(value: Any, field: str) -> int:
    """Require a genuine positive integer, excluding boolean/text/float coercion."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def text(value: Any, field: str) -> str:
    """Require trimmed nonempty text without ASCII controls."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError(f"{field} must be nonempty trimmed text without controls")
    return value


def digest(value: Any, field: str) -> str:
    """Require a lowercase64hex SHA declaration without coercion or whitespace."""
    declared = text(value, field)
    if len(declared) != 64 or any(c not in "0123456789abcdef" for c in declared):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return declared


def sha256_file(path: str | Path) -> str:
    """Stream selected local bytes into SHA-256; supported path/IO failures become ValueError.

    >>> len(sha256_file(Path(__file__)))
    64
    """
    try:
        result = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                result.update(chunk)
        return result.hexdigest()
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"cannot hash selected dataset bytes: {exc}") from exc


def sha256_json(payload: dict[str, Any]) -> str:
    """Hash sorted compact finite ASCII JSON, without implicitly replacing any payload field."""
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"dataset payload must be finite JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuse duplicate JSON keys at every nesting depth."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite(text_value: str) -> float:
    """Refuse overflowed JSON exponents before they become infinities."""
    value = float(text_value)
    if not math.isfinite(value):
        raise ValueError("candidate JSON numbers must be finite")
    return value


def _constant(text_value: str) -> Any:
    """Refuse NaN/Infinity decoder extensions."""
    raise ValueError(f"candidate JSON numbers must be finite: {text_value}")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load finite unique-key UTF-8 object metadata; IO/decode/depth failures become ValueError."""
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique,
            parse_float=_finite,
            parse_constant=_constant,
        )
    except (OSError, ValueError, RecursionError, RuntimeError) as exc:
        raise ValueError(f"cannot read candidate metadata: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("candidate metadata must contain a JSON object")
    return payload


def contained_path(path: Path, storage_root: Path) -> Path:
    """Resolve a selected local path within canonical storage, refusing escaped/dangling-loop aliases."""
    try:
        resolved = path.resolve()
        resolved.relative_to(storage_root.resolve())
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"selected path must remain within storage_root: {path}: {exc}") from exc
    return resolved


def safe_storage_reference(path_text: str, storage_root: Path) -> str:
    """Return a storage-relative local reference; external paths are refused instead of published."""
    return (
        contained_path(Path(text(path_text, "reference path")), storage_root)
        .relative_to(storage_root.resolve())
        .as_posix()
    )


def candidate_reference(value: Any, storage_root: Path) -> Path:
    """Resolve producer absolute or safe storage-relative references, rejecting traversal/foreign drives/escapes.

    The producer records absolute paths with forward slashes. A drive is
    foreign unless it makes the path absolute on this platform, which only a
    Windows host's own drive does; containment in storage is checked after.
    """
    value = text(value, "shot.output_path")
    path = Path(value)
    if (
        (PureWindowsPath(value).drive and not path.is_absolute())
        or "\\" in value
        or any(part in ("", ".", "..") for part in value.split("/")[int(path.is_absolute()) :])
    ):
        raise ValueError("shot.output_path must be a safe local storage reference")
    return contained_path(path if path.is_absolute() else storage_root / path, storage_root)


def read_candidate(inputs: DatasetInput) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Path]]]:
    """Validate exact supported successful converter declarations, digest, split coverage and contained input paths.

    SHA checks bind the later selected bytes, not authentic source physics.
    Every configured shot must occur exactly once; all three split partitions
    are explicit. Missing corpus refuses; it is never manufactured.
    """
    contained_path(inputs.candidate_report, inputs.storage_root)
    contained_path(inputs.output_npz, inputs.storage_root)
    payload = load_json(inputs.candidate_report)
    if payload.get("schema_version") != CANDIDATE_SCHEMA or payload.get("status") != "pass":
        raise ValueError("candidate must have supported schema_version and pass conversion status")
    if (
        payload.get("admission_ready") is not False
        or payload.get("target_schema_status") != "reference_only_no_prediction_metrics"
    ):
        raise ValueError("candidate must preserve reference-only blocked predictive admission")
    if payload.get("source") != "documented_public_reference" or payload.get("errors") != []:
        raise ValueError("candidate must declare public reference source with no conversion errors")
    text(payload.get("reference_dataset_id"), "reference_dataset_id")
    digest(payload.get("payload_sha256"), "candidate.payload_sha256")
    if payload["payload_sha256"] != sha256_json({**payload, "payload_sha256": None}):
        raise ValueError("candidate payload_sha256 does not match its contents")
    shots = payload.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("candidate must contain nonempty shots")
    result: list[tuple[dict[str, Any], Path]] = []
    ids: list[int] = []
    total = 0
    for shot in shots:
        if not isinstance(shot, dict):
            raise ValueError("candidate shot must be an object")
        ids.append(integer(shot.get("shot_id"), "shot_id"))
        total += integer(shot.get("selected_time_count"), "selected_time_count")
        grid = shot.get("grid_shape")
        if not isinstance(grid, list) or len(grid) != 2:
            raise ValueError("candidate grid_shape must contain two positive dimensions")
        for value in grid:
            integer(value, "grid_shape")
        integer(shot.get("lcfs_points"), "lcfs_points")
        if shot.get("status") != "reference_candidate":
            raise ValueError("shot status must be reference_candidate")
        digest(shot.get("sha256"), "shot.sha256")
        result.append((shot, candidate_reference(shot.get("output_path"), inputs.storage_root)))
    selected = set((*inputs.train_shots, *inputs.validation_shots, *inputs.test_shots))
    if len(set(ids)) != len(ids) or set(ids) != selected:
        raise ValueError("candidate shots must match the distinct configured shot partition")
    if integer(payload.get("reference_equilibria_count"), "reference_equilibria_count") != total:
        raise ValueError("candidate reference_equilibria_count must match shot counts")
    return payload, sorted(result, key=lambda item: item[0]["shot_id"])


def ensure_distinct_outputs(paths: list[Path], *, protected: list[Path] | None = None) -> None:
    """Refuse path/symlink/hardlink output aliases, including selected input or weight custody.

    Resolution is checked before persistence/training in main. This is not an
    atomic multi-file snapshot; callers must control concurrent filesystem changes.

    >>> ensure_distinct_outputs([Path(__file__), Path(__file__)])
    Traceback (most recent call last):
        ...
    ValueError: training outputs must be distinct and must not overwrite protected inputs/weights
    """
    for index, output in enumerate(paths):
        output.resolve()
        for other in [*paths[:index], *(protected or [])]:
            if output.resolve() == other.resolve() or (output.exists() and other.exists() and output.samefile(other)):
                raise ValueError("training outputs must be distinct and must not overwrite protected inputs/weights")
