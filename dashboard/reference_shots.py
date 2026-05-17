# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard reference shots
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Safe reference-shot loading for dashboard replay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dashboard.replay import ShotData, build_replay_frame
from dashboard.state import MACHINE_PRESETS


ReferenceValue = NDArray[np.float64] | bool | int | str


def list_reference_shots(directory: Path) -> tuple[Path, ...]:
    """Return sorted ``.npz`` reference shot files from a directory."""
    if not directory.exists():
        return ()
    if not directory.is_dir():
        raise ValueError(f"Reference shot path is not a directory: {directory}")
    return tuple(sorted(path for path in directory.glob("*.npz") if path.is_file()))


def _normalise_loaded_value(value: np.ndarray) -> ReferenceValue:
    if value.shape == ():
        scalar = value.item()
        if isinstance(scalar, np.bool_ | bool):
            return bool(scalar)
        if isinstance(scalar, np.integer | int):
            return int(scalar)
        if isinstance(scalar, np.str_ | str):
            return str(scalar)
        if isinstance(scalar, np.floating | float):
            return np.asarray([float(scalar)], dtype=np.float64)
        raise ValueError(f"Unsupported scalar value type in reference shot: {type(scalar).__name__}")
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"Reference shot vector must be numeric, got {value.dtype}.")
    vector = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError("Reference shot vectors must contain only finite values.")
    return vector


def load_reference_shot(path: Path) -> ShotData:
    """Load and validate one reference shot without enabling pickle decoding."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix != ".npz":
        raise ValueError(f"Reference shot must be an .npz file: {path}")

    with np.load(path, allow_pickle=False) as loaded:
        shot: dict[str, ReferenceValue] = {
            key: _normalise_loaded_value(np.asarray(loaded[key])) for key in loaded.files
        }

    build_replay_frame(
        shot_data=shot,
        machine=MACHINE_PRESETS["DIII-D"],
        shot_label=path.stem,
        step_idx=0,
        profile_points=2,
    )
    return shot


__all__ = ["ReferenceValue", "list_reference_shots", "load_reference_shot"]
