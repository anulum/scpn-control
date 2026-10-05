# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reference features for MAST EFM diagnostic evaluation.

"""Use the supervised producer's twelve feature definitions for diagnostic inference.

FF-prime requires the training campaign's explicit normalisation reference.
Reference-derived geometry and profiles are inputs, so this projection cannot
establish independent predictive accuracy for those same reference quantities.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from validation.neural_equilibrium_dataset_contracts import FEATURE_NAMES
from validation.neural_equilibrium_dataset_features import build_feature_matrix


@dataclass(frozen=True)
class FeatureProjection:
    """Ordered float64 features and per-column provenance notes for inference.

    Features have shape (N, 12); feature names retain the accelerator's order.
    Arrays and notes are mutable despite the frozen field bindings. Neither
    source authentication nor model compatibility is certified by this object.
    """

    features: NDArray[np.float64]
    feature_names: tuple[str, ...]
    mapping_notes: dict[str, str]


def build_feature_projection(
    data: Mapping[str, NDArray[Any]], *, ffprime_reference: float | None = None
) -> FeatureProjection:
    """Derive N by 12 features using the same rules as the supervised producer.

    Present Ip_MA/Bt_T arrays must contain N finite source observations; absent
    keys retain the documented synthetic-domain defaults 8 MA and 5 T.
    FF-prime uses clipped sourced RMS divided by a supplied positive finite
    campaign reference, or the neutral scale 1 when either source or reference
    is absent. Never infer a campaign normalisation from one evaluation shot.
    Profiles and LCFS geometry follow the producer's masks, defaults and units.
    Invalid ranks, lengths, dtypes or nonfinite final float64 features raise
    ValueError. Mapping input includes an open non-pickled NumPy NPZ archive.
    No input arrays are modified and predictive admission remains separate.

    >>> build_feature_projection({})
    Traceback (most recent call last):
        ...
    ValueError: feature input must contain psirz_Wb_per_rad
    """
    arrays = dict(data)
    features = build_feature_matrix(arrays, ffprime_reference=ffprime_reference)
    notes = {
        "Ip_MA": "fallback: unavailable in converted EFM bundle; synthetic-domain centre used",
        "Bt_T": "fallback: unavailable in converted EFM bundle; synthetic-domain centre used",
        "R_axis_m": "source: magnetic_axis_r_m",
        "Z_axis_m": "source: magnetic_axis_z_m",
        "pprime_scale": "source: masked pprime mean magnitude / positive bundle median; missing rows use magnitude 1",
        "ffprime_scale": "fallback: absent source or campaign reference; neutral scale used",
        "simag_Wb": "source: converted EFM psi_axis_Wb_per_rad",
        "sibry_Wb": "source: converted EFM psi_boundary_Wb_per_rad",
        "kappa": "derived: finite LCFS vertical minor radius relative to magnetic axis",
        "delta_upper": "derived: finite LCFS upper triangularity relative to magnetic axis",
        "delta_lower": "derived: finite LCFS lower triangularity relative to magnetic axis",
        "q95": "source: last finite converted EFM q_profile value; not an interpolated q at 95% flux",
    }
    for feature in ("Ip_MA", "Bt_T"):
        if feature in arrays:
            notes[feature] = f"source: {feature}"
    for feature, key, fallback in (
        ("R_axis_m", "magnetic_axis_r_m", 1.0),
        ("Z_axis_m", "magnetic_axis_z_m", 0.0),
        ("simag_Wb", "psi_axis_Wb_per_rad", 0.0),
        ("sibry_Wb", "psi_boundary_Wb_per_rad", 1.0),
    ):
        if key not in arrays:
            notes[feature] = f"fallback: absent {key}; value {fallback:g}"
        elif not np.all(np.isfinite(arrays[key])):
            notes[feature] = f"source: {key}; nonfinite entries use fallback {fallback:g}"
    if "ffprime_rms_T_rad" in arrays and ffprime_reference is not None:
        notes["ffprime_scale"] = f"source: ffprime_rms_T_rad / campaign reference {ffprime_reference:.17g}"
    return FeatureProjection(features, FEATURE_NAMES, notes)
