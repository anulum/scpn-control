# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST evaluator feature source parity tests.

"""Public evaluator projections must match supervised features and report source gaps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from validation.evaluate_mast_efm_neural_equilibrium import build_feature_projection
from validation.neural_equilibrium_dataset_features import build_feature_matrix


def reference_arrays() -> dict[str, NDArray[Any]]:
    """Return real-valued per-equilibrium observations with nonconstant acquired diagnostics."""
    return {
        "psirz_Wb_per_rad": np.zeros((2, 3, 4)),
        "Ip_MA": np.array([0.2, 0.3]),
        "Bt_T": np.array([-0.8, -0.7]),
        "ffprime_rms_T_rad": np.array([0.5, 1.5]),
        "pprime_Pa_per_Wb_rad": np.array([[2.0, 4.0], [6.0, 8.0]]),
        "pprime_valid_mask": np.ones((2, 2), dtype=bool),
        "magnetic_axis_r_m": np.array([0.7, 0.71]),
        "magnetic_axis_z_m": np.zeros(2),
    }


def test_source_projection_matches_producer_and_preserves_observations(tmp_path: Path) -> None:
    """Decode actual NPZ observations, use the explicit campaign scale and match all twelve producer columns."""
    arrays = reference_arrays()
    original = {key: value.copy() for key, value in arrays.items()}
    path = tmp_path / "reference.npz"
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as data:
        projection = build_feature_projection(data, ffprime_reference=1.0)
    expected = build_feature_matrix(arrays, ffprime_reference=1.0)
    np.testing.assert_array_equal(projection.features, expected)
    np.testing.assert_array_equal(projection.features[:, :2], np.array([[0.2, -0.8], [0.3, -0.7]]))
    np.testing.assert_array_equal(projection.features[:, 5], np.array([0.5, 1.5]))
    assert projection.mapping_notes["Ip_MA"] == "source: Ip_MA"
    assert projection.mapping_notes["Bt_T"] == "source: Bt_T"
    assert "campaign reference 1" in projection.mapping_notes["ffprime_scale"]
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, original[key])


@pytest.mark.parametrize("missing", ["Ip_MA", "Bt_T", "ffprime_rms_T_rad"])
def test_absent_source_uses_only_declared_fallback(missing: str) -> None:
    """Wholly absent source keys retain their documented defaults rather than inventing observations."""
    arrays = reference_arrays()
    arrays.pop(missing)
    projection = build_feature_projection(arrays, ffprime_reference=1.0)
    column = {"Ip_MA": 0, "Bt_T": 1, "ffprime_rms_T_rad": 5}[missing]
    feature = projection.feature_names[column]
    np.testing.assert_array_equal(projection.features[:, column], {0: 8.0, 1: 5.0, 5: 1.0}[column])
    assert projection.mapping_notes[feature].startswith("fallback:")


def test_missing_campaign_scale_remains_explicit_without_shot_normalisation() -> None:
    """A sourced FF-prime vector alone does not define the normalisation used by a multi-shot training campaign."""
    projection = build_feature_projection(reference_arrays())
    np.testing.assert_array_equal(projection.features[:, 5], np.ones(2))
    assert projection.mapping_notes["ffprime_scale"].startswith("fallback:")


@pytest.mark.parametrize("key", ["Ip_MA", "Bt_T", "ffprime_rms_T_rad"])
@pytest.mark.parametrize("bad", [np.array([np.nan, 1.0]), np.ones(3), np.array(["0.2", "0.3"])])
def test_present_invalid_diagnostics_refuse_instead_of_fallback(key: str, bad: NDArray[Any]) -> None:
    """Malformed present acquired channels cannot be silently replaced by synthetic defaults."""
    arrays = reference_arrays()
    arrays[key] = bad
    with pytest.raises(ValueError):
        build_feature_projection(arrays, ffprime_reference=1.0)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_campaign_scale_refuses(scale: float) -> None:
    """Explicit campaign references must be finite and positive before feature inference."""
    with pytest.raises(ValueError, match="ffprime_reference"):
        build_feature_projection(reference_arrays(), ffprime_reference=scale)


def test_representable_extreme_pressure_profiles_remain_finite() -> None:
    """The evaluator shares the producer's scaled mean and median for large but representable observations."""
    arrays = reference_arrays()
    arrays["pprime_Pa_per_Wb_rad"] = np.full((2, 2), 1.0e308)
    with np.errstate(over="raise", invalid="raise"):
        projection = build_feature_projection(arrays)
    np.testing.assert_array_equal(projection.features[:, 4], np.ones(2))


def test_missing_and_nonfinite_axis_scalars_report_actual_fallbacks() -> None:
    """Metadata exposes scalar defaults rather than declaring absent axis/flux observations sourced."""
    arrays = reference_arrays()
    arrays["magnetic_axis_r_m"] = np.array([0.7, np.nan])
    projection = build_feature_projection(arrays)
    assert "nonfinite entries use fallback 1" in projection.mapping_notes["R_axis_m"]
    assert projection.mapping_notes["simag_Wb"].startswith("fallback:")
    np.testing.assert_array_equal(projection.features[:, 2], np.array([0.7, 1.0]))
