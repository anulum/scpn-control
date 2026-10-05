# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary objective metrics tests
"""Exercise stable objective metrics through the public tracking controller."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from scpn_control.control.free_boundary_tracking import FreeBoundaryTrackingController
from scpn_control.control.free_boundary_tracking_metrics import evaluate_objective_metrics
from scpn_control.control.free_boundary_tracking_observation import ObjectiveBlock
from scpn_control.core.fusion_kernel import CoilSet


class _MetricKernel:
    """One-coil kernel contract for public metric evaluation."""

    def __init__(self, config_file: str) -> None:
        del config_file
        self.cfg: dict[str, object] = {"coils": [{"current": 0.0}], "free_boundary": {}}

    def build_coilset_from_config(self) -> CoilSet:
        """Provide one shape-flux target with a finite zero baseline."""
        return CoilSet(
            positions=[(1.0, 1.0)],
            currents=np.zeros(1, dtype=np.float64),
            turns=[1],
            current_limits=np.ones(1, dtype=np.float64),
            target_flux_points=np.array([[1.0, 0.0]], dtype=np.float64),
            target_flux_values=np.zeros(1, dtype=np.float64),
        )


def test_public_metrics_preserve_large_finite_rms_and_norms() -> None:
    """Large finite observations retain finite error metrics and a failed tolerance."""
    controller = FreeBoundaryTrackingController(
        "dummy.json", kernel_factory=_MetricKernel, verbose=False, objective_tolerances={"shape_rms": 1.0}
    )
    metrics = controller.evaluate_objectives(np.array([1e308], dtype=np.float64))
    assert metrics["tracking_error_norm"] == 1e308
    assert metrics["shape_rms"] == 1e308
    assert metrics["control_error_norm"] == 1e308
    assert metrics["objective_converged"] is False


def test_metrics_leaf_and_controller_facade_agree_for_nominal_input() -> None:
    """The extracted leaf preserves the controller's public metric mapping."""
    controller = FreeBoundaryTrackingController("dummy.json", kernel_factory=_MetricKernel, verbose=False)
    observation = np.array([0.25], dtype=np.float64)
    leaf = evaluate_objective_metrics(
        observation,
        target_vector=controller.target_vector,
        objective_blocks=controller.objective_blocks,
        objective_tolerances=controller.objective_tolerances,
        control_objective_weights=controller.control_objective_weights,
    )
    assert leaf == controller.evaluate_objectives(observation)
    assert leaf["shape_rms"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("observation", "target", "weights"),
    [(-1e308, 1e308, 1.0), (0.0, 1e308, 1e308)],
)
def test_metrics_leaf_refuses_unrepresentable_error_or_weight(
    observation: float, target: float, weights: float
) -> None:
    """Finite inputs cannot silently publish a nonfinite metric."""
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate_objective_metrics(
            np.array([observation], dtype=np.float64),
            target_vector=np.array([target], dtype=np.float64),
            objective_blocks=(ObjectiveBlock("shape_flux", 0, 1),),
            objective_tolerances={},
            control_objective_weights=np.array([weights], dtype=np.float64),
        )


def test_metrics_leaf_refuses_unrepresentable_vector_norm() -> None:
    """Several finite errors whose true norm exceeds float64 cannot be admitted."""
    with pytest.raises(ValueError, match="objective norm is nonfinite"):
        evaluate_objective_metrics(
            np.full(4, 1e308, dtype=np.float64),
            target_vector=np.zeros(4, dtype=np.float64),
            objective_blocks=(ObjectiveBlock("shape_flux", 0, 4),),
            objective_tolerances={},
            control_objective_weights=np.ones(4, dtype=np.float64),
        )


def test_metrics_leaf_refuses_empty_objective_block() -> None:
    """An empty declared shape block cannot publish an undefined RMS."""
    with pytest.raises(ValueError, match="objective block must not be empty"):
        evaluate_objective_metrics(
            np.zeros(1, dtype=np.float64),
            target_vector=np.zeros(1, dtype=np.float64),
            objective_blocks=(ObjectiveBlock("shape_flux", 0, 0),),
            objective_tolerances={},
            control_objective_weights=np.ones(1, dtype=np.float64),
        )


@pytest.mark.parametrize(
    ("observation", "weights"),
    [([0.0, 1.0], [1.0]), ([0.0], [1.0, 1.0])],
)
def test_metrics_leaf_refuses_mismatched_objective_widths(observation: list[float], weights: list[float]) -> None:
    """Both observation and weight vectors must match the declared target."""
    with pytest.raises(ValueError, match="match the target vector shape"):
        evaluate_objective_metrics(
            np.asarray(observation, dtype=np.float64),
            target_vector=np.zeros(1, dtype=np.float64),
            objective_blocks=(ObjectiveBlock("shape_flux", 0, 1),),
            objective_tolerances={},
            control_objective_weights=np.asarray(weights, dtype=np.float64),
        )


@pytest.mark.parametrize(
    ("observation", "target", "weights"),
    [(float("nan"), 0.0, 1.0), (0.0, float("inf"), 1.0), (0.0, 0.0, float("nan"))],
)
def test_metrics_leaf_refuses_nonfinite_inputs(observation: float, target: float, weights: float) -> None:
    """Malformed objective inputs fail before any convergence verdict exists."""
    with pytest.raises(ValueError, match="objective inputs contain nonfinite"):
        evaluate_objective_metrics(
            np.array([observation], dtype=np.float64),
            target_vector=np.array([target], dtype=np.float64),
            objective_blocks=(ObjectiveBlock("shape_flux", 0, 1),),
            objective_tolerances={},
            control_objective_weights=np.array([weights], dtype=np.float64),
        )


@pytest.mark.parametrize(
    "tolerances",
    [
        {"shape_rms": float("inf")},
        {"shape_rms": float("nan")},
        {"shape_rms": -1.0},
        {"typo": 1.0},
        {"shape_rms": None},
        {"shape_rms": "invalid"},
        {"shape_rms": 10**400},
        [("shape_rms", 1.0)],
        None,
    ],
)
def test_metrics_leaf_refuses_invalid_tolerances(tolerances: Any) -> None:
    """Invalid thresholds cannot create a convergence verdict or deactivate rows."""
    with pytest.raises(ValueError, match="objective_tolerances"):
        evaluate_objective_metrics(
            np.array([10.0], dtype=np.float64),
            target_vector=np.zeros(1, dtype=np.float64),
            objective_blocks=(ObjectiveBlock("shape_flux", 0, 1),),
            objective_tolerances=tolerances,
            control_objective_weights=np.ones(1, dtype=np.float64),
        )


@pytest.mark.parametrize("threshold", [0.0, 1.0, 5.0])
def test_metrics_leaf_preserves_all_block_errors_and_control_activation(threshold: float) -> None:
    """All four objective blocks retain signed errors, exact checks and weighted activation."""
    error = np.array([3.0, 4.0, 3.0, 4.0, -2.0, 0.0, 4.0], dtype=np.float64)
    blocks = (
        ObjectiveBlock("shape_flux", 0, 2),
        ObjectiveBlock("x_point_position", 2, 4),
        ObjectiveBlock("x_point_flux", 4, 5),
        ObjectiveBlock("divertor_flux", 5, 7),
    )
    tolerances = MappingProxyType(
        dict.fromkeys(
            ["shape_rms", "shape_max_abs", "x_point_position", "x_point_flux", "divertor_rms", "divertor_max_abs"],
            threshold,
        )
    )
    result = evaluate_objective_metrics(
        -error,
        target_vector=np.zeros(7, dtype=np.float64),
        objective_blocks=blocks,
        objective_tolerances=tolerances,
        control_objective_weights=np.full(7, 2.0, dtype=np.float64),
    )
    expected_metrics = {
        "shape_rms": np.sqrt(12.5),
        "shape_max_abs": 4.0,
        "x_point_position": 5.0,
        "x_point_flux": 2.0,
        "divertor_rms": np.sqrt(8.0),
        "divertor_max_abs": 4.0,
    }
    assert result["shape_rms"] == pytest.approx(expected_metrics["shape_rms"])
    assert result["shape_max_abs"] == 4.0
    assert result["x_point_position_error"] == 5.0
    assert result["x_point_flux_error"] == 2.0
    assert result["divertor_rms"] == pytest.approx(expected_metrics["divertor_rms"])
    assert result["divertor_max_abs"] == 4.0
    assert result["tracking_error_norm"] == pytest.approx(np.sqrt(70.0))
    assert result["objective_checks"] == {name: value <= threshold for name, value in expected_metrics.items()}
    assert result["objective_convergence_active"] is True
    assert result["objective_converged"] is (threshold == 5.0)
    assert result["control_error_norm"] == pytest.approx(0.0 if threshold == 5.0 else 2.0 * np.sqrt(70.0))
    assert result["active_control_rows"] == (0 if threshold == 5.0 else 6)
    assert dict(tolerances) == result["objective_tolerances"]


@pytest.mark.parametrize("configured", [False, True])
def test_metrics_leaf_zero_errors_and_empty_thresholds(configured: bool) -> None:
    """Exact agreement accepts zero tolerances; absent tolerances keep checks inactive."""
    result = evaluate_objective_metrics(
        np.zeros(2, dtype=np.float64),
        target_vector=np.zeros(2, dtype=np.float64),
        objective_blocks=(ObjectiveBlock("shape_flux", 0, 1), ObjectiveBlock("divertor_flux", 1, 2)),
        objective_tolerances={"shape_rms": 0.0, "divertor_rms": 0.0} if configured else {},
        control_objective_weights=np.ones(2, dtype=np.float64),
    )
    assert result["shape_rms"] == result["divertor_rms"] == 0.0
    assert result["tracking_error_norm"] == result["control_error_norm"] == 0.0
    assert result["active_control_rows"] == 0
    assert result["objective_convergence_active"] is configured
    assert result["objective_converged"] is True


def test_metrics_leaf_keeps_undeclared_threshold_blocks_inactive() -> None:
    """Configured tolerances for absent blocks cannot manufacture an evaluated check."""
    result = evaluate_objective_metrics(
        np.array([1.0], dtype=np.float64),
        target_vector=np.zeros(1, dtype=np.float64),
        objective_blocks=(),
        objective_tolerances={"shape_rms": 0.0},
        control_objective_weights=np.ones(1, dtype=np.float64),
    )
    assert result["objective_checks"] == {}
    assert result["objective_convergence_active"] is False
    assert result["objective_converged"] is True
    assert result["control_error_norm"] == 1.0


def test_metrics_leaf_refuses_unknown_objective_blocks() -> None:
    """An unknown block cannot publish metrics without a defined activation policy."""
    with pytest.raises(ValueError, match="Unknown objective block"):
        evaluate_objective_metrics(
            np.array([1.0], dtype=np.float64),
            target_vector=np.zeros(1, dtype=np.float64),
            objective_blocks=(ObjectiveBlock("unknown", 0, 1),),
            objective_tolerances={},
            control_objective_weights=np.ones(1, dtype=np.float64),
        )
