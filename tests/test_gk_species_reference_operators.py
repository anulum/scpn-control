# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK species reference validation tests

"""Exercise real Bessel, quadrature and pitch-angle reference contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_gk_species_reference_domains import declaration, inspect

from scpn_control.core.gk_species import VelocityGrid, pitch_angle_operator


@pytest.mark.parametrize("field", ["argument", "expected"])
@pytest.mark.parametrize("value", [None, [], "1", True, float("nan"), float("inf"), 10**400])
def test_bessel_scalars(tmp_path: Path, field: str, value: object) -> None:
    """Bessel comparison scalars refuse failed conversion without serializing NaN into report hashes."""
    payload = declaration()
    payload["operator_checks"]["bessel_j0"][0][field] = value
    assert inspect(tmp_path, payload)["status"] == "fail"


@pytest.mark.parametrize(
    "field", ["n_energy", "n_lambda", "energy_weight_sum", "lambda_weight_sum", "energy_min", "energy_max"]
)
@pytest.mark.parametrize("value", [None, [], "1", True, float("nan"), float("inf"), 10**400])
def test_velocity_scalars(tmp_path: Path, field: str, value: object) -> None:
    """Velocity declarations require finite scalars before actual quadrature construction, with no huge allocation probes."""
    payload = declaration()
    payload["operator_checks"]["velocity_grid"][field] = value
    assert inspect(tmp_path, payload)["status"] == "fail"


@pytest.mark.parametrize("field", ["B_ratio", "constant_nullspace_max_abs", "tridiagonal_nonzero_entries"])
@pytest.mark.parametrize("value", [None, [], "1", True, float("nan"), float("inf"), 10**400])
def test_pitch_scalars(tmp_path: Path, field: str, value: object) -> None:
    """Pitch operator metadata refuses malformed/nonfinite scalars through the real persisted reader."""
    payload = declaration()
    payload["operator_checks"]["pitch_angle_operator"][field] = value
    assert inspect(tmp_path, payload)["status"] == "fail"


@pytest.mark.parametrize("value", [None, [], "1", True, float("nan"), float("inf"), 10**400, -0.1, 1.1, 0.2])
def test_pitch_grid(tmp_path: Path, value: object) -> None:
    """The unchanged pitch operator enforces finite ordered lambda points and its physical interval."""
    payload = declaration()
    payload["operator_checks"]["pitch_angle_operator"]["lambda_grid"][0] = value
    assert inspect(tmp_path, payload)["status"] == "fail"


@pytest.mark.parametrize("block", ["bessel_j0", "velocity_grid", "pitch_angle_operator"])
@pytest.mark.parametrize("value", [None, [], {}, [None]])
def test_operator_shapes(tmp_path: Path, block: str, value: object) -> None:
    """Each operator declaration requires its original nonempty array or object structure."""
    payload = declaration()
    payload["operator_checks"][block] = value
    assert inspect(tmp_path, payload)["status"] == "fail"


def test_operator_root_and_grid_shape(tmp_path: Path) -> None:
    """Missing operator object and empty lambda grid refuse without numerical calls on malformed input."""
    payload = declaration()
    payload["operator_checks"] = None
    assert inspect(tmp_path, payload)["status"] == "fail"
    payload = declaration()
    payload["operator_checks"]["pitch_angle_operator"]["lambda_grid"] = []
    assert inspect(tmp_path, payload)["status"] == "fail"


def test_original_count_coercion(tmp_path: Path) -> None:
    """Original finite fractional grid and sparsity metadata preserve int truncation to the actual reference dimensions."""
    payload = declaration()
    payload["operator_checks"]["velocity_grid"]["n_energy"] = 4.9
    payload["operator_checks"]["pitch_angle_operator"]["tridiagonal_nonzero_entries"] = 9.9
    assert inspect(tmp_path, payload)["status"] == "pass"


@pytest.mark.parametrize(
    ("block", "field", "value"),
    [("velocity_grid", "n_energy", 1), ("pitch_angle_operator", "B_ratio", 0), ("pitch_angle_operator", "B_ratio", 2)],
)
def test_operator_physical_domains(tmp_path: Path, block: str, field: str, value: float) -> None:
    """Actual quadrature minimum and trapped-passing limits become fixed operator findings."""
    payload = declaration()
    payload["operator_checks"][block][field] = value
    assert inspect(tmp_path, payload)["status"] == "fail"


def test_declared_constant_nullspace_drift(tmp_path: Path) -> None:
    """A real tridiagonal operator cannot validate an incorrect persisted constant-nullspace residual."""
    payload = declaration()
    payload["operator_checks"]["pitch_angle_operator"]["constant_nullspace_max_abs"] = 1e-3
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail"
    assert any(
        error["field"] == "operator_checks.pitch_angle_operator.constant_nullspace_max_abs"
        for error in report["errors"]
    )


@pytest.mark.parametrize("n_lambda", [2, 3, 5, 8, 24])
@pytest.mark.parametrize("B_ratio", [0.5, 1.0, 2.0])
def test_public_pitch_operator_tridiagonal_construction(n_lambda: int, B_ratio: float) -> None:
    """Real quadrature pitch grids preserve the producer's exact tridiagonal support across local magnetic-field ratios."""
    grid = VelocityGrid(n_energy=4, n_lambda=n_lambda)
    matrix = pitch_angle_operator(n_lambda, grid.lam / max(B_ratio, 1.0), B_ratio=B_ratio)
    assert matrix.shape == (n_lambda, n_lambda)
    assert np.count_nonzero(np.triu(matrix, k=2)) == 0
    assert np.count_nonzero(np.tril(matrix, k=-2)) == 0
    assert np.all(np.isfinite(matrix))
    np.testing.assert_allclose(matrix @ np.ones(n_lambda), 0.0, atol=1e-10)
