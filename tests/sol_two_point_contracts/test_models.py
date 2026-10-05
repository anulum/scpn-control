# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SOL diagnostic geometry and numerical input contracts.
"""Exercise public model diagnostics with real geometry and numerical values."""

from __future__ import annotations

from typing import cast

import pytest

from validation.sol_two_point_contracts.models import (
    SOLConfig,
    conduction_integral_rel_error,
    connection_length_rel_error,
    default_config,
    detachment_boundary,
    eich_scaling_checks,
    flux_mapping_rel_error,
    peak_heat_flux_rel_error,
    pressure_balance_rel_error,
    validate_sol_two_point,
)


@pytest.mark.parametrize("value", [True, False, "1", None, -1.0, 0.0, float("nan"), float("inf"), 10**400])
def test_aggregate_refuses_invalid_tolerances(value: object) -> None:
    """A declared strict tolerance cannot use coercion, nonfinite values or zero."""
    with pytest.raises(ValueError, match="exact_tol"):
        validate_sol_two_point(exact_tol=cast(float, value))


@pytest.mark.parametrize("points", [None, "10,3", b"10,3", (None,), ((1.0,),), ((1.0, 2.0, 3.0),), ("ab",)])
def test_aggregate_refuses_invalid_pair_shapes(points: object) -> None:
    """Refuse malformed public operating-point structure before solving."""
    with pytest.raises(ValueError, match="operating[ _]point"):
        validate_sol_two_point(operating_points=cast(tuple[tuple[float, float], ...], points))


def test_aggregate_refuses_foreign_geometry_type() -> None:
    """A false or unrelated object cannot silently select default geometry."""
    with pytest.raises(ValueError, match="config"):
        validate_sol_two_point(config=cast(SOLConfig, False))


@pytest.mark.parametrize("epsilon", [0.49, 0.5, 0.9])
def test_scaling_keeps_every_valid_aspect_ratio_in_domain(epsilon: float) -> None:
    """Probe doubling or halving preserves production epsilon's open domain."""
    config = SOLConfig(2.0, 2.0 * epsilon, 3.5, 0.4)
    result = validate_sol_two_point(config=config, operating_points=((10.0, 3.0),))
    assert result.passed and all(check.rel_error < 1e-12 for check in result.scaling)
    factor = 2.0 if epsilon < 0.5 else 0.5
    assert result.scaling[-1].expected_ratio == factor**0.42


def test_each_direct_diagnostic_refuses_invalid_declared_power_or_density() -> None:
    """The public scalar checks share numeric MW/density domains."""
    config = default_config()
    for check in (flux_mapping_rel_error, conduction_integral_rel_error, pressure_balance_rel_error):
        with pytest.raises(ValueError, match="P_SOL_MW"):
            check(config, 0.0, 3.0)
        with pytest.raises(ValueError, match="n_u_19"):
            check(config, 10.0, cast(float, "3"))
    with pytest.raises(ValueError, match="P_SOL_MW"):
        peak_heat_flux_rel_error(config, -1.0)
    with pytest.raises(ValueError, match="P_SOL_MW"):
        eich_scaling_checks(config, float("nan"))
    with pytest.raises(ValueError, match="l_par"):
        detachment_boundary(100.0, 0.0)


def test_connection_refuses_overflowing_derived_length() -> None:
    """Finite declared geometry cannot admit an infinite derived connection length."""
    with pytest.raises(ValueError, match="connection length"):
        connection_length_rel_error(SOLConfig(1e308, 1.0, 3.5, 0.4))


def test_geometry_refuses_underflowed_inverse_aspect_ratio() -> None:
    """Positive radii must retain a representable positive ratio for Eich probes."""
    with pytest.raises(ValueError, match="epsilon"):
        SOLConfig(1e308, 5e-324, 3.5, 0.4)


def test_detachment_refuses_nonfinite_derived_density() -> None:
    """An extreme finite flux cannot emit a nonfinite density boundary."""
    with pytest.raises(ValueError, match="critical density"):
        detachment_boundary(1e308, 20.0)
