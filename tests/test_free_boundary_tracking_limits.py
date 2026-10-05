# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for FB tracking limit resolvers

"""Drive production free-boundary tracking limit/tolerance resolvers."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_control.control.free_boundary_tracking_limits as limits
from scpn_control.control.free_boundary_tracking import FreeBoundaryTrackingController
from scpn_control.core.fusion_kernel import CoilSet


class _FractionalHoldKernel:
    """One-coil contract exposing a malformed public tracking configuration."""

    def __init__(self, config_file: str) -> None:
        del config_file
        self.cfg: dict[str, object] = {
            "coils": [{"current": 0.0}],
            "free_boundary": {},
            "free_boundary_tracking": {"hold_steps_after_reject": 1.5},
        }

    def build_coilset_from_config(self) -> CoilSet:
        """Provide a real CoilSet so the controller reaches config resolution."""
        return CoilSet(
            positions=[(1.0, 1.0)],
            currents=np.zeros(1, dtype=np.float64),
            turns=[1],
            current_limits=np.ones(1, dtype=np.float64),
            target_flux_points=np.array([[1.0, 0.0]], dtype=np.float64),
            target_flux_values=np.zeros(1, dtype=np.float64),
        )


def test_owner_objective_and_supervisor_wrappers_match_leaf() -> None:
    """Controller static resolvers are the production leaf functions."""
    cfg = {"shape_rms": 0.2}
    override = {"x_point_flux": 0.01}
    leaf_obj = limits.resolve_objective_tolerances(cfg, override)
    owner_obj = FreeBoundaryTrackingController._resolve_objective_tolerances(cfg, override)
    assert owner_obj == leaf_obj
    leaf_sup = limits.resolve_supervisor_limits({"shape_rms": 0.1}, {"max_abs_coil_current": 2})
    owner_sup = FreeBoundaryTrackingController._resolve_supervisor_limits(
        {"shape_rms": 0.1}, {"max_abs_coil_current": 2}
    )
    assert owner_sup == leaf_sup == {"shape_rms": 0.1, "max_abs_coil_current": 2.0}


def test_resolve_objective_tolerances_fail_closed() -> None:
    """Malformed objective tolerances raise ValueError on the production path."""
    with pytest.raises(ValueError, match="mapping"):
        limits.resolve_objective_tolerances("not-a-mapping", None)
    with pytest.raises(ValueError, match="Unknown"):
        limits.resolve_objective_tolerances({"bogus": 1.0}, None)
    with pytest.raises(ValueError, match=">= 0"):
        limits.resolve_objective_tolerances({"shape_rms": -1.0}, None)


def test_resolve_supervisor_limits_fail_closed() -> None:
    """Malformed supervisor limits raise ValueError on the production path."""
    with pytest.raises(ValueError, match="mapping"):
        limits.resolve_supervisor_limits("not-a-mapping", None)
    with pytest.raises(ValueError, match="Unknown"):
        limits.resolve_supervisor_limits({"bogus_key": 1.0}, None)
    with pytest.raises(ValueError, match=">= 0"):
        limits.resolve_supervisor_limits({"shape_rms": -0.1}, None)


def test_scalar_resolvers_defaults_overrides_and_rejections() -> None:
    """Positive/non-negative float/int and fraction resolvers cover success and fail-closed."""
    assert limits.resolve_positive_float(None, None, default=2.0, name="x") == 2.0
    assert limits.resolve_positive_float(3.0, 5.0, default=2.0, name="x") == 5.0
    with pytest.raises(ValueError):
        limits.resolve_positive_float(0.0, None, default=2.0, name="x")
    assert limits.resolve_nonnegative_int(None, None, default=4, name="n") == 4
    with pytest.raises(ValueError):
        limits.resolve_nonnegative_int(-1, None, default=4, name="n")
    assert limits.resolve_nonnegative_float(float("inf"), default=1.0, name="f") == float("inf")
    with pytest.raises(ValueError):
        limits.resolve_nonnegative_float(-0.1, default=1.0, name="f")
    with pytest.raises(ValueError, match="finite or infinity"):
        limits.resolve_nonnegative_float(float("nan"), default=1.0, name="f")
    assert limits.resolve_fraction(0.7, default=0.3, name="frac") == 0.7
    with pytest.raises(ValueError):
        limits.resolve_fraction(1.1, default=0.3, name="frac")


@pytest.mark.parametrize("invalid", [True, 1.5, "2"])
def test_nonnegative_int_refuses_coercion(invalid: object) -> None:
    """Boolean, fractional and text settings cannot silently become step counts."""
    with pytest.raises(ValueError, match="non-negative integer"):
        limits.resolve_nonnegative_int(invalid, None, default=0, name="hold_steps_after_reject")


def test_public_controller_refuses_fractional_hold_steps() -> None:
    """Malformed config is rejected through the public controller constructor."""
    with pytest.raises(ValueError, match="hold_steps_after_reject must be a non-negative integer"):
        FreeBoundaryTrackingController("dummy.json", kernel_factory=_FractionalHoldKernel, verbose=False)


def test_coil_slew_and_fallback_currents() -> None:
    """Per-coil slew limits and fallback currents validate shape and bounds."""
    n = 4
    coil_limits = np.full(n, 5.0)
    slew = limits.resolve_coil_slew_limits(n, None, 0.5)
    assert slew.shape == (n,)
    np.testing.assert_allclose(slew, 0.5)
    with pytest.raises(ValueError, match="scalar or match"):
        limits.resolve_coil_slew_limits(n, None, [1.0, 2.0])
    with pytest.raises(ValueError, match="finite values"):
        limits.resolve_coil_slew_limits(n, None, [1.0, 2.0, 0.0, 4.0])
    assert limits.resolve_fallback_currents(n, coil_limits, None) is None
    ok = limits.resolve_fallback_currents(n, coil_limits, [1.0, 2.0, 3.0, 4.0])
    assert ok is not None
    np.testing.assert_allclose(ok, [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="match the number of coils"):
        limits.resolve_fallback_currents(n, coil_limits, [1.0, 2.0])
    with pytest.raises(ValueError, match="must be finite"):
        limits.resolve_fallback_currents(n, coil_limits, [1.0, 2.0, float("nan"), 4.0])
    with pytest.raises(ValueError, match="respect CoilSet.current_limits"):
        limits.resolve_fallback_currents(n, coil_limits, [6.0, 0.0, 0.0, 0.0])
