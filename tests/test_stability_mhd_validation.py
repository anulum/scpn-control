# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MHD Stability Validator Branch Tests
"""Branch coverage for the MHD stability scalar/profile validators and the
q-profile self-consistency checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_control.core.stability_mhd import (
    QProfile,
    _require_finite_scalar,
    _require_normalised_radius,
    _require_profile_array,
    _validate_q_profile,
)


def _qprofile(**overrides: Any) -> QProfile:
    rho = np.linspace(0.0, 1.0, 50)
    q = 1.0 + 2.0 * rho
    fields: dict[str, Any] = {
        "rho": rho,
        "q": q,
        "shear": 2.0 * rho / (1.0 + 2.0 * rho),
        "alpha_mhd": np.zeros_like(rho),
        "q_min": float(q.min()),
        "q_min_rho": float(rho[np.argmin(q)]),
        "q_edge": float(q[-1]),
    }
    fields.update(overrides)
    return QProfile(**fields)


class TestScalarAndProfileValidators:
    def test_finite_scalar_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            _require_finite_scalar("q_min", float("nan"))

    def test_profile_array_rejects_too_few_points(self) -> None:
        with pytest.raises(ValueError, match="at least two points"):
            _require_profile_array("q", np.array([1.0]))

    def test_profile_array_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="only finite values"):
            _require_profile_array("q", np.array([1.0, np.nan]))

    def test_normalised_radius_rejects_out_of_interval(self) -> None:
        with pytest.raises(ValueError, match=r"normalised interval \[0, 1\]"):
            _require_normalised_radius(np.array([0.0, 1.5]))


class TestQProfileConsistency:
    def test_rejects_q_min_rho_above_unit_interval(self) -> None:
        with pytest.raises(ValueError, match=r"q_min_rho must stay within the normalised interval"):
            _validate_q_profile(_qprofile(q_min_rho=1.5))

    def test_rejects_q_min_not_matching_profile_minimum(self) -> None:
        with pytest.raises(ValueError, match="q_min must match the q-profile minimum"):
            _validate_q_profile(_qprofile(q_min=0.5))

    def test_rejects_q_edge_not_matching_last_point(self) -> None:
        with pytest.raises(ValueError, match="q_edge must match the last q-profile point"):
            _validate_q_profile(_qprofile(q_edge=99.0))
