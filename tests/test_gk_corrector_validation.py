# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK Corrector Validation Branch Tests
"""Branch coverage for the GK corrector record errors, config and input guards."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.gk_corrector import CorrectionRecord, CorrectorConfig, GKCorrector


def _record(*, rho_idx: int = 0, rho: float = 0.5, chi_e_gk: float = 1.0) -> CorrectionRecord:
    return CorrectionRecord(
        rho_idx=rho_idx,
        rho=rho,
        chi_i_surrogate=1.0,
        chi_i_gk=1.0,
        chi_e_surrogate=1.0,
        chi_e_gk=chi_e_gk,
        D_e_surrogate=1.0,
        D_e_gk=1.0,
    )


class TestCorrectionRecord:
    def test_rel_error_chi_e_with_nonzero_gk(self) -> None:
        record = _record(chi_e_gk=2.0)
        assert record.rel_error_chi_e == pytest.approx((1.0 - 2.0) / 2.0)


class TestStaticValidators:
    def test_transport_coefficients_reject_non_finite(self) -> None:
        with pytest.raises(ValueError, match="transport values must be finite"):
            GKCorrector._validate_transport_coefficients((np.nan, 1.0))

    def test_config_rejects_negative_replace_threshold(self) -> None:
        with pytest.raises(ValueError, match="replace_threshold must be finite and >= 0"):
            GKCorrector(nr=4, config=CorrectorConfig(replace_threshold=-1.0))


class TestUpdateGuards:
    def _corrector(self) -> GKCorrector:
        return GKCorrector(nr=4)

    def test_rejects_wrong_rho_shape(self) -> None:
        with pytest.raises(ValueError, match=r"rho must have shape"):
            self._corrector().update([_record()], np.zeros(3, dtype=np.float64))

    def test_rejects_non_finite_rho(self) -> None:
        rho = np.array([0.0, np.nan, 0.5, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="rho must contain only finite values"):
            self._corrector().update([_record()], rho)

    def test_rejects_record_index_out_of_bounds(self) -> None:
        rho = np.linspace(0.0, 1.0, 4, dtype=np.float64)
        with pytest.raises(ValueError, match="rho_idx out of bounds"):
            self._corrector().update([_record(rho_idx=9)], rho)

    def test_rejects_non_finite_record_rho(self) -> None:
        rho = np.linspace(0.0, 1.0, 4, dtype=np.float64)
        with pytest.raises(ValueError, match="record rho must be finite"):
            self._corrector().update([_record(rho=float("nan"))], rho)


class TestCorrectUnsupportedMode:
    def test_correct_rejects_mutated_invalid_mode(self) -> None:
        corrector = GKCorrector(nr=4)
        # config is a mutable dataclass; a post-construction tamper must be caught.
        corrector.config.mode = "bogus"
        profile = np.ones(4, dtype=np.float64)
        with pytest.raises(ValueError, match="unsupported correction mode"):
            corrector.correct(profile, profile, profile)
