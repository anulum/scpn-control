# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK Scheduler Validation Branch Tests
"""Config-validation and step-guard branch coverage for the GK spot-check scheduler."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.gk_ood_detector import OODResult
from scpn_control.core.gk_scheduler import GKScheduler, SchedulerConfig


class TestConfigValidation:
    def test_rejects_negative_budget(self) -> None:
        with pytest.raises(ValueError, match="budget must be >= 0"):
            GKScheduler(SchedulerConfig(budget=-1))

    def test_rejects_out_of_range_pedestal_rho(self) -> None:
        with pytest.raises(ValueError, match="pedestal_rho must be finite and within"):
            GKScheduler(SchedulerConfig(pedestal_rho=1.5))

    def test_rejects_out_of_range_axis_rho(self) -> None:
        with pytest.raises(ValueError, match="axis_rho must be finite and within"):
            GKScheduler(SchedulerConfig(axis_rho=-0.1))

    def test_rejects_axis_not_below_pedestal(self) -> None:
        with pytest.raises(ValueError, match="axis_rho must be strictly less than pedestal_rho"):
            GKScheduler(SchedulerConfig(axis_rho=0.9, pedestal_rho=0.8))

    def test_rejects_negative_chi_change_threshold(self) -> None:
        with pytest.raises(ValueError, match="chi_change_threshold must be finite and >= 0"):
            GKScheduler(SchedulerConfig(chi_change_threshold=-1.0))


class TestStepGuards:
    def _scheduler(self) -> GKScheduler:
        return GKScheduler(SchedulerConfig())

    def test_rejects_non_1d_rho(self) -> None:
        with pytest.raises(ValueError, match="rho must be a 1D array"):
            self._scheduler().step(np.zeros((2, 2)), np.zeros(4))

    def test_rejects_non_1d_chi(self) -> None:
        with pytest.raises(ValueError, match="chi_i must be a 1D array"):
            self._scheduler().step(np.zeros(4), np.zeros((2, 2)))

    def test_rejects_empty_rho(self) -> None:
        with pytest.raises(ValueError, match="rho must be non-empty"):
            self._scheduler().step(np.zeros(0), np.zeros(0))

    def test_rejects_non_finite_rho(self) -> None:
        with pytest.raises(ValueError, match="rho must contain only finite values"):
            self._scheduler().step(np.array([0.0, np.nan, 1.0]), np.zeros(3))


class TestAdaptiveOODTrigger:
    def test_ood_surface_is_scheduled(self) -> None:
        scheduler = GKScheduler(SchedulerConfig(strategy="adaptive", budget=5))
        rho = np.linspace(0.1, 0.9, 4, dtype=np.float64)
        ood_results = [
            OODResult(is_ood=False, confidence=0.1, method="mahalanobis", details={}),
            OODResult(is_ood=True, confidence=0.9, method="mahalanobis", details={}),
            OODResult(is_ood=False, confidence=0.2, method="mahalanobis", details={}),
            OODResult(is_ood=False, confidence=0.1, method="mahalanobis", details={}),
        ]
        request = scheduler.step(rho, np.full(4, 1.0), ood_results=ood_results)
        assert request is not None
        assert request.reasons[1] == "ood_mahalanobis"


class TestCriticalRegionEmptySelection:
    def test_zero_budget_critical_region_returns_none(self) -> None:
        # budget=0 admits no critical surfaces and no anchors, so the step yields
        # an empty selection and returns None.
        scheduler = GKScheduler(SchedulerConfig(strategy="critical_region", budget=0))
        rho = np.linspace(0.2, 0.8, 5, dtype=np.float64)
        assert scheduler.step(rho, np.full(5, 1.0)) is None
