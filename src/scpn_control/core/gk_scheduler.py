# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK Spot-Check Scheduler
"""
Scheduler for GK spot-check validation of surrogate transport models.

Decides *when* and *where* to invoke the expensive GK solver based on
configurable strategies: periodic, adaptive (OOD-triggered), or
critical-region policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_control.core.gk_ood_detector import OODResult


@dataclass
class SchedulerConfig:
    """Scheduler parameters."""

    strategy: str = "adaptive"  # "periodic" / "adaptive" / "critical_region"
    period: int = 5  # validate every N transport steps (periodic mode)
    budget: int = 5  # max GK calls per transport step
    anchor_rho: tuple[float, ...] = (0.3, 0.5, 0.8)  # always-validated surfaces
    pedestal_rho: float = 0.85  # critical-region inner edge
    axis_rho: float = 0.15  # critical-region axis edge
    chi_change_threshold: float = 0.5  # adaptive: flag if |delta chi|/chi > this


@dataclass
class SpotCheckRequest:
    """Which flux surfaces to validate and why."""

    rho_indices: list[int]
    reasons: dict[int, str]  # index → reason string
    step_number: int


class GKScheduler:
    """Spot-check scheduler for hybrid surrogate+GK transport."""

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()
        self._validate_config(self.config)
        self._step = 0
        self._prev_chi_i: NDArray[np.float64] | None = None

    @staticmethod
    def _validate_config(config: SchedulerConfig) -> None:
        valid_strategies = {"periodic", "adaptive", "critical_region"}
        if config.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {sorted(valid_strategies)}, received {config.strategy!r}")
        if int(config.period) < 1:
            raise ValueError("period must be >= 1")
        if int(config.budget) < 0:
            raise ValueError("budget must be >= 0")
        if not np.isfinite(config.pedestal_rho) or not (0.0 <= float(config.pedestal_rho) <= 1.0):
            raise ValueError("pedestal_rho must be finite and within [0, 1]")
        if not np.isfinite(config.axis_rho) or not (0.0 <= float(config.axis_rho) <= 1.0):
            raise ValueError("axis_rho must be finite and within [0, 1]")
        if float(config.axis_rho) >= float(config.pedestal_rho):
            raise ValueError("axis_rho must be strictly less than pedestal_rho")
        if not np.isfinite(config.chi_change_threshold) or float(config.chi_change_threshold) < 0.0:
            raise ValueError("chi_change_threshold must be finite and >= 0")
        for anchor in config.anchor_rho:
            if not np.isfinite(anchor) or not (0.0 <= float(anchor) <= 1.0):
                raise ValueError("anchor_rho entries must be finite and within [0, 1]")

    def step(
        self,
        rho: NDArray[np.float64],
        chi_i: NDArray[np.float64],
        ood_results: list[OODResult] | None = None,
    ) -> SpotCheckRequest | None:
        """Decide whether to run GK spot-checks this step.

        Parameters
        ----------
        rho : array
            Radial grid.
        chi_i : array
            Current ion diffusivity profile (from surrogate).
        ood_results : list of OODResult or None
            Per-surface OOD detector results (adaptive mode).

        Returns
        -------
        SpotCheckRequest or None
            None if no validation needed this step.
        """
        rho_arr = np.asarray(rho, dtype=np.float64)
        chi_arr = np.asarray(chi_i, dtype=np.float64)
        if rho_arr.ndim != 1:
            raise ValueError(f"rho must be a 1D array, received ndim={rho_arr.ndim}")
        if chi_arr.ndim != 1:
            raise ValueError(f"chi_i must be a 1D array, received ndim={chi_arr.ndim}")
        if len(rho_arr) == 0:
            raise ValueError("rho must be non-empty")
        if len(rho_arr) != len(chi_arr):
            raise ValueError(f"rho/chi_i length mismatch: {len(rho_arr)} vs {len(chi_arr)}")
        if not np.all(np.isfinite(rho_arr)):
            raise ValueError("rho must contain only finite values")
        if not np.all(np.isfinite(chi_arr)):
            raise ValueError("chi_i must contain only finite values")
        if ood_results is not None and len(ood_results) != len(rho_arr):
            raise ValueError(f"ood_results length mismatch: expected {len(rho_arr)}, received {len(ood_results)}")
        if self._prev_chi_i is not None and len(self._prev_chi_i) != len(chi_arr):
            raise ValueError(
                "chi_i grid length changed between scheduler steps; "
                "call reset() before stepping with a different radial grid"
            )

        self._step += 1
        indices: dict[int, str] = {}

        if self.config.strategy == "periodic":
            if self._step % self.config.period != 0:
                self._prev_chi_i = chi_arr.copy()
                return None
            # Add anchor surfaces
            for rho_val in self.config.anchor_rho:
                idx = int(np.argmin(np.abs(rho_arr - rho_val)))
                indices[idx] = "periodic_anchor"

        elif self.config.strategy == "adaptive":
            # OOD-triggered surfaces
            if ood_results is not None:
                for i, result in enumerate(ood_results):
                    if result.is_ood and len(indices) < self.config.budget:
                        indices[i] = f"ood_{result.method}"

            # Large chi change
            if self._prev_chi_i is not None:
                safe_prev = np.maximum(np.abs(self._prev_chi_i), 1e-10)
                rel_change = np.abs(chi_arr - self._prev_chi_i) / safe_prev
                big_change = np.where(rel_change > self.config.chi_change_threshold)[0]
                for idx in big_change:
                    if len(indices) < self.config.budget:
                        indices[int(idx)] = "chi_change"

            # Always add anchors if budget allows
            for rho_val in self.config.anchor_rho:
                idx = int(np.argmin(np.abs(rho_arr - rho_val)))
                if idx not in indices and len(indices) < self.config.budget:
                    indices[idx] = "anchor"

            if not indices:
                self._prev_chi_i = chi_arr.copy()
                return None

        elif self.config.strategy == "critical_region":
            for i, r in enumerate(rho_arr):
                if (r > self.config.pedestal_rho or r < self.config.axis_rho) and len(indices) < self.config.budget:
                    indices[i] = "critical_region"
            for rho_val in self.config.anchor_rho:
                idx = int(np.argmin(np.abs(rho_arr - rho_val)))
                if idx not in indices and len(indices) < self.config.budget:
                    indices[idx] = "anchor"

        self._prev_chi_i = chi_arr.copy()

        if not indices:
            return None

        # Enforce budget
        selected = dict(list(indices.items())[: self.config.budget])
        return SpotCheckRequest(
            rho_indices=list(selected.keys()),
            reasons=selected,
            step_number=self._step,
        )

    def reset(self) -> None:
        self._step = 0
        self._prev_chi_i = None
