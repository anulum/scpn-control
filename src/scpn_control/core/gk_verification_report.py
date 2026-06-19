# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK Verification Report
"""
Per-session verification report for hybrid surrogate+GK transport.

Tracks spot-check statistics, correction factors, OOD trigger rates,
and error distributions across a simulation session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scpn_control.core.gk_corrector import CorrectionRecord


@dataclass
class VerificationReport:
    """Accumulated verification statistics for one simulation session."""

    total_steps: int = 0
    steps_verified: int = 0
    total_spot_checks: int = 0
    ood_triggers: int = 0
    records: list[CorrectionRecord] = field(default_factory=list)
    correction_factors: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.total_steps < 0 or self.steps_verified < 0 or self.total_spot_checks < 0 or self.ood_triggers < 0:
            raise ValueError("report counters must be >= 0")
        if self.steps_verified > self.total_steps:
            raise ValueError("steps_verified cannot exceed total_steps")

    @property
    def verification_fraction(self) -> float:
        """Fraction of steps that were verified against the GK reference."""
        if self.total_steps == 0:
            return 0.0
        return self.steps_verified / self.total_steps

    @property
    def max_rel_error(self) -> float:
        """Largest absolute relative error in ion heat diffusivity across records."""
        if not self.records:
            return 0.0
        return max(abs(r.rel_error_chi_i) for r in self.records)

    @property
    def mean_rel_error(self) -> float:
        """Mean absolute relative error in ion heat diffusivity across records."""
        if not self.records:
            return 0.0
        return float(np.mean([abs(r.rel_error_chi_i) for r in self.records]))

    def add_step(self, verified: bool, n_spot_checks: int = 0, n_ood: int = 0) -> None:
        """Record one simulation step and its spot-check and OOD counts.

        Parameters
        ----------
        verified
            Whether this step was verified against the GK reference.
        n_spot_checks
            Number of GK spot checks performed this step; must be non-negative.
        n_ood
            Number of out-of-distribution triggers this step; must be
            non-negative.
        """
        n_spot_checks_i = int(n_spot_checks)
        n_ood_i = int(n_ood)
        if n_spot_checks_i < 0:
            raise ValueError("n_spot_checks must be >= 0")
        if n_ood_i < 0:
            raise ValueError("n_ood must be >= 0")
        self.total_steps += 1
        if verified:
            self.steps_verified += 1
        self.total_spot_checks += n_spot_checks_i
        self.ood_triggers += n_ood_i

    def add_records(self, records: list[CorrectionRecord]) -> None:
        """Append GK correction records to the session log.

        Parameters
        ----------
        records
            Correction records to append; each must be a ``CorrectionRecord``.

        Raises
        ------
        ValueError
            If any entry is not a ``CorrectionRecord``.
        """
        for rec in records:
            if not isinstance(rec, CorrectionRecord):
                raise ValueError("records must contain only CorrectionRecord instances")
        self.records.extend(records)

    def add_correction_factor(self, factor: float) -> None:
        """Append a surrogate-to-GK correction factor.

        Parameters
        ----------
        factor
            The correction factor; must be finite.
        """
        factor_f = float(factor)
        if not np.isfinite(factor_f):
            raise ValueError("correction factor must be finite")
        self.correction_factors.append(factor_f)

    def to_dict(self) -> dict[str, float]:
        """Return a rounded summary of the session verification statistics."""
        return {
            "total_steps": self.total_steps,
            "steps_verified": self.steps_verified,
            "verification_fraction": round(self.verification_fraction, 4),
            "total_spot_checks": self.total_spot_checks,
            "ood_triggers": self.ood_triggers,
            "max_rel_error_chi_i": round(self.max_rel_error, 4),
            "mean_rel_error_chi_i": round(self.mean_rel_error, 4),
            "n_correction_records": len(self.records),
            "mean_correction_factor": round(float(np.mean(self.correction_factors)), 4)
            if self.correction_factors
            else 0.0,
        }

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialise the report to JSON, optionally writing it to a file.

        Parameters
        ----------
        path
            Optional destination path; when given the JSON is written there.

        Returns
        -------
        str
            The JSON text of the report summary.
        """
        text = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            Path(path).write_text(text)
        return text
