# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Confinement reference comparison.

"""Compare actual IPB98(y,2) predictions with hash-bound local calibration rows.

Source classification stays caller-declared; digests establish byte identity,
not external authenticity, held-out empirical validation or facility authority.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from scpn_control.core.scaling_laws import ipb98y2_tau_e


def _coefficient_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys at every depth of a bound coefficient document."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"coefficient JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _coefficient_json_float(token: str) -> float:
    """Reject nonstandard constants and numeric tokens outside finite float64."""
    value = float(token)
    if not np.isfinite(value):
        raise ValueError("coefficient JSON numbers must be finite")
    return value


@dataclass(frozen=True)
class ConfinementReference:
    """Bind one CSV operating point and coefficient file for scaling comparison.

    Expected digests bind bytes, not external authenticity. Source classification
    is caller-declared. This adapter supports bounded calibration/design/synthetic
    comparisons only; it cannot grant held-out empirical or facility acceptance.
    The operating point comes entirely from the selected row, not a preset name.
    """

    csv_path: Path
    csv_sha256: str
    coefficients_path: Path
    coefficients_sha256: str
    row_index: int
    machine: str
    shot: str
    source_class: Literal["derived_calibration", "design_reference", "synthetic"]
    relative_tolerance: float

    def evaluate(self) -> str:
        """Run actual IPB98(y,2) on a hash-bound reference row and return JSON.

        The zero-based row index excludes the header. Machine and shot must match
        the selected row. Ploss, isotope mass and line-averaged density are taken
        explicitly from that row; auxiliary power/profile averages are not used.
        Both files are read once; parsing and hashing use those same bytes.
        Coefficient JSON rejects duplicate keys at every depth and nonfinite
        numeric tokens, including unused metadata, rather than selecting one
        of several possible model interpretations.

        Returns
        -------
        str
            Canonical JSON containing inputs, units, source row and file hashes,
            prediction/reference seconds, relative error and declared tolerance.
            Comparison success does not grant action or facility authority.

        Raises
        ------
        ValueError
            Digests, schema, identity, source classification or numeric inputs are
            invalid. No fallback coefficients, implicit tolerance or passing
            result is substituted. Boolean tolerances, including NumPy booleans,
            are invalid. File access failures propagate as OSError.

        Examples
        --------
        Run the shipped calibration rows with their actual bytes. Computing
        these digests identifies local files; it supplies no independent trust
        anchor or held-out provenance. JET fails the same 10% comparison that
        the ITER design row passes.

        >>> from dataclasses import replace
        >>> directory = Path(__file__).resolve().parent / "reference_data/itpa"
        >>> reference = directory / "hmode_confinement.csv"
        >>> coefficients = directory / "ipb98y2_coefficients.json"
        >>> bound = ConfinementReference(
        ...     reference, hashlib.sha256(reference.read_bytes()).hexdigest(),
        ...     coefficients, hashlib.sha256(coefficients.read_bytes()).hexdigest(),
        ...     0, "ITER", "design", "derived_calibration", 0.1,
        ... )
        >>> design = json.loads(bound.evaluate())
        >>> round(design["predicted_tau_E_s"], 3), design["comparison_pass"]
        (3.606, True)
        >>> shot = replace(bound, row_index=1, machine="JET", shot="92436")
        >>> json.loads(shot.evaluate())["comparison_pass"]
        False
        >>> design["actionable"], design["facility_validated"]
        (False, False)
        """
        if self.source_class not in ("derived_calibration", "design_reference", "synthetic"):
            raise ValueError("unsupported reference source classification")
        if type(self.row_index) is not int or self.row_index < 0:
            raise ValueError("row_index must be a nonnegative integer")
        if (
            isinstance(self.relative_tolerance, (bool, np.bool_))
            or not np.isfinite(self.relative_tolerance)
            or self.relative_tolerance <= 0
        ):
            raise ValueError("relative_tolerance must be finite and positive")
        payloads = []
        for path, expected in ((self.csv_path, self.csv_sha256), (self.coefficients_path, self.coefficients_sha256)):
            if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
                raise ValueError("expected SHA256 must be 64 lowercase hexadecimal characters")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("reference or coefficient SHA256 mismatch")
            payloads.append(raw)
        reader = csv.DictReader(io.StringIO(payloads[0].decode("utf-8")))
        fields = reader.fieldnames
        numeric = ("Ip_MA", "BT_T", "ne19_1e19m3", "Ploss_MW", "R_m", "a_m", "kappa", "M_AMU", "tau_E_s")
        required = {"machine", "shot", "source", *numeric}
        if fields is None or len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError("reference CSV requires unique headers and all operating-point columns")
        rows = list(reader)
        if self.row_index >= len(rows):
            raise ValueError("reference row_index is outside the CSV")
        row = rows[self.row_index]
        if None in row or any(value is None for value in row.values()):
            raise ValueError("reference row has a different width from its header")
        if (
            not self.machine.strip()
            or not self.shot.strip()
            or row["machine"] != self.machine
            or row["shot"] != self.shot
        ):
            raise ValueError("reference machine/shot identity mismatch")
        if not row["source"].strip():
            raise ValueError("reference source must be declared")
        values = {key: float(row[key]) for key in numeric}
        if any(not np.isfinite(value) or value <= 0 for value in values.values()) or values["a_m"] >= values["R_m"]:
            raise ValueError("operating-point values must be finite positive with a < R")
        predicted = ipb98y2_tau_e(
            values["Ip_MA"],
            values["BT_T"],
            values["ne19_1e19m3"],
            values["Ploss_MW"],
            values["R_m"],
            values["kappa"],
            values["a_m"] / values["R_m"],
            values["M_AMU"],
            coefficients=json.loads(
                payloads[1],
                object_pairs_hook=_coefficient_json_object,
                parse_float=_coefficient_json_float,
                parse_constant=_coefficient_json_float,
            ),
        )
        relative_error = abs(predicted - values["tau_E_s"]) / values["tau_E_s"]
        if not np.isfinite(relative_error):
            raise ValueError("relative comparison error is not representable")
        result = {
            "schema": "scpn-control.confinement-reference.v1",
            "model": "scpn_control.core.scaling_laws.ipb98y2_tau_e",
            "reference_sha256": self.csv_sha256,
            "coefficients_sha256": self.coefficients_sha256,
            "row_index": self.row_index,
            "source_row": row,
            "source_class": self.source_class,
            "source_class_verified": False,
            "inputs": {key: values[key] for key in numeric if key != "tau_E_s"},
            "units": {
                "Ip_MA": "MA",
                "BT_T": "T",
                "ne19_1e19m3": "1e19 m^-3",
                "Ploss_MW": "MW",
                "R_m": "m",
                "a_m": "m",
                "kappa": "1",
                "M_AMU": "u",
                "tau_E": "s",
            },
            "predicted_tau_E_s": predicted,
            "reference_tau_E_s": values["tau_E_s"],
            "relative_error": relative_error,
            "relative_tolerance": self.relative_tolerance,
            "comparison_pass": relative_error <= self.relative_tolerance,
            "training_domain": "not_assessed",
            "actionable": False,
            "facility_validated": False,
        }
        return json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
