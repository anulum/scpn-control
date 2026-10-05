# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Machine input contracts.

"""Capture analytic machine scalars/profiles as explicit-unit immutable JSON.

Presets identify example/design inputs, not measured facility records. A snapshot
binds sampled values without attesting callback code or external provenance.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from validation.synthetic_diagnostics import SyntheticDiagnosticSuite


@dataclass
class MachineConfig:
    """Analytic machine inputs; profiles use keV and density in 1e19 m^-3.

    Presets describe design/example inputs, not measured facility records.
    ``snapshot`` samples callbacks into a reproducible immutable JSON string;
    it does not attest provenance or establish model/reference validation.
    """

    name: str
    R0: float
    a: float
    B0: float
    kappa: float
    delta: float
    Ip_MA: float
    P_aux_MW: float
    ne_profile: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    Te_profile: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    Ti_profile: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    diagnostics: SyntheticDiagnosticSuite = dataclasses.field(default_factory=SyntheticDiagnosticSuite)

    def snapshot(self, rho: npt.NDArray[np.float64]) -> str:
        """Capture validated scalar inputs and sampled profiles as canonical JSON.

        Parameters
        ----------
        rho
            Finite, strictly increasing one-dimensional normalised radius from
            exactly zero to one, with at least three points. Each callback gets
            an independent copy; its output is captured before the next call.

        Returns
        -------
        str
            Schema-versioned JSON with explicit units and sampled numeric values.
            Hash these UTF-8 bytes to bind subsequent model runs to these inputs.
            A digest binds values only, not callback code or external provenance.

        Raises
        ------
        ValueError
            Geometry, scalar domains, grid or sampled profiles are invalid.
            Zero edge density/temperature is retained, never silently floored;
            downstream models must declare their own admissible boundary inputs.

        Examples
        --------
        Capture the analytic example without changing its zero edge values.
        The snapshot is an input record, not a machine-validation result.

        >>> rho = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        >>> inputs = json.loads(iter_15ma().snapshot(rho))
        >>> inputs["units"]["ne"], inputs["profiles"]["ne"][-1]
        ('1e19 m^-3', 0.0)
        >>> inputs["scalars"]["Ip_MA"]
        15.0
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("machine name must be a nonempty string")
        name = self.name
        scalars = {}
        for key in ("R0", "a", "B0", "kappa", "delta", "Ip_MA", "P_aux_MW"):
            raw = getattr(self, key)
            if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, float, np.integer, np.floating)):
                raise ValueError(f"{key} must be a finite real scalar")
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
            scalars[key] = value
        if any(scalars[key] <= 0.0 for key in ("R0", "a", "B0", "kappa", "Ip_MA")):
            raise ValueError("radii, B0, elongation and plasma current magnitude must be positive")
        if scalars["a"] >= scalars["R0"] or abs(scalars["delta"]) >= 1.0 or scalars["P_aux_MW"] < 0.0:
            raise ValueError("require a < R0, abs(delta) < 1 and nonnegative auxiliary power")
        raw_grid = np.asarray(rho)
        if raw_grid.dtype.kind not in "fiu":
            raise ValueError("rho must contain real numeric values")
        grid = np.array(raw_grid, dtype=np.float64, copy=True)
        if (
            grid.ndim != 1
            or grid.size < 3
            or not np.all(np.isfinite(grid))
            or grid[0] != 0.0
            or grid[-1] != 1.0
            or not np.all(np.diff(grid) > 0.0)
        ):
            raise ValueError("rho must increase strictly from 0 to 1 with at least three finite points")
        callbacks = (("ne", self.ne_profile), ("Te", self.Te_profile), ("Ti", self.Ti_profile))
        profiles = {}
        for key, callback in callbacks:
            sampled = np.asarray(callback(grid.copy()))
            if sampled.dtype.kind not in "fiu" or sampled.shape != grid.shape:
                raise ValueError(f"{key} profile must be real numeric values matching rho shape")
            with np.errstate(over="ignore", invalid="ignore"):
                sampled = sampled.astype(np.float64)
            if not np.all(np.isfinite(sampled)) or np.any(sampled < 0.0):
                raise ValueError(f"{key} profile must be finite and nonnegative in float64")
            profiles[key] = sampled.tolist()
        payload = {
            "schema": "scpn-control.machine-inputs.v1",
            "machine": name,
            "scalars": scalars,
            "rho": grid.tolist(),
            "profiles": profiles,
            "units": {
                "R0": "m",
                "a": "m",
                "B0": "T",
                "kappa": "1",
                "delta": "1",
                "Ip_MA": "MA",
                "P_aux_MW": "MW",
                "rho": "1",
                "ne": "1e19 m^-3",
                "Te": "keV",
                "Ti": "keV",
            },
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def iter_15ma() -> MachineConfig:
    """Return the analytic ITER example with Ip=15 MA and P_aux=50 MW.

    Profiles use normalised rho in [0, 1], density in 1e19 m^-3 and temperatures
    in keV. All profiles vanish at the edge. No shot identity, measured profile,
    boundary/coils or external reference is attached; the name is not provenance.
    """
    return MachineConfig(
        "ITER",
        6.2,
        2.0,
        5.3,
        1.7,
        0.33,
        15.0,
        50.0,
        lambda rho: 10.0 * (1.0 - rho**2) ** 0.5,
        lambda rho: 20.0 * (1.0 - rho**2) ** 2,
        lambda rho: 20.0 * (1.0 - rho**2) ** 2,
    )


def jet_high_performance() -> MachineConfig:
    """Return the analytic JET example with Ip=3.5 MA and P_aux=30 MW.

    Profiles use normalised rho in [0, 1], density in 1e19 m^-3 and temperatures
    in keV. All profiles vanish at the edge. No shot identity, measured profile,
    boundary/coils or external reference is attached; the name is not provenance.
    """
    return MachineConfig(
        "JET",
        2.96,
        1.25,
        3.45,
        1.68,
        0.3,
        3.5,
        30.0,
        lambda rho: 5.0 * (1.0 - rho**2) ** 0.5,
        lambda rho: 10.0 * (1.0 - rho**2) ** 1.5,
        lambda rho: 12.0 * (1.0 - rho**2) ** 1.5,
    )


def diiid_h_mode() -> MachineConfig:
    """Return the analytic DIII-D example with Ip=1.5 MA and P_aux=15 MW.

    Profiles use normalised rho in [0, 1], density in 1e19 m^-3 and temperatures
    in keV. All profiles vanish at the edge. No shot identity, measured profile,
    boundary/coils or external reference is attached; the name is not provenance.
    """
    return MachineConfig(
        "DIII-D",
        1.67,
        0.67,
        2.1,
        1.8,
        0.4,
        1.5,
        15.0,
        lambda rho: 4.0 * (1.0 - rho**2) ** 0.5,
        lambda rho: 4.0 * (1.0 - rho**2) ** 1.5,
        lambda rho: 4.0 * (1.0 - rho**2) ** 1.5,
    )


def sparc_baseline() -> MachineConfig:
    """Return the analytic SPARC example with Ip=8.7 MA and P_aux=25 MW.

    Profiles use normalised rho in [0, 1], density in 1e19 m^-3 and temperatures
    in keV. All profiles vanish at the edge. No shot identity, measured profile,
    boundary/coils or external reference is attached; the name is not provenance.
    """
    return MachineConfig(
        "SPARC",
        1.85,
        0.57,
        12.2,
        1.97,
        0.54,
        8.7,
        25.0,
        lambda rho: 30.0 * (1.0 - rho**2) ** 0.5,
        lambda rho: 15.0 * (1.0 - rho**2) ** 2,
        lambda rho: 15.0 * (1.0 - rho**2) ** 2,
    )


def nstx_u_standard() -> MachineConfig:
    """Return the analytic NSTX-U example with Ip=1 MA and P_aux=10 MW.

    Profiles use normalised rho in [0, 1], density in 1e19 m^-3 and temperatures
    in keV. All profiles vanish at the edge. No shot identity, measured profile,
    boundary/coils or external reference is attached; the name is not provenance.
    """
    return MachineConfig(
        "NSTX-U",
        0.93,
        0.58,
        1.0,
        2.0,
        0.4,
        1.0,
        10.0,
        lambda rho: 5.0 * (1.0 - rho**2) ** 0.5,
        lambda rho: 1.5 * (1.0 - rho**2),
        lambda rho: 1.5 * (1.0 - rho**2),
    )
