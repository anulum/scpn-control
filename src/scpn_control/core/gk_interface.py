# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Gyrokinetic Solver Interface
"""
Abstract base class for external and native gyrokinetic solvers.

Defines the universal input/output dataclasses (GKLocalParams, GKOutput)
and the solver contract (GKSolverBase) that all concrete implementations
must satisfy.  The design follows the TGLF-10 parameterisation extended
with Miller geometry fields for flux-tube solvers.
"""

from __future__ import annotations

import abc
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class GKLocalParams:
    """Local plasma parameters at a single flux surface.

    The first 11 fields match the GyrokineticsParams convention already
    used by `gyrokinetic_transport.py`.  The remaining fields add Miller
    geometry and dimensional quantities needed by external GK codes.
    """

    # Normalised gradients
    R_L_Ti: float
    R_L_Te: float
    R_L_ne: float

    # Magnetic geometry
    q: float
    s_hat: float
    alpha_MHD: float = 0.0

    # Ratios / dimensionless
    Te_Ti: float = 1.0
    Z_eff: float = 1.5
    nu_star: float = 0.1
    beta_e: float = 0.01
    epsilon: float = 0.1  # r / R

    # Miller shaping
    kappa: float = 1.0
    delta: float = 0.0

    # Dimensional
    rho: float = 0.5  # normalised toroidal flux
    R0: float = 6.2  # major radius [m]
    a: float = 2.0  # minor radius [m]
    B0: float = 5.3  # toroidal field [T]
    n_e: float = 10.0  # electron density [10^19 m^-3]
    T_e_keV: float = 8.0  # electron temperature [keV]
    T_i_keV: float = 8.0  # ion temperature [keV]


@dataclass
class GKOutput:
    """Gyrokinetic solver output for a single flux surface.

    Fluxes are in physical units [m^2/s].  Growth rate arrays are
    normalised to c_s / a.
    """

    chi_i: float  # ion thermal diffusivity [m^2/s]
    chi_e: float  # electron thermal diffusivity [m^2/s]
    D_e: float  # particle diffusivity [m^2/s]
    D_i: float = 0.0  # ion particle diffusivity [m^2/s]

    gamma: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    omega_r: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    k_y: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))

    dominant_mode: str = "stable"  # ITG / TEM / ETG / KBM / MTM / stable
    converged: bool = True
    electromagnetic: bool = False


class GKSolverBase(abc.ABC):
    """Abstract base for gyrokinetic solvers.

    Concrete subclasses must implement ``prepare_input``, ``run``,
    and ``is_available``.
    """

    @abc.abstractmethod
    def prepare_input(self, params: GKLocalParams) -> Path:
        """Write solver-specific input deck to a temporary directory.

        Returns the path to the input file or directory. A temporary
        directory created here belongs to the solver; pass it to
        :meth:`release_input` once its output has been read.
        """

    @abc.abstractmethod
    def run(self, input_path: Path, *, timeout_s: float = 30.0) -> GKOutput:
        """Execute the solver and parse output into GKOutput."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if the solver binary/library is installed."""

    def run_from_params(self, params: GKLocalParams, *, timeout_s: float = 30.0) -> GKOutput:
        """Prepare and run in one call.

        A temporary directory the solver created for the call is removed
        afterwards, whether the run succeeds or fails. A configured working
        directory is kept.

        Parameters
        ----------
        params : GKLocalParams
            Local plasma parameters at one flux surface.
        timeout_s : float
            Time limit for the solver run, in seconds.

        Returns
        -------
        GKOutput
            The parsed solver result.
        """
        input_path = self.prepare_input(params)
        try:
            return self.run(input_path, timeout_s=timeout_s)
        finally:
            self.release_input(input_path)

    def release_input(self, input_path: Path) -> None:
        """Remove ``input_path`` if this solver created it as a temporary directory.

        Anything else, such as a configured working directory, is left alone.

        Parameters
        ----------
        input_path : pathlib.Path
            A path returned by :meth:`prepare_input`.
        """
        owned = self._owned_directories()
        if input_path in owned:
            owned.discard(input_path)
            shutil.rmtree(input_path, ignore_errors=True)

    def _input_directory(self, work_dir: Path | None, prefix: str) -> Path:
        """Return ``work_dir``, created if missing, or a new temporary directory.

        Parameters
        ----------
        work_dir : pathlib.Path or None
            The configured working directory, kept after every run.
        prefix : str
            Name prefix of the temporary directory used when ``work_dir`` is None.

        Returns
        -------
        pathlib.Path
            The directory that receives the input deck.
        """
        if work_dir is not None:
            work_dir.mkdir(parents=True, exist_ok=True)
            return work_dir
        created = Path(tempfile.mkdtemp(prefix=prefix))
        self._owned_directories().add(created)
        return created

    def _owned_directories(self) -> set[Path]:
        """Return the temporary directories this solver created and has not removed.

        The set lives on the instance and is created on first use, so
        subclasses need not call a base constructor.
        """
        owned: set[Path] | None = self.__dict__.get("_gk_owned_directories")
        if owned is None:
            owned = set()
            self.__dict__["_gk_owned_directories"] = owned
        return owned
