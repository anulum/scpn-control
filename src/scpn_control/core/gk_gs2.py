# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GS2 External Gyrokinetic Solver
"""
GS2 external solver interface.

Reference: Kotschenreuther et al., Comp. Phys. Comm. 88 (1995) 128.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from scpn_control.core.gk_interface import GKLocalParams, GKOutput, GKSolverBase

_logger = logging.getLogger(__name__)


def generate_gs2_input(params: GKLocalParams) -> str:
    """Render a GS2 ``gs2.in`` deck from local GK parameters.

    Parameters
    ----------
    params
        Local gyrokinetic input parameters.

    Returns
    -------
    str
        The GS2 input-deck text.
    """
    R0_over_a = params.R0 / max(params.a, 0.01)
    return f"""\
&theta_grid_eik_knobs
 itor = 1
 iflux = 0
 irho = 2
 local_eq = .true.
 bishop = 4
 s_hat_input = {params.s_hat:.6f}
 beta_prime_input = 0.0
 ntheta = 32
 nperiod = 1
/
&theta_grid_parameters
 rhoc = {params.rho:.6f}
 qinp = {params.q:.6f}
 shat = {params.s_hat:.6f}
 akappa = {params.kappa:.6f}
 tri = {params.delta:.6f}
 rmaj = {R0_over_a:.6f}
 shift = 0.0
/
&species_knobs
 nspec = 2
/
&species_parameters_1
 z = 1
 mass = 1.0
 dens = 1.0
 temp = 1.0
 tprim = {params.R_L_Ti:.6f}
 fprim = {params.R_L_ne:.6f}
 type = 'ion'
/
&species_parameters_2
 z = -1
 mass = 2.7234e-4
 dens = 1.0
 temp = {params.Te_Ti:.6f}
 tprim = {params.R_L_Te:.6f}
 fprim = {params.R_L_ne:.6f}
 type = 'electron'
/
&kt_grids_knobs
 grid_option = 'single'
/
&kt_grids_single_parameters
 aky = 0.3
 theta0 = 0.0
/
"""


def parse_gs2_output(run_dir: Path) -> GKOutput:
    """Parse GS2 NetCDF or text output."""
    omega_file = run_dir / "gs2.omega"
    if omega_file.exists():
        try:
            data = np.loadtxt(omega_file)
            if data.ndim == 1 and len(data) >= 3:
                ky, gamma, omega_r = data[0], data[1], data[2]
                return GKOutput(
                    chi_i=max(float(gamma), 0.0),
                    chi_e=max(float(gamma) * 0.8, 0.0),
                    D_e=0.0,
                    gamma=np.array([gamma]),
                    omega_r=np.array([omega_r]),
                    k_y=np.array([ky]),
                    dominant_mode="ITG" if omega_r < 0 else "TEM",
                    converged=True,
                )
        except (ValueError, OSError) as exc:
            _logger.warning("GS2 parse error: %s", exc)

    return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)


class GS2Solver(GKSolverBase):
    """GS2 external solver."""

    def __init__(
        self,
        binary: str = "gs2",
        work_dir: Path | None = None,
        *,
        allow_fallback: bool = False,
        allow_legacy_fallback: bool = False,
    ) -> None:
        if allow_fallback and not allow_legacy_fallback:
            raise ValueError(
                "allow_fallback=True requires allow_legacy_fallback=True; legacy GS2 fallback is disabled by default."
            )
        self.binary = binary
        self.work_dir = work_dir
        self.allow_fallback = bool(allow_fallback)
        self.allow_legacy_fallback = bool(allow_legacy_fallback)

    def is_available(self) -> bool:
        """Return whether the GS2 binary is on the PATH."""
        return shutil.which(self.binary) is not None

    def prepare_input(self, params: GKLocalParams) -> Path:
        """Write the GS2 ``gs2.in`` deck and return its working directory.

        Parameters
        ----------
        params
            Local gyrokinetic input parameters.

        Returns
        -------
        Path
            The working directory containing ``gs2.in``.
        """
        base = self.work_dir or Path(tempfile.mkdtemp(prefix="gs2_"))
        base.mkdir(parents=True, exist_ok=True)
        (base / "gs2.in").write_text(generate_gs2_input(params))
        return base

    def run(self, input_path: Path, *, timeout_s: float = 30.0) -> GKOutput:
        """Execute GS2 on a prepared input directory.

        Fails closed when the binary is unavailable, the run fails, or the output
        is non-converged, unless the explicit legacy fallback is enabled.

        Parameters
        ----------
        input_path
            Working directory holding ``gs2.in``.
        timeout_s
            Subprocess timeout in seconds.

        Returns
        -------
        GKOutput
            The parsed GS2 result.

        Raises
        ------
        RuntimeError
            If GS2 is unavailable, fails, or returns non-converged output and the
            legacy fallback is disabled.
        """
        if not self.is_available():
            if not self.allow_fallback:
                raise RuntimeError(
                    "GS2 binary is unavailable; legacy fallback is disabled. "
                    "Set allow_fallback=True and allow_legacy_fallback=True for explicit "
                    "degraded-mode operation."
                )
            return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        try:
            subprocess.run(
                [self.binary, str(input_path / "gs2.in")],
                cwd=str(input_path),
                capture_output=True,
                timeout=timeout_s,
                check=True,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    "GS2 execution failed; legacy fallback is disabled. "
                    "Set allow_fallback=True and allow_legacy_fallback=True for explicit "
                    "degraded-mode operation."
                ) from exc
            return GKOutput(chi_i=0.0, chi_e=0.0, D_e=0.0, converged=False)
        result = parse_gs2_output(input_path)
        if not result.converged and not self.allow_fallback:
            raise RuntimeError(
                "GS2 completed without converged output; legacy fallback is disabled. "
                "Set allow_fallback=True and allow_legacy_fallback=True for explicit "
                "degraded-mode operation."
            )
        return result
