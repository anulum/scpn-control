# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Equilibrium execution evidence.

"""Retain configured-unit evidence from an actual hash-bound fixed-boundary solve.

Numerical evidence does not establish a machine's physical identity or an
external reference comparison. Nonfinite output is refused before JSON return.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scpn_control.core.fusion_kernel import FusionKernel


@dataclass(frozen=True)
class EquilibriumExecution:
    """Run a hash-bound equilibrium configuration and retain numerical evidence.

    This measures configured-unit solver behaviour. It does not infer a physical
    machine from a reactor name or compare against an external equilibrium.
    """

    config_path: Path
    config_sha256: str
    reactor_name: str
    max_gs_rms: float

    def evaluate(self, scratch_directory: Path) -> str:
        """Execute the fixed-boundary solver on a captured configuration.

        Parameters
        ----------
        scratch_directory
            Existing writable directory for an isolated temporary config copy.
            The copy is removed after execution; source files are untouched.

        Returns
        -------
        str
            JSON with exact config bytes, effective config, returned flux/current
            fields and grids, solver histories and independently recomputed GS
            RMS. Numerical acceptance requires reported convergence and the
            declared RMS bound. All physical/action authority remains false.

        Raises
        ------
        ValueError
            Hash, identity, threshold or fixed-boundary configuration is invalid.
        RuntimeError
            Solver failure or nonfinite output prevents valid evidence. No
            passing placeholder is returned; the underlying error propagates.

        Notes
        -----
        A digest identifies bytes, not the numerical validity of every kernel
        control. After actual solver execution, nonfinite flux/current/grids or
        independently computed GS RMS are refused before returning any JSON.
        Solver convergence alone cannot bypass these output checks. Kernel
        configuration/parser and solver errors propagate without replacement.
        """
        if not self.reactor_name.strip():
            raise ValueError("reactor_name must be explicit")
        if isinstance(self.max_gs_rms, (bool, np.bool_)) or not np.isfinite(self.max_gs_rms) or self.max_gs_rms <= 0:
            raise ValueError("max_gs_rms must be finite and positive")
        raw = self.config_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.config_sha256:
            raise ValueError("equilibrium configuration SHA256 mismatch")
        with tempfile.TemporaryDirectory(prefix="equilibrium-", dir=scratch_directory) as temporary:
            captured = Path(temporary) / "config.json"
            captured.write_bytes(raw)
            kernel = FusionKernel(captured)
            if kernel.cfg["reactor_name"] != self.reactor_name:
                raise ValueError("equilibrium reactor identity mismatch")
            if kernel.boundary_variant != "fixed_boundary" or kernel.cfg["solver"]["max_iterations"] <= 0:
                raise ValueError("require a fixed-boundary configuration with positive iteration budget")
            result = kernel.solve_fixed_boundary()
        psi = np.asarray(result["psi"], dtype=np.float64)
        current = np.asarray(kernel.J_phi, dtype=np.float64)
        r, z = kernel.R, kernel.Z
        if not all(np.all(np.isfinite(array)) for array in (psi, current, r, z)):
            raise RuntimeError("equilibrium returned nonfinite fields")
        dr, dz = float(r[1] - r[0]), float(z[1] - z[0])
        radial = ((psi[1:-1, 2:] - psi[1:-1, 1:-1]) / dr - (psi[1:-1, 1:-1] - psi[1:-1, :-2]) / dr) / dr
        vertical = ((psi[2:, 1:-1] - psi[1:-1, 1:-1]) / dz - (psi[1:-1, 1:-1] - psi[:-2, 1:-1]) / dz) / dz
        gradient = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dr)
        permeability = kernel.cfg["physics"].get("vacuum_permeability", 1.0)
        source = -permeability * r[None, 1:-1] * current[1:-1, 1:-1]
        residual = radial + vertical - gradient / r[None, 1:-1] - source
        rms = float(np.linalg.norm(residual) / np.sqrt(residual.size))
        if not np.isfinite(rms):
            raise RuntimeError("equilibrium GS residual is not finite")
        evidence = {
            "schema": "scpn-control.equilibrium-execution.v1",
            "config_sha256": self.config_sha256,
            "config_utf8": raw.decode("utf-8"),
            "effective_config": kernel.cfg,
            "model": "scpn_control.core.fusion_kernel.FusionKernel.solve_fixed_boundary",
            "solver_result": {key: value for key, value in result.items() if key != "psi"},
            "fields": {"psi": psi.tolist(), "J_phi": current.tolist(), "R": r.tolist(), "Z": z.tolist()},
            "independent_gs_rms": rms,
            "max_gs_rms": self.max_gs_rms,
            "numerical_pass": bool(result["converged"]) and rms <= self.max_gs_rms,
            "unit_convention": "configuration-defined; no SI conversion inferred",
            "external_reference_compared": False,
            "actionable": False,
            "facility_validated": False,
        }
        return json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
