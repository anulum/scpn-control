# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Analytic Solver
"""Analytic control-law helpers used by deterministic controller tests and examples."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

logger = logging.getLogger(__name__)

_PRESERVED_METADATA_KEYS = (
    "license",
    "spdx_license_id",
    "commercial_license",
    "concepts_copyright",
    "code_copyright",
    "orcid",
    "contact",
    "file",
)


def _require_positive_scalar(name: str, value: float) -> float:
    """Return a finite positive scalar or fail closed."""
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and > 0.")
    return scalar


def _require_nonnegative_scalar(name: str, value: float) -> float:
    """Return a finite non-negative scalar or fail closed."""
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.")
    return scalar


def _require_finite_scalar(name: str, value: float) -> float:
    """Return a finite scalar or fail closed."""
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


try:
    from scpn_control.core._rust_compat import FusionKernel
except ImportError:
    try:
        from scpn_control.core.fusion_kernel import FusionKernel
    except ImportError as exc:  # pragma: no cover - import-guard path
        raise ImportError(
            "Unable to import FusionKernel. Run with PYTHONPATH=src "
            "or use `python -m scpn_control.control.analytic_solver`."
        ) from exc


class AnalyticEquilibriumSolver:
    """
    Analytic vertical-field target and least-norm coil-current solve.
    """

    def __init__(
        self,
        config_path: str,
        *,
        kernel_factory: Callable[[str], Any] = FusionKernel,
        verbose: bool = True,
    ) -> None:
        self.kernel = kernel_factory(str(config_path))
        self.config_path = str(config_path)
        self.verbose = bool(verbose)

    def _log(self, message: str) -> None:
        if self.verbose:
            logger.info(message)

    def calculate_required_Bv(
        self,
        R_geo: float,
        a_min: float,
        Ip_MA: float,
        *,
        beta_p: float = 0.5,
        li: float = 0.8,
    ) -> float:
        """
        Shafranov radial-force balance vertical field estimate.
        """
        R_geo = _require_positive_scalar("R_geo", R_geo)
        a_min = _require_positive_scalar("a_min", a_min)
        Ip_MA = _require_positive_scalar("Ip_MA", Ip_MA)
        beta_p = _require_nonnegative_scalar("beta_p", beta_p)
        li = _require_nonnegative_scalar("li", li)
        if R_geo <= a_min:
            raise ValueError("R_geo must exceed a_min for tokamak aspect geometry.")

        mu0 = 4.0 * np.pi * 1e-7
        Ip = Ip_MA * 1e6
        term_log = np.log(8.0 * R_geo / a_min)
        term_physics = beta_p + (li / 2.0) - 1.5
        Bv = -((mu0 * Ip) / (4.0 * np.pi * R_geo)) * (term_log + term_physics)

        self._log("--- SHAFRANOV EQUILIBRIUM CHECK ---")
        self._log(f"Target Radius: {R_geo:.3f} m")
        self._log(f"Plasma Current: {Ip_MA:.3f} MA")
        self._log(f"Required Vertical Field (Bv): {Bv:.6f} Tesla")
        return float(Bv)

    def compute_coil_efficiencies(
        self,
        target_R: float,
        *,
        target_Z: float = 0.0,
    ) -> FloatArray:
        """
        Compute dBz/dI per coil at target location using kernel vacuum-field map.
        """
        coils = self.kernel.cfg.get("coils", [])
        n_coils = len(coils)
        if n_coils == 0:
            raise ValueError("Kernel config has no coils.")

        target_R = _require_positive_scalar("target_R", target_R)
        target_Z = _require_finite_scalar("target_Z", target_Z)
        r_grid = np.asarray(self.kernel.R, dtype=np.float64)
        z_grid = np.asarray(self.kernel.Z, dtype=np.float64)
        if r_grid.ndim != 1 or r_grid.size < 3 or not np.all(np.isfinite(r_grid)):
            raise ValueError("kernel R grid must be finite with at least 3 points.")
        if z_grid.ndim != 1 or z_grid.size < 1 or not np.all(np.isfinite(z_grid)):
            raise ValueError("kernel Z grid must be finite with at least 1 point.")
        r_steps = np.diff(r_grid)
        if not (np.all(r_steps > 0.0) or np.all(r_steps < 0.0)):
            raise ValueError("kernel R grid must be strictly monotonic.")
        if not np.allclose(r_steps, r_steps[0], rtol=1e-9, atol=1e-12):
            raise ValueError("kernel R grid must be uniformly spaced.")
        r_min = float(min(r_grid[1], r_grid[-2]))
        r_max = float(max(r_grid[1], r_grid[-2]))
        if not (r_min <= target_R <= r_max):
            raise ValueError("target_R must lie inside the kernel R grid interior.")
        z_min = float(np.min(z_grid))
        z_max = float(np.max(z_grid))
        if not (z_min <= target_Z <= z_max):
            raise ValueError("target_Z must lie inside the kernel Z grid.")

        original_currents = [float(c.get("current", 0.0)) for c in coils]
        eff = np.zeros(n_coils, dtype=np.float64)

        idx_r = int(np.argmin(np.abs(r_grid - target_R)))
        idx_z = int(np.argmin(np.abs(z_grid - target_Z)))
        idx_r = int(np.clip(idx_r, 1, len(self.kernel.R) - 2))
        dR = float(getattr(self.kernel, "dR", float(self.kernel.R[1] - self.kernel.R[0])))
        if dR <= 0.0:
            raise ValueError("Kernel grid spacing dR must be > 0.")

        self._log("\nCalculating Coil Influence Matrix (Green's Functions)...")
        try:
            for i in range(n_coils):
                for c in coils:
                    c["current"] = 0.0
                coils[i]["current"] = 1.0

                psi_vac = np.asarray(self.kernel.calculate_vacuum_field(), dtype=np.float64)
                expected_shape = (len(self.kernel.Z), len(self.kernel.R))
                if psi_vac.shape != expected_shape:
                    raise ValueError("vacuum field shape must match kernel grid.")
                if not np.all(np.isfinite(psi_vac)):
                    raise ValueError("vacuum field must contain only finite values.")
                dpsi = (psi_vac[idx_z, idx_r + 1] - psi_vac[idx_z, idx_r - 1]) / (2.0 * dR)
                bz_unit = float((1.0 / target_R) * dpsi)
                eff[i] = bz_unit

                name = str(coils[i].get("name", f"coil_{i}"))
                self._log(f"  Coil {name} Efficiency: {bz_unit:.6f} T/MA")
        finally:
            for c, current in zip(coils, original_currents):
                c["current"] = float(current)

        return eff

    def solve_coil_currents(
        self,
        target_Bv: float,
        target_R: float,
        *,
        target_Z: float = 0.0,
        ridge_lambda: float = 0.0,
    ) -> FloatArray:
        """
        Solve least-norm coil currents for desired vertical field target.
        """
        eff = self.compute_coil_efficiencies(target_R, target_Z=target_Z)
        target_Bv = _require_finite_scalar("target_Bv", target_Bv)
        ridge_lambda = _require_finite_scalar("ridge_lambda", ridge_lambda)
        ridge_lambda = max(ridge_lambda, 0.0)
        if not np.all(np.isfinite(eff)):
            raise ValueError("coil influence matrix must contain only finite values.")
        influence_norm_sq = float(np.dot(eff, eff))
        if influence_norm_sq <= 1e-24:
            if abs(target_Bv) > 1e-12:
                raise ValueError("nonzero target_Bv requires nonzero coil influence.")
            return np.zeros_like(eff, dtype=np.float64)

        g = eff.reshape(1, -1)
        if ridge_lambda > 0.0:
            denom = max(influence_norm_sq + ridge_lambda, 1e-12)
            currents = (eff * target_Bv) / denom
        else:
            currents = np.linalg.pinv(g).dot(np.array([target_Bv], dtype=np.float64)).reshape(-1)

        self._log("\n--- ANALYTIC SOLUTION (Least Norm) ---")
        for i, val in enumerate(currents):
            name = str(self.kernel.cfg["coils"][i].get("name", f"coil_{i}"))
            self._log(f"  {name}: {float(val):.6f} MA")
        return np.asarray(currents, dtype=np.float64)

    def apply_currents(self, currents: AnyFloatArray) -> None:
        """Set the kernel coil currents from a current vector.

        Parameters
        ----------
        currents
            Coil currents, one per kernel coil; must be finite and match the
            coil count.

        Raises
        ------
        ValueError
            If the length mismatches the coils or any value is non-finite.
        """
        arr = np.asarray(currents, dtype=np.float64).reshape(-1)
        coils = self.kernel.cfg.get("coils", [])
        if arr.size != len(coils):
            raise ValueError("Current vector length mismatch with kernel coils.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("currents must contain only finite values.")
        for i, val in enumerate(arr):
            coils[i]["current"] = float(val)

    def apply_and_save(
        self,
        currents: AnyFloatArray,
        output_path: str | None = None,
    ) -> str:
        """Apply coil currents and save the resulting kernel config to JSON.

        Parameters
        ----------
        currents
            Coil currents to apply (see :meth:`apply_currents`).
        output_path
            Destination JSON path; defaults to
            ``validation/iter_analytic_config.json`` under the repo root.
            Preserved metadata keys in an existing file are retained.

        Returns
        -------
        str
            The path to the written configuration file.
        """
        self.apply_currents(currents)
        if output_path is None:
            repo_root = Path(__file__).resolve().parents[3]
            out_path = repo_root / "validation" / "iter_analytic_config.json"
        else:
            out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = dict(self.kernel.cfg)
        if out_path.exists():
            with out_path.open(encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                preserved = {
                    key: existing[key] for key in _PRESERVED_METADATA_KEYS if key in existing and key not in cfg
                }
                cfg = {**preserved, **cfg}
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
            f.write("\n")
        self._log(f"Saved analytic configuration: {out_path}")
        return str(out_path)


def run_analytic_solver(
    config_path: str | None = None,
    *,
    target_r: float = 6.2,
    target_z: float = 0.0,
    a_minor: float = 2.0,
    ip_target_ma: float = 15.0,
    beta_p: float = 0.5,
    li: float = 0.8,
    ridge_lambda: float = 0.0,
    save_config: bool = True,
    output_config_path: str | None = None,
    verbose: bool = True,
    allow_config_fallback: bool = False,
    allow_legacy_config_fallback: bool = False,
    kernel_factory: Callable[[str], Any] = FusionKernel,
) -> Dict[str, Any]:
    """
    Run analytic equilibrium solve and return deterministic summary.
    """
    if allow_config_fallback and not allow_legacy_config_fallback:
        raise ValueError(
            "allow_config_fallback=True requires allow_legacy_config_fallback=True; "
            "legacy analytic config fallback is disabled by default."
        )

    repo_root = Path(__file__).resolve().parents[3]
    if config_path is None:
        preferred = repo_root / "calibration" / "iter_genetic_temp.json"
        fallback = repo_root / "validation" / "iter_validated_config.json"
        if preferred.exists():
            config_path = str(preferred)
        elif allow_config_fallback:
            if not fallback.exists():
                raise FileNotFoundError(
                    f"Legacy analytic config fallback was enabled but no validated config exists at {fallback}."
                )
            config_path = str(fallback)
        else:
            raise FileNotFoundError(
                "Default analytic configuration is missing at "
                f"{preferred}. Provide config_path explicitly or set both "
                "allow_config_fallback=True and allow_legacy_config_fallback=True."
            )

    solver = AnalyticEquilibriumSolver(
        str(config_path),
        kernel_factory=kernel_factory,
        verbose=verbose,
    )
    required_bv = solver.calculate_required_Bv(
        target_r,
        a_minor,
        ip_target_ma,
        beta_p=beta_p,
        li=li,
    )
    currents = solver.solve_coil_currents(
        required_bv,
        target_r,
        target_Z=target_z,
        ridge_lambda=ridge_lambda,
    )

    written_path: str | None = None
    if save_config:
        written_path = solver.apply_and_save(currents, output_path=output_config_path)
    else:
        solver.apply_currents(currents)

    names = [str(c.get("name", f"coil_{i}")) for i, c in enumerate(solver.kernel.cfg["coils"])]
    summary_currents = {name: float(currents[i]) for i, name in enumerate(names)}
    return {
        "config_path": str(config_path),
        "output_config_path": written_path,
        "target_r_m": float(target_r),
        "target_z_m": float(target_z),
        "a_minor_m": float(a_minor),
        "ip_target_ma": float(ip_target_ma),
        "required_bv_t": float(required_bv),
        "coil_currents_ma": summary_currents,
        "coil_current_l2_norm": float(np.linalg.norm(currents)),
        "max_abs_coil_current_ma": float(np.max(np.abs(currents))) if currents.size else 0.0,
    }


if __name__ == "__main__":
    run_analytic_solver()
