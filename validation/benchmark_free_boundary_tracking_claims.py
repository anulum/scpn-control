# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary tracking claim-admission benchmark

"""Publish controller plumbing evidence using the shipped linear response fixture."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from scpn_control.benchmark_records import require_recorded_campaign
from scpn_control.control.free_boundary_tracking import run_free_boundary_tracking
from scpn_control.control.free_boundary_tracking_claims import (
    free_boundary_tracking_claim_evidence,
    save_free_boundary_tracking_claim_evidence,
)
from scpn_control.core.fusion_kernel import CoilSet

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "free_boundary_tracking_claims.json"
MARKDOWN_REPORT = REPORT_DIR / "free_boundary_tracking_claims.md"


class _BenchmarkFreeBoundaryKernel:
    """Fixed linear fixture implementing the controller's kernel protocol.

    Parameters
    ----------
    _config_file : str
        Accepted for factory compatibility and ignored; no file is opened.

    Notes
    -----
    The state has eight entries: three boundary fluxes, X-position R/Z,
    X-flux and two divertor fluxes. Coordinates follow metres, fluxes follow
    the consumer Wb/rad convention and currents follow MA; these are fixture
    conventions without an independently calibrated magnetic field. A fixed
    8-by-4 response matrix maps four currents onto state and a fixed bias.
    The 8-by-8 Psi grid remains zero. There is no Grad-Shafranov solve,
    equilibrium null search or interpolation from that grid. Each instance
    owns mutable arrays/configuration; sharing it requires caller coordination.
    """

    def __init__(self, _config_file: str) -> None:
        """Allocate this instance's preset arrays/configuration and initialise its state."""
        self._boundary_points = np.array([[3.4, -0.1], [4.0, 0.3], [4.6, -0.4]], dtype=np.float64)
        self._divertor_points = np.array([[3.1, -2.6], [4.9, -2.6]], dtype=np.float64)
        self._x_target = np.array([4.2, -1.4], dtype=np.float64)
        self._target_vector = np.array([0.12, 0.18, 0.10, 4.2, -1.4, 0.15, 0.15, 0.15], dtype=np.float64)
        self._response_matrix = np.array(
            [
                [0.60, -0.20, 0.15, 0.05],
                [0.10, 0.55, -0.15, 0.05],
                [-0.20, 0.10, 0.50, 0.10],
                [0.12, -0.08, 0.03, 0.01],
                [-0.04, 0.02, 0.11, -0.09],
                [0.22, 0.10, -0.05, 0.04],
                [0.14, -0.12, 0.18, 0.05],
                [0.11, 0.09, -0.04, 0.16],
            ],
            dtype=np.float64,
        )
        self._bias = self._response_matrix @ np.array([0.45, -0.35, 0.30, -0.20], dtype=np.float64)
        self.cfg: dict[str, Any] = {
            "physics": {"drift_scale": 0.0},
            "coils": [
                {"name": "PF1", "current": 0.0},
                {"name": "PF2", "current": 0.0},
                {"name": "PF3", "current": 0.0},
                {"name": "PF4", "current": 0.0},
            ],
            "free_boundary": {
                "objective_tolerances": {
                    "shape_rms": 0.025,
                    "x_point_position": 0.08,
                    "x_point_flux": 0.03,
                    "divertor_rms": 0.025,
                }
            },
            "free_boundary_tracking": {"measurement_latency_steps": 1, "latency_compensation_gain": 0.75},
        }
        self.R = np.linspace(3.0, 5.2, 8)
        self.Z = np.linspace(-3.0, 1.0, 8)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self._state = self._target_vector + self._bias
        self.solve()

    def build_coilset_from_config(self) -> CoilSet:
        """Return a fresh four-coil fixture, independent of cfg current mutations.

        Returns
        -------
        CoilSet
            Positions in metres, zero MA currents, 12 turns each and 3 MA
            limits. Boundary/divertor targets and X-position use copied arrays.

        Notes
        -----
        Despite the method name, cfg is not read. Every call resets returned
        currents to zero and preserves this instance's fixed target values.
        """
        return CoilSet(
            positions=[(3.0, 2.2), (3.6, -2.1), (4.4, 2.0), (5.0, -2.2)],
            currents=np.zeros(4, dtype=np.float64),
            turns=[12, 12, 12, 12],
            current_limits=np.full(4, 3.0, dtype=np.float64),
            target_flux_points=self._boundary_points.copy(),
            target_flux_values=self._target_vector[:3].copy(),
            x_point_target=self._x_target.copy(),
            x_point_flux_target=float(self._target_vector[5]),
            divertor_strike_points=self._divertor_points.copy(),
            divertor_flux_values=self._target_vector[6:].copy(),
        )

    def solve(
        self,
        *,
        boundary_variant: str | None = None,
        coils: CoilSet | None = None,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        optimize_shape: bool = False,
        tikhonov_alpha: float = 1e-4,
    ) -> dict[str, float | bool | str]:
        """Apply the fixed linear current response, without iterative solving.

        Parameters
        ----------
        boundary_variant : str or None
            Report label only; None emits free_boundary.
        coils : CoilSet or None
            Four current entries, in fixture MA. None builds zero currents.
        max_outer_iter, tol, optimize_shape, tikhonov_alpha
            Accepted for protocol compatibility and ignored.

        Returns
        -------
        dict[str, float | bool | str]
            Unconditional converged=True, outer_iterations=1 and final_diff
            equal to the norm of the linear current contribution, not a PDE
            residual or convergence test.

        Raises
        ------
        ValueError
            Current conversion or incompatible four-column matrix dimensions
            fail in NumPy.

        Notes
        -----
        Replace the eight-entry state, mutate cfg coil currents and zero Psi.
        No input-domain, bound, finiteness or tolerance validation is added.
        The controller applies its own current limits outside this method.
        """
        active_coils = coils if coils is not None else self.build_coilset_from_config()
        currents = np.asarray(active_coils.currents, dtype=np.float64).reshape(-1)
        self._state = self._target_vector + self._bias + self._response_matrix @ currents
        for idx, current in enumerate(currents):
            self.cfg["coils"][idx]["current"] = float(current)
        self.Psi.fill(0.0)
        return {
            "boundary_variant": "free_boundary" if boundary_variant is None else str(boundary_variant),
            "converged": True,
            "outer_iterations": 1,
            "final_diff": float(np.linalg.norm(self._response_matrix @ currents)),
        }

    def _sample_flux_at_points(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Copy stored boundary or divertor values for the exact fixture probes.

        Parameters
        ----------
        points : numpy.ndarray
            Three boundary or two divertor (R, Z) pairs in fixture metres.

        Returns
        -------
        numpy.ndarray
            Fresh length-three or length-two state slice in fixture flux units.

        Raises
        ------
        ValueError
            Converted points match neither preset shape/NumPy allclose test.

        Notes
        -----
        No grid interpolation is performed; NumPy allclose defaults apply.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape == self._boundary_points.shape and np.allclose(pts, self._boundary_points):
            return self._state[:3].copy()
        if pts.shape == self._divertor_points.shape and np.allclose(pts, self._divertor_points):
            return self._state[6:].copy()
        raise ValueError("unexpected benchmark probe points")

    def find_x_point(self, _psi: npt.NDArray[np.float64]) -> tuple[tuple[float, float], float]:
        """Return the stored X-position and flux without inspecting _psi.

        Parameters
        ----------
        _psi : numpy.ndarray
            Ignored protocol-compatible input.

        Returns
        -------
        tuple[tuple[float, float], float]
            Stored (R, Z) in fixture metres and stored fixture flux.
        """
        return (float(self._state[3]), float(self._state[4])), float(self._state[5])

    def _interp_psi(self, r_pt: float, z_pt: float) -> float:
        """Return fixture X-flux at its target, otherwise mean boundary flux.

        Parameters
        ----------
        r_pt, z_pt : float
            Requested position in fixture metres; NumPy allclose defaults
            decide whether it matches the X-target.

        Returns
        -------
        float
            Stored fixture flux; Psi and grid interpolation are unused.
        """
        if np.allclose([r_pt, z_pt], self._x_target):
            return float(self._state[5])
        return float(np.mean(self._state[:3]))


def main() -> None:
    """Write bounded five-step tracking evidence from the shipped linear fixture.

    Returns
    -------
    None
        Write free_boundary_tracking_claims.json, its Markdown sibling and
        then the same JSON again under this module's reports directory.

    Raises
    ------
    RuntimeError
        Persistent destinations lack a recorded-campaign identifier.
    ValueError
        Identifier syntax, controller inputs or claim evidence is refused.
    OSError
        Directory creation or a sequential UTF-8 write fails; earlier output
        may remain. There is no atomic replacement or multi-file transaction.

    Notes
    -----
    No CLI parameters are parsed. A fresh fixture feeds the actual public
    controller for five steps, gain 0.5, without early convergence stopping.
    Its config_path is only a report label: the fixture ignores that file.
    One-step measurement latency and gain-0.75 compensation are configured.
    Currents use fixture MA and coordinates metres. Residual flux follows the
    consumer convention without independently calibrated physical units.
    The fixture runs no magnetic equilibrium/PDE solve. No reference artifact
    is supplied and facility_claim_allowed stays False. State is local but
    output names are shared without locks. The destination guard checks
    identifier presence/syntax; recorded custody is provided by the wrapper.
    """
    require_recorded_campaign(JSON_REPORT, MARKDOWN_REPORT, repository_root=REPORT_DIR.parents[1])
    summary = run_free_boundary_tracking(
        "validation/free_boundary_tracking_claims_fixture.json",
        kernel_factory=_BenchmarkFreeBoundaryKernel,
        shot_steps=5,
        gain=0.5,
        verbose=False,
        stop_on_convergence=False,
    )
    evidence = free_boundary_tracking_claim_evidence(
        summary,
        source="repository_free_boundary_regression",
        source_id="free-boundary-tracking-claim-benchmark-v1",
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_free_boundary_tracking_claim_evidence(evidence, JSON_REPORT)
    payload = asdict(evidence)
    MARKDOWN_REPORT.write_text(
        "\n".join(
            [
                "# Free-boundary Tracking Claim-Admission Benchmark",
                "",
                "This report records bounded kernel-in-loop evidence for the",
                "free-boundary tracking claim boundary. It captures objective",
                "residuals, true hidden residuals, response-rank health, actuator",
                "bounds, latency compensation status, supervisor actions, and the",
                "explicit facility-claim boundary.",
                "",
                "This fixture is a fixed 8-by-4 linear response model; no Grad-Shafranov solve is run.",
                "",
                f"- Claim status: `{payload['claim_status']}`",
                f"- Facility claim allowed: `{payload['facility_claim_allowed']}`",
                f"- Steps: `{payload['steps']}`",
                f"- True shape RMS: `{payload['true_shape_rms']:.12g}`",
                f"- True X-point position error: `{payload['true_x_point_position_error_m']:.12g}` m",
                f"- True X-point flux error: `{payload['true_x_point_flux_error']:.12g}`",
                f"- True divertor RMS: `{payload['true_divertor_rms']:.12g}`",
                f"- Minimum response rank: `{payload['min_response_rank']}`",
                f"- Response degeneracy count: `{payload['response_degenerate_count']}`",
                "",
                "Bounded repository regression evidence is not commissioned facility-control validation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
