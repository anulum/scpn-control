# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary tracking claim-admission benchmark

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scpn_control.control.free_boundary_tracking import (
    free_boundary_tracking_claim_evidence,
    run_free_boundary_tracking,
    save_free_boundary_tracking_claim_evidence,
)
from scpn_control.core.fusion_kernel import CoilSet

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "free_boundary_tracking_claims.json"
MARKDOWN_REPORT = REPORT_DIR / "free_boundary_tracking_claims.md"


class _BenchmarkFreeBoundaryKernel:
    """Deterministic free-boundary plant for claim-admission benchmarking."""

    def __init__(self, _config_file: str) -> None:
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
        self.cfg = {
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

    def _sample_flux_at_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape == self._boundary_points.shape and np.allclose(pts, self._boundary_points):
            return self._state[:3].copy()
        if pts.shape == self._divertor_points.shape and np.allclose(pts, self._divertor_points):
            return self._state[6:].copy()
        raise ValueError("unexpected benchmark probe points")

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], float]:
        return (float(self._state[3]), float(self._state[4])), float(self._state[5])

    def _interp_psi(self, r_pt: float, z_pt: float) -> float:
        if np.allclose([r_pt, z_pt], self._x_target):
            return float(self._state[5])
        return float(np.mean(self._state[:3]))


def main() -> None:
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
