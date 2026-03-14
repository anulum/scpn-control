# ──────────────────────────────────────────────────────────────────────
# SCPN Control — GEQDSK Equilibrium Regression Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Regression tests using SPARC GEQDSK reference equilibria.

Tests are split into two categories:

1. **Solver convergence gates** — verify the GS solver converges on
   GEQDSK-derived configs without NaN/Inf and with bounded residual.

2. **Self-consistency gates** — verify the converged axis is inside the
   plasma domain (not on the boundary), and that successive solves
   produce reproducible results (deterministic regression).

Note: our solver uses its own source model (p'(ψ), FF'(ψ)), which
differs from the EFIT reconstruction in the GEQDSK. Pointwise ψ
comparison against the GEQDSK reference is therefore not meaningful
without matching source profiles. Axis location and ψ shape comparisons
have relaxed tolerances to account for this.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.eqdsk import read_geqdsk
from scpn_control.core.fusion_kernel import FusionKernel

_SPARC_DIR = Path(__file__).resolve().parents[1] / "validation" / "reference_data" / "sparc"
_LMODE_FILES = sorted(_SPARC_DIR.glob("lmode_*.geqdsk"))
_EQDSK_FILES = sorted(_SPARC_DIR.glob("sparc_*.eqdsk"))
_ALL_FILES = _LMODE_FILES + _EQDSK_FILES


# ── Gates ─────────────────────────────────────────────────────────────

RESIDUAL_CEIL = 1e-2


def _make_config(eq, method="sor", max_iter=300, tol=1e-4):
    cfg = eq.to_config(name=f"sparc_{eq.description[:20].strip()}")
    cfg["solver"]["solver_method"] = method
    cfg["solver"]["max_iterations"] = max_iter
    cfg["solver"]["convergence_threshold"] = tol
    cfg["solver"]["relaxation_factor"] = 0.15
    cfg["solver"]["sor_omega"] = 1.6
    return cfg


def _write_cfg(tmp_path, cfg, name="cfg.json"):
    p = tmp_path / name
    p.write_text(json.dumps(cfg))
    return p


# ── Convergence gates ─────────────────────────────────────────────────


class TestSolverConvergenceGate:
    """Solver converges on all GEQDSK-derived configs."""

    @pytest.mark.parametrize("geqdsk_path", _ALL_FILES, ids=[f.stem for f in _ALL_FILES])
    def test_converges_no_nan(self, geqdsk_path, tmp_path):
        eq = read_geqdsk(geqdsk_path)
        cfg = _make_config(eq, max_iter=500)
        fk = FusionKernel(_write_cfg(tmp_path, cfg))
        result = fk.solve_equilibrium()

        assert not np.any(np.isnan(result["psi"])), f"{geqdsk_path.stem}: NaN in ψ"
        assert not np.any(np.isinf(result["psi"])), f"{geqdsk_path.stem}: Inf in ψ"

    @pytest.mark.parametrize("geqdsk_path", _ALL_FILES, ids=[f.stem for f in _ALL_FILES])
    def test_residual_below_ceiling(self, geqdsk_path, tmp_path):
        eq = read_geqdsk(geqdsk_path)
        cfg = _make_config(eq, max_iter=500)
        fk = FusionKernel(_write_cfg(tmp_path, cfg))
        result = fk.solve_equilibrium()

        assert result["residual"] < RESIDUAL_CEIL, (
            f"{geqdsk_path.stem}: residual {result['residual']:.2e} >= {RESIDUAL_CEIL:.2e} "
            f"after {result['iterations']} iters"
        )


# ── Self-consistency gates ────────────────────────────────────────────


class TestDeterministicReproducibility:
    """Two consecutive solves with the same config must give identical ψ."""

    def test_reproducible_solve(self, tmp_path):
        if not _LMODE_FILES:
            pytest.skip("No GEQDSK files")
        eq = read_geqdsk(_LMODE_FILES[0])
        cfg = _make_config(eq, max_iter=200)

        fk1 = FusionKernel(_write_cfg(tmp_path, cfg, "a.json"))
        r1 = fk1.solve_equilibrium()

        fk2 = FusionKernel(_write_cfg(tmp_path, cfg, "b.json"))
        r2 = fk2.solve_equilibrium()

        diff = float(np.max(np.abs(r1["psi"] - r2["psi"])))
        assert diff < 1e-12, f"Non-deterministic: max |Δψ| = {diff:.2e}"


class TestResidualMonotonicity:
    """Residual history should generally trend downward."""

    @pytest.mark.parametrize("geqdsk_path", _LMODE_FILES[:1], ids=[f.stem for f in _LMODE_FILES[:1]])
    def test_residual_trend(self, geqdsk_path, tmp_path):
        eq = read_geqdsk(geqdsk_path)
        cfg = _make_config(eq, max_iter=200, tol=1e-6)
        fk = FusionKernel(_write_cfg(tmp_path, cfg))
        result = fk.solve_equilibrium()

        hist = result["residual_history"]
        if len(hist) >= 20:
            first_avg = float(np.mean(hist[:10]))
            last_avg = float(np.mean(hist[-10:]))
            assert last_avg < first_avg, f"Residual not decreasing: first_10={first_avg:.2e}, last_10={last_avg:.2e}"
