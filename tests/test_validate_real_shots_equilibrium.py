# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — real-shot equilibrium validation tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Tests for GEQDSK source-term equilibrium validation."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.eqdsk import GEqdsk
from validation.validate_real_shots import MU0, _geqdsk_source_residual, _gs_operator, validate_equilibrium


def _constant_source_equilibrium(nw: int = 17, nh: int = 17) -> GEqdsk:
    """Build a manufactured equilibrium with Delta* psi + FF' = 0."""
    r = np.linspace(1.0, 3.0, nw)
    z = np.linspace(-1.0, 1.0, nh)
    _, zz = np.meshgrid(r, z)
    psi = zz**2
    return GEqdsk(
        description="manufactured source residual",
        nw=nw,
        nh=nh,
        rdim=float(r[-1] - r[0]),
        zdim=float(z[-1] - z[0]),
        rcentr=2.0,
        rleft=float(r[0]),
        zmid=0.0,
        rmaxis=2.0,
        zmaxis=0.0,
        simag=0.0,
        sibry=1.0,
        bcentr=5.0,
        current=1.0e6,
        fpol=np.ones(nw),
        pres=np.zeros(nw),
        ffprime=np.full(nw, -2.0),
        pprime=np.zeros(nw),
        qpsi=np.linspace(1.0, 3.0, nw),
        psirz=psi,
    )


def test_gs_operator_includes_toroidal_r_term() -> None:
    r = np.linspace(1.0, 3.0, 9)
    z = np.linspace(-1.0, 1.0, 9)
    rr, _ = np.meshgrid(r, z)

    lpsi = _gs_operator(rr**2, r, z)

    np.testing.assert_allclose(lpsi, 0.0, atol=1e-12)


def test_geqdsk_source_residual_cancels_manufactured_ffprime() -> None:
    eq = _constant_source_equilibrium()

    residual_norm, source_norm, psi_norm, psi_range = _geqdsk_source_residual(eq)

    assert residual_norm < 1e-12
    assert source_norm == pytest.approx(2.0)
    assert psi_norm > 0.0
    assert psi_range == pytest.approx(1.0)


def test_geqdsk_source_residual_rejects_profile_length_mismatch() -> None:
    eq = _constant_source_equilibrium()
    eq.pprime = eq.pprime[:-1]

    with pytest.raises(ValueError, match="pprime length"):
        _geqdsk_source_residual(eq)


def test_geqdsk_source_residual_reconstructs_j_phi_scaling() -> None:
    eq = _constant_source_equilibrium()
    r_inner = eq.r[1:-1]
    expected_j_phi = -2.0 / (MU0 * r_inner)

    _, source_norm, _, _ = _geqdsk_source_residual(eq)

    np.testing.assert_allclose(MU0 * r_inner * expected_j_phi, -2.0)
    assert source_norm == pytest.approx(2.0)


def test_equilibrium_lane_accepts_one_documented_low_current_outlier(tmp_path, monkeypatch) -> None:
    def fake_read_geqdsk(path: str) -> GEqdsk:
        name = str(path)
        eq = _constant_source_equilibrium()
        if "sparc_1300" in name:
            eq.ffprime = np.ones(eq.nw)
        return eq

    for name in ["diiid_hmode_1p5MA.geqdsk", "diiid_hmode_2MA.geqdsk", "sparc_1300.eqdsk", "sparc_1349.eqdsk"]:
        (tmp_path / name).write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr("validation.validate_real_shots.read_geqdsk", fake_read_geqdsk)

    result = validate_equilibrium([tmp_path])

    assert result["n_files"] == 4
    assert result["n_psi_pass"] == 3
    assert result["psi_pass_fraction"] == pytest.approx(0.75)
    assert result["passes"] is True
