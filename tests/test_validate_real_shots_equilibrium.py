# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — real-shot equilibrium validation tests.

"""Tests for GEQDSK source-term equilibrium validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.eqdsk import GEqdsk
from validation.validate_real_shots import (
    MU0,
    _geqdsk_source_residual,
    _gs_operator,
    _guess_machine,
    nrmse,
    validate_equilibrium,
)


def test_nrmse_uses_range_and_zero_range_floor() -> None:
    """NRMSE is range-normalised and finite for a constant reference."""
    assert nrmse(np.array([0.0, 2.0]), np.array([0.0, 0.0])) == pytest.approx(np.sqrt(2.0) / 2.0)
    assert nrmse(np.ones(2), np.zeros(2)) == pytest.approx(1.0e12)


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
    """The discrete operator includes the cylindrical first-derivative term."""
    r = np.linspace(1.0, 3.0, 9)
    z = np.linspace(-1.0, 1.0, 9)
    rr, _ = np.meshgrid(r, z)

    lpsi = _gs_operator(rr**2, r, z)

    np.testing.assert_allclose(lpsi, 0.0, atol=1e-12)


def test_gs_operator_rejects_invalid_grid_contracts() -> None:
    """Shape, minimum-size, and monotonic-grid violations fail closed."""
    with pytest.raises(ValueError, match="shape must match"):
        _gs_operator(np.zeros((3, 4)), np.arange(3.0), np.arange(3.0))
    with pytest.raises(ValueError, match="at least 3x3"):
        _gs_operator(np.zeros((2, 2)), np.arange(2.0), np.arange(2.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        _gs_operator(np.zeros((3, 3)), np.array([0.0, 0.0, 1.0]), np.arange(3.0))


def test_geqdsk_source_residual_cancels_manufactured_ffprime() -> None:
    """A manufactured FF-prime source cancels the GS operator."""
    eq = _constant_source_equilibrium()

    residual_norm, source_norm, psi_norm, psi_range = _geqdsk_source_residual(eq)

    assert residual_norm < 1e-12
    assert source_norm == pytest.approx(2.0)
    assert psi_norm > 0.0
    assert psi_range == pytest.approx(1.0)


def test_geqdsk_source_residual_rejects_profile_length_mismatch() -> None:
    """Profile arrays must match the declared equilibrium grid width."""
    eq = _constant_source_equilibrium()
    eq.pprime = eq.pprime[:-1]

    with pytest.raises(ValueError, match="pprime length"):
        _geqdsk_source_residual(eq)


def test_geqdsk_source_residual_rejects_ffprime_and_flux_span_mismatch() -> None:
    """FF-prime length and degenerate flux spans are rejected independently."""
    eq = _constant_source_equilibrium()
    eq.ffprime = eq.ffprime[:-1]
    with pytest.raises(ValueError, match="ffprime length"):
        _geqdsk_source_residual(eq)

    eq = _constant_source_equilibrium()
    eq.sibry = eq.simag
    with pytest.raises(ValueError, match="degenerate psi range"):
        _geqdsk_source_residual(eq)


def test_geqdsk_source_residual_reconstructs_j_phi_scaling() -> None:
    """The reconstructed toroidal-current source keeps the expected scaling."""
    eq = _constant_source_equilibrium()
    r_inner = eq.r[1:-1]
    expected_j_phi = -2.0 / (MU0 * r_inner)

    _, source_norm, _, _ = _geqdsk_source_residual(eq)

    np.testing.assert_allclose(MU0 * r_inner * expected_j_phi, -2.0)
    assert source_norm == pytest.approx(2.0)


def test_equilibrium_lane_accepts_one_documented_low_current_outlier(tmp_path, monkeypatch) -> None:
    """The computational threshold permits one outlier in four local proxies."""

    def fake_read_geqdsk(path: str) -> GEqdsk:
        name = str(path)
        eq = _constant_source_equilibrium()
        if "sparc_1300" in name:
            eq.ffprime = np.ones(eq.nw)
        return eq

    for name in ["diiid_hmode_1p5MA.geqdsk", "diiid_hmode_2MA.geqdsk", "sparc_1300.eqdsk", "sparc_1349.eqdsk"]:
        (tmp_path / name).write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr("validation.validate_real_shots.read_geqdsk", fake_read_geqdsk)

    result = validate_equilibrium([tmp_path], evidence_class="local_proxy")

    assert result["n_files"] == 4
    assert result["n_psi_pass"] == 3
    assert result["psi_pass_fraction"] == pytest.approx(0.75)
    assert result["evidence_class"] == "local_proxy"
    assert result["data_provenance_pass"] is True
    assert result["computational_pass"] is True


def test_equilibrium_lane_records_reader_errors_and_empty_q_profile(tmp_path, monkeypatch) -> None:
    """Per-file reader failures are reported and empty q profiles remain bounded."""
    (tmp_path / "jet_empty_q.geqdsk").write_text("placeholder", encoding="utf-8")
    (tmp_path / "unknown_broken.geqdsk").write_text("placeholder", encoding="utf-8")

    def fake_read_geqdsk(path: str) -> GEqdsk:
        if "broken" in path:
            raise ValueError("malformed fixture")
        eq = _constant_source_equilibrium()
        eq.qpsi = np.array([], dtype=np.float64)
        return eq

    monkeypatch.setattr("validation.validate_real_shots.read_geqdsk", fake_read_geqdsk)
    result = validate_equilibrium([tmp_path], evidence_class="local_proxy")

    assert result["data_provenance_pass"] is False
    assert result["computational_pass"] is False
    assert np.isnan(result["results"][0]["q95"])
    assert result["results"][1]["error"] == "malformed fixture"
    assert _guess_machine(Path("jet/case.geqdsk")) == "JET"
    assert _guess_machine(Path("other/case.geqdsk")) == "unknown"


def test_equilibrium_lane_fails_closed_without_files(tmp_path) -> None:
    """An empty equilibrium source cannot pass provenance or computation."""
    result = validate_equilibrium([tmp_path], evidence_class="local_proxy")

    assert result["n_files"] == 0
    assert result["data_provenance_pass"] is False
    assert result["computational_pass"] is False
