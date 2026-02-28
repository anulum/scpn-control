# ──────────────────────────────────────────────────────────────────────
# SCPN Control — SPI Mitigation Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Deterministic regression tests for SPI mitigation reduced models."""

from __future__ import annotations

import numpy as np

from scpn_control.control.spi_mitigation import ShatteredPelletInjection


def test_estimate_z_eff_increases_with_neon_quantity() -> None:
    low = ShatteredPelletInjection.estimate_z_eff(0.02)
    high = ShatteredPelletInjection.estimate_z_eff(0.20)
    assert low >= 1.0
    assert high > low


def test_estimate_z_eff_cocktail_reflects_species_mix() -> None:
    neon_only = ShatteredPelletInjection.estimate_z_eff_cocktail(neon_quantity_mol=0.12)
    argon_only = ShatteredPelletInjection.estimate_z_eff_cocktail(
        argon_quantity_mol=0.12
    )
    xenon_only = ShatteredPelletInjection.estimate_z_eff_cocktail(
        xenon_quantity_mol=0.12
    )
    assert neon_only >= 1.0
    assert argon_only > neon_only
    assert xenon_only > argon_only


def test_estimate_tau_cq_is_iter_like_for_cold_impure_plasma() -> None:
    tau = ShatteredPelletInjection.estimate_tau_cq(te_keV=0.1, z_eff=2.0)
    assert 0.015 <= tau <= 0.025

    tau_hot = ShatteredPelletInjection.estimate_tau_cq(te_keV=0.5, z_eff=2.0)
    tau_impure = ShatteredPelletInjection.estimate_tau_cq(te_keV=0.1, z_eff=4.0)
    assert tau_hot > tau
    assert tau_impure < tau


def test_estimate_mitigation_cocktail_shifts_to_heavier_species_with_risk() -> None:
    low = ShatteredPelletInjection.estimate_mitigation_cocktail(
        risk_score=0.2,
        disturbance=0.2,
        action_bias=0.0,
    )
    high = ShatteredPelletInjection.estimate_mitigation_cocktail(
        risk_score=0.9,
        disturbance=0.9,
        action_bias=1.0,
    )
    assert np.isclose(
        low["total_quantity_mol"],
        low["neon_quantity_mol"]
        + low["argon_quantity_mol"]
        + low["xenon_quantity_mol"],
    )
    assert high["xenon_quantity_mol"] > low["xenon_quantity_mol"]
    assert high["total_quantity_mol"] >= low["total_quantity_mol"]


def test_trigger_mitigation_returns_finite_histories_and_diagnostics() -> None:
    spi = ShatteredPelletInjection(Plasma_Energy_MJ=300.0, Plasma_Current_MA=15.0)
    t, w, i, diag = spi.trigger_mitigation(
        neon_quantity_mol=0.1,
        argon_quantity_mol=0.02,
        xenon_quantity_mol=0.01,
        return_diagnostics=True,
    )
    assert len(t) == len(w) == len(i)
    assert len(t) > 100
    assert np.isfinite(np.asarray(w, dtype=float)).all()
    assert np.isfinite(np.asarray(i, dtype=float)).all()
    assert i[-1] < i[0]
    assert diag["z_eff"] >= 1.0
    assert diag["tau_cq_ms_mean"] > 0.0
    assert diag["tau_cq_ms_p95"] > 0.0
    assert np.isclose(diag["argon_quantity_mol"], 0.02)
    assert np.isclose(diag["xenon_quantity_mol"], 0.01)
    assert np.isclose(diag["total_impurity_mol"], 0.13)


# ── Constructor validation ────────────────────────────────────────

import pytest


def test_rejects_nonpositive_energy() -> None:
    with pytest.raises(ValueError, match="Plasma_Energy_MJ"):
        ShatteredPelletInjection(Plasma_Energy_MJ=0.0)


def test_rejects_nonpositive_current() -> None:
    with pytest.raises(ValueError, match="Plasma_Current_MA"):
        ShatteredPelletInjection(Plasma_Current_MA=-1.0)


def test_rejects_nan_energy() -> None:
    with pytest.raises(ValueError, match="Plasma_Energy_MJ"):
        ShatteredPelletInjection(Plasma_Energy_MJ=float("nan"))


# ── _require_non_negative ─────────────────────────────────────────

def test_require_non_negative_rejects_inf() -> None:
    with pytest.raises(ValueError, match="finite"):
        ShatteredPelletInjection._require_non_negative("x", float("inf"))


def test_require_non_negative_rejects_negative() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        ShatteredPelletInjection._require_non_negative("x", -0.1)


# ── estimate_mitigation_cocktail validation ───────────────────────

def test_cocktail_rejects_nan_risk() -> None:
    with pytest.raises(ValueError, match="risk_score"):
        ShatteredPelletInjection.estimate_mitigation_cocktail(
            risk_score=float("nan"), disturbance=0.5,
        )


def test_cocktail_rejects_nan_disturbance() -> None:
    with pytest.raises(ValueError, match="disturbance"):
        ShatteredPelletInjection.estimate_mitigation_cocktail(
            risk_score=0.5, disturbance=float("inf"),
        )


def test_cocktail_rejects_nan_action_bias() -> None:
    with pytest.raises(ValueError, match="action_bias"):
        ShatteredPelletInjection.estimate_mitigation_cocktail(
            risk_score=0.5, disturbance=0.5, action_bias=float("nan"),
        )


# ── trigger_mitigation validation & modes ─────────────────────────

def test_trigger_rejects_bad_duration() -> None:
    spi = ShatteredPelletInjection()
    with pytest.raises(ValueError, match="duration_s"):
        spi.trigger_mitigation(duration_s=0.0, verbose=False)


def test_trigger_rejects_bad_dt() -> None:
    spi = ShatteredPelletInjection()
    with pytest.raises(ValueError, match="dt_s"):
        spi.trigger_mitigation(dt_s=-1e-5, verbose=False)


def test_trigger_without_diagnostics_returns_three() -> None:
    spi = ShatteredPelletInjection()
    result = spi.trigger_mitigation(
        neon_quantity_mol=0.1, return_diagnostics=False,
        duration_s=0.01, verbose=False,
    )
    assert len(result) == 3
    t, w, i = result
    assert len(t) == len(w) == len(i)


def test_trigger_rejects_negative_impurity_mol() -> None:
    spi = ShatteredPelletInjection()
    with pytest.raises(ValueError, match="neon_quantity_mol"):
        spi.trigger_mitigation(neon_quantity_mol=-0.1, verbose=False)


# ── run_spi_mitigation convenience function ───────────────────────

from scpn_control.control.spi_mitigation import run_spi_mitigation


def test_run_spi_mitigation_smoke() -> None:
    summary = run_spi_mitigation(
        save_plot=False, verbose=False,
        duration_s=0.05, dt_s=1e-4,
    )
    for key in (
        "plasma_energy_mj", "plasma_current_ma",
        "z_eff", "tau_cq_ms_mean", "plot_saved",
        "initial_current_ma", "final_current_ma",
    ):
        assert key in summary
    assert summary["plot_saved"] is False
    assert summary["final_current_ma"] < summary["initial_current_ma"]


def test_run_spi_mitigation_deterministic() -> None:
    kwargs = dict(save_plot=False, verbose=False, duration_s=0.05, dt_s=1e-4)
    a = run_spi_mitigation(**kwargs)
    b = run_spi_mitigation(**kwargs)
    assert a["z_eff"] == b["z_eff"]
    assert a["final_current_ma"] == b["final_current_ma"]


def test_run_spi_mitigation_cocktail_fields() -> None:
    summary = run_spi_mitigation(
        argon_quantity_mol=0.05, xenon_quantity_mol=0.02,
        save_plot=False, verbose=False, duration_s=0.05, dt_s=1e-4,
    )
    assert summary["argon_quantity_mol"] == 0.05
    assert summary["xenon_quantity_mol"] == 0.02
