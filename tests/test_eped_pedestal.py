# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — EPED pedestal model tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.eped_pedestal import (
    EPEDConfig,
    EPEDValidationPoint,
    EpedPedestalModel,
    PedestalProfileGenerator,
    _shaping_factor,
    eped1_predict,
    eped1_scan,
    eped_validation_database,
)


def test_eped1_predict_iter():
    # ITER Baseline: B_pol_ped ≈ μ₀ Ip / (2π a √((1+κ²)/2)) ≈ 0.8 T
    config = EPEDConfig(R0=6.2, a=2.0, B0=5.3, kappa=1.7, delta=0.33, Ip_MA=15.0, ne_ped_19=6.0, B_pol_ped=0.8)

    res = eped1_predict(config)

    # Wesson Eq. 3.6.8 q95 (no spurious sqrt) gives q95 ~ 4.2 for ITER,
    # reducing p_ped vs. the old formula. Simplified model yields ~5-50 kPa.
    assert 5.0 < res.p_ped_kPa < 120.0

    # T_ped at ITER-class densities with corrected q95
    assert 0.2 < res.T_ped_keV < 8.0

    # Normalised pedestal width in ρ_tor (narrower with corrected q95)
    assert 0.01 < res.delta_ped < 0.1


def test_eped_self_consistency():
    config = EPEDConfig(R0=6.2, a=2.0, B0=5.3, kappa=1.7, delta=0.33, Ip_MA=15.0, ne_ped_19=6.0, B_pol_ped=0.8)
    res = eped1_predict(config)

    # Check KBM constraint manually
    delta_kbm = config.C_KBM * np.sqrt(res.beta_p_ped)
    assert np.isclose(delta_kbm, res.delta_ped, rtol=0.05)


def test_eped_density_scan():
    config = EPEDConfig(R0=6.2, a=2.0, B0=5.3, kappa=1.7, delta=0.33, Ip_MA=15.0, ne_ped_19=6.0, B_pol_ped=0.8)
    n_range = np.linspace(3.0, 10.0, 5)
    results = eped1_scan(config, n_range)

    assert len(results) == 5

    # Higher density should lead to lower T_ped to maintain roughly constant p_ped
    # p_ped may shift slightly with collisionality, but T drops materially
    assert results[-1].T_ped_keV < results[0].T_ped_keV


def test_eped_validation_db():
    db = eped_validation_database()
    assert len(db) >= 5

    # Check error within 30%
    for pt in db:
        err = abs(pt.p_ped_eped_kPa - pt.p_ped_measured_kPa) / pt.p_ped_measured_kPa
        assert err < 0.3


def test_pedestal_profile_generator():
    config = EPEDConfig(R0=6.2, a=2.0, B0=5.3, kappa=1.7, delta=0.33, Ip_MA=15.0, ne_ped_19=6.0, B_pol_ped=0.8)
    res = eped1_predict(config)

    gen = PedestalProfileGenerator(res, Te_sep_eV=100.0, ne_sep_19=0.3)
    rho = np.linspace(0, 1, 100)

    Te, ne = gen.generate(rho)

    # Core matches pedestal top
    assert np.isclose(Te[0], res.T_ped_keV, rtol=0.05)
    assert np.isclose(ne[0], res.n_ped_19, rtol=0.05)

    # Edge matches separatrix
    assert np.isclose(Te[-1], 0.1, rtol=0.05)
    assert np.isclose(ne[-1], 0.3, rtol=0.05)


# ── New physics tests ─────────────────────────────────────────────────────────


def _iter_base() -> EPEDConfig:
    return EPEDConfig(
        R0=6.2,
        a=2.0,
        B0=5.3,
        kappa=1.7,
        delta=0.33,
        Ip_MA=15.0,
        ne_ped_19=6.0,
        B_pol_ped=0.8,
    )


def test_collisionality_narrows_pedestal():
    """Higher ν*_e → narrower Δ_ped. Snyder et al. 2011, Nucl. Fusion 51, 103016, Fig. 7."""
    base = _iter_base()
    res_low = eped1_predict(base)
    res_high = eped1_predict(EPEDConfig(**{**vars(base), "nu_star_e": 5.0}))

    assert res_high.delta_ped < res_low.delta_ped
    # Collisionless width stored separately must equal the nu=0 result
    assert np.isclose(res_low.delta_ped, res_low.delta_ped_collisionless, rtol=1e-9)


def test_triangularity_raises_alpha_crit():
    """Higher δ → larger F_shape → higher α_crit. Connor et al. 1998, PPCF 40, 531."""
    kappa = 1.7
    delta_lo, delta_hi = 0.1, 0.6
    f_lo = _shaping_factor(kappa, delta_lo)
    f_hi = _shaping_factor(kappa, delta_hi)
    assert f_hi > f_lo

    # Verify α_crit in full EPED solve also increases
    base = _iter_base()
    res_lo = eped1_predict(EPEDConfig(**{**vars(base), "delta": delta_lo}))
    res_hi = eped1_predict(EPEDConfig(**{**vars(base), "delta": delta_hi}))
    assert res_hi.alpha_crit > res_lo.alpha_crit


def test_kbm_width_scaling():
    """Δ ∝ sqrt(β_p). Snyder et al. 2009, Phys. Plasmas 16, 056118, Eq. 4."""
    # At collisionless limit, the converged solution must satisfy Δ ≈ C_KBM * sqrt(β_p)
    config = _iter_base()
    res = eped1_predict(config)

    delta_expected = config.C_KBM * np.sqrt(res.beta_p_ped)
    # Allow 5% tolerance due to the peeling correction shifting the fixed point slightly
    assert np.isclose(delta_expected, res.delta_ped, rtol=0.05)

    # Scaling check: double β_p by doubling B_pol (halves β_p denominator... use a
    # controlled parametric shift via C_KBM instead, holding everything else fixed)
    c2 = EPEDConfig(**{**vars(config), "C_KBM": config.C_KBM * 2.0})
    res2 = eped1_predict(c2)
    # Wider C_KBM → wider Δ (sqrt scaling preserved)
    assert res2.delta_ped > res.delta_ped


def test_eped_shaping_scan():
    """
    α_crit increases monotonically with κ.

    p_ped = α_crit B₀² a Δ / (2μ₀ q_95² R₀), and q_95 ∝ √((1+κ²)/2) also
    grows with κ, so p_ped can decrease even as α_crit rises.  The physically
    correct observable for this shaping scan is α_crit, not p_ped.

    Connor et al. 1998, PPCF 40, 531 — ballooning boundary increases with
    elongation. Snyder et al. 2009, Phys. Plasmas 16, 056118 — F_shape factor.
    """
    base = _iter_base()
    kappa_values = np.linspace(1.0, 2.0, 6)
    alpha_crits = []
    for kappa in kappa_values:
        cfg = EPEDConfig(**{**vars(base), "kappa": float(kappa)})
        alpha_crits.append(eped1_predict(cfg).alpha_crit)

    diffs = np.diff(alpha_crits)
    assert np.all(diffs > 0), f"α_crit not monotonically increasing: {alpha_crits}"


def test_eped_config_rejects_nonphysical_geometry_and_inputs():
    base = _iter_base()

    with pytest.raises(ValueError, match="a must be smaller"):
        eped1_predict(EPEDConfig(**{**vars(base), "a": base.R0}))

    with pytest.raises(ValueError, match="delta"):
        eped1_predict(EPEDConfig(**{**vars(base), "delta": 1.0}))

    with pytest.raises(ValueError, match="ne_ped_19"):
        eped1_predict(EPEDConfig(**{**vars(base), "ne_ped_19": 0.0}))

    with pytest.raises(ValueError, match="B_pol_ped"):
        eped1_predict(EPEDConfig(**{**vars(base), "B_pol_ped": 0.0}))

    with pytest.raises(ValueError, match="C_KBM"):
        eped1_predict(EPEDConfig(**{**vars(base), "C_KBM": 0.0}))

    with pytest.raises(ValueError, match="mode bounds"):
        eped1_predict(EPEDConfig(**{**vars(base), "n_mode_min": 31, "n_mode_max": 30}))

    with pytest.raises(ValueError, match="mode bounds"):
        eped1_predict(EPEDConfig(**{**vars(base), "n_mode_min": True}))

    with pytest.raises(ValueError, match="nu_star_e"):
        eped1_predict(EPEDConfig(**{**vars(base), "nu_star_e": -0.1}))


def test_eped_scan_rejects_invalid_density_axis():
    base = _iter_base()

    with pytest.raises(ValueError, match="ne_ped_range"):
        eped1_scan(base, np.array([[3.0, 4.0]]))

    with pytest.raises(ValueError, match="ne_ped_range"):
        eped1_scan(base, np.array([3.0, np.nan]))

    with pytest.raises(ValueError, match="ne_ped_range"):
        eped1_scan(base, np.array([3.0, 0.0]))


def test_pedestal_profile_generator_rejects_invalid_boundaries():
    res = eped1_predict(_iter_base())

    with pytest.raises(ValueError, match="Te_sep_eV"):
        PedestalProfileGenerator(res, Te_sep_eV=res.T_ped_keV * 1000.0)

    with pytest.raises(ValueError, match="ne_sep_19"):
        PedestalProfileGenerator(res, ne_sep_19=res.n_ped_19)

    gen = PedestalProfileGenerator(res)
    with pytest.raises(ValueError, match="rho"):
        gen.generate(np.array([[0.0, 1.0]]))

    with pytest.raises(ValueError, match="rho"):
        gen.generate(np.array([0.0, 0.5, np.nan]))

    with pytest.raises(ValueError, match="rho"):
        gen.generate(np.array([0.0, 0.8, 0.7]))

    with pytest.raises(ValueError, match="strictly increasing"):
        gen.generate(np.array([0.0, 0.5, 0.5, 1.0]))


def test_integrated_wrapper_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="a must be smaller"):
        EpedPedestalModel(R0=2.0, a=2.0, B0=5.0, Ip_MA=10.0)

    with pytest.raises(ValueError, match="Z_eff"):
        EpedPedestalModel(R0=6.2, a=2.0, B0=5.0, Ip_MA=10.0, Z_eff=0.0)

    model = EpedPedestalModel(R0=6.2, a=2.0, B0=5.0, Ip_MA=10.0)

    with pytest.raises(ValueError, match="ne_ped_19"):
        model.predict(0.0)

    with pytest.raises(ValueError, match="nu_star_e"):
        model.predict(6.0, nu_star_e=-1.0)


def test_eped_validation_point_rejects_nonphysical_records():
    """Validation records reject impossible provenance and pedestal quantities."""
    with pytest.raises(ValueError, match="machine"):
        EPEDValidationPoint("", 1, 10.0, 10.5, 0.04, 0.042)

    with pytest.raises(ValueError, match="shot"):
        EPEDValidationPoint("DIII-D", False, 10.0, 10.5, 0.04, 0.042)

    with pytest.raises(ValueError, match="p_ped_measured"):
        EPEDValidationPoint("DIII-D", 1, 0.0, 10.5, 0.04, 0.042)

    with pytest.raises(ValueError, match="normalised minor-radius"):
        EPEDValidationPoint("DIII-D", 1, 10.0, 10.5, 1.0, 0.042)
