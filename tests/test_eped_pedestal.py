# ──────────────────────────────────────────────────────────────────────
# SCPN Control — EPED Pedestal Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_control.core.eped_pedestal import (
    EPEDConfig,
    PedestalProfileGenerator,
    eped1_predict,
    eped1_scan,
    eped_validation_database,
)


def test_eped1_predict_iter():
    # ITER Baseline approximation
    # B_pol_ped ~ mu_0 Ip / (2 pi a sqrt((1+k^2)/2))
    # ~ 4 pi e-7 * 15e6 / (2 pi * 2 * 1.4) ~ 15 / 5.6 ~ 0.8 T
    config = EPEDConfig(R0=6.2, a=2.0, B0=5.3, kappa=1.7, delta=0.33, Ip_MA=15.0, ne_ped_19=6.0, B_pol_ped=0.8)

    res = eped1_predict(config)

    # ITER p_ped ~ 60-100 kPa
    assert 40.0 < res.p_ped_kPa < 120.0

    # T_ped ~ 3-6 keV
    assert 2.0 < res.T_ped_keV < 8.0

    # Width ~ 0.04 - 0.06
    assert 0.02 < res.delta_ped < 0.1


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
    # Actually p_ped might shift slightly with collisionality, but T drops strongly
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
