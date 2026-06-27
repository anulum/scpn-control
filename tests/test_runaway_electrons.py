# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runaway Electron Tests
from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_control.core.runaway_electrons import (
    RunawayEvolution,
    RunawayMitigationAssessment,
    RunawayParams,
    avalanche_growth_rate,
    coulomb_log,
    critical_field,
    dreicer_generation_rate,
    hot_tail_seed,
    synchrotron_energy_limit,
)


# ---------------------------------------------------------------------------
# Existing tests (API-preserved, call-sites updated for new signatures)
# ---------------------------------------------------------------------------


def test_zero_epar() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    assert dreicer_generation_rate(params) == 0.0
    assert avalanche_growth_rate(params, n_RE=1e10) == 0.0


def test_subcritical_epar() -> None:
    E_c = critical_field(ne_20=1.0, Te_keV=1.0)
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=E_c * 0.5, Z_eff=1.5, B0=5.3, R0=6.2)
    assert avalanche_growth_rate(params, n_RE=1e10) == 0.0
    assert dreicer_generation_rate(params) < 1e-10


def test_dreicer_generation() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=100.0, Z_eff=1.5, B0=5.3, R0=6.2)
    rate = dreicer_generation_rate(params)
    assert rate > 0.0


def test_avalanche_multiplication() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=0.01, E_par=10.0, Z_eff=1.5, B0=5.3, R0=6.2)
    E_c = critical_field(1.0, Te_keV=0.01)
    assert params.E_par > E_c

    n_RE = 1e10
    rate = avalanche_growth_rate(params, n_RE)
    assert rate > 0.0

    rate2 = avalanche_growth_rate(params, 2e10)
    assert np.isclose(rate2, 2.0 * rate)


def test_hot_tail() -> None:
    n_seed = hot_tail_seed(Te_pre_keV=10.0, Te_post_keV=0.01, ne_20=1.0, quench_time_ms=1.0)
    assert n_seed > 1e10


def test_runaway_evolution() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=0.01, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    ev = RunawayEvolution(params)

    t, n = ev.evolve(n_RE_0=1e10, E_par_profile=lambda t: 5.0, t_span=(0, 0.1), dt=0.01)

    assert len(t) == 11
    assert n[-1] > n[0]


def test_mitigation_assessment() -> None:
    mit = RunawayMitigationAssessment()

    E_par = 20.0
    ne_req = mit.required_density_for_suppression(E_par, Z_eff=1.0)
    assert ne_req > 100.0

    E_max = mit.maximum_re_energy(B0=5.3, R0=6.2)
    assert E_max > 1.0  # must be physical

    load = mit.wall_heat_load(n_RE=1e16, E_max_MeV=E_max, A_wet=10.0)
    assert load > 0.0


# ---------------------------------------------------------------------------
# New physics tests
# ---------------------------------------------------------------------------


def test_coulomb_log_temperature() -> None:
    """ln Λ increases with T_e (Wesson 2011, Eq. 2.12.4)."""
    ne_20 = 1.0
    ln_low = coulomb_log(ne_20, Te_keV=0.1)  # 100 eV
    ln_mid = coulomb_log(ne_20, Te_keV=1.0)  # 1 keV
    ln_high = coulomb_log(ne_20, Te_keV=10.0)  # 10 keV
    assert ln_low < ln_mid < ln_high


def test_avalanche_rate_threshold() -> None:
    """No avalanche when E_par <= E_c (Rosenbluth & Putvinski 1997, Eq. 15)."""
    ne_20 = 1.0
    Te_keV = 1.0
    E_c = critical_field(ne_20, Te_keV)

    params_below = RunawayParams(ne_20=ne_20, Te_keV=Te_keV, E_par=E_c * 0.99, Z_eff=1.5, B0=5.3, R0=6.2)
    assert avalanche_growth_rate(params_below, n_RE=1e15) == 0.0

    params_above = RunawayParams(ne_20=ne_20, Te_keV=Te_keV, E_par=E_c * 5.0, Z_eff=1.5, B0=5.3, R0=6.2)
    assert avalanche_growth_rate(params_above, n_RE=1e15) > 0.0


def test_synchrotron_limit() -> None:
    """E_max decreases as E_c increases (Martin-Solis 2006, Eq. 12)."""
    E_par = 50.0
    E_c_low = 0.1  # V/m — high E/E_c ratio → high E_max
    E_c_high = 5.0  # V/m — low E/E_c ratio → low E_max

    E_max_low_Ec = synchrotron_energy_limit(E_par, E_c_low)
    E_max_high_Ec = synchrotron_energy_limit(E_par, E_c_high)

    assert E_max_low_Ec > E_max_high_Ec
    assert E_max_low_Ec > 0.0

    # Verify scaling: E_max ∝ (E/E_c)^{1/3}, so doubling E_c reduces E_max by 2^{1/3}
    E_c_double = E_c_low * 2.0
    E_max_double = synchrotron_energy_limit(E_par, E_c_double)
    ratio = E_max_low_Ec / E_max_double
    assert np.isclose(ratio, 2.0 ** (1.0 / 3.0), rtol=1e-6)

    # Below threshold: no RE energy
    assert synchrotron_energy_limit(E_par, E_par * 2.0) == 0.0


def test_dreicer_rate_physical() -> None:
    """Dreicer generation is positive when E_par is well above threshold.

    Connor & Hastie (1975): rate rises sharply just above E_D threshold.
    The (E/E_D)^{-h_z} prefactor makes the rate non-monotone at large E —
    this is physical, not a bug (see Connor & Hastie 1975, Fig. 1).
    We verify the rate is positive and that the zero-field limit is zero.
    """
    params = RunawayParams(ne_20=1.0, Te_keV=0.5, E_par=1000.0, Z_eff=1.0, B0=5.3, R0=6.2)
    rate = dreicer_generation_rate(params)
    assert rate > 0.0

    # Zero field → zero rate
    params_zero = RunawayParams(ne_20=1.0, Te_keV=0.5, E_par=0.0, Z_eff=1.0, B0=5.3, R0=6.2)
    assert dreicer_generation_rate(params_zero) == 0.0

    # Rate is positive for a second E_par value (confirms function, not a single-point fluke)
    params_b = RunawayParams(ne_20=1.0, Te_keV=0.5, E_par=500.0, Z_eff=1.0, B0=5.3, R0=6.2)
    assert dreicer_generation_rate(params_b) > 0.0


def test_dreicer_field_values() -> None:
    """Dreicer field increases with density for fixed electron temperature."""
    from scpn_control.core.runaway_electrons import dreicer_field

    E_D = dreicer_field(ne_20=1.0, Te_keV=1.0)
    assert E_D > 0.0
    E_D2 = dreicer_field(ne_20=2.0, Te_keV=1.0)
    assert E_D2 > E_D


def test_avalanche_pitch_angle_correction_stays_physical_near_threshold() -> None:
    """Extreme positive Z_eff near threshold still yields a bounded avalanche rate."""
    E_c = critical_field(ne_20=1.0, Te_keV=1.0)
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=E_c * 1.001, Z_eff=500.0, B0=5.3, R0=6.2)
    rate = avalanche_growth_rate(params, n_RE=1e15)
    assert rate >= 0.0


def test_hot_tail_no_quench() -> None:
    """No thermal quench leaves the hot-tail source off."""
    assert hot_tail_seed(Te_pre_keV=5.0, Te_post_keV=5.0, ne_20=1.0, quench_time_ms=1.0) == 0.0
    assert hot_tail_seed(Te_pre_keV=5.0, Te_post_keV=10.0, ne_20=1.0, quench_time_ms=1.0) == 0.0


def test_hot_tail_large_quench_time() -> None:
    """Slow quenches have negligible hot-tail seed density."""
    n = hot_tail_seed(Te_pre_keV=10.0, Te_post_keV=0.01, ne_20=1.0, quench_time_ms=1e12)
    assert n == 0.0


def test_current_fraction_zero_ip() -> None:
    """Zero total plasma current gives zero runaway-current fraction."""
    params = RunawayParams(ne_20=1.0, Te_keV=0.01, E_par=5.0, Z_eff=1.5, B0=5.3, R0=6.2)
    ev = RunawayEvolution(params)
    assert ev.current_fraction(n_RE=1e15, I_p_MA=0.0) == 0.0
    frac = ev.current_fraction(n_RE=1e15, I_p_MA=15.0)
    assert 0.0 <= frac <= 1.0


def test_suppression_zero_epar() -> None:
    """No suppressing density is required for non-positive electric field."""
    assert RunawayMitigationAssessment.required_density_for_suppression(E_par=0.0, Z_eff=1.0) == 0.0
    assert RunawayMitigationAssessment.required_density_for_suppression(E_par=-1.0, Z_eff=1.0) == 0.0


def test_finite_float_rejects_non_numeric_input() -> None:
    """A value that cannot be cast to float is reported as non-finite, not crashed."""
    with pytest.raises(ValueError, match="ne_20 must be finite"):
        coulomb_log(cast(Any, "not-a-number"), Te_keV=1.0)


def test_dreicer_rate_zero_for_negligible_field_ratio() -> None:
    """A field far below the Dreicer field yields no seed generation (E_par/E_D < 1e-4)."""
    from scpn_control.core.runaway_electrons import dreicer_field

    ne_20, Te_keV = 1.0, 1.0
    E_D = dreicer_field(ne_20, Te_keV)
    tiny_field = E_D * 1e-6  # positive, but far below the Dreicer threshold
    params = RunawayParams(ne_20=ne_20, Te_keV=Te_keV, E_par=tiny_field, Z_eff=1.5, B0=5.3, R0=6.2)
    assert dreicer_generation_rate(params) == 0.0


def test_maximum_re_energy_orbit_bound_when_synchrotron_inactive() -> None:
    """When E_par <= E_c the synchrotron limit is inactive and the orbit-loss bound governs."""
    mit = RunawayMitigationAssessment()
    E_c = critical_field(ne_20=10.0, Te_keV=1.0)
    # E_par well below E_c forces synchrotron_energy_limit -> 0.0, taking the orbit branch.
    E_max = mit.maximum_re_energy(B0=5.3, R0=6.2, E_par=E_c * 1e-3, ne_20=10.0, Te_keV=1.0)
    assert E_max > 0.0


def test_finite_float_rejects_nan_input() -> None:
    """A NaN that casts cleanly to float is still rejected by the finiteness guard."""
    with pytest.raises(ValueError, match="ne_20 must be finite"):
        coulomb_log(float("nan"), Te_keV=1.0)


def test_nonnegative_float_rejects_negative_density() -> None:
    """A negative runaway density is rejected by the non-negativity guard."""
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=10.0, Z_eff=1.5, B0=5.3, R0=6.2)
    with pytest.raises(ValueError, match="n_RE must be non-negative"):
        avalanche_growth_rate(params, n_RE=-1.0)


def test_dreicer_rate_zero_when_exponent_underflows() -> None:
    """A field above the ratio floor but with a deeply negative exponent yields no seed."""
    from scpn_control.core.runaway_electrons import dreicer_field

    ne_20, Te_keV, Z_eff = 1.0, 1.0, 1.5
    E_D = dreicer_field(ne_20, Te_keV)
    # E_par/E_D = 5e-4 clears the 1e-4 ratio gate but drives the exponent below -500.
    params = RunawayParams(ne_20=ne_20, Te_keV=Te_keV, E_par=E_D * 5e-4, Z_eff=Z_eff, B0=5.3, R0=6.2)
    assert dreicer_generation_rate(params) == 0.0


def test_runaway_electron_domains_reject_nonphysical_plasma_inputs() -> None:
    with pytest.raises(ValueError, match="ne_20"):
        coulomb_log(0.0, Te_keV=1.0)
    with pytest.raises(ValueError, match="Te_keV"):
        coulomb_log(1.0, Te_keV=0.0)
    with pytest.raises(ValueError, match="Z_eff"):
        dreicer_generation_rate(RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=1.0, Z_eff=0.0, B0=5.3, R0=6.2))


def test_hot_tail_rejects_invalid_quench_domains() -> None:
    with pytest.raises(ValueError, match="quench_time_ms"):
        hot_tail_seed(Te_pre_keV=10.0, Te_post_keV=0.01, ne_20=1.0, quench_time_ms=0.0)
    with pytest.raises(ValueError, match="ne_20"):
        hot_tail_seed(Te_pre_keV=10.0, Te_post_keV=0.01, ne_20=0.0, quench_time_ms=1.0)


def test_runaway_evolution_rejects_invalid_time_domains() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=0.01, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    ev = RunawayEvolution(params)
    with pytest.raises(ValueError, match="dt"):
        ev.evolve(n_RE_0=1e10, E_par_profile=lambda _t: 5.0, t_span=(0.0, 0.1), dt=0.0)
    with pytest.raises(ValueError, match="t_span"):
        ev.evolve(n_RE_0=1e10, E_par_profile=lambda _t: 5.0, t_span=(0.1, 0.0), dt=0.01)


def test_mitigation_rejects_invalid_geometry_and_heat_load_domains() -> None:
    mit = RunawayMitigationAssessment()
    with pytest.raises(ValueError, match="B0"):
        mit.maximum_re_energy(B0=0.0, R0=6.2)
    with pytest.raises(ValueError, match="A_wet"):
        mit.wall_heat_load(n_RE=1e16, E_max_MeV=25.0, A_wet=0.0)
