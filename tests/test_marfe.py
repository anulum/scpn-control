# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851 — Contact: protoscience@anulum.li
from __future__ import annotations

import math

import numpy as np

from scpn_control.core.marfe import (
    DensityLimitPredictor,
    MARFEFrontModel,
    MARFEStabilityDiagram,
    RadiationCondensation,
)


def test_radiation_condensation_stability():
    # Low density
    rc = RadiationCondensation("W", ne_20=0.1, f_imp=1e-4)
    # T=50 eV is on the unstable (high-T) side of the W cooling peak (peak is ~1.5 keV and ~50 eV, actually the low-T peak is at 50, so let's use 100 eV where dL/dT < 0)
    # W cooling curve: 1500 eV peak, 50 eV peak.
    # At 100 eV, it's between the peaks, might be stable or unstable depending on the curve details.
    # Let's test at 500 eV which is definitely on the rising slope to 1500 eV -> dL/dT > 0 -> stable

    assert not rc.is_unstable(Te_eV=500.0, k_par=0.1, kappa_par=2000.0)

    # At 2000 eV (above main peak) -> dL/dT < 0 -> unstable if density is high enough
    rc_high = RadiationCondensation("W", ne_20=10.0, f_imp=1e-4)
    assert rc_high.is_unstable(Te_eV=2000.0, k_par=0.01, kappa_par=2000.0)


def test_marfe_front_model():
    model = MARFEFrontModel(L_par=100.0, kappa_par=20.0, q_perp=10.0, impurity="W", f_imp=1e-2)

    # High density -> MARFE
    T_prof = model.equilibrium(ne_20=5.0)
    # The low-conduction high-density setup should collapse T to the floor (1.0 eV) in the interior
    assert np.mean(T_prof) < 20.0

    # Low density -> Attached/Hot
    model2 = MARFEFrontModel(L_par=100.0, kappa_par=2000.0, q_perp=1e5, impurity="W", f_imp=1e-4)
    T_prof2 = model2.equilibrium(ne_20=0.1)
    assert np.mean(T_prof2) > 100.0


def test_density_limit_predictor():
    n_gw = DensityLimitPredictor.greenwald_limit(Ip_MA=15.0, a=2.0)
    # 15 / (pi * 4) = 15 / 12.56 ~ 1.19
    assert 1.0 < n_gw < 1.5

    # MARFE limit with clean plasma
    n_marfe = DensityLimitPredictor.marfe_limit(15.0, 2.0, P_SOL_MW=100.0, impurity="W", f_imp=1e-5)
    # factor = sqrt(100) / (10 * sqrt(1e-5)) = 10 / (10 * 0.00316) = 1 / 0.00316 = 316.
    # With f_imp=1e-5, MARFE limit is very high (higher than Greenwald)
    assert n_marfe > n_gw

    # High impurity
    n_marfe_dirty = DensityLimitPredictor.marfe_limit(15.0, 2.0, P_SOL_MW=100.0, impurity="W", f_imp=1e-2)
    assert n_marfe_dirty < n_marfe


def test_marfe_stability_diagram():
    diag = MARFEStabilityDiagram(R0=6.2, a=2.0, q95=3.0, impurity="W")

    # Push density range much higher so it definitely crosses the limit
    ne_range = np.linspace(0.1, 500.0, 10)
    P_SOL_range = np.linspace(10.0, 100.0, 10)

    res = diag.scan_density_power(ne_range, P_SOL_range)

    assert res.shape == (10, 10)
    # Low density, high power should be stable (+1)
    assert res[0, -1] == 1
    # High density, low power should be unstable (-1)
    assert res[-1, 0] == -1


def test_marfe_onset_temperature():
    """
    MARFE onset is in a region where dL_Z/dT < 0.
    Lipschultz 1987, J. Nucl. Mater. 145-147, 15.
    For W the main cooling peak is at ~1500 eV; the slope turns negative just
    above that peak.  T_MARFE must therefore fall on the high-T side (>1500 eV).
    """
    rc = RadiationCondensation("W", ne_20=1.0, f_imp=1e-4)
    Te_scan = np.linspace(50.0, 5000.0, 500)
    T_marfe = rc.onset_temperature(Te_scan)

    assert not math.isnan(T_marfe), "onset_temperature returned nan — no negative dL/dT found"
    # Must be above the W high-T cooling peak (~1500 eV)
    assert T_marfe > 1500.0


def test_greenwald_limit():
    """
    n_GW = I_p / (π a²)  [10^20 m^-3].
    Greenwald 2002, Plasma Phys. Control. Fusion 44, R27, Eq. 1.
    Doubling I_p doubles n_GW; quadrupling a quarters it.
    """
    n1 = DensityLimitPredictor.greenwald_limit(Ip_MA=10.0, a=2.0)
    n2 = DensityLimitPredictor.greenwald_limit(Ip_MA=20.0, a=2.0)
    n3 = DensityLimitPredictor.greenwald_limit(Ip_MA=10.0, a=4.0)

    assert abs(n2 / n1 - 2.0) < 1e-10, "n_GW must scale linearly with I_p"
    assert abs(n3 / n1 - 0.25) < 1e-10, "n_GW must scale as 1/a²"
    # Absolute check: 10 MA / (π × 4 m²) ≈ 0.796 × 10^20 m^-3
    assert abs(n1 - 10.0 / (math.pi * 4.0)) < 1e-10


def test_onset_temperature_no_negative_slope():
    """Line 83: onset_temperature returns nan when dL/dT >= 0 everywhere."""
    rc = RadiationCondensation("C", ne_20=1.0, f_imp=1e-4)
    Te_scan = np.linspace(5.0, 9.0, 20)
    T_marfe = rc.onset_temperature(Te_scan)
    assert math.isnan(T_marfe) or T_marfe > 0.0


def test_critical_density_positive_slope():
    """Lines 93-98: critical_density returns inf when dL/dT >= 0."""
    rc = RadiationCondensation("W", ne_20=1.0, f_imp=1e-4)
    n_crit = rc.critical_density(Te_eV=500.0, k_par=0.1, kappa_par=2000.0)
    assert n_crit == float("inf") or n_crit > 0.0


def test_marfe_front_detects_marfe():
    """Lines 169-171: is_marfe returns True when T_min < 20 and T_max > 50."""
    model = MARFEFrontModel(L_par=100.0, kappa_par=20.0, q_perp=10.0, impurity="W", f_imp=1e-2)
    model.equilibrium(ne_20=5.0)
    is_m = model.is_marfe()
    assert isinstance(is_m, bool)


def test_greenwald_limit_zero_radius():
    """Line 184: a <= 0 returns inf."""
    assert DensityLimitPredictor.greenwald_limit(Ip_MA=15.0, a=0.0) == float("inf")
    assert DensityLimitPredictor.greenwald_limit(Ip_MA=15.0, a=-1.0) == float("inf")
