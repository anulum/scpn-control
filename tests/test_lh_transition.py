# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# Contact: protoscience@anulum.li
from __future__ import annotations

import numpy as np

from scpn_control.core.lh_transition import (
    IPhaseDetector,
    IPhaseFrequency,
    LHTransitionController,
    LHTrigger,
    MartinThreshold,
    PredatorPreyModel,
)


def test_predator_prey_l_mode():
    model = PredatorPreyModel()
    res = model.evolve(Q_heating=1.0, t_span=(0.0, 1.0), dt=0.001)

    assert res.regime == "L_MODE"
    assert res.epsilon_trace[-1] > 1e4
    assert res.V_ZF_trace[-1] < 10.0


def test_predator_prey_h_mode():
    model = PredatorPreyModel(gamma_damp=0.01, alpha3=1e-3)
    # High enough Q to push pressure and thus turbulence into the saturation regime where ZF suppress it
    res = model.evolve(Q_heating=1000.0, t_span=(0.0, 10.0), dt=0.001)

    # We should have triggered H-mode -> high V_ZF, low epsilon
    assert res.regime == "H_MODE"
    assert res.V_ZF_trace[-1] > 100.0


def test_lh_trigger_threshold():
    model = PredatorPreyModel()
    trigger = LHTrigger(model)

    Q_range = np.linspace(1.0, 500.0, 10)
    Q_th = trigger.find_threshold(Q_range)

    assert 1.0 < Q_th <= 500.0


def test_martin_scaling():
    # ITER geometry: a=2.0 m, R=6.2 m, κ=1.7 → S ≈ 840 m².
    # ne_19=5.0 → n₂₀=0.5, B_T=5.3 T.
    # Martin et al. 2008 Eq. (1): P_LH ≈ 60–70 MW.
    S_iter = 840.0
    P_th = MartinThreshold.power_threshold_MW(ne_19=5.0, B_T=5.3, S_m2=S_iter)
    assert 50.0 < P_th < 100.0


def test_i_phase_detector():
    det = IPhaseDetector(window_size=100)

    # Stable trace -> no I-phase
    trace_stable = np.ones(200) * 1e4
    assert not det.detect(trace_stable)

    # Oscillating trace -> I-phase
    t = np.linspace(0, 10, 200)
    trace_osc = 1e4 * (1.0 + 0.5 * np.sin(t))
    assert det.detect(trace_osc)


def test_transition_controller():
    model = PredatorPreyModel()
    ctrl = LHTransitionController(model, Q_target=50.0)

    # Still L-mode -> ramp
    Q1 = ctrl.step(epsilon_measured=1e5, Q_current=10.0, dt=0.1)
    assert Q1 > 10.0

    # Hit H-mode -> jump to target
    Q2 = ctrl.step(epsilon_measured=1e4, Q_current=20.0, dt=0.1)
    assert Q2 == 50.0


# ─── New physics tests ───────────────────────────────────────────────────────


def test_martin_scaling_iter():
    # ITER parameters: n_e ≈ 5×10¹⁹ m⁻³ (ne_19=5), B_T=5.3 T, S≈680 m².
    # Martin et al. 2008 predicts P_LH ≈ 50–80 MW for these values.
    # Geometry: minor a=2.0 m, R=6.2 m, κ=1.7 → S ≈ 4π²aRκ ≈ 830 m².
    # Using S=680 m² (ITER official surface area) gives a lower bound.
    P_th = MartinThreshold.power_threshold_MW(ne_19=5.0, B_T=5.3, S_m2=680.0)
    assert 40.0 < P_th < 100.0, f"ITER P_LH = {P_th:.1f} MW, expected 40–100 MW"


def test_low_density_branch():
    # Ryter et al. 2014, Nucl. Fusion 54, 083003, Fig. 3.
    # ITER-like: I_p=15 MA, a=2.0 m → n_GW_19 ≈ 11.9, n_min_19 ≈ 4.8.
    # ne=2.0 is deep below n_min → boost ≈ 5.7× → P_LH ≫ above-threshold case.
    # ne=8.0 is above n_min → plain Martin scaling.
    kwargs = dict(B_T=5.3, S_m2=680.0, I_p_MA=15.0, a_m=2.0)
    P_low = MartinThreshold.power_threshold_with_low_density_branch_MW(ne_19=2.0, **kwargs)
    P_above = MartinThreshold.power_threshold_with_low_density_branch_MW(ne_19=8.0, **kwargs)
    assert P_low > P_above, f"Low-density branch: {P_low:.1f} MW must exceed above-n_min: {P_above:.1f} MW"


def test_predator_prey_oscillation():
    # Kim & Diamond 2003: turbulence and ZF are anti-correlated in limit-cycle regime.
    # Use a moderately high Q to drive oscillations.
    model = PredatorPreyModel(gamma_damp=500.0, alpha3=5e-7)
    res = model.evolve(Q_heating=200.0, t_span=(0.0, 5.0), dt=5e-5)
    # Pearson correlation between ε and V_ZF should be negative (predator-prey).
    corr = np.corrcoef(res.epsilon_trace, res.V_ZF_trace)[0, 1]
    assert corr < 0.0, f"ε–V_ZF correlation = {corr:.3f}, expected < 0 (predator-prey)"


def test_i_phase_frequency():
    # Tynan et al. 2013: f_I ≈ γ_ZF / (2π); must be positive for physical parameters.
    model = PredatorPreyModel()
    f_I = IPhaseFrequency.estimate_hz(model)
    assert f_I > 0.0, f"I-phase frequency {f_I} Hz must be positive"
