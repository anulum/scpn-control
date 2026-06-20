# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# Contact: protoscience@anulum.li
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest

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


def test_lh_trigger_all_l_mode():
    """Threshold search returns last Q when no H-mode is found."""
    model = PredatorPreyModel()
    Q_range = np.array([0.1, 0.5, 1.0])
    Q_th = LHTrigger(model).find_threshold(Q_range)
    assert Q_th == 1.0


def test_martin_threshold_rejects_nonphysical_inputs():
    """Martin scaling rejects inputs outside its published positive domain."""
    with pytest.raises(ValueError, match="ne_19"):
        MartinThreshold.power_threshold_MW(ne_19=0.0, B_T=5.3, S_m2=680.0)
    with pytest.raises(ValueError, match="B_T"):
        MartinThreshold.power_threshold_MW(ne_19=5.0, B_T=0.0, S_m2=680.0)
    with pytest.raises(ValueError, match="S_m2"):
        MartinThreshold.power_threshold_MW(ne_19=5.0, B_T=5.3, S_m2=0.0)
    with pytest.raises(ValueError, match="ne_19"):
        MartinThreshold.power_threshold_MW(ne_19=float("nan"), B_T=5.3, S_m2=680.0)


def test_low_density_branch_rejects_nonphysical_geometry():
    """Low-density branch requires valid Greenwald geometry."""
    with pytest.raises(ValueError, match="I_p_MA"):
        MartinThreshold.power_threshold_with_low_density_branch_MW(ne_19=5.0, B_T=5.3, S_m2=680.0, I_p_MA=0.0, a_m=2.0)
    with pytest.raises(ValueError, match="a_m"):
        MartinThreshold.power_threshold_with_low_density_branch_MW(ne_19=5.0, B_T=5.3, S_m2=680.0, I_p_MA=15.0, a_m=0.0)


def test_low_density_branch_zero_density_is_non_triggerable():
    """Zero density with valid machine parameters cannot enter H-mode."""
    p_lh = MartinThreshold.power_threshold_with_low_density_branch_MW(
        ne_19=0.0, B_T=5.3, S_m2=680.0, I_p_MA=15.0, a_m=2.0
    )
    assert np.isinf(p_lh)


def test_i_phase_detector_short_trace():
    """Trace shorter than window returns False."""
    det = IPhaseDetector(window_size=100)
    assert not det.detect(np.ones(50))


def test_predator_prey_rejects_nonphysical_domains():
    with pytest.raises(ValueError, match="gamma_L"):
        PredatorPreyModel(gamma_L=float("nan"))

    model = PredatorPreyModel()
    with pytest.raises(ValueError, match="epsilon"):
        model.confinement_time(-1.0)
    with pytest.raises(ValueError, match="state"):
        model.step(np.array([1.0, -1.0, 1.0]), dt=1e-3, Q_heating=1.0)
    with pytest.raises(ValueError, match="dt"):
        model.step(np.array([1.0, 1.0, 1.0]), dt=0.0, Q_heating=1.0)
    with pytest.raises(ValueError, match="Q_heating"):
        model.evolve(Q_heating=-1.0, t_span=(0.0, 1.0), dt=1e-3)
    with pytest.raises(ValueError, match="t_span"):
        model.evolve(Q_heating=1.0, t_span=(1.0, 0.0), dt=1e-3)


def test_lh_trigger_rejects_malformed_heating_scan():
    trigger = LHTrigger(PredatorPreyModel())

    with pytest.raises(ValueError, match="non-empty"):
        trigger.find_threshold(np.array([]))
    with pytest.raises(ValueError, match="strictly increasing"):
        trigger.find_threshold(np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        trigger.find_threshold(np.array([1.0, np.nan]))


def test_i_phase_detector_rejects_malformed_trace():
    with pytest.raises(ValueError, match="window_size"):
        IPhaseDetector(window_size=1)

    det = IPhaseDetector(window_size=10)
    with pytest.raises(ValueError, match="one-dimensional"):
        det.detect(np.ones((2, 10)))
    with pytest.raises(ValueError, match="finite"):
        det.detect(np.array([1.0, np.nan] * 10))
    with pytest.raises(ValueError, match="non-negative"):
        det.detect(np.array([1.0, -1.0] * 10))


def test_transition_controller_rejects_nonphysical_inputs():
    model = PredatorPreyModel()
    with pytest.raises(ValueError, match="Q_target"):
        LHTransitionController(model, Q_target=-1.0)

    ctrl = LHTransitionController(model, Q_target=50.0)
    with pytest.raises(ValueError, match="epsilon_measured"):
        ctrl.step(epsilon_measured=float("nan"), Q_current=10.0, dt=0.1)
    with pytest.raises(ValueError, match="Q_current"):
        ctrl.step(epsilon_measured=1e5, Q_current=-1.0, dt=0.1)
    with pytest.raises(ValueError, match="dt"):
        ctrl.step(epsilon_measured=1e5, Q_current=10.0, dt=0.0)


def test_find_threshold_rejects_negative_heating_scan() -> None:
    trigger = LHTrigger(PredatorPreyModel())
    with pytest.raises(ValueError, match="values must be non-negative"):
        trigger.find_threshold(np.array([-1.0, 0.0, 1.0]))


def test_find_threshold_returns_first_h_mode_heating(monkeypatch: pytest.MonkeyPatch) -> None:
    trigger = LHTrigger(PredatorPreyModel())

    def fake_evolve(Q_heating: float, t_span: tuple[float, float], dt: float) -> SimpleNamespace:
        return SimpleNamespace(regime="H_MODE" if Q_heating >= 2.0 else "L_MODE")

    monkeypatch.setattr(trigger.model, "evolve", fake_evolve)
    threshold = trigger.find_threshold(np.array([1.0, 2.0, 3.0]))
    assert threshold == 2.0


@pytest.mark.parametrize(
    ("state", "match"),
    [
        (np.array([1.0, 1.0]), "state must contain epsilon, V_ZF, and pressure"),
        (np.array([np.nan, 1.0, 1.0]), "state values must be finite"),
    ],
)
def test_step_rejects_invalid_state(state: npt.NDArray[np.float64], match: str) -> None:
    model = PredatorPreyModel()
    with pytest.raises(ValueError, match=match):
        model.step(state, dt=1.0e-3, Q_heating=1.0)


def test_evolve_rejects_malformed_time_span() -> None:
    model = PredatorPreyModel()
    with pytest.raises(ValueError, match="t_span must contain start and end times"):
        model.evolve(Q_heating=1.0, t_span=(0.0, 1.0, 2.0), dt=1.0e-3)  # type: ignore[arg-type]


def test_evolve_rejects_step_larger_than_duration() -> None:
    model = PredatorPreyModel()
    with pytest.raises(ValueError, match="dt must not exceed the evolution duration"):
        model.evolve(Q_heating=1.0, t_span=(0.0, 1.0), dt=2.0)


def test_evolve_rejects_too_few_steps() -> None:
    model = PredatorPreyModel()
    with pytest.raises(ValueError, match="evolution must contain at least two steps"):
        model.evolve(Q_heating=1.0, t_span=(0.0, 1.0), dt=0.6)


def test_low_density_branch_rejects_nonfinite_density() -> None:
    with pytest.raises(ValueError, match="ne_19 must be finite"):
        MartinThreshold.power_threshold_with_low_density_branch_MW(
            ne_19=np.nan, B_T=2.0, S_m2=100.0, I_p_MA=15.0, a_m=2.0
        )
