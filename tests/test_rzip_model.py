# ──────────────────────────────────────────────────────────────────────
# SCPN Control — RZIP Model Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.rzip_model import RZIPController, RZIPModel
from scpn_control.core.vessel_model import VesselElement, VesselModel


@pytest.fixture
def simple_vessel():
    # Two loops above and below midplane to act as a wall
    elements = [
        VesselElement(R=2.0, Z=0.5, resistance=1e-3, cross_section=0.1, inductance=1e-5),
        VesselElement(R=2.0, Z=-0.5, resistance=1e-3, cross_section=0.1, inductance=1e-5),
    ]
    return VesselModel(elements)


@pytest.fixture
def active_coils():
    return [
        VesselElement(R=2.5, Z=1.0, resistance=1e-2, cross_section=0.05, inductance=5e-5),
        VesselElement(R=2.5, Z=-1.0, resistance=1e-2, cross_section=0.05, inductance=5e-5),
    ]


def test_stable_n_index(simple_vessel):
    # n_index > 0 means vertically stable
    rzip = RZIPModel(R0=2.0, a=0.5, kappa=1.0, Ip_MA=1.0, B0=1.0, n_index=0.5, vessel=simple_vessel)
    gamma = rzip.vertical_growth_rate()
    assert gamma <= 0.0


def test_unstable_n_index(simple_vessel):
    # n_index < 0 means vertically unstable
    rzip = RZIPModel(R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-1.0, vessel=simple_vessel)
    gamma = rzip.vertical_growth_rate()
    assert gamma > 0.0

    # Growth time should be realistic (e.g. 10s to 1000s of ms)
    tau = rzip.vertical_growth_time()
    assert 1.0 < tau < 10000.0


def test_wall_slows_growth():
    elements_highly_resistive = [
        VesselElement(R=2.0, Z=0.5, resistance=1e3, cross_section=0.1, inductance=1e-5),
        VesselElement(R=2.0, Z=-0.5, resistance=1e3, cross_section=0.1, inductance=1e-5),
    ]
    elements_conductive = [
        VesselElement(R=2.0, Z=0.5, resistance=1e-5, cross_section=0.1, inductance=1e-5),
        VesselElement(R=2.0, Z=-0.5, resistance=1e-5, cross_section=0.1, inductance=1e-5),
    ]
    vessel_res = VesselModel(elements_highly_resistive)
    vessel_cond = VesselModel(elements_conductive)

    rzip_res = RZIPModel(R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-1.0, vessel=vessel_res)
    rzip_cond = RZIPModel(R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-1.0, vessel=vessel_cond)

    # Conductive wall should lower the growth rate (slow down the instability)
    assert rzip_cond.vertical_growth_rate() < rzip_res.vertical_growth_rate()


def test_state_space_dimensions(simple_vessel, active_coils):
    rzip = RZIPModel(
        R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-1.0, vessel=simple_vessel, active_coils=active_coils
    )
    A, B, C, D = rzip.build_state_space()

    n_wall = len(simple_vessel.elements)
    n_coils = len(active_coils)
    n_states = 2 + n_wall + n_coils

    assert A.shape == (n_states, n_states)
    assert B.shape == (n_states, n_coils)
    assert C.shape == (1, n_states)
    assert D.shape == (1, n_coils)


def test_feedback_stabilization(active_coils):
    # Use a resistive wall so it doesn't shield the active coils completely
    elements = [
        VesselElement(R=2.0, Z=0.5, resistance=1e0, cross_section=0.1, inductance=1e-5),
        VesselElement(R=2.0, Z=-0.5, resistance=1e0, cross_section=0.1, inductance=1e-5),
    ]
    resistive_vessel = VesselModel(elements)

    rzip = RZIPModel(
        R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-0.5, vessel=resistive_vessel, active_coils=active_coils
    )

    assert rzip.vertical_growth_rate() > 0.0

    ctrl = RZIPController(rzip, Kp=0.0, Kd=-1e4)
    eigvals = ctrl.closed_loop_eigenvalues()

    max_real = np.max(np.real(eigvals))
    assert max_real < rzip.vertical_growth_rate()  # Growth rate should at least decrease


def test_singular_m_mat(simple_vessel):
    """Lines 90-91: build_state_space handles singular M_mat with zero fallback."""
    # Create vessel with zero inductance to force singular M_mat
    elements = [
        VesselElement(R=2.0, Z=0.5, resistance=1e-3, cross_section=0.1, inductance=0.0),
        VesselElement(R=2.0, Z=-0.5, resistance=1e-3, cross_section=0.1, inductance=0.0),
    ]
    vessel = VesselModel(elements)
    rzip = RZIPModel(R0=2.0, a=0.5, kappa=1.0, Ip_MA=1.0, B0=1.0, n_index=0.5, vessel=vessel)
    A, B, C, D = rzip.build_state_space()
    assert np.all(np.isfinite(A))


def test_vertical_growth_time_stable(simple_vessel):
    """Line 127: vertical_growth_time returns inf for stable plasma (gamma <= 0)."""
    rzip = RZIPModel(R0=2.0, a=0.5, kappa=1.0, Ip_MA=1.0, B0=1.0, n_index=0.5, vessel=simple_vessel)
    tau = rzip.vertical_growth_time()
    assert tau == float("inf")


def test_stability_margin(simple_vessel):
    """Line 132: stability_margin returns n_index."""
    rzip = RZIPModel(R0=2.0, a=0.5, kappa=1.0, Ip_MA=1.0, B0=1.0, n_index=-0.5, vessel=simple_vessel)
    assert rzip.stability_margin() == -0.5


def test_compute_n_index():
    """Line 142: VerticalStabilityAnalysis.compute_n_index returns stub."""
    from scpn_control.control.rzip_model import VerticalStabilityAnalysis

    result = VerticalStabilityAnalysis.compute_n_index(np.zeros(10), np.zeros(10), np.zeros(10), 2.0)
    assert result == -1.0


def test_passive_stability_margin():
    """Line 146: VerticalStabilityAnalysis.passive_stability_margin returns n_index."""
    from scpn_control.control.rzip_model import VerticalStabilityAnalysis

    assert VerticalStabilityAnalysis.passive_stability_margin(-0.3, 0.01) == -0.3


def test_required_feedback_gain():
    """Line 150: VerticalStabilityAnalysis.required_feedback_gain returns 1.0."""
    from scpn_control.control.rzip_model import VerticalStabilityAnalysis

    assert VerticalStabilityAnalysis.required_feedback_gain(10.0, 0.01, 1e-4) == 1.0


def test_controller_step_zero_dt(active_coils):
    """Lines 178-181: RZIPController.step with dt=0 sets dZ_dt=0."""
    elements = [
        VesselElement(R=2.0, Z=0.5, resistance=1e0, cross_section=0.1, inductance=1e-5),
        VesselElement(R=2.0, Z=-0.5, resistance=1e0, cross_section=0.1, inductance=1e-5),
    ]
    vessel = VesselModel(elements)
    rzip = RZIPModel(
        R0=2.0, a=0.5, kappa=1.7, Ip_MA=1.0, B0=1.0, n_index=-0.5, vessel=vessel, active_coils=active_coils
    )
    ctrl = RZIPController(rzip, Kp=1.0, Kd=1.0)
    V = ctrl.step(0.1, dt=0.0)
    assert V.shape == (len(active_coils),)
    assert np.all(np.isfinite(V))
