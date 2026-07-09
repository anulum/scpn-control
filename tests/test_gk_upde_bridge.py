# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK → UPDE Bridge Tests
from __future__ import annotations

import logging

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_control.core.gk_interface import GKOutput
from scpn_control.phase.gk_upde_bridge import GKCouplingGains, adaptive_knm, gk_natural_frequencies


@pytest.fixture
def k_base():
    return 0.3 * np.ones((8, 8))


@pytest.fixture
def gk_output_unstable():
    return GKOutput(
        chi_i=2.5,
        chi_e=1.8,
        D_e=0.4,
        gamma=np.array([0.1, 0.25, 0.15]),
        omega_r=np.array([-0.3, -0.5, 0.2]),
        k_y=np.array([0.1, 0.3, 0.5]),
        dominant_mode="ITG",
    )


@pytest.fixture
def gk_output_stable():
    return GKOutput(
        chi_i=0.01,
        chi_e=0.01,
        D_e=0.001,
        gamma=np.array([0.0, 0.0]),
        omega_r=np.array([0.0, 0.0]),
        k_y=np.array([0.1, 0.3]),
        dominant_mode="stable",
    )


def test_adaptive_knm_shape(k_base, gk_output_unstable):
    K = adaptive_knm(k_base, gk_output_unstable)
    assert K.shape == (8, 8)


def test_adaptive_knm_symmetric_coupling(k_base, gk_output_unstable):
    K = adaptive_knm(k_base, gk_output_unstable)
    assert K[0, 1] == K[1, 0]
    assert K[1, 4] == K[4, 1]


def test_adaptive_knm_increases_with_gamma(k_base, gk_output_unstable, gk_output_stable):
    K_unstable = adaptive_knm(k_base, gk_output_unstable)
    K_stable = adaptive_knm(k_base, gk_output_stable)
    # P0↔P1 coupling should be stronger with instability
    assert K_unstable[0, 1] >= K_stable[0, 1]


def test_adaptive_knm_damped_growth_does_not_suppress_baseline_coupling(k_base):
    gk = GKOutput(
        chi_i=0.1,
        chi_e=0.1,
        D_e=0.01,
        gamma=np.array([-0.8, -0.2]),
        omega_r=np.array([-0.3, 0.1]),
        k_y=np.array([0.2, 0.4]),
        dominant_mode="damped",
    )
    K = adaptive_knm(k_base, gk)
    assert K[0, 1] == pytest.approx(k_base[0, 1])
    assert K[1, 0] == pytest.approx(k_base[1, 0])


def test_adaptive_knm_pedestal_ratio(k_base, gk_output_unstable):
    chi_i_profile = np.ones(50)
    chi_i_profile[-10:] = 0.1  # low pedestal transport
    K = adaptive_knm(k_base, gk_output_unstable, chi_i_profile=chi_i_profile)
    # P3↔P4 should be modified
    assert K[3, 4] != k_base[3, 4]


def test_adaptive_knm_rejects_nonfinite_transport_inputs(k_base, gk_output_unstable):
    bad = GKOutput(
        chi_i=gk_output_unstable.chi_i,
        chi_e=np.nan,
        D_e=gk_output_unstable.D_e,
        gamma=gk_output_unstable.gamma,
        omega_r=gk_output_unstable.omega_r,
        k_y=gk_output_unstable.k_y,
        dominant_mode=gk_output_unstable.dominant_mode,
    )
    with pytest.raises(ValueError, match="chi_e"):
        adaptive_knm(k_base, bad)


def test_adaptive_knm_rejects_nonfinite_growth_rates(k_base, gk_output_unstable):
    bad = GKOutput(
        chi_i=gk_output_unstable.chi_i,
        chi_e=gk_output_unstable.chi_e,
        D_e=gk_output_unstable.D_e,
        gamma=np.array([0.1, np.inf]),
        omega_r=gk_output_unstable.omega_r,
        k_y=gk_output_unstable.k_y,
        dominant_mode=gk_output_unstable.dominant_mode,
    )
    with pytest.raises(ValueError, match="gamma"):
        adaptive_knm(k_base, bad)


def test_adaptive_knm_rejects_invalid_pedestal_profile(k_base, gk_output_unstable):
    chi_i_profile = np.ones(50)
    chi_i_profile[-1] = np.nan
    with pytest.raises(ValueError, match="chi_i_profile"):
        adaptive_knm(k_base, gk_output_unstable, chi_i_profile=chi_i_profile)


def test_adaptive_knm_small_matrix_warns_and_returns_unchanged(caplog):
    K_small = np.eye(4) * 0.3
    gk = GKOutput(chi_i=1.0, chi_e=0.8, D_e=0.1, gamma=np.array([0.2]), omega_r=np.array([-0.3]), k_y=np.array([0.3]))
    with caplog.at_level(logging.WARNING, logger="scpn_control.phase.gk_upde_bridge"):
        K_out = adaptive_knm(K_small, gk)
    np.testing.assert_array_equal(K_out, K_small)
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    assert "unmodulated" in caplog.text


def test_gk_coupling_gains_defaults_are_documented_values():
    gains = GKCouplingGains()
    assert gains.p0_p1_turbulence == 0.5
    assert gains.p1_p4_transport == 0.3
    assert gains.p1_p4_chi_clip_max == 2.0
    assert gains.p3_p4_pedestal == 0.4
    assert gains.diffusivity_floor == 1e-10


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"p0_p1_turbulence": -0.1}, "p0_p1_turbulence must be finite and non-negative"),
        ({"p1_p4_transport": float("nan")}, "p1_p4_transport must be finite and non-negative"),
        ({"p3_p4_pedestal": float("-inf")}, "p3_p4_pedestal must be finite and non-negative"),
        ({"p1_p4_chi_clip_max": 0.0}, "p1_p4_chi_clip_max must be finite and positive"),
        ({"diffusivity_floor": 0.0}, "diffusivity_floor must be finite and positive"),
    ],
)
def test_gk_coupling_gains_rejects_invalid(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GKCouplingGains(**kwargs)


def test_adaptive_knm_custom_gains_strengthen_coupling(k_base, gk_output_unstable):
    default = adaptive_knm(k_base, gk_output_unstable)
    stronger = adaptive_knm(k_base, gk_output_unstable, gains=GKCouplingGains(p0_p1_turbulence=1.0))
    # A larger turbulence gain yields stronger P0<->P1 coupling for the same drive.
    assert stronger[0, 1] > default[0, 1]
    assert stronger[1, 0] == stronger[0, 1]


def test_gk_natural_frequencies(gk_output_unstable):
    omega_base = np.ones(8)
    omega = gk_natural_frequencies(omega_base, gk_output_unstable)
    assert omega[0] > omega_base[0]  # layer 0 frequency increased
    np.testing.assert_array_equal(omega[1:], omega_base[1:])  # others unchanged


def test_gk_natural_frequencies_stable(gk_output_stable):
    omega_base = np.ones(8)
    omega = gk_natural_frequencies(omega_base, gk_output_stable)
    assert omega[0] == omega_base[0]  # no growth → no change


def test_gk_natural_frequencies_uses_positive_growth_drive_only():
    gk = GKOutput(
        chi_i=0.1,
        chi_e=0.1,
        D_e=0.01,
        gamma=np.array([-0.5, -0.1]),
        omega_r=np.array([0.0, 0.0]),
        k_y=np.array([0.2, 0.4]),
        dominant_mode="damped",
    )
    omega_base = np.ones(8)
    omega = gk_natural_frequencies(omega_base, gk)
    np.testing.assert_array_equal(omega, omega_base)


def _gk_output(gamma: NDArray[np.float64]) -> GKOutput:
    """Build a minimal GKOutput carrying the given growth-rate spectrum."""
    size = gamma.size
    return GKOutput(
        chi_i=1.0,
        chi_e=0.8,
        D_e=0.1,
        gamma=gamma,
        omega_r=np.zeros(size),
        k_y=np.linspace(0.1, 0.5, size, dtype=np.float64) if size else np.zeros(0, dtype=np.float64),
        dominant_mode="ITG",
    )


def test_adaptive_knm_rejects_non_square_coupling_matrix() -> None:
    """A non-square baseline coupling matrix is rejected before modulation."""
    with pytest.raises(ValueError, match="square matrix"):
        adaptive_knm(np.ones((3, 4)), _gk_output(np.array([0.2])))


def test_adaptive_knm_rejects_nonfinite_coupling_matrix() -> None:
    """A baseline coupling matrix with a non-finite entry is rejected."""
    k_base = 0.3 * np.ones((8, 8))
    k_base[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite values"):
        adaptive_knm(k_base, _gk_output(np.array([0.2])))


def test_adaptive_knm_accepts_empty_growth_spectrum() -> None:
    """An empty growth-rate spectrum yields zero drive and baseline coupling."""
    k_base = 0.3 * np.ones((8, 8))
    K = adaptive_knm(k_base, _gk_output(np.zeros(0)))
    assert K.shape == (8, 8)
    assert K[0, 1] == pytest.approx(k_base[0, 1])


def test_adaptive_knm_rejects_nonpositive_reference_scales() -> None:
    """Reference growth and transport scales must be strictly positive."""
    k_base = 0.3 * np.ones((8, 8))
    gk = _gk_output(np.array([0.2]))
    with pytest.raises(ValueError, match="gamma_ref must be finite and positive"):
        adaptive_knm(k_base, gk, gamma_ref=0.0)
    with pytest.raises(ValueError, match="chi_ref must be finite and positive"):
        adaptive_knm(k_base, gk, chi_ref=-1.0)


def test_gk_natural_frequencies_rejects_nonfinite_base_frequencies() -> None:
    """Base natural frequencies must all be finite."""
    omega_base = np.ones(8)
    omega_base[2] = np.nan
    with pytest.raises(ValueError, match="omega_base"):
        gk_natural_frequencies(omega_base, _gk_output(np.array([0.2])))


def test_gk_natural_frequencies_rejects_negative_growth_scale() -> None:
    """A negative growth-scale factor is rejected as non-physical."""
    with pytest.raises(ValueError, match="gamma_scale"):
        gk_natural_frequencies(np.ones(8), _gk_output(np.array([0.2])), gamma_scale=-0.1)
