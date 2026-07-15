# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Blob Transport Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.blob_transport import (
    BlobEvent,
    BlobPopulation,
    BlobDetector,
    BlobDynamics,
    BlobEnsemble,
    SOLBlobProfile,
)


def test_blob_dynamics_critical_size():
    dyn = BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0)

    # Check critical size
    delta_star = dyn.critical_size(L_parallel=10.0)
    assert 0.001 < delta_star < 0.1  # Typically ~cm scale

    # Regimes
    v_sheath, reg_sheath = dyn.blob_velocity(delta_star * 0.5, 1e19, 10.0)
    assert reg_sheath == "sheath"

    v_inertial, reg_inertial = dyn.blob_velocity(delta_star * 2.0, 1e19, 10.0)
    assert reg_inertial == "inertial"

    # Velocity at delta_star should be the crossing point
    v_star = dyn.max_velocity(10.0)
    assert v_star > 0.0


def test_blob_ensemble_generation():
    dyn = BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0)
    ens = BlobEnsemble(dyn, n_blobs=100)
    rng = np.random.default_rng(42)

    pop = ens.generate(0.01, 0.002, 1.0, 1e-4, rng)

    assert len(pop.sizes) == 100
    assert len(pop.birth_times) == 100
    assert np.all(pop.sizes > 0)
    assert np.all(pop.amplitudes > 0)
    assert np.all(np.diff(pop.birth_times) > 0.0)

    # Verify fluxes
    gamma = ens.radial_flux(pop)
    q = ens.heat_flux(pop, 20.0)

    assert gamma > 0.0
    assert q > 0.0


def test_sol_blob_profile() -> None:
    # Without blobs
    r_arr = np.linspace(0.0, 0.05, 8)
    prof_clean = SOLBlobProfile.radial_density(r=r_arr, Gamma_blob=0.0, D_perp=1.0, lambda_n=0.02)

    # With blobs
    prof_blobs = SOLBlobProfile.radial_density(r=r_arr, Gamma_blob=1e20, D_perp=1.0, lambda_n=0.02)

    assert np.all((0.0 < prof_clean) & (prof_clean <= 1.0))
    assert np.all((0.0 < prof_blobs) & (prof_blobs <= 1.0))
    assert np.all(np.diff(prof_clean) <= 0.0)
    assert np.all(np.diff(prof_blobs) <= 0.0)
    assert prof_blobs[-1] > prof_clean[-1]
    # Wall flux
    wall_clean = SOLBlobProfile.wall_flux(r_wall=0.1, Gamma_blob=1e18, lambda_n=0.02)
    wall_dirty = SOLBlobProfile.wall_flux(r_wall=0.1, Gamma_blob=1e20, lambda_n=0.02)

    assert wall_dirty > wall_clean


def test_blob_detector():
    det = BlobDetector()

    # Create a synthetic signal with a spike
    sig = np.random.randn(1000) * 0.1
    # Spike at index 500
    sig[500:510] += 5.0

    events = det.detect_blobs(sig, dt=1e-6, threshold=2.5)

    assert len(events) >= 1
    # Spike is around index 500
    assert 495 < events[0].start_idx < 505

    avg = det.conditional_average(sig, events, window=20)
    assert len(avg) == 41
    assert avg[20] > 1.0  # Center of average is the spike


# ── New citation-driven tests ─────────────────────────────────────────


def test_blob_velocity_positive():
    """Blob radial velocity is positive for outward propagation.

    Krasheninnikov 2001, Phys. Lett. A 283, 368, Eq. 5:
        v_b = 2 T_e / (e B R δ_b) × c_s/Ω_i > 0  for δ_b > 0.
    D'Ippolito et al. 2011, Phys. Plasmas 18, 060501: both regimes outward.
    """
    dyn = BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0)
    L_par = 10.0

    # sheath regime
    v_sh, regime_sh = dyn.blob_velocity(dyn.critical_size(L_par) * 0.5, 1e19, L_par)
    assert v_sh > 0.0
    assert regime_sh == "sheath"

    # inertial regime
    v_in, regime_in = dyn.blob_velocity(dyn.critical_size(L_par) * 2.0, 1e19, L_par)
    assert v_in > 0.0
    assert regime_in == "inertial"


def test_blob_size_scaling():
    """Critical blob size increases with connection length.

    Myra et al. 2006, Phys. Plasmas 13, 112502, Eq. 12:
        δ_b* ≈ 2 ρ_s (L_∥ / (R ρ_s))^(1/5) — monotone in L_∥.
    """
    dyn = BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0)

    delta_short = dyn.critical_size(L_parallel=5.0)
    delta_long = dyn.critical_size(L_parallel=50.0)

    assert delta_long > delta_short
    assert delta_short > 0.0


def test_critical_size_zero_l_par():
    """Non-positive connection length is outside the sheath-transition domain."""
    dyn = BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0)
    with pytest.raises(ValueError, match="L_parallel"):
        dyn.critical_size(L_parallel=0.0)
    with pytest.raises(ValueError, match="L_parallel"):
        dyn.critical_size(L_parallel=-1.0)


def test_sheath_velocity_zero_delta():
    """Zero-radius filaments have no outward sheath velocity; negative radius is invalid."""
    dyn = BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0)
    assert dyn.sheath_velocity(0.0) == 0.0
    with pytest.raises(ValueError, match="delta_b"):
        dyn.sheath_velocity(-0.01)


def test_sol_blob_profile_zero_d_perp():
    """Zero cross-field diffusivity falls back to the base exponential profile."""
    r = np.array([0.05])
    prof = SOLBlobProfile.radial_density(r, Gamma_blob=1e20, D_perp=0.0, lambda_n=0.02)
    expected = np.exp(-r / 0.02)
    np.testing.assert_allclose(prof, expected)


def test_blob_detector_flat_signal():
    """A flat signal has no intermittent blob events."""
    det = BlobDetector()
    flat = np.ones(200)
    events = det.detect_blobs(flat)
    assert events == []


def test_conditional_average_empty():
    """Empty event lists produce a zero conditional-average waveform."""
    det = BlobDetector()
    sig = np.random.randn(200)
    avg = det.conditional_average(sig, [], window=10)
    assert len(avg) == 21
    assert np.allclose(avg, 0.0)


def test_conditional_average_event_outside_window_bounds():
    """An event whose window overruns the signal contributes nothing, leaving a zero waveform."""
    det = BlobDetector()
    sig = np.ones(10)
    # center = 5; window = 8 puts center - window at -3, so the event fits no full window.
    event = BlobEvent(start_idx=4, end_idx=6, peak_amplitude=1.0, duration=1e-6, size_estimate=0.01)
    avg = det.conditional_average(sig, [event], window=8)
    assert len(avg) == 17
    assert np.allclose(avg, 0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"R0": 0.0, "B0": 5.3, "Te_eV": 20.0, "Ti_eV": 20.0}, "R0"),
        ({"R0": 6.2, "B0": -1.0, "Te_eV": 20.0, "Ti_eV": 20.0}, "B0"),
        ({"R0": 6.2, "B0": 5.3, "Te_eV": -1.0, "Ti_eV": 20.0}, "non-negative"),
        ({"R0": 6.2, "B0": 5.3, "Te_eV": 0.0, "Ti_eV": 0.0}, "must be positive"),
        ({"R0": 6.2, "B0": 5.3, "Te_eV": 20.0, "Ti_eV": 20.0, "mi_amu": 0.0}, "mi_amu"),
    ),
)
def test_blob_dynamics_rejects_nonphysical_constructor_inputs(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        BlobDynamics(**kwargs)


def test_blob_dynamics_rejects_nonfinite_and_invalid_velocity_domains() -> None:
    with pytest.raises(ValueError, match="R0"):
        BlobDynamics(R0=float("nan"), B0=5.3, Te_eV=20.0, Ti_eV=20.0)

    dyn = BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0)
    with pytest.raises(ValueError, match="L_parallel"):
        dyn.critical_size(float("nan"))
    with pytest.raises(ValueError, match="delta_b"):
        dyn.sheath_velocity(float("nan"))
    with pytest.raises(ValueError, match="delta_b"):
        dyn.inertial_velocity(-0.01)
    with pytest.raises(ValueError, match="n_e"):
        dyn.blob_velocity(delta_b=0.01, n_e=0.0, L_parallel=10.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {"delta_b_mean": 0.0, "delta_b_sigma": 0.002, "amplitude_mean": 1.0, "waiting_time_mean": 1e-4},
            "delta_b_mean",
        ),
        (
            {"delta_b_mean": 0.01, "delta_b_sigma": -0.1, "amplitude_mean": 1.0, "waiting_time_mean": 1e-4},
            "delta_b_sigma",
        ),
        (
            {"delta_b_mean": 0.01, "delta_b_sigma": 0.002, "amplitude_mean": 0.0, "waiting_time_mean": 1e-4},
            "amplitude_mean",
        ),
        (
            {"delta_b_mean": 0.01, "delta_b_sigma": 0.002, "amplitude_mean": 1.0, "waiting_time_mean": 0.0},
            "waiting_time_mean",
        ),
    ),
)
def test_blob_ensemble_rejects_nonphysical_distribution_inputs(kwargs, message) -> None:
    dyn = BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0)
    ens = BlobEnsemble(dyn, n_blobs=10)

    with pytest.raises(ValueError, match=message):
        ens.generate(rng=np.random.default_rng(7), **kwargs)


def test_blob_flux_empty_population_is_zero() -> None:
    dyn = BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0)
    ens = BlobEnsemble(dyn, n_blobs=0)
    pop = BlobPopulation(np.array([]), np.array([]), np.array([]), np.array([]))

    assert ens.radial_flux(pop) == 0.0
    assert ens.heat_flux(pop, Te_eV=20.0) == 0.0


def test_blob_flux_rejects_nonpositive_elapsed_time() -> None:
    dyn = BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0)
    ens = BlobEnsemble(dyn, n_blobs=1)
    pop = BlobPopulation(np.array([0.01]), np.array([1.0]), np.array([100.0]), np.array([0.0]))

    with pytest.raises(ValueError, match="positive elapsed time"):
        ens.radial_flux(pop)


def test_blob_flux_rejects_malformed_population_before_empty_time_shortcut() -> None:
    dyn = BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0)
    ens = BlobEnsemble(dyn, n_blobs=1)
    pop = BlobPopulation(np.array([0.01]), np.array([1.0]), np.array([100.0]), np.array([]))

    with pytest.raises(ValueError, match="same length"):
        ens.radial_flux(pop)


def test_blob_population_rejects_malformed_or_nonfinite_arrays() -> None:
    dyn = BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0)
    ens = BlobEnsemble(dyn, n_blobs=1)

    mismatched = BlobPopulation(np.array([0.01]), np.array([1.0, 2.0]), np.array([100.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="same length"):
        ens.radial_flux(mismatched)

    nonfinite = BlobPopulation(np.array([0.01]), np.array([np.nan]), np.array([100.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="finite"):
        ens.radial_flux(nonfinite)

    negative = BlobPopulation(np.array([-0.01]), np.array([1.0]), np.array([100.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="sizes"):
        ens.radial_flux(negative)

    nonmonotonic_births = BlobPopulation(
        np.array([0.01, 0.02]),
        np.array([1.0, 1.0]),
        np.array([100.0, 120.0]),
        np.array([1.0, 0.5]),
    )
    with pytest.raises(ValueError, match="birth times must be strictly increasing"):
        ens.radial_flux(nonmonotonic_births)


@pytest.mark.parametrize(
    ("r_wall", "gamma", "lambda_n", "message"),
    (
        (-0.1, 1e18, 0.02, "r_wall"),
        (0.1, -1.0, 0.02, "Gamma_blob"),
        (0.1, 1e18, 0.0, "lambda_n"),
    ),
)
def test_sol_blob_profile_rejects_nonphysical_wall_flux_inputs(r_wall, gamma, lambda_n, message) -> None:
    with pytest.raises(ValueError, match=message):
        SOLBlobProfile.wall_flux(r_wall=r_wall, Gamma_blob=gamma, lambda_n=lambda_n)


def test_sol_blob_profile_rejects_nonphysical_density_inputs() -> None:
    r = np.array([0.05])

    with pytest.raises(ValueError, match="non-empty"):
        SOLBlobProfile.radial_density(np.array([]), Gamma_blob=1e20, D_perp=1.0, lambda_n=0.02)
    with pytest.raises(ValueError, match="lambda_n"):
        SOLBlobProfile.radial_density(r, Gamma_blob=1e20, D_perp=1.0, lambda_n=0.0)
    with pytest.raises(ValueError, match="Gamma_blob"):
        SOLBlobProfile.radial_density(r, Gamma_blob=-1.0, D_perp=1.0, lambda_n=0.02)
    with pytest.raises(ValueError, match="r"):
        SOLBlobProfile.radial_density(np.array([0.0, np.nan]), Gamma_blob=1e20, D_perp=1.0, lambda_n=0.02)
    with pytest.raises(ValueError, match="D_perp"):
        SOLBlobProfile.radial_density(r, Gamma_blob=1e20, D_perp=-1.0, lambda_n=0.02)
    with pytest.raises(ValueError, match="ordered"):
        SOLBlobProfile.radial_density(np.array([0.02, 0.01]), Gamma_blob=1e20, D_perp=1.0, lambda_n=0.02)
    with pytest.raises(ValueError, match="strictly ordered"):
        SOLBlobProfile.radial_density(np.array([0.0, 0.01, 0.01]), Gamma_blob=1e20, D_perp=1.0, lambda_n=0.02)


def test_blob_ensemble_rejects_invalid_population_count_at_construction() -> None:
    dyn = BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0)

    for n_blobs in (-1, True):
        with pytest.raises(ValueError, match="n_blobs"):
            BlobEnsemble(dyn, n_blobs=n_blobs)


def test_blob_detector_closes_event_at_signal_boundary() -> None:
    det = BlobDetector()
    signal = np.zeros(40)
    signal[30:] = 5.0

    events = det.detect_blobs(signal, dt=2e-6, threshold=1.0)

    assert len(events) == 1
    assert events[0].start_idx == 30
    assert events[0].end_idx == len(signal)
    assert events[0].duration == pytest.approx(20e-6)


def test_blob_detector_rejects_nonphysical_event_parameters() -> None:
    det = BlobDetector()
    signal = np.zeros(10)
    malformed_event = BlobEvent(start_idx=8, end_idx=4, peak_amplitude=1.0, duration=-4e-6, size_estimate=-0.004)

    with pytest.raises(ValueError, match="dt"):
        det.detect_blobs(signal, dt=0.0)
    with pytest.raises(ValueError, match="threshold"):
        det.detect_blobs(signal, threshold=0.0)
    with pytest.raises(ValueError, match="window"):
        det.conditional_average(signal, [], window=-1)
    with pytest.raises(ValueError, match="signal"):
        det.detect_blobs(np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="signal"):
        det.conditional_average(np.ones((2, 2)), [], window=1)
    with pytest.raises(ValueError, match="event"):
        det.conditional_average(signal, [malformed_event], window=1)


def test_blob_flux_rejects_negative_amplitude() -> None:
    """A negative blob amplitude trips the non-negative array guard.

    Sizes use the strictly-positive check; amplitudes, velocities and birth
    times use the non-negative check, so a valid positive size with a negative
    amplitude exercises the distinct non-negative branch.
    """
    ens = BlobEnsemble(BlobDynamics(R0=6.2, B0=5.3, Te_eV=20.0, Ti_eV=20.0, mi_amu=2.0), n_blobs=1)
    pop = BlobPopulation(np.array([0.01]), np.array([-1.0]), np.array([100.0]), np.array([1.0]))

    with pytest.raises(ValueError, match="must be non-negative"):
        ens.radial_flux(pop)


def test_conditional_average_rejects_non_integer_event_indices() -> None:
    """Boolean event indices are rejected even though bool is an int subtype.

    ``BlobEvent`` indices must be true integers; the validator explicitly
    excludes ``bool`` so a truthy index cannot masquerade as a sample position.
    """
    det = BlobDetector()
    signal = np.zeros(10)
    bool_index_event = BlobEvent(start_idx=True, end_idx=4, peak_amplitude=1.0, duration=4e-6, size_estimate=0.004)

    with pytest.raises(ValueError, match="event indices must be integers"):
        det.conditional_average(signal, [bool_index_event], window=1)
