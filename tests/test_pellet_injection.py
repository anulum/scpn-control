# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Pellet Injection
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Pellet Injection Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.pellet_injection import (
    PelletFuelingController,
    PelletParams,
    PelletTrajectory,
    ngs_ablation_rate,
    pellet_pacing_elm_control,
)


def test_ablation_rate():
    # Scaling tests
    rate1 = ngs_ablation_rate(0.002, 1e20, 1000.0, 2.0)
    rate2 = ngs_ablation_rate(0.002, 1e20, 2000.0, 2.0)

    # Te^(5/3)
    assert np.isclose(rate2 / rate1, 2.0 ** (5.0 / 3.0))


def test_pellet_trajectory_penetration():
    params1 = PelletParams(r_p_mm=4.0, v_p_m_s=300.0)  # Slower
    params2 = PelletParams(r_p_mm=4.0, v_p_m_s=1000.0)  # Faster

    rho = np.linspace(0, 1, 50)
    ne = np.ones(50) * 5.0  # 5e19
    Te = np.ones(50) * 10000.0  # High temp so they ablate

    traj1 = PelletTrajectory(params1, R0=6.2, a=2.0, B0=5.3)
    res1 = traj1.simulate(rho, ne, Te)

    traj2 = PelletTrajectory(params2, R0=6.2, a=2.0, B0=5.3)
    res2 = traj2.simulate(rho, ne, Te)

    # A faster pellet will always reach a *smaller* or equal final rho.
    assert res2.penetration_depth <= res1.penetration_depth


def test_pellet_drift():
    params_hfs = PelletParams(r_p_mm=4.0, v_p_m_s=300.0, injection_side="HFS")
    params_lfs = PelletParams(r_p_mm=4.0, v_p_m_s=300.0, injection_side="LFS")

    rho = np.linspace(0, 1, 50)
    ne = np.ones(50) * 5.0
    Te = np.ones(50) * 2000.0

    t_hfs = PelletTrajectory(params_hfs, 6.2, 2.0, 5.3)
    res_hfs = t_hfs.simulate(rho, ne, Te)

    t_lfs = PelletTrajectory(params_lfs, 6.2, 2.0, 5.3)
    res_lfs = t_lfs.simulate(rho, ne, Te)

    assert res_hfs.drift_displacement < 0.0  # Inward
    assert res_lfs.drift_displacement > 0.0  # Outward


def test_pellet_pacing():
    # Natural: 5 Hz, 20 MJ
    # Pacing: 20 Hz
    f, w = pellet_pacing_elm_control(20.0, 5.0, 20.0)

    assert f == 20.0
    # W should be reduced by 5/20 = 1/4 -> 5 MJ
    assert np.isclose(w, 5.0)


def test_fueling_controller():
    params = PelletParams(4.0, 300.0)
    ctrl = PelletFuelingController(target_density=10.0, pellet_params=params)

    rho = np.linspace(0, 1, 50)
    ne_low = np.ones(50) * 5.0
    Te = np.ones(50) * 5000.0
    V = 800.0

    # Wait for period
    for _ in range(10):
        ctrl.step(ne_low, Te, 0.1, V)

    cmd = ctrl.step(ne_low, Te, 1.0, V)
    assert cmd is not None
    assert cmd.pellet_params.r_p_mm == 4.0


def test_pellet_trajectory_edge_clamp():
    """Trajectory interpolation remains finite at radial-grid boundaries."""
    params = PelletParams(r_p_mm=2.0, v_p_m_s=500.0)
    rho = np.linspace(0, 1, 10)
    ne = np.ones(10) * 1.0
    Te = np.ones(10) * 500.0
    traj = PelletTrajectory(params, R0=6.2, a=2.0, B0=5.3)
    res = traj.simulate(rho, ne, Te)
    assert res.total_particles > 0.0


def test_ablation_clamp():
    """A high-temperature plasma produces finite positive ablation."""
    rate = ngs_ablation_rate(0.001, 1e21, 50000.0, 2.0)
    assert rate > 0.0


def test_pellet_shift_positive():
    """LFS injection shifts deposition outward under the drift model."""
    params_lfs = PelletParams(r_p_mm=4.0, v_p_m_s=300.0, injection_side="LFS")
    rho = np.linspace(0, 1, 50)
    ne = np.ones(50) * 5.0
    Te = np.ones(50) * 2000.0
    traj = PelletTrajectory(params_lfs, R0=6.2, a=2.0, B0=5.3)
    res = traj.simulate(rho, ne, Te)
    assert res.deposition_profile is not None
    assert res.drift_displacement > 0.0


def test_pellet_pacing_no_trigger():
    """Pellet frequency below the pacing threshold returns natural ELM values."""
    f, w = pellet_pacing_elm_control(5.0, 5.0, 20.0)
    assert f == 5.0
    assert w == 20.0


def test_ablation_rejects_nonphysical_domains():
    with pytest.raises(ValueError, match="r_p"):
        ngs_ablation_rate(0.0, 1e20, 1000.0, 2.0)
    with pytest.raises(ValueError, match="ne"):
        ngs_ablation_rate(0.002, float("nan"), 1000.0, 2.0)
    with pytest.raises(ValueError, match="Te_eV"):
        ngs_ablation_rate(0.002, 1e20, 0.0, 2.0)
    with pytest.raises(ValueError, match="M_p"):
        ngs_ablation_rate(0.002, 1e20, 1000.0, 0.0)


def test_pellet_params_and_trajectory_reject_nonphysical_domains():
    with pytest.raises(ValueError, match="r_p_mm"):
        PelletParams(r_p_mm=0.0, v_p_m_s=300.0)
    with pytest.raises(ValueError, match="injection_side"):
        PelletParams(r_p_mm=4.0, v_p_m_s=300.0, injection_side="BAD")

    params = PelletParams(r_p_mm=4.0, v_p_m_s=300.0)
    with pytest.raises(ValueError, match="R0"):
        PelletTrajectory(params, R0=0.0, a=2.0, B0=5.3)


def test_pellet_trajectory_rejects_malformed_profiles():
    params = PelletParams(r_p_mm=4.0, v_p_m_s=300.0)
    traj = PelletTrajectory(params, R0=6.2, a=2.0, B0=5.3)
    rho = np.linspace(0.0, 1.0, 50)
    ne = np.ones(50) * 5.0
    Te = np.ones(50) * 1000.0

    with pytest.raises(ValueError, match="rho"):
        traj.simulate(rho[::-1], ne, Te)
    with pytest.raises(ValueError, match="ne"):
        traj.simulate(rho, np.array([1.0, np.nan] + [1.0] * 48), Te)
    with pytest.raises(ValueError, match="Te_eV"):
        traj.simulate(rho, ne, -Te)


def test_fueling_controller_and_pacing_reject_nonphysical_domains():
    params = PelletParams(4.0, 300.0)
    with pytest.raises(ValueError, match="target_density"):
        PelletFuelingController(target_density=0.0, pellet_params=params)

    ctrl = PelletFuelingController(target_density=10.0, pellet_params=params)
    with pytest.raises(ValueError, match="tau_p"):
        ctrl.required_frequency(ne_current=5.0, tau_p=0.0, V_plasma=800.0)
    with pytest.raises(ValueError, match="dt"):
        ctrl.step(np.ones(50), np.ones(50), dt=0.0, V_plasma=800.0)

    with pytest.raises(ValueError, match="f_pellet_Hz"):
        pellet_pacing_elm_control(-1.0, 5.0, 20.0)
    with pytest.raises(ValueError, match="f_elm_natural_Hz"):
        pellet_pacing_elm_control(20.0, 0.0, 20.0)


# ── Coverage completion: validators, trajectory branches, controller guards ──


def test_trajectory_rejects_minor_radius_not_smaller_than_major():
    params = PelletParams(r_p_mm=4.0, v_p_m_s=300.0)
    with pytest.raises(ValueError, match="a must be smaller than R0"):
        PelletTrajectory(params, R0=2.0, a=2.0, B0=5.3)


def test_simulate_rejects_profile_shape_mismatch():
    params = PelletParams(r_p_mm=4.0, v_p_m_s=300.0)
    traj = PelletTrajectory(params, R0=6.2, a=2.0, B0=5.3)
    rho = np.linspace(0.0, 1.0, 50)
    with pytest.raises(ValueError, match="ne must match rho shape"):
        traj.simulate(rho, np.ones(49), np.ones(50))


def test_simulate_rejects_degenerate_or_non_finite_rho():
    params = PelletParams(r_p_mm=4.0, v_p_m_s=300.0)
    traj = PelletTrajectory(params, R0=6.2, a=2.0, B0=5.3)
    with pytest.raises(ValueError, match="one-dimensional grid with at least two points"):
        traj.simulate(np.array([0.5]), np.ones(1), np.ones(1))
    with pytest.raises(ValueError, match="rho must contain only finite values"):
        traj.simulate(np.array([0.0, np.nan]), np.ones(2), np.ones(2))


def test_simulate_handles_grid_interior_to_unit_interval():
    # rho spanning (0.1, 0.9) exercises both the idx==0 (inner) and idx>=len
    # (outer) interpolation clamps as the pellet crosses the grid boundaries.
    params = PelletParams(r_p_mm=4.0, v_p_m_s=300.0)
    traj = PelletTrajectory(params, R0=6.2, a=2.0, B0=5.3)
    rho = np.linspace(0.1, 0.9, 20)
    res = traj.simulate(rho, np.ones(20) * 1.0, np.ones(20) * 300.0)
    assert res.deposition_profile.shape == rho.shape
    assert np.all(np.isfinite(res.deposition_profile))


def test_simulate_fully_ablates_small_pellet_in_hot_dense_plasma():
    # A very small, slow pellet in a hot dense plasma is fully consumed, exercising
    # the per-step mass clamp (dN > remaining) and the zeroed-radius branch.
    params = PelletParams(r_p_mm=0.002, v_p_m_s=20.0)
    traj = PelletTrajectory(params, R0=6.2, a=2.0, B0=5.3)
    res = traj.simulate(np.linspace(0.0, 1.0, 40), np.ones(40) * 40.0, np.ones(40) * 2.0e5)
    # The pellet is consumed entirely: deposited count == initial inventory.
    assert res.total_particles == pytest.approx(traj.N_initial, rel=1e-9)


def test_simulate_shifts_deposition_outward_for_lfs_drift():
    params = PelletParams(r_p_mm=4.0, v_p_m_s=300.0, injection_side="LFS")
    traj = PelletTrajectory(params, R0=6.2, a=2.0, B0=0.5)
    res = traj.simulate(np.linspace(0.0, 1.0, 50), np.ones(50) * 8.0, np.ones(50) * 8.0e4)
    assert res.drift_displacement > 0.0


def test_simulate_shifts_deposition_inward_for_hfs_drift():
    params = PelletParams(r_p_mm=4.0, v_p_m_s=300.0, injection_side="HFS")
    traj = PelletTrajectory(params, R0=6.2, a=2.0, B0=0.5)
    res = traj.simulate(np.linspace(0.0, 1.0, 50), np.ones(50) * 8.0, np.ones(50) * 8.0e4)
    assert res.drift_displacement < 0.0


def test_fueling_controller_step_rejects_malformed_density_profile():
    ctrl = PelletFuelingController(target_density=10.0, pellet_params=PelletParams(4.0, 300.0))
    with pytest.raises(ValueError, match="ne_profile must be a non-empty one-dimensional array"):
        ctrl.step(np.ones((2, 2)), np.ones((2, 2)), dt=0.1, V_plasma=800.0)
    with pytest.raises(ValueError, match="ne_profile must contain only finite non-negative values"):
        ctrl.step(np.array([1.0, -1.0, 1.0]), np.ones(3), dt=0.1, V_plasma=800.0)
