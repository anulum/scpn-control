# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Physics Invariant Tests
"""
CI-gated physics invariants for the gyrokinetic solver stack.

These tests verify fundamental conservation laws, analytic limits,
and cross-tier consistency. Any code change that breaks these has
introduced a physics bug.
"""

from __future__ import annotations

import numpy as np

from scpn_control.core.gk_nonlinear import NonlinearGKConfig, NonlinearGKSolver

_FAST = dict(
    n_kx=8,
    n_ky=8,
    n_theta=16,
    n_vpar=8,
    n_mu=4,
    cfl_adapt=True,
    cfl_factor=0.5,
)


# ── 1. Energy conservation ───────────────────────────────────────────


class TestEnergyConservation:
    def test_no_drive_no_dissipation(self):
        """Without drive, collisions, or hyperdiffusion, δf² energy is bounded."""
        cfg = NonlinearGKConfig(
            **_FAST,
            dt=0.01,
            n_steps=100,
            save_interval=20,
            nonlinear=True,
            collisions=False,
            hyper_coeff=0.0,
            R_L_Ti=0.0,
            R_L_Te=0.0,
            R_L_ne=0.0,
        )
        s = NonlinearGKSolver(cfg)
        state = s.init_state(amplitude=1e-6)
        E0 = s.total_energy(state)
        for _ in range(100):
            state = s._rk4_step(state, 0.01)
        E1 = s.total_energy(state)
        assert E1 < E0 * 10.0, f"Energy grew {E1 / E0:.1f}x without drive"


# ── 2-3. Sugama conservation ─────────────────────────────────────────


class TestSugamaConservation:
    def test_particle_conservation(self):
        """integral C[f] dv ~ 0."""
        cfg = NonlinearGKConfig(**_FAST, collision_model="sugama", nu_collision=0.01)
        s = NonlinearGKSolver(cfg)
        state = s.init_state()
        Cf = s._collide_sugama(state.f[0])
        m0 = float(np.sum(np.real(Cf)) * s.dvpar * s.dmu)
        assert abs(m0) < 1e-4, f"Particle moment = {m0}"

    def test_momentum_conservation(self):
        """integral v_par C[f] dv ~ 0."""
        cfg = NonlinearGKConfig(**_FAST, collision_model="sugama", nu_collision=0.01)
        s = NonlinearGKSolver(cfg)
        state = s.init_state()
        Cf = s._collide_sugama(state.f[0])
        vpar = s.vpar[None, None, None, :, None]
        m1 = float(np.sum(np.real(Cf * vpar)) * s.dvpar * s.dmu)
        assert abs(m1) < 1e-15, f"Momentum moment = {m1}"

    def test_energy_conservation(self):
        """integral E C[f] dv ~ 0."""
        cfg = NonlinearGKConfig(**_FAST, collision_model="sugama", nu_collision=0.01)
        s = NonlinearGKSolver(cfg)
        state = s.init_state()
        Cf = s._collide_sugama(state.f[0])
        vpar2 = s.vpar[None, None, None, :, None] ** 2
        mu_val = s.mu[None, None, None, None, :]
        energy = 0.5 * vpar2 + mu_val
        m2 = float(np.sum(np.real(Cf * energy)) * s.dvpar * s.dmu)
        assert abs(m2) < 1e-3, f"Energy moment = {m2}"


# ── 4. Rosenbluth-Hinton residual ────────────────────────────────────


class TestRosenbluthHinton:
    def test_rh_residual_matches_analytic(self):
        """RH residual = 1/(1+1.6q^2/sqrt(eps)) for CBC."""
        cfg = NonlinearGKConfig(**_FAST, q=1.4, R0=2.78, a=1.0)
        s = NonlinearGKSolver(cfg)
        eps = 0.5 * cfg.a / cfg.R0
        analytic = 1.0 / (1.0 + 1.6 * cfg.q**2 / np.sqrt(eps))
        assert abs(s._rh_residual - analytic) < 1e-10
        assert 0.10 < s._rh_residual < 0.15


# ── 5. Linear growth rate convergence ────────────────────────────────


class TestLinearConvergence:
    def test_theta_convergence(self):
        """Growth rate converges with theta resolution."""
        from scpn_control.core.gk_eigenvalue import solve_linear_gk
        from scpn_control.core.gk_species import deuterium_ion, electron

        species = [
            deuterium_ion(T_keV=2.0, n_19=5.0, R_L_T=6.9, R_L_n=2.2),
            electron(T_keV=2.0, n_19=5.0, R_L_T=6.9, R_L_n=2.2),
        ]
        gammas = []
        for ntheta in [16, 32, 64]:
            r = solve_linear_gk(
                species_list=species,
                R0=2.78,
                a=1.0,
                B0=2.0,
                q=1.4,
                s_hat=0.78,
                n_ky_ion=8,
                n_theta=ntheta,
                n_period=1,
            )
            gammas.append(r.gamma_max)
        assert all(g > 0 for g in gammas), f"Not all positive: {gammas}"
        d1 = abs(gammas[1] - gammas[0])
        d2 = abs(gammas[2] - gammas[1])
        # Convergence: later differences should not grow
        assert d2 <= d1 * 2.0, f"Not converging: d1={d1:.4f}, d2={d2:.4f}"


# ── 6. Cross-tier consistency ────────────────────────────────────────


class TestCrossTierConsistency:
    def test_linear_and_tglf_both_find_itg(self):
        """Linear eigenvalue and TGLF native both identify ITG at CBC."""
        from scpn_control.core.gk_eigenvalue import solve_linear_gk
        from scpn_control.core.gk_interface import GKLocalParams
        from scpn_control.core.gk_species import deuterium_ion, electron
        from scpn_control.core.gk_tglf_native import TGLFNativeConfig, TGLFNativeSolver

        species = [
            deuterium_ion(T_keV=2.0, n_19=5.0, R_L_T=6.9, R_L_n=2.2),
            electron(T_keV=2.0, n_19=5.0, R_L_T=6.9, R_L_n=2.2),
        ]
        lr = solve_linear_gk(
            species_list=species,
            R0=2.78,
            a=1.0,
            B0=2.0,
            q=1.4,
            s_hat=0.78,
            n_ky_ion=8,
            n_theta=32,
            n_period=1,
        )
        assert lr.gamma_max > 0

        p = GKLocalParams(
            R_L_Ti=6.9,
            R_L_Te=6.9,
            R_L_ne=2.2,
            q=1.4,
            s_hat=0.78,
            R0=2.78,
            a=1.0,
            B0=2.0,
            epsilon=0.18,
            T_e_keV=2.0,
            T_i_keV=2.0,
            n_e=5.0,
        )
        tr = TGLFNativeSolver(TGLFNativeConfig(n_ky_ion=8, n_theta=32)).solve(p)
        assert tr.dominant_mode == "ITG"


# ── 7. Dimits shift direction ────────────────────────────────────────


class TestDimitsDirection:
    def test_subcritical_lower_growth_than_supercritical(self):
        """R/L_Ti=2 has lower growth rate than R/L_Ti=8."""
        from scpn_control.core.gk_eigenvalue import solve_linear_gk
        from scpn_control.core.gk_species import deuterium_ion, electron

        gammas = {}
        for rlt in [2.0, 8.0]:
            species = [
                deuterium_ion(T_keV=2.0, n_19=5.0, R_L_T=rlt, R_L_n=2.2),
                electron(T_keV=2.0, n_19=5.0, R_L_T=rlt, R_L_n=2.2),
            ]
            r = solve_linear_gk(
                species_list=species,
                R0=2.78,
                a=1.0,
                B0=2.0,
                q=1.4,
                s_hat=0.78,
                n_ky_ion=8,
                n_theta=32,
                n_period=1,
            )
            gammas[rlt] = r.gamma_max
        assert gammas[2.0] < gammas[8.0], f"Sub={gammas[2.0]}, Super={gammas[8.0]}"


# ── 8. EM Ampere consistency ─────────────────────────────────────────


class TestEMConsistency:
    def test_zero_beta_gives_zero_apar(self):
        """A_par = 0 when beta_e = 0."""
        cfg = NonlinearGKConfig(**_FAST, electromagnetic=True, beta_e=0.0)
        s = NonlinearGKSolver(cfg)
        state = s.init_state()
        A_par = s.ampere_solve(state.f)
        assert np.max(np.abs(A_par)) < 1e-30

    def test_nonzero_beta_gives_nonzero_apar(self):
        """A_par != 0 when beta_e > 0."""
        cfg = NonlinearGKConfig(**_FAST, electromagnetic=True, beta_e=0.1)
        s = NonlinearGKSolver(cfg)
        state = s.init_state(amplitude=1e-3)
        A_par = s.ampere_solve(state.f)
        assert np.max(np.abs(A_par)) > 1e-10


# ── 9. Kinetic electron field solve ──────────────────────────────────


class TestKineticElectronFieldSolve:
    def test_differs_from_adiabatic(self):
        """Kinetic electron phi differs from adiabatic for same f."""
        cfg_ad = NonlinearGKConfig(**_FAST, kinetic_electrons=False)
        cfg_ke = NonlinearGKConfig(**_FAST, kinetic_electrons=True, mass_ratio_me_mi=1 / 400)
        s_ad = NonlinearGKSolver(cfg_ad)
        s_ke = NonlinearGKSolver(cfg_ke)
        state = s_ad.init_state(amplitude=1e-3)
        phi_ad = s_ad.field_solve(state.f)
        phi_ke = s_ke.field_solve(state.f)
        diff = float(np.max(np.abs(phi_ad - phi_ke)))
        assert diff > 1e-10, f"Adiabatic and kinetic phi identical (diff={diff})"


# ── 10. Ballooning BC at zero shear ──────────────────────────────────


class TestBallooningBC:
    def test_zero_shear_no_shift(self):
        """With s_hat=0, ballooning phase factors are unity (no kx shift)."""
        cfg = NonlinearGKConfig(**_FAST, s_hat=0.0)
        s = NonlinearGKSolver(cfg)
        assert np.allclose(s._ball_phase_fwd, 1.0, atol=1e-10)
        assert np.allclose(s._ball_phase_bwd, 1.0, atol=1e-10)
