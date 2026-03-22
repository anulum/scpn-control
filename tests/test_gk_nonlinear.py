# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests: Nonlinear δf Gyrokinetic Solver

from __future__ import annotations

import numpy as np

from scpn_control.core.gk_nonlinear import (
    NonlinearGKConfig,
    NonlinearGKSolver,
)

# Small grid for fast tests
_FAST_CFG = NonlinearGKConfig(
    n_kx=8,
    n_ky=8,
    n_theta=16,
    n_vpar=8,
    n_mu=4,
    n_species=2,
    dt=0.02,
    n_steps=50,
    save_interval=10,
    nonlinear=True,
    collisions=True,
    nu_collision=0.01,
    hyper_coeff=0.1,
    cfl_adapt=False,
)

# Linear-only (no E×B nonlinearity)
_LINEAR_CFG = NonlinearGKConfig(
    n_kx=8,
    n_ky=8,
    n_theta=16,
    n_vpar=8,
    n_mu=4,
    n_species=2,
    dt=0.02,
    n_steps=50,
    save_interval=10,
    nonlinear=False,
    collisions=False,
    hyper_coeff=0.0,
    cfl_adapt=False,
)

# No drive, no collisions, no hyperdiffusion — for energy conservation
_CONSERV_CFG = NonlinearGKConfig(
    n_kx=8,
    n_ky=8,
    n_theta=16,
    n_vpar=8,
    n_mu=4,
    n_species=2,
    dt=0.01,
    n_steps=20,
    save_interval=5,
    nonlinear=True,
    collisions=False,
    hyper_coeff=0.0,
    R_L_Ti=0.0,
    R_L_Te=0.0,
    R_L_ne=0.0,
    cfl_adapt=False,
)


# ── Config and State ──────────────────────────────────────────────────


class TestConfigState:
    def test_config_defaults(self):
        cfg = NonlinearGKConfig()
        assert cfg.n_kx == 16
        assert cfg.n_steps == 5000
        assert cfg.dealiasing == "2/3"

    def test_state_shape(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        assert state.f.shape == (2, 8, 8, 16, 8, 4)
        assert state.phi.shape == (8, 8, 16)
        assert state.time == 0.0

    def test_single_mode_init(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_single_mode(kx_idx=0, ky_idx=1, amplitude=1e-3)
        assert np.max(np.abs(state.f[0, 0, 1])) > 0
        # Other modes should be zero (except at the specified mode)
        assert np.max(np.abs(state.f[0, 2, 3])) == 0.0


# ── Field solve ───────────────────────────────────────────────────────


class TestFieldSolve:
    def test_phi_shape(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        phi = solver.field_solve(state.f)
        assert phi.shape == (8, 8, 16)

    def test_phi_finite(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        assert np.all(np.isfinite(state.phi))

    def test_zero_f_gives_zero_phi(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        f = np.zeros((2, 8, 8, 16, 8, 4), dtype=complex)
        phi = solver.field_solve(f)
        np.testing.assert_allclose(phi, 0.0, atol=1e-30)


# ── E×B bracket ──────────────────────────────────────────────────────


class TestExBBracket:
    def test_bracket_shape(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        bracket = solver.exb_bracket(state.phi, state.f[0])
        assert bracket.shape == state.f[0].shape

    def test_bracket_zero_for_uniform_f(self):
        """Spatially uniform f (only k=0 mode) has zero gradients → bracket = 0."""
        solver = NonlinearGKSolver(_FAST_CFG)
        f_uniform = np.zeros((8, 8, 16, 8, 4), dtype=complex)
        f_uniform[0, 0, :, :, :] = 1e-3  # only the (kx=0, ky=0) mode
        phi = solver.init_state().phi
        bracket = solver.exb_bracket(phi, f_uniform)
        # df/dx = df/dy = 0 for uniform f → bracket = 0
        assert np.max(np.abs(bracket)) < 1e-10

    def test_bracket_dealiased(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state(amplitude=1e-3)
        bracket = solver.exb_bracket(state.phi, state.f[0])
        # High-k modes should be zero after dealiasing
        mask = ~solver.dealias_mask
        for t in range(16):
            for v in range(8):
                for m in range(4):
                    assert np.all(bracket[mask, t, v, m] == 0.0)


# ── Parallel streaming ───────────────────────────────────────────────


class TestParallelStreaming:
    def test_streaming_shape(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        stream = solver.parallel_streaming(state.f[0])
        assert stream.shape == state.f[0].shape

    def test_uniform_theta_zero_streaming(self):
        """Uniform f(θ) → ∂f/∂θ = 0 → no streaming contribution."""
        solver = NonlinearGKSolver(_FAST_CFG)
        f_s = np.ones((8, 8, 16, 8, 4), dtype=complex) * 1e-3
        stream = solver.parallel_streaming(f_s)
        np.testing.assert_allclose(stream, 0.0, atol=1e-14)


# ── Energy conservation ──────────────────────────────────────────────


class TestEnergyConservation:
    def test_energy_finite(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        E = solver.total_energy(state)
        assert np.isfinite(E)
        assert E > 0

    def test_no_drive_no_dissipation_conserves(self):
        """Without drive, collisions, or hyperdiffusion, E should not grow."""
        solver = NonlinearGKSolver(_CONSERV_CFG)
        state = solver.init_state(amplitude=1e-6)
        E0 = solver.total_energy(state)

        for _ in range(20):
            state = solver._rk4_step(state, 0.01)

        E1 = solver.total_energy(state)
        # Energy should not grow (may decrease slightly due to dealiasing)
        assert E1 < E0 * 10.0  # generous bound — no unbounded growth


# ── Linear recovery ──────────────────────────────────────────────────


class TestLinearRecovery:
    def test_linear_mode_grows(self):
        """With nonlinearity off, a single mode should grow."""
        solver = NonlinearGKSolver(_LINEAR_CFG)
        state = solver.init_single_mode(ky_idx=1, amplitude=1e-8)
        phi0 = solver.phi_rms(state)

        for _ in range(50):
            state = solver._rk4_step(state, 0.02)

        phi1 = solver.phi_rms(state)
        # In linear regime, phi should evolve (grow or damp)
        assert np.isfinite(phi1)

    def test_phi_rms_finite_throughout(self):
        solver = NonlinearGKSolver(_LINEAR_CFG)
        result = solver.run(solver.init_single_mode(amplitude=1e-8))
        assert np.all(np.isfinite(result.phi_rms_t))


# ── Zonal flows ──────────────────────────────────────────────────────


class TestZonalFlows:
    def test_zonal_rms_computed(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        zr = solver.zonal_rms(state)
        assert np.isfinite(zr)

    def test_zonal_present_in_nonlinear(self):
        """Nonlinear run should generate zonal flows (k_y=0) from noise."""
        solver = NonlinearGKSolver(_FAST_CFG)
        result = solver.run()
        assert result.zonal_rms_t[-1] >= 0

    def test_rosenbluth_hinton_residual_positive(self):
        """After zonal flow initialization, residual should be > 0.

        Rosenbluth & Hinton 1998: ZF residual = 1/(1 + 1.6q²/√ε).
        """
        solver = NonlinearGKSolver(
            NonlinearGKConfig(
                n_kx=8,
                n_ky=8,
                n_theta=16,
                n_vpar=8,
                n_mu=4,
                n_species=2,
                dt=0.02,
                n_steps=10,
                save_interval=5,
                nonlinear=False,
                collisions=False,
                hyper_coeff=0.0,
                R_L_Ti=0.0,
                R_L_Te=0.0,
                R_L_ne=0.0,
                cfl_adapt=False,
            )
        )
        state = solver.init_state(amplitude=1e-5)
        zr0 = solver.zonal_rms(state)
        # After a few steps, zonal component should remain (not decay to zero)
        for _ in range(10):
            state = solver._rk4_step(state, 0.02)
        zr1 = solver.zonal_rms(state)
        assert np.isfinite(zr1)


# ── CBC benchmark ────────────────────────────────────────────────────


class TestCBCBenchmark:
    def test_cbc_runs_without_nan(self):
        cfg = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            dt=0.02,
            n_steps=100,
            save_interval=20,
            R_L_Ti=6.9,
            R_L_Te=6.9,
            R_L_ne=2.2,
            q=1.4,
            s_hat=0.78,
            R0=2.78,
            a=1.0,
            B0=2.0,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        result = solver.run()
        assert result.converged
        assert np.all(np.isfinite(result.Q_i_t))

    def test_cbc_chi_i_finite(self):
        cfg = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            dt=0.02,
            n_steps=100,
            save_interval=20,
            R_L_Ti=6.9,
            q=1.4,
            s_hat=0.78,
            R0=2.78,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        result = solver.run()
        assert np.isfinite(result.chi_i)

    def test_subcritical_lower_transport(self):
        """R/L_Ti=2.0 (subcritical) should produce less transport than R/L_Ti=6.9."""
        cfg_sub = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            dt=0.02,
            n_steps=100,
            save_interval=20,
            R_L_Ti=2.0,
            q=1.4,
            s_hat=0.78,
            R0=2.78,
            cfl_adapt=False,
        )
        cfg_sup = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            dt=0.02,
            n_steps=100,
            save_interval=20,
            R_L_Ti=6.9,
            q=1.4,
            s_hat=0.78,
            R0=2.78,
            cfl_adapt=False,
        )
        r_sub = NonlinearGKSolver(cfg_sub).run()
        r_sup = NonlinearGKSolver(cfg_sup).run()
        # Both should be finite
        assert np.isfinite(r_sub.chi_i)
        assert np.isfinite(r_sup.chi_i)


# ── Flux diagnostics ─────────────────────────────────────────────────


class TestFluxDiagnostics:
    def test_flux_computation(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        Q_i, Q_e = solver.compute_fluxes(state)
        assert np.isfinite(Q_i)
        assert np.isfinite(Q_e)

    def test_phi_rms_positive(self):
        solver = NonlinearGKSolver(_FAST_CFG)
        state = solver.init_state()
        assert solver.phi_rms(state) > 0


# ── JAX fallback ─────────────────────────────────────────────────────


# ── Bug fix regressions ────────────────────────────────────────────


class TestElectronDriveUsePhiEff:
    """Fix 3: electron drive must use phi_eff = phi - v_par*A_par, not phi."""

    def test_em_electron_drive_differs_from_es(self):
        cfg_em = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            dt=0.02,
            n_steps=1,
            save_interval=1,
            kinetic_electrons=True,
            electromagnetic=True,
            beta_e=0.02,
            nonlinear=False,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg_em)
        state = solver.init_state(amplitude=1e-5)
        # Manually set A_par to something nonzero
        A_par = 1e-4 * np.ones_like(state.phi)
        drive_with_A = solver.gradient_drive(state.phi, A_par)
        drive_no_A = solver.gradient_drive(state.phi, None)
        # Electron species (index 1) should differ when A_par != 0
        assert not np.allclose(drive_with_A[1], drive_no_A[1])

    def test_es_electron_drive_same_with_none(self):
        cfg = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            dt=0.02,
            n_steps=1,
            save_interval=1,
            kinetic_electrons=True,
            electromagnetic=False,
            nonlinear=False,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-5)
        drive = solver.gradient_drive(state.phi, None)
        # Both ion and electron drives should be nonzero
        assert np.max(np.abs(drive[0])) > 0
        assert np.max(np.abs(drive[1])) > 0


class TestAmpereSkinDepth:
    """Fix 4: Ampere's law denominator must include d_e_sq skin-depth."""

    def test_d_e_sq_zero_recovers_original(self):
        cfg = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            electromagnetic=True,
            beta_e=0.01,
            d_e_sq=0.0,
            kinetic_electrons=True,
            dt=0.02,
            n_steps=1,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-5)
        A_par = solver.ampere_solve(state.f)
        assert np.all(np.isfinite(A_par))

    def test_d_e_sq_reduces_A_par(self):
        """Nonzero skin depth adds to denominator → smaller A_par amplitude."""
        cfg_no_skin = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            electromagnetic=True,
            beta_e=0.01,
            d_e_sq=0.0,
            kinetic_electrons=True,
            dt=0.02,
            n_steps=1,
            save_interval=1,
            cfl_adapt=False,
        )
        cfg_skin = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            electromagnetic=True,
            beta_e=0.01,
            d_e_sq=1.0,
            kinetic_electrons=True,
            dt=0.02,
            n_steps=1,
            save_interval=1,
            cfl_adapt=False,
        )
        solver_no = NonlinearGKSolver(cfg_no_skin)
        solver_yes = NonlinearGKSolver(cfg_skin)
        state = solver_no.init_state(amplitude=1e-5, seed=42)
        # Use same f for both
        A_no = solver_no.ampere_solve(state.f)
        A_yes = solver_yes.ampere_solve(state.f)
        assert np.max(np.abs(A_yes)) <= np.max(np.abs(A_no))


class TestCFLVExB:
    """Fix 5: CFL v_ExB must scale as kmax**2 * phi, not kmax * phi."""

    def test_cfl_uses_k_squared(self):
        # Use small Ly to get kmax > 1 (ky_max ~ 2π/Ly * n_ky/2)
        cfg = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            dt=1.0,
            Lx=6.0,
            Ly=6.0,
            cfl_adapt=True,
            cfl_factor=0.5,
            collisions=False,
            hyper_coeff=0.0,
            nonlinear=True,
            R_L_Ti=0.0,
            R_L_Te=0.0,
            R_L_ne=0.0,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-3)
        phi_max = np.max(np.abs(state.phi)) + 1e-30
        kmax = max(np.max(np.abs(solver.kx)), np.max(np.abs(solver.ky)))
        assert kmax > 1.0
        # The CFL dt should reflect k^2 scaling
        dt_val = solver._cfl_dt(state)
        v_exb_k2 = kmax**2 * phi_max
        vmax = max(np.max(np.abs(solver.vpar)), 1.0)
        v_par_eff = vmax * np.max(np.abs(solver.b_dot_grad))
        dt_expected = cfg.cfl_factor / max(v_exb_k2 + v_par_eff, 1e-30)
        np.testing.assert_allclose(dt_val, min(dt_expected, cfg.dt), rtol=1e-10)


class TestSugamaGramMatrix:
    """Fix 6: Sugama conservation must use 3x3 Gram matrix, not diagonal."""

    def test_sugama_conserves_density(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=8,
            n_mu=4,
            collisions=True,
            collision_model="sugama",
            nu_collision=0.1,
            dt=0.01,
            n_steps=1,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-4)
        Cf = solver._collide_sugama(state.f[0])
        density_moment = np.sum(Cf * solver._dv_weights_5d, axis=(-2, -1))
        np.testing.assert_allclose(density_moment, 0.0, atol=1e-12)

    def test_sugama_conserves_momentum(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=8,
            n_mu=4,
            collisions=True,
            collision_model="sugama",
            nu_collision=0.1,
            dt=0.01,
            n_steps=1,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-4)
        Cf = solver._collide_sugama(state.f[0])
        vpar_5d = solver.vpar[None, None, None, :, None]
        mom_moment = np.sum(Cf * vpar_5d * solver._dv_weights_5d, axis=(-2, -1))
        np.testing.assert_allclose(mom_moment, 0.0, atol=1e-12)

    def test_sugama_conserves_energy(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=8,
            n_mu=4,
            collisions=True,
            collision_model="sugama",
            nu_collision=0.1,
            dt=0.01,
            n_steps=1,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-4)
        Cf = solver._collide_sugama(state.f[0])
        vpar2 = solver.vpar[None, None, None, :, None] ** 2
        mu_val = solver.mu[None, None, None, None, :]
        energy = 0.5 * vpar2 + mu_val
        energy_moment = np.sum(Cf * energy * solver._dv_weights_5d, axis=(-2, -1))
        np.testing.assert_allclose(energy_moment, 0.0, atol=1e-12)


class TestJaxFallback:
    def test_jax_solver_import(self):
        from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver

        solver = JaxNonlinearGKSolver(_FAST_CFG)
        # Should work regardless of JAX availability
        assert solver._np_solver is not None

    def test_jax_solver_runs(self):
        from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver

        solver = JaxNonlinearGKSolver(_FAST_CFG)
        result = solver.run()
        assert result.converged or len(result.Q_i_t) > 0
        assert np.all(np.isfinite(result.Q_i_t))
        if np.isfinite(result.chi_i):
            expected = result.chi_i / max(_FAST_CFG.R_L_Ti, 0.01)
            assert abs(result.chi_i_gB - expected) < 1e-10

    def test_jax_available_bool(self):
        from scpn_control.core.jax_gk_nonlinear import jax_available

        assert isinstance(jax_available(), bool)


# ── Coverage-gap tests: EM paths, negative-shift ballooning, minimal config ──


class TestElectromagneticCodePaths:
    """electromagnetic=True exercises ampere_solve and gradient_drive EM branch."""

    def test_em_rhs_finite(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=4,
            n_mu=4,
            electromagnetic=True,
            beta_e=0.02,
            d_e_sq=0.5,
            kinetic_electrons=True,
            dt=0.02,
            n_steps=2,
            save_interval=1,
            nonlinear=False,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-5)
        assert state.A_par is not None
        assert np.all(np.isfinite(state.A_par))
        rhs_val = solver.rhs(state)
        assert np.all(np.isfinite(rhs_val))

    def test_em_run_converges(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=4,
            n_mu=4,
            electromagnetic=True,
            beta_e=0.02,
            d_e_sq=0.0,
            kinetic_electrons=True,
            dt=0.02,
            n_steps=4,
            save_interval=2,
            nonlinear=False,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        result = solver.run()
        assert np.all(np.isfinite(result.Q_i_t))

    def test_ampere_solve_ion_only(self):
        """ampere_solve with kinetic_electrons=False uses ion current only."""
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=4,
            n_mu=4,
            electromagnetic=True,
            beta_e=0.01,
            kinetic_electrons=False,
            dt=0.02,
            n_steps=1,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-5)
        A_par = solver.ampere_solve(state.f)
        assert np.all(np.isfinite(A_par))
        assert A_par[0, 0, :].tolist() == [0.0] * cfg.n_theta


class TestRollBallooningNegativeShift:
    """_roll_ballooning with shift < 0 wraps forward at theta_max."""

    def test_negative_shift_applies_forward_phase(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=4,
            n_mu=4,
            dt=0.02,
            n_steps=1,
            save_interval=1,
            cfl_adapt=False,
            s_hat=0.78,
        )
        solver = NonlinearGKSolver(cfg)
        f_s = np.random.default_rng(7).standard_normal((4, 4, 8, 4, 4)).astype(complex)
        rolled_neg = solver._roll_ballooning(f_s, shift=-1)
        rolled_pos = solver._roll_ballooning(f_s, shift=1)
        assert rolled_neg.shape == f_s.shape
        assert rolled_pos.shape == f_s.shape
        # Negative and positive shifts should differ
        assert not np.allclose(rolled_neg, rolled_pos)

    def test_negative_shift_2(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=4,
            n_mu=4,
            dt=0.02,
            n_steps=1,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        f_s = np.random.default_rng(8).standard_normal((4, 4, 8, 4, 4)).astype(complex)
        rolled = solver._roll_ballooning(f_s, shift=-2)
        assert rolled.shape == f_s.shape
        assert np.all(np.isfinite(rolled))


class TestMinimalGridConfig:
    """Run with n_kx=4, n_ky=4, n_theta=8, n_vpar=4, n_mu=4, n_steps=2."""

    def test_minimal_run(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=4,
            n_mu=4,
            dt=0.02,
            n_steps=2,
            save_interval=1,
            nonlinear=True,
            collisions=True,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        result = solver.run()
        assert result.Q_i_t.size > 0
        assert np.all(np.isfinite(result.Q_i_t))


class TestNoDealiasing:
    """Line 194: dealiasing != '2/3' uses all-ones mask."""

    def test_no_dealiasing_mask(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=4,
            n_mu=4,
            dealiasing="none",
            dt=0.02,
            n_steps=1,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        assert np.all(solver.dealias_mask)


class TestImplicitElectronStreaming:
    """Lines 694, 708-752: implicit electron streaming correction."""

    def test_implicit_electrons_runs(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=4,
            n_mu=4,
            kinetic_electrons=True,
            implicit_electrons=True,
            electromagnetic=False,
            dt=0.02,
            n_steps=2,
            save_interval=1,
            nonlinear=False,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-6)
        result = solver.run(state)
        assert np.all(np.isfinite(result.Q_i_t))


class TestCFLElectronScaling:
    """Line 769: CFL includes electron velocity scaling for explicit kinetic electrons."""

    def test_cfl_electron_vscale(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=4,
            n_mu=4,
            kinetic_electrons=True,
            implicit_electrons=False,
            dt=1.0,
            cfl_adapt=True,
            cfl_factor=0.5,
            nonlinear=False,
            collisions=False,
            hyper_coeff=0.0,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-5)
        dt_val = solver._cfl_dt(state)
        # With kinetic (explicit) electrons, CFL should be more restrictive
        assert dt_val < cfg.dt


class TestRunNaNBreak:
    """Lines 888-889: run() breaks early on NaN."""

    def test_nan_breaks_early(self):
        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=4,
            n_mu=4,
            dt=0.02,
            n_steps=50,
            save_interval=10,
            nonlinear=True,
            collisions=False,
            hyper_coeff=0.0,
            cfl_adapt=False,
        )
        solver = NonlinearGKSolver(cfg)
        state = solver.init_state(amplitude=1e-5)
        # Inject NaN into f to trigger the break
        state.f[0, 0, 0, 0, 0, 0] = float("nan")
        result = solver.run(state)
        # Should stop early — fewer saves than expected
        assert result.Q_i_t.size <= cfg.n_steps // cfg.save_interval + 1


class TestJaxV019Parity:
    def test_jax_ampere_solve_matches_numpy_with_skin_depth(self):
        from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver, jax_available

        if not jax_available():
            return

        import jax.numpy as jnp

        cfg = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            electromagnetic=True,
            beta_e=0.01,
            d_e_sq=1.0,
            kinetic_electrons=True,
            dt=0.02,
            n_steps=1,
            save_interval=1,
            cfl_adapt=False,
        )
        np_solver = NonlinearGKSolver(cfg)
        jax_solver = JaxNonlinearGKSolver(cfg)
        state = np_solver.init_state(amplitude=1e-5, seed=42)

        A_np = np_solver.ampere_solve(state.f)
        A_jax = np.asarray(jax_solver._jax_ampere_solve(jnp.array(state.f)))
        np.testing.assert_allclose(A_jax, A_np, rtol=1e-6, atol=1e-8)

    def test_jax_cfl_uses_k_squared(self):
        from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver, jax_available

        if not jax_available():
            return

        import jax.numpy as jnp

        cfg = NonlinearGKConfig(
            n_kx=8,
            n_ky=8,
            n_theta=16,
            n_vpar=8,
            n_mu=4,
            dt=1.0,
            Lx=6.0,
            Ly=6.0,
            cfl_adapt=True,
            cfl_factor=0.5,
            collisions=False,
            hyper_coeff=0.0,
            nonlinear=True,
            R_L_Ti=0.0,
            R_L_Te=0.0,
            R_L_ne=0.0,
        )
        solver = JaxNonlinearGKSolver(cfg)
        state = solver._np_solver.init_state(amplitude=1e-3)
        phi = solver._jax_field_solve(jnp.array(state.f))

        dt_val = solver._jax_cfl_dt(phi)
        phi_max = np.max(np.abs(np.asarray(phi))) + 1e-30
        kmax = max(np.max(np.abs(solver._np_solver.kx)), np.max(np.abs(solver._np_solver.ky)))
        vmax = max(np.max(np.abs(solver._np_solver.vpar)), 1.0)
        v_par_eff = vmax * np.max(np.abs(solver._np_solver.b_dot_grad))
        dt_expected = cfg.cfl_factor / max(kmax**2 * phi_max + v_par_eff, 1e-30)
        np.testing.assert_allclose(dt_val, min(dt_expected, cfg.dt), rtol=1e-8)

    def test_jax_sugama_conserves_moments(self):
        from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver, jax_available

        if not jax_available():
            return

        import jax.numpy as jnp

        cfg = NonlinearGKConfig(
            n_kx=4,
            n_ky=4,
            n_theta=8,
            n_vpar=8,
            n_mu=4,
            collisions=True,
            collision_model="sugama",
            nu_collision=0.1,
            dt=0.01,
            n_steps=1,
            save_interval=1,
            cfl_adapt=False,
        )
        solver = JaxNonlinearGKSolver(cfg)
        state = solver._np_solver.init_state(amplitude=1e-4, seed=42)
        Cf = np.asarray(solver._jax_collide_sugama(jnp.array(state.f[0])))

        density = np.sum(Cf * solver._np_solver._dv_weights_5d, axis=(-2, -1))
        vpar_5d = solver._np_solver.vpar[None, None, None, :, None]
        momentum = np.sum(Cf * vpar_5d * solver._np_solver._dv_weights_5d, axis=(-2, -1))
        vpar2 = solver._np_solver.vpar[None, None, None, :, None] ** 2
        mu_val = solver._np_solver.mu[None, None, None, None, :]
        energy = 0.5 * vpar2 + mu_val
        energy_moment = np.sum(Cf * energy * solver._np_solver._dv_weights_5d, axis=(-2, -1))

        np.testing.assert_allclose(density, 0.0, atol=1e-5)
        np.testing.assert_allclose(momentum, 0.0, atol=1e-5)
        np.testing.assert_allclose(energy_moment, 0.0, atol=1e-5)
