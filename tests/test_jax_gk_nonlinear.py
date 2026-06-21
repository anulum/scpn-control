# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JAX Nonlinear Gyrokinetic Solver Tests
"""Targeted coverage for the JAX nonlinear gyrokinetic solver branches."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from scpn_control.core.gk_nonlinear import NonlinearGKConfig  # noqa: E402
from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver  # noqa: E402


def _config(**overrides: object) -> NonlinearGKConfig:
    base: dict[str, object] = {
        "n_kx": 8,
        "n_ky": 8,
        "n_theta": 16,
        "n_vpar": 8,
        "n_mu": 4,
        "n_species": 2,
        "dt": 0.01,
        "n_steps": 2,
        "save_interval": 1,
        "cfl_adapt": False,
    }
    base.update(overrides)
    return NonlinearGKConfig(**base)  # type: ignore[arg-type]


def test_jax_ampere_solve_is_zero_in_electrostatic_limit() -> None:
    """The JAX Ampère solve short-circuits to a zero A_par field when electrostatic."""
    solver = JaxNonlinearGKSolver(_config())  # electromagnetic defaults to False
    f = jnp.array(solver._np_solver.init_state().f)

    a_par = np.asarray(solver._jax_ampere_solve(f))

    assert a_par.shape == (8, 8, 16)
    assert np.all(a_par == 0.0)


def test_jax_collide_dispatches_to_sugama() -> None:
    """The JAX collision dispatch routes to the Sugama operator when configured."""
    solver = JaxNonlinearGKSolver(_config(collision_model="sugama", nu_collision=0.1, n_steps=1))
    f_s = jnp.array(solver._np_solver.init_state().f[0])

    collided = np.asarray(solver._jax_collide(f_s))

    assert collided.shape == f_s.shape
    assert np.all(np.isfinite(collided))


def test_jax_run_with_explicit_kinetic_electrons() -> None:
    """The CFL estimate includes the electron drift scale for explicit kinetic electrons."""
    solver = JaxNonlinearGKSolver(_config(kinetic_electrons=True, implicit_electrons=False))

    result = solver.run()

    assert result.final_state is not None
    assert np.all(np.isfinite(result.final_state.f))


def test_jax_run_breaks_on_non_finite_state() -> None:
    """A non-finite initial state halts the integration loop at the NaN guard."""
    solver = JaxNonlinearGKSolver(_config())
    state = solver._np_solver.init_state()
    state.f[0, 0, 0, 0, 0] = np.inf

    result = solver.run(state)

    assert result.converged is False
    assert result.Q_i_t.size == 0
