# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual CPU manual GK runner boundaries

"""Exercise short real JAX CPU diagnostics; these are not Dimits physics evidence."""

from __future__ import annotations

import math
from contextlib import chdir
from pathlib import Path

import jax
import pytest

from scpn_control.core.gk_nonlinear import NonlinearGKConfig
from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver
from tools.em_and_dimits import run_jax

GRID = dict(n_kx=8, n_ky=4, n_theta=8, n_vpar=4, n_mu=2, save_interval=1)


def test_empty_actual_cpu_history_is_refused() -> None:
    """A zero-step real solver result raises the explicit runner error."""
    assert all(device.platform == "cpu" for device in jax.devices())
    with pytest.raises(RuntimeError, match="produced no saved history"):
        run_jax("actual empty CPU history", **GRID, n_steps=0)


def test_one_saved_cpu_sample_retains_nonfinite_transport_display() -> None:
    """One real sample retains the solver's nonfinite tail mean as None."""
    with pytest.warns(RuntimeWarning):
        result = run_jax("actual one CPU sample", **GRID, n_steps=1)
    assert result["chi_i_gB"] is None
    assert result["converged"] is False
    assert result["late_growth"] == 0.0
    assert result["time_final"] == pytest.approx(0.05)
    assert math.isfinite(result["phi_final"]) and result["phi_final"] > 0.0


def test_short_cpu_diagnostics_do_not_write_a_report(tmp_path: Path) -> None:
    """Two actual saved samples report finite diagnostics without report I/O."""
    with chdir(tmp_path):
        result = run_jax("actual two CPU samples", **GRID, n_steps=2)
    assert result["label"] == "actual two CPU samples"
    assert result["converged"] is True
    assert result["late_growth"] == 0.0
    assert result["chi_i_gB"] is not None and math.isfinite(result["chi_i_gB"])
    assert math.isfinite(result["phi_final"]) and result["phi_final"] > 0.0
    assert result["time_final"] == pytest.approx(0.1)
    assert math.isfinite(result["elapsed_s"]) and result["elapsed_s"] > 0.0
    assert not list(tmp_path.iterdir())


def test_late_growth_matches_an_independent_fresh_public_cpu_run() -> None:
    """The displayed endpoint rate agrees with a fresh actual public solver run."""
    result = run_jax("actual CPU endpoint rate", **GRID, n_steps=16)
    cfg = NonlinearGKConfig(n_kx=8, n_ky=4, n_theta=8, n_vpar=4, n_mu=2, save_interval=1, n_steps=16, hyper_coeff=0.2)
    reference = JaxNonlinearGKSolver(cfg).run()
    start = 3 * len(reference.phi_rms_t) // 4
    assert len(reference.phi_rms_t[start:]) > 2 and reference.phi_rms_t[start] > 0.0
    rate = (reference.phi_rms_t[-1] - reference.phi_rms_t[start]) / (
        reference.phi_rms_t[start] * max(reference.time[-1] - reference.time[start], 0.01)
    )
    assert result["late_growth"] == pytest.approx(float(rate))
    assert result["phi_final"] == pytest.approx(float(reference.phi_rms_t[-1]))
    assert result["chi_i_gB"] == pytest.approx(reference.chi_i / max(cfg.R_L_Ti, 0.01))


def test_zero_mode_cpu_history_has_no_endpoint_growth() -> None:
    """The real zero-mode-only grid retains zero potential and zero growth."""
    result = run_jax("actual zero-mode CPU history", **(GRID | {"n_kx": 1, "n_ky": 1}), n_steps=16)
    assert result["phi_final"] == 0.0
    assert result["late_growth"] == 0.0
    assert result["time_final"] > 0.0


def test_unknown_override_follows_the_actual_config_error() -> None:
    """An unsupported public override is refused without a backend substitute."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        run_jax("unknown actual config field", unsupported_manual_option=1)
