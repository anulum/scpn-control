# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual fixed CBC model reports and persistent-output refusal.

"""Exercise real native/NumPy/JAX public reports and unrecorded campaign refusal."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.gk_eigenvalue import solve_linear_gk
from scpn_control.core.gk_interface import GKLocalParams
from scpn_control.core.gk_nonlinear import NonlinearGKConfig, NonlinearGKSolver
from scpn_control.core.gk_species import deuterium_ion, electron
from scpn_control.core.gk_tglf_native import TGLFNativeConfig, TGLFNativeSolver
from tools import gpu_cbc_benchmark as tool


def test_actual_fixed_model_reports() -> None:
    """Bind every report's numeric fields to independent actual public provider executions."""
    linear = tool.run_linear_benchmark()
    spectrum = solve_linear_gk(
        species_list=[
            deuterium_ion(T_keV=2.0, n_19=5.0, R_L_T=6.9, R_L_n=2.2),
            electron(T_keV=2.0, n_19=5.0, R_L_T=6.9, R_L_n=2.2, adiabatic=True),
        ],
        R0=2.78,
        a=1.0,
        B0=2.0,
        q=1.4,
        s_hat=0.78,
        n_ky_ion=16,
        n_theta=64,
        n_period=2,
    )
    assert set(linear) == {"gamma_max", "k_y_max", "elapsed_s", "gamma", "k_y", "mode_type"}
    assert linear["gamma_max"] == spectrum.gamma_max and linear["k_y_max"] == spectrum.k_y_max
    assert linear["gamma"] == spectrum.gamma.tolist() and linear["k_y"] == spectrum.k_y.tolist()
    assert linear["mode_type"] == spectrum.mode_type and len(linear["gamma"]) == 16
    assert math.isfinite(linear["elapsed_s"]) and linear["elapsed_s"] >= 0

    tglf = tool.run_tglf_native()
    flux = TGLFNativeSolver(TGLFNativeConfig(sat_model="SAT1", n_ky_ion=16, n_theta=64)).solve(
        GKLocalParams(
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
    )
    assert set(tglf) == {"chi_i", "chi_e", "D_e", "dominant_mode", "elapsed_s", "gamma", "k_y"}
    assert (tglf["chi_i"], tglf["chi_e"], tglf["D_e"]) == (flux.chi_i, flux.chi_e, flux.D_e)
    assert tglf["dominant_mode"] == flux.dominant_mode
    assert tglf["gamma"] == flux.gamma.tolist() and tglf["k_y"] == flux.k_y.tolist()
    assert math.isfinite(tglf["elapsed_s"]) and tglf["elapsed_s"] >= 0

    for steps in (0, 1):
        config = NonlinearGKConfig(
            n_kx=16,
            n_ky=16,
            n_theta=64,
            n_vpar=16,
            n_mu=8,
            dt=0.02,
            n_steps=steps,
            save_interval=50,
            R_L_Ti=6.9,
            R_L_Te=6.9,
            R_L_ne=2.2,
            q=1.4,
            s_hat=0.78,
            R0=2.78,
            a=1.0,
            B0=2.0,
            cfl_adapt=False,
            nonlinear=True,
            collisions=True,
            nu_collision=0.01,
            hyper_coeff=0.1,
        )
        reports = [(tool.run_nonlinear_numpy(steps, "actual NumPy"), NonlinearGKSolver(config).run())]
        if importlib.util.find_spec("jax") is None:
            assert tool.run_nonlinear_jax(steps, "absent JAX") == {
                "label": "absent JAX",
                "skipped": True,
                "reason": "JAX not available",
            }
        else:
            from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver

            actual_jax = tool.run_nonlinear_jax(steps, "actual JAX")
            assert "skipped" not in actual_jax
            reports.append((actual_jax, JaxNonlinearGKSolver(config).run()))
        for actual, expected in reports:
            assert set(actual) == {
                "label",
                "chi_i",
                "chi_e",
                "converged",
                "elapsed_s",
                "n_steps",
                "phi_rms",
                "zonal_rms",
                "Q_i",
                "time",
            }
            assert actual["n_steps"] == steps and actual["converged"] == expected.converged
            np.testing.assert_array_equal([actual["chi_i"], actual["chi_e"]], [expected.chi_i, expected.chi_e])
            assert actual["phi_rms"] == expected.phi_rms_t.tolist()
            assert actual["zonal_rms"] == expected.zonal_rms_t.tolist()
            assert actual["Q_i"] == expected.Q_i_t.tolist() and actual["time"] == expected.time.tolist()
            assert math.isfinite(actual["elapsed_s"]) and actual["elapsed_s"] >= 0
            assert len(actual["time"]) == steps and actual["converged"] is False
            if steps:
                assert actual["time"] == [0.02] and math.isnan(actual["chi_i"])
            else:
                assert actual["chi_i"] == 0.0 and actual["time"] == []


def test_unrecorded_canonical_main_refuses_before_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise real API/available-provider CLI refusal while preserving the existing output."""
    root = Path(__file__).resolve().parents[1]
    output = root / "gpu_results/gk_nonlinear_cbc_gpu.json"
    assert not output.is_symlink()
    before = output.read_bytes() if output.is_file() else None
    monkeypatch.delenv("SCPN_BENCHMARK_CAMPAIGN_ID", raising=False)
    monkeypatch.chdir(root)
    with pytest.raises(RuntimeError, match="persistent benchmark output requires"):
        tool.main()
    assert (output.read_bytes() if output.is_file() else None) == before
    env = os.environ.copy()
    env.pop("SCPN_BENCHMARK_CAMPAIGN_ID", None)
    env["PYTHONPATH"] = str(root) + os.pathsep + str(root / "src")
    commands = [[sys.executable, "-c", "from tools.gpu_cbc_benchmark import main; main()"]]
    from scpn_control.core.jax_gk_nonlinear import jax_available

    if jax_available():
        commands.append([sys.executable, str(root / "tools/gpu_cbc_benchmark.py")])
    for command in commands:
        refused = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, check=False)
        assert refused.returncode != 0 and "persistent benchmark output requires" in refused.stderr
        assert "=== Linear GK" not in refused.stdout and "Installing JAX" not in refused.stdout
        assert (output.read_bytes() if output.is_file() else None) == before
