# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual JAX convergence-runner and scratch-writer contracts.

"""Exercise real available JAX runs or real missing-provider import refusal, without skips."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Windows reports a directory opened as a file as a permission error.
DIRECTORY_AS_FILE_ERROR: type[OSError] = PermissionError if os.name == "nt" else IsADirectoryError


def test_actual_provider_workflow_and_public_scratch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Compare actual zero/one/two-sample providers, then exercise the real declared writer and failures."""
    root = Path(__file__).resolve().parents[1]
    if importlib.util.find_spec("jax") is None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root) + os.pathsep + str(root / "src")
        refusal = subprocess.run(
            [sys.executable, "-c", "import tools.gk_convergence_benchmark"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert refusal.returncode != 0 and "ModuleNotFoundError" in refusal.stderr and "jax" in refusal.stderr
        assert list(tmp_path.iterdir()) == []
        return

    from scpn_control.core.gk_nonlinear import NonlinearGKConfig
    from scpn_control.core.jax_gk_nonlinear import JaxNonlinearGKSolver
    from tools import gk_convergence_benchmark as tool

    results: dict[str, object] = {}
    for steps in (0, 1, 2):
        config = NonlinearGKConfig(
            n_kx=8,
            n_ky=4,
            n_theta=8,
            n_vpar=4,
            n_mu=2,
            n_steps=steps,
            save_interval=1,
        )
        actual = tool.run_benchmark(f"real software case {steps}", config)
        independent = JaxNonlinearGKSolver(config).run()
        assert set(actual) == {"chi_i_gB", "converged", "wall_s", "Q_i"}
        assert actual["Q_i"] == independent.Q_i_t.tolist()
        assert actual["converged"] == independent.converged
        if math.isnan(independent.chi_i_gB):
            assert actual["chi_i_gB"] is None
        else:
            assert actual["chi_i_gB"] == independent.chi_i_gB
        assert math.isfinite(actual["wall_s"]) and actual["wall_s"] >= 0
        assert len(actual["Q_i"]) == steps
        assert actual["converged"] is (steps > 1)
        assert independent.final_state is not None and np.all(np.isfinite(independent.final_state.f))
        results[str(steps)] = actual
    with pytest.raises(IndexError):
        tool.run_benchmark("real invalid parallel grid", NonlinearGKConfig(n_theta=1, n_steps=1))

    original_path = tool.RESULTS_FILE
    output = tmp_path / "scratch.json"
    try:
        tool.RESULTS_FILE = str(output)
        tool.save(results)
        assert json.loads(output.read_text()) == results
        tool.save({"actual_replacement": results["2"]})
        assert json.loads(output.read_text()) == {"actual_replacement": results["2"]}
        tool.save({"declared_nonstandard_float": math.nan})
        assert math.isnan(json.loads(output.read_text())["declared_nonstandard_float"])
        before_failure = output.read_bytes()
        circular: dict[str, object] = {}
        circular["self"] = circular
        with pytest.raises(ValueError, match="Circular reference"):
            tool.save(circular)
        assert output.is_file() and output.read_bytes() != before_failure
        with pytest.raises(TypeError):
            tool.save({"unsupported": Path("not JSON")})
        tool.RESULTS_FILE = str(tmp_path / "missing-parent/scratch.json")
        with pytest.raises(FileNotFoundError):
            tool.save(results)
        tool.RESULTS_FILE = str(tmp_path)
        with pytest.raises(DIRECTORY_AS_FILE_ERROR):
            tool.save(results)
    finally:
        tool.RESULTS_FILE = original_path
    stdout = capsys.readouterr().out
    assert "real software case" in stdout and "chi_i_gB" in stdout and "saved to" in stdout
