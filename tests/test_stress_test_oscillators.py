# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native oscillator capacity command contracts.
"""Exercise the actual capacity API and CLI with their installed dependencies."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.stress_test_oscillators import stress_test

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_REFUSAL = "Oscillator capacity benchmark dependencies are unavailable.\n"


def _assert_capacity_table(output: str) -> None:
    """Check native sweep sizes and its actual completion or capacity stop."""
    rows = re.findall(r"^\s*(\d+)\s*\|\s*(\d+)\s*\|", output, re.MULTILINE)
    counts = [int(size) for size, _total in rows]
    expected = [10, 50, 100, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    assert counts and counts == expected[: len(counts)]
    assert all(int(total) == 16 * int(size) for size, total in rows)
    if len(counts) == len(expected) and "Capacity limit reached" not in output:
        assert "Completed full sweep." in output
    else:
        assert "Capacity limit reached" in output


@pytest.mark.parametrize("isolated", [False, True])
def test_capacity_cli_runs_native_or_authors_dependency_refusal(tmp_path: Path, isolated: bool) -> None:
    """Run the actual standalone command, including a standard-library context."""
    argv = [sys.executable]
    if isolated:
        argv.append("-S")
    argv.append(str(ROOT / "tools/stress_test_oscillators.py"))
    environment = dict(os.environ)
    if isolated:
        environment.pop("PYTHONPATH", None)
    result = subprocess.run(argv, cwd=tmp_path, env=environment, capture_output=True, text=True, check=False)
    native_available = not isolated and importlib.util.find_spec("scpn_control_rs") is not None
    if native_available:
        assert result.returncode == 0
        assert result.stderr == ""
        _assert_capacity_table(result.stdout)
    else:
        assert result.returncode == 1
        assert result.stdout == ""
        assert result.stderr == DEPENDENCY_REFUSAL


def test_capacity_api_preserves_native_import_failure_or_prints_sweep(capsys: pytest.CaptureFixture[str]) -> None:
    """Call the real public function and preserve the caller's global RNG state."""
    state = np.random.get_state()
    try:
        if importlib.util.find_spec("scpn_control_rs") is None:
            with pytest.raises(ImportError, match="scpn_control_rs"):
                stress_test()
            assert capsys.readouterr().out == ""
        else:
            stress_test()
            _assert_capacity_table(capsys.readouterr().out)
    finally:
        np.random.set_state(state)
