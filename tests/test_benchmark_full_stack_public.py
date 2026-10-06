# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public full-stack native timing probes

"""Exercise actual script/API probes against an existing real native extension, without claiming physics acceptance."""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", ["script", "api"])
def test_real_native_probes_and_readout_export(tmp_path: Path, entry: str) -> None:
    """Real script/native solve and outside-checkout API/solver failure both finish Rust controller and oscillator probes."""
    binary = ROOT / "scpn-control-rs/target/debug/libscpn_control_rs.so"
    if not binary.is_file():
        pytest.skip("needs the locally built debug Rust extension; there is no mock or automatic install")
    code = (
        "import importlib.util,importlib,sys,runpy; "
        "s=importlib.util.spec_from_file_location('scpn_control_rs',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s);sys.modules['scpn_control_rs']=m;s.loader.exec_module(m); "
    )
    if entry == "script":
        code += "runpy.run_path(sys.argv[2],run_name='__main__')"
    else:
        code += "m=importlib.import_module('tools.benchmark_full_stack'); assert m.benchmark_full_stack() is None"
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(binary), str(ROOT / "tools/benchmark_full_stack.py")],
        cwd=ROOT if entry == "script" else tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert "SCPN-CONTROL: Full-Stack Physics Benchmark" in result.stdout
    assert "Backend:      rust" in result.stdout
    assert "3. Raw Kuramoto Sync (Rust)" in result.stdout
    assert ("Solve time:" in result.stdout) is (entry == "script")
    assert ("Error:" in result.stdout) is (entry == "api")
    for label in [r"Control step:\s+([0-9.]+)", r"Throughput:\s+([0-9.]+)", r"Single step \(L=16, N=50\):\s+([0-9.]+)"]:
        match = re.search(label, result.stdout)
        assert match is not None
        value = float(match.group(1))
        assert math.isfinite(value) and value > 0
