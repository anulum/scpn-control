# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real native output-buffer contract tests.

"""Exercise public HPC buffer contracts against the packaged C++ solver.

This native profile requires a working C++ compiler; absence is a failed
precondition, never a passing native result. Compilation writes only pytest's
temporary directory. All Python calls load the actual source-bound library.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_control.core import hpc_bridge
from scpn_control.core.hpc_bridge import HPCBridge, NativeSolverError, NativeSolverStatus


@pytest.fixture(scope="module")
def native_library(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Compile admitted package sources into a caller-owned native library."""
    compiler = shutil.which("g++")
    assert compiler is not None, "real HPC contract tests require a C++ compiler"
    directory = tmp_path_factory.mktemp("native-hpc-contract")
    root = Path(hpc_bridge.__file__).resolve().parent
    manifest = json.loads((root / "solver_manifest.json").read_text(encoding="utf-8"))
    for name in ("solver.cpp", "solver.h"):
        path = root / name
        assert not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[name]["sha256"]

    system = platform.system()
    library = directory / ("scpn_solver.dll" if system == "Windows" else "libscpn_solver.so")
    argv = [
        compiler,
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-dynamiclib" if system == "Darwin" else "-shared",
        "-DSCPN_SOLVER_BUILD=1",
        "-O3",
        "-fstack-protector-strong",
        str(root / "solver.cpp"),
        "-o",
        str(library),
    ]
    if system == "Windows":
        argv.append("-static")
    else:
        argv.append("-fPIC")
    subprocess.run(argv, check=True, capture_output=True, text=True, timeout=120)
    return str(library)


@pytest.fixture
def native_bridge(native_library: str) -> Iterator[HPCBridge]:
    """Own a real five-by-seven state until the public context manager closes."""
    with HPCBridge(native_library) as bridge:
        assert bridge.is_available()
        bridge.initialize(7, 5, (1.0, 4.0), (-2.0, 1.0), boundary_value=0.25)
        yield bridge
    assert bridge.solver_ptr is None


def _run_into(
    bridge: HPCBridge,
    source: NDArray[np.float64],
    output: NDArray[np.float64],
    mode: str,
) -> object:
    """Invoke one public state-mutating output API with a bounded sweep cap."""
    if mode == "fixed":
        return bridge.solve_into(source, output, iterations=3)
    return bridge.solve_until_converged_into(source, output, max_iterations=3, tolerance=0.0, omega=1.2)


@pytest.mark.parametrize("mode", ["fixed", "converged"])
def test_native_readonly_output_refuses_before_state_mutation(native_bridge: HPCBridge, mode: str) -> None:
    """A read-only output remains unchanged and the actual native state is preserved."""
    source = np.full((5, 7), -1.0)
    state = native_bridge.solve(source, iterations=0)
    assert state is not None
    output = np.full_like(source, -97.0)
    output.setflags(write=False)
    source_bytes, output_bytes = source.tobytes(), output.tobytes()
    with pytest.raises(ValueError, match="psi_out must be writable"):
        _run_into(native_bridge, source, output, mode)
    assert not output.flags.writeable and output.tobytes() == output_bytes
    assert source.tobytes() == source_bytes
    after = native_bridge.solve(source, iterations=0)
    assert after is not None
    np.testing.assert_array_equal(after, state)


@pytest.mark.parametrize("mode", ["fixed", "converged"])
@pytest.mark.parametrize("kind", ["non_array", "wrong_dtype", "fortran", "wrong_shape", "overlap", "partial_overlap"])
def test_native_invalid_output_refuses_before_state_mutation(native_bridge: HPCBridge, mode: str, kind: str) -> None:
    """Malformed or overlapping actual buffers are refused without a native sweep."""
    source = np.full((5, 7), -1.0)
    expected = {
        "non_array": "numpy.ndarray",
        "wrong_dtype": "dtype float64",
        "fortran": "C-contiguous",
        "wrong_shape": "shape mismatch",
        "overlap": "must not overlap",
        "partial_overlap": "must not overlap",
    }
    if kind == "non_array":
        output = cast(NDArray[np.float64], [[0.0] * 7] * 5)
    elif kind == "wrong_dtype":
        output = cast(NDArray[np.float64], np.full((5, 7), -97.0, dtype=np.float32))
    elif kind == "fortran":
        output = np.full((5, 7), -97.0, order="F")
    elif kind == "wrong_shape":
        output = np.full((6, 7), -97.0)
    elif kind == "overlap":
        output = source
    else:
        backing = np.full(36, -1.0)
        source, output = backing[:-1].reshape(5, 7), backing[1:].reshape(5, 7)
    state = native_bridge.solve(source, iterations=0)
    assert state is not None
    source_bytes = source.tobytes()
    output_bytes = output.tobytes() if isinstance(output, np.ndarray) else None
    with pytest.raises(ValueError, match=expected[kind]):
        _run_into(native_bridge, source, output, mode)
    assert source.tobytes() == source_bytes
    if isinstance(output, np.ndarray):
        assert output.tobytes() == output_bytes
    after = native_bridge.solve(source, iterations=0)
    assert after is not None
    np.testing.assert_array_equal(after, state)


@pytest.mark.parametrize("mode", ["fixed", "converged"])
def test_native_readonly_source_and_writable_output(native_bridge: HPCBridge, mode: str) -> None:
    """Const input is accepted while the caller's output receives a finite solution."""
    source = np.full((5, 7), -1.0)
    source.setflags(write=False)
    original = source.tobytes()
    output = np.full_like(source, -97.0)
    result = _run_into(native_bridge, source, output, mode)
    if mode == "fixed":
        assert result is output
    else:
        assert isinstance(result, tuple) and result[0] == 3
        assert np.isfinite(result[1])
    assert source.tobytes() == original and not source.flags.writeable
    assert np.all(np.isfinite(output)) and np.all(output[0, :] == 0.25)
    assert np.all(output[-1, :] == 0.25) and np.all(output[:, 0] == 0.25)
    assert np.all(output[:, -1] == 0.25)
    assert float(output[2, 3]) != -97.0


@pytest.mark.parametrize("mode", ["fixed", "converged"])
def test_native_rejected_source_preserves_output(native_bridge: HPCBridge, mode: str) -> None:
    """Actual non-finite source data cannot enter the native solver or alter output."""
    source = np.full((5, 7), -1.0)
    source[2, 3] = np.nan
    output = np.full_like(source, -97.0)
    before = output.tobytes()
    with pytest.raises(ValueError, match="only finite"):
        _run_into(native_bridge, source, output, mode)
    assert output.tobytes() == before


def test_native_manufactured_elliptic_solution(native_library: str) -> None:
    """A real bounded RHS solve recovers a polynomial solution with zero edge flux."""
    nr, nz = 13, 11
    radius = np.linspace(1.0, 4.0, nr)[None, :]
    height = np.linspace(-2.0, 1.0, nz)[:, None]
    radial = (radius - 1.0) * (4.0 - radius)
    vertical = (height + 2.0) * (1.0 - height)
    exact = np.asarray(radial * vertical)
    # The SOR kernel solves -psi_RR + psi_R/R - psi_ZZ = source.
    source = np.asarray(2.0 * vertical + (5.0 - 2.0 * radius) * vertical / radius + 2.0 * radial)
    source_bytes = source.tobytes()
    with HPCBridge(native_library) as bridge:
        assert bridge.is_available()
        bridge.initialize(nr, nz, (1.0, 4.0), (-2.0, 1.0))
        result = bridge.solve_until_converged(source, max_iterations=2000, tolerance=1e-12, omega=1.7)
        assert result is not None
        actual, used, delta = result
        assert 1 < used < 2000 and 0.0 <= delta <= 1e-12
        np.testing.assert_allclose(actual, exact, rtol=0.0, atol=1e-10)
        assert source.tobytes() == source_bytes


def test_native_state_lifetime_and_status_errors(native_library: str) -> None:
    """Public native allocation/refusal/close/reinitialisation preserve typed statuses."""
    with HPCBridge(native_library) as bridge:
        assert bridge.is_available()
        with pytest.raises(NativeSolverError) as invalid:
            bridge.initialize(2, 5, (1.0, 4.0), (-2.0, 1.0))
        assert invalid.value.status is NativeSolverStatus.INVALID_DIMENSIONS
        assert bridge.solver_ptr is None
        bridge.initialize(7, 5, (1.0, 4.0), (-2.0, 1.0))
        with pytest.raises(RuntimeError, match="already initialized"):
            bridge.initialize(7, 5, (1.0, 4.0), (-2.0, 1.0))
        source = np.zeros((5, 7))
        with pytest.raises(NativeSolverError) as negative:
            bridge.solve(source, iterations=-1)
        assert negative.value.status is NativeSolverStatus.INVALID_ARGUMENT
        bridge.close()
        bridge.close()
        assert bridge.solve(source) is None
        bridge.initialize(7, 5, (1.0, 4.0), (-2.0, 1.0))
        result = bridge.solve(source, iterations=1)
        assert result is not None and np.all(result == 0.0)
