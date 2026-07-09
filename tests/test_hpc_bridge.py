# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: Tests for native C++ solver bridge safety and low-copy behaviour.
"""Unit tests for HPC bridge safety and low-copy behaviour."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import ctypes
import sys
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

import numpy as np
import pytest

from scpn_control.core import hpc_bridge as hpc_mod
from scpn_control.core.hpc_bridge import HPCBridge, _as_contiguous_f64

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ArrayF64 = np.ndarray[Any, np.dtype[np.float64]]


class _DummyLib:
    def __init__(self) -> None:
        self.called = False
        self.called_converged = False
        self.last_size: int | None = None
        self.last_iterations: int | None = None
        self.last_max_iterations: int | None = None
        self.last_omega: float | None = None
        self.last_tolerance: float | None = None
        self.destroyed: object | None = None
        self.boundary_value: float | None = None
        self.last_psi_ref: ArrayF64 | None = None

    def run_step(
        self,
        solver_ptr: object,
        j_array: ArrayF64,
        psi_array: ArrayF64,
        size: int,
        iterations: int,
    ) -> None:
        del solver_ptr
        self.called = True
        self.last_size = int(size)
        self.last_iterations = int(iterations)
        self.last_psi_ref = psi_array
        psi_array[:] = 0.5 * j_array

    def run_step_converged(
        self,
        solver_ptr: object,
        j_array: ArrayF64,
        psi_array: ArrayF64,
        size: int,
        max_iterations: int,
        omega: float,
        tolerance: float,
        final_delta_ptr: Any,
    ) -> int:
        del solver_ptr
        self.called_converged = True
        self.last_size = int(size)
        self.last_max_iterations = int(max_iterations)
        self.last_omega = float(omega)
        self.last_tolerance = float(tolerance)
        self.last_psi_ref = psi_array
        psi_array[:] = 0.25 * j_array
        final_delta_ptr._obj.value = 2.5e-4
        return 7

    def destroy_solver(self, solver_ptr: object) -> None:
        self.destroyed = solver_ptr

    def set_boundary_dirichlet(self, solver_ptr: object, value: float) -> None:
        del solver_ptr
        self.boundary_value = float(value)


class _DummyDeleteLib:
    def __init__(self) -> None:
        self.deleted: object | None = None

    def delete_solver(self, solver_ptr: object) -> None:
        self.deleted = solver_ptr


def _make_bridge(nr: int = 2, nz: int = 3) -> HPCBridge:
    bridge = HPCBridge.__new__(HPCBridge)
    bridge.lib = cast(Any, _DummyLib())
    bridge.solver_ptr = cast(Any, 12345)
    bridge.loaded = True
    bridge._destroy_symbol = "destroy_solver"
    bridge._has_converged_api = True
    bridge._has_boundary_api = True
    bridge.nr = nr
    bridge.nz = nz
    return bridge


def _dummy_lib(bridge: HPCBridge) -> _DummyLib:
    return cast(_DummyLib, bridge.lib)


def _delete_lib(bridge: HPCBridge) -> _DummyDeleteLib:
    return cast(_DummyDeleteLib, bridge.lib)


def _write_solver_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: bytes = b'extern "C" void create_solver() {}\n',
) -> Path:
    module_file = tmp_path / "hpc_bridge.py"
    module_file.write_text("# test module path\n", encoding="utf-8")
    source_path = tmp_path / "solver.cpp"
    source_path.write_bytes(source)
    digest = hashlib.sha256(source).hexdigest()
    (tmp_path / "solver_manifest.json").write_text(
        json.dumps({"solver.cpp": {"sha256": digest}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(hpc_mod, "__file__", str(module_file))
    return source_path


def _admit_test_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hpc_mod, "_native_build_compiler", lambda: "/usr/bin/g++")


def test_as_contiguous_f64_reuses_float64_c_contiguous() -> None:
    arr = np.zeros((3, 2), dtype=np.float64)
    out = _as_contiguous_f64(arr)
    assert out is arr


def test_as_contiguous_f64_converts_dtype_and_layout() -> None:
    arr = np.asfortranarray(np.zeros((3, 2), dtype=np.float32))
    out = _as_contiguous_f64(arr)
    assert out.dtype == np.float64
    assert out.flags.c_contiguous


def test_solve_returns_none_when_solver_not_initialized() -> None:
    bridge = _make_bridge()
    bridge.solver_ptr = None
    result = bridge.solve(np.zeros((3, 2), dtype=np.float64))
    assert result is None


def test_solve_raises_on_shape_mismatch() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    with pytest.raises(ValueError, match="shape mismatch"):
        bridge.solve(np.zeros((2, 2), dtype=np.float64))


def test_solve_rejects_non_2d_input() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    with pytest.raises(ValueError, match="must be a 2D array"):
        bridge.solve(np.zeros((6,), dtype=np.float64))


def test_solve_rejects_nonfinite_input() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    bad = np.arange(6, dtype=np.float64).reshape(3, 2)
    bad[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        bridge.solve(bad)


def test_solve_runs_and_returns_expected_shape() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out = bridge.solve(j_phi, iterations=17)

    assert out is not None
    assert out.shape == (3, 2)
    assert np.allclose(out, 0.5 * j_phi)
    lib = _dummy_lib(bridge)
    assert lib.called
    assert lib.last_size == 6
    assert lib.last_iterations == 17


def test_solve_into_reuses_output_buffer() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out_buf = np.empty((3, 2), dtype=np.float64)
    out = bridge.solve_into(j_phi, out_buf, iterations=5)

    assert out is out_buf
    assert _dummy_lib(bridge).last_psi_ref is out_buf
    assert np.allclose(out_buf, 0.5 * j_phi)
    assert _dummy_lib(bridge).last_iterations == 5


def test_solve_into_rejects_noncontiguous_output() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out_buf = np.empty((2, 3), dtype=np.float64).T
    with pytest.raises(ValueError, match="C-contiguous"):
        bridge.solve_into(j_phi, out_buf, iterations=3)


def test_solve_into_rejects_wrong_shape_output() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out_buf = np.empty((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="shape mismatch"):
        bridge.solve_into(j_phi, out_buf, iterations=3)


def test_solve_into_rejects_non_ndarray_output() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    with pytest.raises(ValueError, match="numpy.ndarray"):
        bridge.solve_into(j_phi, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], iterations=3)  # type: ignore[arg-type]


def test_solve_until_converged_uses_native_api() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out = bridge.solve_until_converged(j_phi, max_iterations=200, tolerance=1e-5, omega=1.7)
    assert out is not None
    psi, iters, delta = out
    assert psi.shape == (3, 2)
    assert np.allclose(psi, 0.25 * j_phi)
    assert iters == 7
    assert abs(delta - 2.5e-4) < 1e-12
    lib = _dummy_lib(bridge)
    assert lib.called_converged
    assert lib.last_max_iterations == 200
    assert lib.last_omega is not None
    assert lib.last_tolerance is not None
    assert abs(lib.last_omega - 1.7) < 1e-12
    assert abs(lib.last_tolerance - 1e-5) < 1e-12


def test_solve_until_converged_into_reuses_output_buffer() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out_buf = np.empty((3, 2), dtype=np.float64)
    out = bridge.solve_until_converged_into(
        j_phi,
        out_buf,
        max_iterations=33,
        tolerance=1e-7,
        omega=1.5,
    )
    assert out is not None
    iters, delta = out
    assert iters == 7
    assert abs(delta - 2.5e-4) < 1e-12
    assert _dummy_lib(bridge).last_psi_ref is out_buf
    assert np.allclose(out_buf, 0.25 * j_phi)
    assert _dummy_lib(bridge).last_max_iterations == 33


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_iterations": 0}, "max_iterations"),
        ({"tolerance": float("nan")}, "tolerance"),
        ({"tolerance": -1.0e-3}, "tolerance"),
        ({"omega": float("inf")}, "omega"),
        ({"omega": 0.0}, "omega"),
        ({"omega": 2.0}, "omega"),
    ],
)
def test_solve_until_converged_into_rejects_invalid_convergence_params(
    kwargs: dict[str, float | int], match: str
) -> None:
    bridge = _make_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out_buf = np.empty((3, 2), dtype=np.float64)
    params: dict[str, float | int] = {
        "max_iterations": 32,
        "tolerance": 1e-7,
        "omega": 1.6,
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        bridge.solve_until_converged_into(
            j_phi,
            out_buf,
            max_iterations=int(params["max_iterations"]),
            tolerance=float(params["tolerance"]),
            omega=float(params["omega"]),
        )


def test_solve_until_converged_falls_back_without_native_api() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    bridge._has_converged_api = False
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out = bridge.solve_until_converged(j_phi, max_iterations=12)
    assert out is not None
    psi, iters, delta = out
    assert psi.shape == (3, 2)
    assert np.allclose(psi, 0.5 * j_phi)
    assert iters == 12
    assert np.isnan(delta)


def test_solve_until_converged_into_fallback_without_native_api() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    bridge._has_converged_api = False
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    out_buf = np.empty((3, 2), dtype=np.float64)
    out = bridge.solve_until_converged_into(j_phi, out_buf, max_iterations=9)
    assert out is not None
    iters, delta = out
    assert iters == 9
    assert np.isnan(delta)
    assert np.allclose(out_buf, 0.5 * j_phi)
    assert _dummy_lib(bridge).last_iterations == 9


def test_set_boundary_dirichlet_calls_native_symbol() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    bridge.set_boundary_dirichlet(1.25)
    assert _dummy_lib(bridge).boundary_value == 1.25


def test_set_boundary_dirichlet_noop_without_support() -> None:
    bridge = _make_bridge(nr=2, nz=3)
    bridge._has_boundary_api = False
    bridge.set_boundary_dirichlet(0.5)
    assert _dummy_lib(bridge).boundary_value is None


def test_close_releases_solver_pointer() -> None:
    bridge = _make_bridge()
    bridge.close()
    assert bridge.solver_ptr is None
    assert _dummy_lib(bridge).destroyed == 12345


def test_close_supports_delete_solver_alias() -> None:
    bridge = HPCBridge.__new__(HPCBridge)
    bridge.lib = cast(Any, _DummyDeleteLib())
    bridge.solver_ptr = cast(Any, 999)
    bridge.loaded = True
    bridge._destroy_symbol = "delete_solver"
    bridge.close()
    assert bridge.solver_ptr is None
    assert _delete_lib(bridge).deleted == 999


def test_init_prefers_env_override_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = tmp_path / "scpn_solver_override.so"

    def _raise_cdll(_path: str) -> object:
        raise OSError("no library")

    monkeypatch.setenv("SCPN_SOLVER_LIB", str(expected))
    monkeypatch.setenv("SCPN_ALLOW_EXTERNAL_SOLVER_LIB", "1")
    monkeypatch.setattr(ctypes, "CDLL", _raise_cdll)
    bridge = HPCBridge()
    assert bridge.lib_path == str(expected.resolve(strict=False))
    assert not bridge.loaded


def test_init_rejects_relative_env_solver_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected_cdll(_path: str) -> object:
        raise AssertionError("ctypes.CDLL must not receive an untrusted solver path")

    monkeypatch.setenv("SCPN_SOLVER_LIB", "relative/libscpn_solver.so")
    monkeypatch.setattr(ctypes, "CDLL", _unexpected_cdll)

    with pytest.raises(ValueError, match="absolute"):
        HPCBridge()


def test_init_rejects_non_library_suffix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    candidate = tmp_path / "solver.txt"

    def _unexpected_cdll(_path: str) -> object:
        raise AssertionError("ctypes.CDLL must not receive a non-library path")

    monkeypatch.setenv("SCPN_SOLVER_LIB", str(candidate))
    monkeypatch.setenv("SCPN_ALLOW_EXTERNAL_SOLVER_LIB", "1")
    monkeypatch.setattr(ctypes, "CDLL", _unexpected_cdll)

    with pytest.raises(ValueError, match="must end with"):
        HPCBridge()


def test_init_rejects_directory_solver_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    candidate = tmp_path / "libscpn_solver.so"
    candidate.mkdir()

    def _unexpected_cdll(_path: str) -> object:
        raise AssertionError("ctypes.CDLL must not receive a directory path")

    monkeypatch.setenv("SCPN_SOLVER_LIB", str(candidate))
    monkeypatch.setenv("SCPN_ALLOW_EXTERNAL_SOLVER_LIB", "1")
    monkeypatch.setattr(ctypes, "CDLL", _unexpected_cdll)

    with pytest.raises(ValueError, match="regular file"):
        HPCBridge()


def test_init_rejects_external_env_solver_path_without_trust_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    external = tmp_path / "libscpn_solver.so"
    external.write_bytes(b"not a real shared object")

    def _unexpected_cdll(_path: str) -> object:
        raise AssertionError("ctypes.CDLL must not receive an untrusted solver path")

    monkeypatch.setenv("SCPN_SOLVER_LIB", str(external))
    monkeypatch.delenv("SCPN_ALLOW_EXTERNAL_SOLVER_LIB", raising=False)
    monkeypatch.setattr(ctypes, "CDLL", _unexpected_cdll)

    with pytest.raises(ValueError, match="trusted package-local"):
        HPCBridge()


def test_init_uses_package_local_default_without_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_cdll(_path: str) -> object:
        raise OSError("no library")

    monkeypatch.delenv("SCPN_SOLVER_LIB", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(ctypes, "CDLL", _raise_cdll)

    bridge = HPCBridge()
    expected = str(Path(hpc_mod.__file__).resolve().parent / "libscpn_solver.so")
    assert bridge.lib_path == expected
    assert not bridge.loaded
    assert not hasattr(bridge, "_cleanup_state")
    bridge.close()


def test_compile_cpp_requires_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCPN_ALLOW_NATIVE_BUILD", raising=False)
    assert hpc_mod.compile_cpp() is None


def test_compile_cpp_builds_in_package_bin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def _fake_run(
        cmd: Sequence[str],
        check: bool,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> None:
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"shared object")
        calls["cmd"] = list(cmd)
        calls["check"] = check
        calls["cwd"] = cwd
        calls["env"] = dict(env)
        calls["timeout"] = timeout

    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    _admit_test_compiler(monkeypatch)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    _write_solver_source(tmp_path, monkeypatch)

    out = hpc_mod.compile_cpp()
    assert out is not None
    assert Path(out).name == "libscpn_solver.so"
    assert Path(out).parent.name == "bin"
    assert calls["check"] is True
    assert isinstance(calls["cmd"], list)
    compiler_cmd = str(calls["cmd"][0])
    assert PurePosixPath(compiler_cmd).is_absolute() or PureWindowsPath(compiler_cmd).is_absolute()
    assert "-march=native" not in calls["cmd"]
    assert "-mtune=generic" in calls["cmd"]
    assert "-fstack-protector-strong" in calls["cmd"]
    assert "-Wl,-z,relro" in calls["cmd"]
    assert calls["timeout"] == hpc_mod._NATIVE_BUILD_TIMEOUT_S
    build_env = cast(dict[str, str], calls["env"])
    assert "PYTHONPATH" not in build_env
    assert "LD_PRELOAD" not in build_env
    assert "LD_LIBRARY_PATH" not in build_env
    assert "DYLD_INSERT_LIBRARIES" not in build_env
    assert Path(str(calls["cwd"])).name == tmp_path.name
    assert Path(out).read_bytes() == b"shared object"


def test_packaged_solver_manifest_matches_source() -> None:
    """The shipped native solver source must match its package manifest."""

    source_path = Path(hpc_mod.__file__).resolve().parent / "solver.cpp"
    manifest_path = Path(hpc_mod.__file__).resolve().parent / "solver_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert source_path.is_file()
    assert manifest == {
        "solver.cpp": {
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
    }
    assert hpc_mod._verify_solver_source(source_path, manifest_path)


def test_packaged_solver_artifacts_are_declared_in_package_data() -> None:
    """The wheel/sdist metadata must include the native source contract."""

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    package_data = cast(list[str], pyproject["tool"]["setuptools"]["package-data"]["scpn_control.core"])

    assert {"solver.cpp", "solver_manifest.json"} <= set(package_data)


def test_compile_cpp_builds_and_loads_packaged_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-in native build path compiles and loads the shipped solver."""

    if shutil.which("g++") is None:
        pytest.skip("g++ is unavailable")

    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    monkeypatch.delenv("SCPN_SOLVER_LIB", raising=False)

    out = hpc_mod.compile_cpp()
    assert out is not None
    out_path = Path(out)
    try:
        with HPCBridge(out) as bridge:
            assert bridge.is_available()
            bridge.initialize(5, 5, (1.0, 5.0), (-2.0, 2.0))
            source = -np.ones((5, 5), dtype=np.float64)

            psi = bridge.solve(source, iterations=20)
            assert psi is not None
            assert psi.shape == source.shape
            assert np.all(np.isfinite(psi))
            assert np.allclose(psi[0, :], 0.0)
            assert np.allclose(psi[-1, :], 0.0)
            assert np.allclose(psi[:, 0], 0.0)
            assert np.allclose(psi[:, -1], 0.0)
            assert abs(float(psi[2, 2])) > 1e-12

            converged = bridge.solve_until_converged(source, max_iterations=5, tolerance=0.0, omega=1.2)
            assert converged is not None
            _, iterations_used, final_delta = converged
            assert 0 <= iterations_used <= 5
            assert np.isfinite(final_delta)
    finally:
        out_path.unlink(missing_ok=True)
        try:
            out_path.parent.rmdir()
        except OSError:
            pass


def test_compile_cpp_windows_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, list[str]] = {}

    def _fake_run(
        cmd: Sequence[str],
        check: bool,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> None:
        del check, cwd, env, timeout
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"shared object")
        calls["cmd"] = list(cmd)

    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    _admit_test_compiler(monkeypatch)
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    _write_solver_source(tmp_path, monkeypatch)

    out = hpc_mod.compile_cpp()
    assert out is not None
    assert Path(out).name == "scpn_solver.dll"
    assert "-mavx2" not in calls["cmd"]


def test_compile_cpp_refuses_missing_compiler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _unexpected_run(
        cmd: Sequence[str],
        check: bool,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> None:
        del cmd, check, cwd, env, timeout
        raise AssertionError("native build must not run without an admitted compiler")

    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(subprocess, "run", _unexpected_run)
    _write_solver_source(tmp_path, monkeypatch)

    assert hpc_mod.compile_cpp() is None


def test_compile_cpp_refuses_symlink_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")

    real_bin = tmp_path / "real-bin"
    real_bin.mkdir()
    (tmp_path / "bin").symlink_to(real_bin, target_is_directory=True)

    def _unexpected_run(
        cmd: Sequence[str],
        check: bool,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> None:
        del cmd, check, cwd, env, timeout
        raise AssertionError("native build must not write into symlink output directory")

    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    _admit_test_compiler(monkeypatch)
    monkeypatch.setattr(subprocess, "run", _unexpected_run)
    _write_solver_source(tmp_path, monkeypatch)

    assert hpc_mod.compile_cpp() is None


def test_compile_cpp_refuses_existing_symlink_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "target.so").write_bytes(b"target")
    (bin_dir / "libscpn_solver.so").symlink_to(bin_dir / "target.so")

    def _unexpected_run(
        cmd: Sequence[str],
        check: bool,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> None:
        del cmd, check, cwd, env, timeout
        raise AssertionError("native build must not overwrite symlink output")

    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    _admit_test_compiler(monkeypatch)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(subprocess, "run", _unexpected_run)
    _write_solver_source(tmp_path, monkeypatch)

    assert hpc_mod.compile_cpp() is None


def test_compile_cpp_refuses_symlink_solver_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink support unavailable")

    module_file = tmp_path / "hpc_bridge.py"
    module_file.write_text("# test module path\n", encoding="utf-8")
    real_source = tmp_path / "real_solver.cpp"
    real_source.write_text("int main() { return 0; }\n", encoding="utf-8")
    (tmp_path / "solver.cpp").symlink_to(real_source)
    digest = hashlib.sha256(real_source.read_bytes()).hexdigest()
    (tmp_path / "solver_manifest.json").write_text(
        json.dumps({"solver.cpp": {"sha256": digest}}),
        encoding="utf-8",
    )

    def _unexpected_run(
        cmd: Sequence[str],
        check: bool,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> None:
        del cmd, check, cwd, env, timeout
        raise AssertionError("symlinked solver source must not be compiled")

    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    _admit_test_compiler(monkeypatch)
    monkeypatch.setattr(hpc_mod, "__file__", str(module_file))
    monkeypatch.setattr(subprocess, "run", _unexpected_run)

    assert hpc_mod.compile_cpp() is None


def test_compile_cpp_refuses_missing_checksum_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_file = tmp_path / "hpc_bridge.py"
    module_file.write_text("# test module path\n", encoding="utf-8")
    (tmp_path / "solver.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")

    def _unexpected_run(
        cmd: Sequence[str],
        check: bool,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> None:
        del cmd, check, cwd, env, timeout
        raise AssertionError("unchecked solver.cpp must not be compiled")

    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    _admit_test_compiler(monkeypatch)
    monkeypatch.setattr(hpc_mod, "__file__", str(module_file))
    monkeypatch.setattr(subprocess, "run", _unexpected_run)

    assert hpc_mod.compile_cpp() is None


def test_compile_cpp_refuses_missing_solver_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module_file = tmp_path / "hpc_bridge.py"
    module_file.write_text("# test module path\n", encoding="utf-8")
    (tmp_path / "solver_manifest.json").write_text(
        json.dumps({"solver.cpp": {"sha256": "0" * 64}}),
        encoding="utf-8",
    )

    def _unexpected_run(
        cmd: Sequence[str],
        check: bool,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> None:
        del cmd, check, cwd, env, timeout
        raise AssertionError("missing solver.cpp must not be compiled")

    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    _admit_test_compiler(monkeypatch)
    monkeypatch.setattr(hpc_mod, "__file__", str(module_file))
    monkeypatch.setattr(subprocess, "run", _unexpected_run)

    assert hpc_mod.compile_cpp() is None


def test_compile_cpp_refuses_unreadable_manifest_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module_file = tmp_path / "hpc_bridge.py"
    module_file.write_text("# test module path\n", encoding="utf-8")
    (tmp_path / "solver.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (tmp_path / "solver_manifest.json").write_text("{", encoding="utf-8")

    def _unexpected_run(
        cmd: Sequence[str],
        check: bool,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> None:
        del cmd, check, cwd, env, timeout
        raise AssertionError("solver.cpp must not be compiled with unreadable manifest")

    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    _admit_test_compiler(monkeypatch)
    monkeypatch.setattr(hpc_mod, "__file__", str(module_file))
    monkeypatch.setattr(subprocess, "run", _unexpected_run)

    assert hpc_mod.compile_cpp() is None


def test_compile_cpp_refuses_manifest_without_valid_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module_file = tmp_path / "hpc_bridge.py"
    module_file.write_text("# test module path\n", encoding="utf-8")
    (tmp_path / "solver.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (tmp_path / "solver_manifest.json").write_text(json.dumps({"solver.cpp": {"sha256": "short"}}), encoding="utf-8")

    def _unexpected_run(
        cmd: Sequence[str],
        check: bool,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> None:
        del cmd, check, cwd, env, timeout
        raise AssertionError("solver.cpp must not be compiled without manifest SHA-256")

    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    _admit_test_compiler(monkeypatch)
    monkeypatch.setattr(hpc_mod, "__file__", str(module_file))
    monkeypatch.setattr(subprocess, "run", _unexpected_run)

    assert hpc_mod.compile_cpp() is None


def test_compile_cpp_refuses_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_file = tmp_path / "hpc_bridge.py"
    module_file.write_text("# test module path\n", encoding="utf-8")
    (tmp_path / "solver.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (tmp_path / "solver_manifest.json").write_text(
        json.dumps({"solver.cpp": {"sha256": "0" * 64}}),
        encoding="utf-8",
    )

    def _unexpected_run(
        cmd: Sequence[str],
        check: bool,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> None:
        del cmd, check, cwd, env, timeout
        raise AssertionError("tampered solver.cpp must not be compiled")

    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    monkeypatch.setattr(hpc_mod, "__file__", str(module_file))
    monkeypatch.setattr(subprocess, "run", _unexpected_run)

    assert hpc_mod.compile_cpp() is None


def test_compile_cpp_handles_build_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fail_run(
        cmd: Sequence[str],
        check: bool,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> None:
        del check, cwd, env, timeout
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    _admit_test_compiler(monkeypatch)
    monkeypatch.setattr(subprocess, "run", _fail_run)
    _write_solver_source(tmp_path, monkeypatch)

    assert hpc_mod.compile_cpp() is None


def test_compile_cpp_refuses_missing_temporary_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_run(
        cmd: Sequence[str],
        check: bool,
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> None:
        del cmd, check, cwd, env, timeout
        return None

    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    _admit_test_compiler(monkeypatch)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    _write_solver_source(tmp_path, monkeypatch)

    assert hpc_mod.compile_cpp() is None


def test_context_manager_protocol() -> None:
    bridge = _make_bridge()
    with bridge as b:
        assert b is bridge
        assert b.solver_ptr is not None
    assert bridge.solver_ptr is None
    assert _dummy_lib(bridge).destroyed == 12345


def test_solve_rejects_empty_input() -> None:
    bridge = _make_bridge(nr=0, nz=0)
    bridge.nr = 2
    bridge.nz = 3
    with pytest.raises(ValueError, match="non-empty"):
        bridge.solve(np.zeros((0, 0), dtype=np.float64))


def test_require_c_contiguous_f64_wrong_dtype() -> None:
    from scpn_control.core.hpc_bridge import _require_c_contiguous_f64

    arr = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="dtype float64"):
        _require_c_contiguous_f64(arr, (3, 2), "test")


def test_sanitize_convergence_params_valid() -> None:
    from scpn_control.core.hpc_bridge import _sanitize_convergence_params

    iters, tol, omega = _sanitize_convergence_params(100, 1e-6, 1.5)
    assert iters == 100
    assert abs(tol - 1e-6) < 1e-12
    assert abs(omega - 1.5) < 1e-12


def test_close_noop_when_no_solver_ptr() -> None:
    bridge = _make_bridge()
    bridge.solver_ptr = None
    bridge.close()
    assert _dummy_lib(bridge).destroyed is None


def test_close_handles_destroy_symbol_failure_without_leaking_pointer() -> None:
    class _BrokenDestroyLib:
        def destroy_solver(self, _solver_ptr: object) -> None:
            raise AttributeError("symbol unavailable")

    bridge = HPCBridge.__new__(HPCBridge)
    bridge.lib = cast(Any, _BrokenDestroyLib())
    bridge.solver_ptr = cast(Any, 321)
    bridge.loaded = True
    bridge._destroy_symbol = "destroy_solver"

    bridge.close()

    assert bridge.solver_ptr is None


def test_close_noop_when_not_loaded() -> None:
    bridge = _make_bridge()
    bridge.loaded = False
    bridge.close()
    assert _dummy_lib(bridge).destroyed is None


def test_bridge_uses_finalizer_instead_of_destructor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _CFunc:
        def __init__(self, func: Callable[..., object]) -> None:
            self._func = func
            self.argtypes: object | None = None
            self.restype: object | None = None

        def __call__(self, *args: object) -> object:
            return self._func(*args)

    class _LoadableLib:
        def __init__(self) -> None:
            self.destroyed: object | None = None
            self.create_solver = _CFunc(lambda nr, nz, r0, r1, z0, z1: 4242)
            self.run_step = _CFunc(lambda solver_ptr, j_array, psi_array, size, iterations: None)
            self.destroy_solver = _CFunc(self._destroy)
            self.set_boundary_dirichlet = _CFunc(lambda solver_ptr, value: None)

        def _destroy(self, solver_ptr: object) -> None:
            self.destroyed = solver_ptr

    loaded = _LoadableLib()

    monkeypatch.setenv("SCPN_SOLVER_LIB", str(tmp_path / "libscpn_solver.so"))
    monkeypatch.setenv("SCPN_ALLOW_EXTERNAL_SOLVER_LIB", "1")
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: loaded)

    bridge = HPCBridge()
    bridge.initialize(2, 3, (0.0, 1.0), (-1.0, 1.0))

    assert "__del__" not in HPCBridge.__dict__
    assert "weakref.finalize" in inspect.getsource(hpc_mod)
    assert bridge._finalizer.alive

    bridge.close()
    assert loaded.destroyed == 4242
    assert bridge.solver_ptr is None
    assert not bridge._finalizer.alive

    loaded.destroyed = None
    bridge.close()
    assert loaded.destroyed is None


def test_native_build_environment_keeps_only_portable_build_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scpn_control.core.hpc_bridge import _native_build_environment

    monkeypatch.setenv("LD_PRELOAD", "/tmp/untrusted.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp")
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/tmp/untrusted.dylib")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/tmp")
    monkeypatch.setenv("DYLD_FRAMEWORK_PATH", "/tmp")
    monkeypatch.setenv("SCPN_SOLVER_LIB", "/tmp/untrusted.so")

    env = _native_build_environment(tmp_path)

    assert env["TMPDIR"] == str(tmp_path)
    assert "PATH" in env
    assert "LD_PRELOAD" not in env
    assert "LD_LIBRARY_PATH" not in env
    assert "DYLD_INSERT_LIBRARIES" not in env
    assert "DYLD_LIBRARY_PATH" not in env
    assert "DYLD_FRAMEWORK_PATH" not in env
    assert "SCPN_SOLVER_LIB" not in env


def test_env_solver_path_external_opt_in_preserves_absolute_library_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scpn_control.core.hpc_bridge import _validate_solver_library_path

    candidate = tmp_path / "libscpn_solver.so"
    candidate.write_bytes(b"not a real shared library")
    monkeypatch.setenv("SCPN_ALLOW_EXTERNAL_SOLVER_LIB", "1")

    resolved = _validate_solver_library_path(str(candidate), source="SCPN_SOLVER_LIB")

    assert resolved == str(candidate.resolve())


# ── Build-helper, signature-variant and native-path contracts ─────────────────


class _Sym:
    """Placeholder native symbol that accepts argtypes/restype assignment."""


class _FakeNativeLib:
    def __init__(self, names: tuple[str, ...]) -> None:
        for name in names:
            setattr(self, name, _Sym())


def _bridge_with_lib(lib: object) -> HPCBridge:
    bridge = HPCBridge.__new__(HPCBridge)
    bridge.lib = cast(Any, lib)
    bridge._destroy_symbol = None
    bridge._has_converged_api = False
    bridge._has_boundary_api = False
    return bridge


def test_is_relative_to_distinguishes_nested_and_unrelated_paths() -> None:
    assert hpc_mod._is_relative_to(Path("/alpha/beta"), Path("/alpha")) is True
    assert hpc_mod._is_relative_to(Path("/alpha/beta"), Path("/gamma")) is False


def test_native_build_environment_includes_system_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SystemRoot", "C:\\Windows")
    env = hpc_mod._native_build_environment(tmp_path)
    assert env["SystemRoot"] == "C:\\Windows"
    assert env["TMPDIR"] == str(tmp_path)


def test_native_build_compiler_rejects_relative_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "g++")
    assert hpc_mod._native_build_compiler() is None


def test_native_build_compiler_rejects_unresolvable_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/nonexistent/dir/g++")
    assert hpc_mod._native_build_compiler() is None


def test_native_build_compiler_rejects_non_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    directory = tmp_path / "compiler_dir"
    directory.mkdir()
    monkeypatch.setattr(shutil, "which", lambda name: str(directory))
    assert hpc_mod._native_build_compiler() is None


def test_prepare_native_output_path_removes_stale_temp(tmp_path: Path) -> None:
    out = tmp_path / "libscpn_solver.so"
    stale = tmp_path / f".{out.name}.build.{os.getpid()}"
    stale.write_text("stale", encoding="utf-8")
    result = hpc_mod._prepare_native_output_path(tmp_path, out)
    assert result == stale
    assert not stale.exists()


def test_prepare_native_output_path_rejects_unremovable_temp(tmp_path: Path) -> None:
    out = tmp_path / "libscpn_solver.so"
    stale = tmp_path / f".{out.name}.build.{os.getpid()}"
    stale.mkdir()
    (stale / "child").write_text("x", encoding="utf-8")
    assert hpc_mod._prepare_native_output_path(tmp_path, out) is None


def test_verify_solver_source_handles_unreadable_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    src = tmp_path / "solver.cpp"
    src.write_bytes(b"code")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"solver.cpp": {"sha256": "a" * 64}}), encoding="utf-8")

    def _raise_read_bytes(self: Path) -> bytes:
        del self
        raise OSError("unreadable")

    monkeypatch.setattr(Path, "read_bytes", _raise_read_bytes)
    assert hpc_mod._verify_solver_source(src, manifest) is False


def test_release_native_solver_tolerates_destroy_error() -> None:
    class _Lib:
        def destroy(self, ptr: object) -> None:
            del ptr
            raise OSError("native cleanup failed")

    state = {"solver_ptr": 99, "loaded": True, "lib": _Lib(), "destroy_symbol": "destroy"}
    hpc_mod._release_native_solver(state)
    assert state["solver_ptr"] is None


def test_setup_signatures_full_lib_enables_all_apis() -> None:
    bridge = _bridge_with_lib(
        _FakeNativeLib(("create_solver", "run_step", "run_step_converged", "set_boundary_dirichlet", "destroy_solver"))
    )
    bridge._setup_signatures()
    assert bridge._has_converged_api is True
    assert bridge._has_boundary_api is True
    assert bridge._destroy_symbol == "destroy_solver"


def test_setup_signatures_minimal_lib_disables_optional_apis() -> None:
    bridge = _bridge_with_lib(_FakeNativeLib(("create_solver", "run_step")))
    bridge._setup_signatures()
    assert bridge._has_converged_api is False
    assert bridge._has_boundary_api is False
    assert bridge._destroy_symbol is None


def test_setup_signatures_uses_delete_solver_alias() -> None:
    bridge = _bridge_with_lib(_FakeNativeLib(("create_solver", "run_step", "delete_solver")))
    bridge._setup_signatures()
    assert bridge._destroy_symbol == "delete_solver"


def test_initialize_noop_when_not_loaded() -> None:
    bridge = _make_bridge()
    bridge.loaded = False
    bridge.solver_ptr = None
    bridge.initialize(2, 3, (2.0, 10.0), (-5.0, 5.0))
    assert bridge.solver_ptr is None


def test_solve_into_returns_none_when_not_loaded() -> None:
    bridge = _make_bridge()
    bridge.loaded = False
    assert bridge.solve_into(np.zeros((3, 2)), np.zeros((3, 2))) is None


def test_solve_until_converged_returns_none_when_not_loaded() -> None:
    bridge = _make_bridge()
    bridge.loaded = False
    assert bridge.solve_until_converged(np.zeros((3, 2))) is None


def test_solve_until_converged_into_returns_none_when_not_loaded() -> None:
    bridge = _make_bridge()
    bridge.loaded = False
    assert bridge.solve_until_converged_into(np.zeros((3, 2)), np.zeros((3, 2))) is None


def test_sync_cleanup_state_without_armed_state() -> None:
    bridge = HPCBridge.__new__(HPCBridge)
    bridge._sync_cleanup_state()  # no _cleanup_state attribute → early return


def test_init_uses_package_local_candidate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SCPN_SOLVER_LIB", raising=False)
    module_file = tmp_path / "hpc_bridge.py"
    module_file.write_text("# test module path\n", encoding="utf-8")
    monkeypatch.setattr(hpc_mod, "__file__", str(module_file))
    lib_name = "scpn_solver.dll" if platform.system() == "Windows" else "libscpn_solver.so"
    bin_lib = tmp_path / "bin" / lib_name
    bin_lib.parent.mkdir()
    bin_lib.write_bytes(b"\x7fELF-not-a-real-library")

    def _raise_cdll(path: str) -> object:
        del path
        raise OSError("not a valid shared object")

    monkeypatch.setattr(ctypes, "CDLL", _raise_cdll)
    bridge = HPCBridge()
    assert bridge.is_available() is False
    assert bridge.lib_path == str(bin_lib)


def test_compile_cpp_returns_none_when_windows_temp_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    _admit_test_compiler(monkeypatch)
    _write_solver_source(tmp_path, monkeypatch)
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(hpc_mod, "_prepare_native_output_path", lambda out_dir, out: None)
    assert hpc_mod.compile_cpp() is None


def test_compile_cpp_tolerates_cleanup_error_on_failed_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    _admit_test_compiler(monkeypatch)
    _write_solver_source(tmp_path, monkeypatch)

    def _failed_run(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise subprocess.CalledProcessError(1, "g++")

    monkeypatch.setattr(subprocess, "run", _failed_run)

    def _raise_unlink(self: Path, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise OSError("temp locked")

    monkeypatch.setattr(Path, "unlink", _raise_unlink)
    assert hpc_mod.compile_cpp() is None


def test_compile_cpp_returns_none_when_publication_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCPN_ALLOW_NATIVE_BUILD", "1")
    _admit_test_compiler(monkeypatch)
    _write_solver_source(tmp_path, monkeypatch)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    temp_out = bin_dir / "temp_build.so"
    monkeypatch.setattr(hpc_mod, "_prepare_native_output_path", lambda out_dir, out: temp_out)

    def _ok_run(*args: object, **kwargs: object) -> None:
        del args, kwargs
        temp_out.write_bytes(b"compiled-library")
        return None

    monkeypatch.setattr(subprocess, "run", _ok_run)

    def _raise_replace(
        src: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        dst: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        del src, dst
        raise OSError("cannot publish")

    monkeypatch.setattr(os, "replace", _raise_replace)
    assert hpc_mod.compile_cpp() is None
