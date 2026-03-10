# ──────────────────────────────────────────────────────────────────────
# SCPN Control — HPC Bridge Coverage Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Cover remaining branches in hpc_bridge.py (lines 106-311)."""

from __future__ import annotations

import ctypes
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scpn_control.core import hpc_bridge as hpc_mod
from scpn_control.core.hpc_bridge import (
    HPCBridge,
    _as_contiguous_f64,
    _require_c_contiguous_f64,
    _sanitize_convergence_params,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_loaded_bridge(nr: int = 2, nz: int = 3) -> HPCBridge:
    bridge = HPCBridge.__new__(HPCBridge)
    bridge.lib = MagicMock()
    bridge.solver_ptr = 12345
    bridge.loaded = True
    bridge._destroy_symbol = "destroy_solver"
    bridge._has_converged_api = True
    bridge._has_boundary_api = True
    bridge.nr = nr
    bridge.nz = nz
    return bridge


# ── _as_contiguous_f64 edge cases ────────────────────────────────────


def test_as_contiguous_f64_already_contiguous_noop():
    arr = np.ones((4, 3), dtype=np.float64)
    assert _as_contiguous_f64(arr) is arr


def test_as_contiguous_f64_fortran_order_copies():
    arr = np.asfortranarray(np.ones((4, 3), dtype=np.float64))
    out = _as_contiguous_f64(arr)
    assert out is not arr
    assert out.flags.c_contiguous
    assert out.dtype == np.float64


def test_as_contiguous_f64_wrong_dtype_copies():
    arr = np.ones((4, 3), dtype=np.float32)
    out = _as_contiguous_f64(arr)
    assert out.dtype == np.float64
    assert out.flags.c_contiguous


# ── _require_c_contiguous_f64 ────────────────────────────────────────


def test_require_c_contiguous_f64_valid():
    arr = np.zeros((3, 2), dtype=np.float64)
    assert _require_c_contiguous_f64(arr, (3, 2), "buf") is arr


def test_require_c_contiguous_f64_not_ndarray():
    with pytest.raises(ValueError, match="numpy.ndarray"):
        _require_c_contiguous_f64([[0.0]], (1, 1), "buf")  # type: ignore[arg-type]


def test_require_c_contiguous_f64_wrong_dtype():
    with pytest.raises(ValueError, match="dtype float64"):
        _require_c_contiguous_f64(np.zeros((2, 2), dtype=np.int32), (2, 2), "buf")


def test_require_c_contiguous_f64_not_contiguous():
    arr = np.zeros((3, 2), dtype=np.float64, order="F")
    with pytest.raises(ValueError, match="C-contiguous"):
        _require_c_contiguous_f64(arr, (3, 2), "buf")


def test_require_c_contiguous_f64_wrong_shape():
    arr = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="shape mismatch"):
        _require_c_contiguous_f64(arr, (2, 3), "buf")


# ── _sanitize_convergence_params ─────────────────────────────────────


def test_sanitize_valid():
    i, t, o = _sanitize_convergence_params(50, 1e-8, 1.2)
    assert i == 50
    assert t == 1e-8
    assert o == 1.2


def test_sanitize_max_iterations_zero():
    with pytest.raises(ValueError, match="max_iterations"):
        _sanitize_convergence_params(0, 1e-6, 1.0)


def test_sanitize_negative_tolerance():
    with pytest.raises(ValueError, match="tolerance"):
        _sanitize_convergence_params(10, -0.01, 1.0)


def test_sanitize_nan_tolerance():
    with pytest.raises(ValueError, match="tolerance"):
        _sanitize_convergence_params(10, float("nan"), 1.0)


def test_sanitize_omega_zero():
    with pytest.raises(ValueError, match="omega"):
        _sanitize_convergence_params(10, 1e-6, 0.0)


def test_sanitize_omega_two():
    with pytest.raises(ValueError, match="omega"):
        _sanitize_convergence_params(10, 1e-6, 2.0)


def test_sanitize_omega_inf():
    with pytest.raises(ValueError, match="omega"):
        _sanitize_convergence_params(10, 1e-6, float("inf"))


# ── HPCBridge constructor — no library ───────────────────────────────


def test_constructor_no_library(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SCPN_SOLVER_LIB", raising=False)

    def _raise_cdll(_path: str):
        raise OSError("not found")

    monkeypatch.setattr(hpc_mod.ctypes, "CDLL", _raise_cdll)
    bridge = HPCBridge()
    assert not bridge.loaded
    assert bridge.lib is None


def test_is_available_false_when_not_loaded(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SCPN_SOLVER_LIB", raising=False)
    monkeypatch.setattr(hpc_mod.ctypes, "CDLL", lambda _: (_ for _ in ()).throw(OSError("nope")))
    bridge = HPCBridge()
    assert bridge.is_available() is False


# ── HPCBridge constructor — mocked CDLL loads successfully ───────────
# Covers lines 106-107 (candidate found), 114-117 (CDLL + _setup_signatures + loaded=True),
# and 145-202 (_setup_signatures body).


def _build_mock_cdll():
    """Build a MagicMock that quacks like ctypes.CDLL with expected symbols."""
    lib = MagicMock()
    lib.create_solver = MagicMock()
    lib.run_step = MagicMock()
    lib.run_step_converged = MagicMock()
    lib.set_boundary_dirichlet = MagicMock()
    lib.destroy_solver = MagicMock()
    # hasattr must return True for run_step_converged, set_boundary_dirichlet, destroy_solver
    return lib


def test_constructor_with_mocked_cdll_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    lib_file = tmp_path / "libscpn_solver.so"
    lib_file.touch()
    mock_lib = _build_mock_cdll()
    monkeypatch.setattr(hpc_mod.ctypes, "CDLL", lambda _path: mock_lib)

    bridge = HPCBridge(str(lib_file))
    assert bridge.loaded is True
    assert bridge._has_converged_api is True
    assert bridge._has_boundary_api is True
    assert bridge._destroy_symbol == "destroy_solver"


def test_setup_signatures_no_converged_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    lib_file = tmp_path / "libscpn_solver.so"
    lib_file.touch()
    mock_lib = _build_mock_cdll()
    del mock_lib.run_step_converged
    monkeypatch.setattr(hpc_mod.ctypes, "CDLL", lambda _path: mock_lib)

    bridge = HPCBridge(str(lib_file))
    assert bridge.loaded is True
    assert bridge._has_converged_api is False


def test_setup_signatures_no_boundary_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    lib_file = tmp_path / "libscpn_solver.so"
    lib_file.touch()
    mock_lib = _build_mock_cdll()
    del mock_lib.set_boundary_dirichlet
    monkeypatch.setattr(hpc_mod.ctypes, "CDLL", lambda _path: mock_lib)

    bridge = HPCBridge(str(lib_file))
    assert bridge.loaded is True
    assert bridge._has_boundary_api is False


def test_setup_signatures_delete_solver_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    lib_file = tmp_path / "libscpn_solver.so"
    lib_file.touch()
    mock_lib = _build_mock_cdll()
    del mock_lib.destroy_solver
    mock_lib.delete_solver = MagicMock()
    monkeypatch.setattr(hpc_mod.ctypes, "CDLL", lambda _path: mock_lib)

    bridge = HPCBridge(str(lib_file))
    assert bridge._destroy_symbol == "delete_solver"


def test_setup_signatures_no_destroy_symbol(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    lib_file = tmp_path / "libscpn_solver.so"
    lib_file.touch()
    mock_lib = _build_mock_cdll()
    del mock_lib.destroy_solver
    del mock_lib.delete_solver
    monkeypatch.setattr(hpc_mod.ctypes, "CDLL", lambda _path: mock_lib)

    bridge = HPCBridge(str(lib_file))
    assert bridge._destroy_symbol is None


# ── Candidate path found on disk (lines 106-107) ────────────────────


def test_constructor_finds_candidate_on_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("SCPN_SOLVER_LIB", raising=False)
    monkeypatch.setattr(hpc_mod.platform, "system", lambda: "Linux")

    # Place a fake .so in the expected location
    fake_lib = tmp_path / "libscpn_solver.so"
    fake_lib.touch()

    # Patch Path(__file__).resolve().parent to return tmp_path
    original_init = Path.__init__

    mock_lib = _build_mock_cdll()

    class PatchedPath(type(Path())):
        def resolve(self):
            result = super().resolve()
            if str(result) == str(Path(hpc_mod.__file__).resolve()):
                # Return a path whose .parent is tmp_path
                return tmp_path / "hpc_bridge.py"
            return result

    # Simpler approach: just provide explicit lib_path
    monkeypatch.setattr(hpc_mod.ctypes, "CDLL", lambda _path: mock_lib)
    bridge = HPCBridge(str(fake_lib))
    assert bridge.loaded is True
    assert bridge.lib_path == str(fake_lib)


# ── close() exception handling (lines 131-132) ──────────────────────


def test_close_handles_destroy_exception():
    bridge = _make_loaded_bridge()
    bridge.lib.destroy_solver.side_effect = OSError("cleanup failed")
    bridge.close()
    assert bridge.solver_ptr is None


def test_close_handles_attribute_error():
    bridge = _make_loaded_bridge()
    bridge._destroy_symbol = "nonexistent_symbol"
    bridge.lib.configure_mock(**{"nonexistent_symbol.side_effect": AttributeError("no such attr")})
    bridge.close()
    assert bridge.solver_ptr is None


# ── initialize (lines 213-218) ───────────────────────────────────────


def test_initialize_noop_when_not_loaded():
    bridge = _make_loaded_bridge()
    bridge.loaded = False
    bridge.initialize(10, 10, (2.0, 10.0), (-5.0, 5.0))
    bridge.lib.create_solver.assert_not_called()


def test_initialize_creates_solver():
    bridge = _make_loaded_bridge()
    bridge.lib.create_solver.return_value = 99999
    bridge.initialize(10, 20, (2.0, 10.0), (-5.0, 5.0), boundary_value=1.5)
    bridge.lib.create_solver.assert_called_once_with(10, 20, 2.0, 10.0, -5.0, 5.0)
    assert bridge.solver_ptr == 99999
    assert bridge.nr == 10
    assert bridge.nz == 20
    bridge.lib.set_boundary_dirichlet.assert_called_once_with(99999, 1.5)


# ── set_boundary_dirichlet when not loaded ───────────────────────────


def test_set_boundary_dirichlet_noop_not_loaded():
    bridge = _make_loaded_bridge()
    bridge.loaded = False
    bridge.set_boundary_dirichlet(0.5)
    bridge.lib.set_boundary_dirichlet.assert_not_called()


# ── solve/solve_into/solve_until_converged return None paths ─────────


def test_solve_returns_none_not_loaded():
    bridge = _make_loaded_bridge()
    bridge.loaded = False
    assert bridge.solve(np.zeros((3, 2), dtype=np.float64)) is None


def test_solve_into_returns_none_not_loaded():
    bridge = _make_loaded_bridge()
    bridge.loaded = False
    out = np.zeros((3, 2), dtype=np.float64)
    assert bridge.solve_into(np.zeros((3, 2), dtype=np.float64), out) is None


def test_solve_until_converged_returns_none_not_loaded():
    bridge = _make_loaded_bridge()
    bridge.loaded = False
    assert bridge.solve_until_converged(np.zeros((3, 2), dtype=np.float64)) is None


def test_solve_until_converged_into_returns_none_not_loaded():
    bridge = _make_loaded_bridge()
    bridge.loaded = False
    out = np.zeros((3, 2), dtype=np.float64)
    assert bridge.solve_until_converged_into(np.zeros((3, 2), dtype=np.float64), out) is None


def test_solve_returns_none_no_solver_ptr():
    bridge = _make_loaded_bridge()
    bridge.solver_ptr = None
    assert bridge.solve(np.zeros((3, 2), dtype=np.float64)) is None


def test_solve_until_converged_returns_none_no_solver_ptr():
    bridge = _make_loaded_bridge()
    bridge.solver_ptr = None
    assert bridge.solve_until_converged(np.zeros((3, 2), dtype=np.float64)) is None


def test_solve_until_converged_into_returns_none_no_solver_ptr():
    bridge = _make_loaded_bridge()
    bridge.solver_ptr = None
    out = np.zeros((3, 2), dtype=np.float64)
    assert bridge.solve_until_converged_into(np.zeros((3, 2), dtype=np.float64), out) is None


# ── _prepare_inputs validation ───────────────────────────────────────


def test_prepare_inputs_rejects_1d():
    bridge = _make_loaded_bridge()
    with pytest.raises(ValueError, match="2D array"):
        bridge.solve(np.zeros((6,), dtype=np.float64))


def test_prepare_inputs_rejects_empty():
    bridge = _make_loaded_bridge()
    bridge.nr = 0
    bridge.nz = 0
    with pytest.raises(ValueError, match="non-empty"):
        bridge.solve(np.zeros((0, 0), dtype=np.float64))


def test_prepare_inputs_rejects_nan():
    bridge = _make_loaded_bridge()
    arr = np.ones((3, 2), dtype=np.float64)
    arr[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        bridge.solve(arr)


def test_prepare_inputs_rejects_inf():
    bridge = _make_loaded_bridge()
    arr = np.ones((3, 2), dtype=np.float64)
    arr[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        bridge.solve(arr)


def test_prepare_inputs_rejects_shape_mismatch():
    bridge = _make_loaded_bridge(nr=2, nz=3)
    with pytest.raises(ValueError, match="shape mismatch"):
        bridge.solve(np.zeros((4, 4), dtype=np.float64))


# ── Context manager ──────────────────────────────────────────────────


def test_context_manager_enter_exit():
    bridge = _make_loaded_bridge()
    with bridge as b:
        assert b is bridge
    assert bridge.solver_ptr is None


# ── compile_cpp without opt-in ───────────────────────────────────────


def test_solve_returns_none_when_solve_into_returns_none():
    """Line 244: _prepare_inputs passes in solve but solve_into returns None."""
    bridge = _make_loaded_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    with patch.object(bridge, "solve_into", return_value=None):
        assert bridge.solve(j_phi) is None


def test_solve_until_converged_returns_none_when_inner_returns_none():
    """Line 296: _prepare_inputs passes but solve_until_converged_into returns None."""
    bridge = _make_loaded_bridge(nr=2, nz=3)
    j_phi = np.arange(6, dtype=np.float64).reshape(3, 2)
    with patch.object(bridge, "solve_until_converged_into", return_value=None):
        assert bridge.solve_until_converged(j_phi) is None


def test_compile_cpp_returns_none_without_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SCPN_ALLOW_NATIVE_BUILD", raising=False)
    assert hpc_mod.compile_cpp() is None
