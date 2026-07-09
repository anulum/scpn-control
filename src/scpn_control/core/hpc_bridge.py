# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native C++ solver bridge and build admission.
"""HPC job bridge for submitting and tracking external validation workloads."""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import weakref
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_SOLVER_LIBRARY_SUFFIXES = {".so", ".dylib", ".dll"}
_ALLOW_EXTERNAL_SOLVER_LIB = "SCPN_ALLOW_EXTERNAL_SOLVER_LIB"
_SOLVER_SOURCE = "solver.cpp"
_SOLVER_MANIFEST = "solver_manifest.json"
_NATIVE_BUILD_TIMEOUT_S = 120
_NATIVE_BUILD_COMPILER = "g++"


def _as_contiguous_f64(array: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return ``array`` as C-contiguous ``float64`` with minimal copying."""
    if isinstance(array, np.ndarray) and array.dtype == np.float64 and array.flags.c_contiguous:
        return cast("NDArray[np.float64]", array)
    return cast("NDArray[np.float64]", np.ascontiguousarray(array, dtype=np.float64))


def _require_c_contiguous_f64(
    array: NDArray[np.floating],
    expected_shape: tuple[int, int],
    name: str,
) -> NDArray[np.float64]:
    """Validate that an output buffer can be written into without copying."""
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be a numpy.ndarray")
    if array.dtype != np.float64:
        raise ValueError(f"{name} must have dtype float64")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if tuple(array.shape) != tuple(expected_shape):
        raise ValueError(f"{name} shape mismatch: expected {expected_shape}, received {tuple(array.shape)}")
    return cast("NDArray[np.float64]", array)


def _sanitize_convergence_params(
    max_iterations: int,
    tolerance: float,
    omega: float,
) -> tuple[int, float, float]:
    """Validate convergence parameters for native calls."""
    max_iters = int(max_iterations)
    if max_iters < 1:
        raise ValueError("max_iterations must be >= 1.")

    tol_safe = float(tolerance)
    if not math.isfinite(tol_safe) or tol_safe < 0.0:
        raise ValueError("tolerance must be finite and >= 0.")

    omega_safe = float(omega)
    if not math.isfinite(omega_safe) or omega_safe <= 0.0 or omega_safe >= 2.0:
        raise ValueError("omega must be finite and in (0, 2).")

    return max_iters, tol_safe, omega_safe


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _trusted_solver_roots() -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    return here, here / "bin"


def _validate_solver_library_path(raw_path: str, *, source: str) -> str:
    """Return a canonical dynamic-library path after applying loader policy."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{source} solver library path must be absolute")

    resolved = path.resolve(strict=False)
    if resolved.suffix.lower() not in _SOLVER_LIBRARY_SUFFIXES:
        raise ValueError(f"{source} solver library path must end with one of {sorted(_SOLVER_LIBRARY_SUFFIXES)}")
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"{source} solver library path must reference a regular file")

    roots = tuple(root.resolve(strict=False) for root in _trusted_solver_roots())
    if source == "SCPN_SOLVER_LIB" and not any(_is_relative_to(resolved, root) for root in roots):
        if os.environ.get(_ALLOW_EXTERNAL_SOLVER_LIB) != "1":
            raise ValueError(
                "SCPN_SOLVER_LIB must point to a trusted package-local solver library "
                f"or set {_ALLOW_EXTERNAL_SOLVER_LIB}=1 for a vetted external path"
            )

    return str(resolved)


def _verify_solver_source(src: Path, manifest_path: Path) -> bool:
    """Return ``True`` only when ``solver.cpp`` matches its SHA-256 manifest."""
    if src.is_symlink() or not src.is_file():
        logger.error("Native solver source missing: %s", src)
        return False
    if manifest_path.is_symlink() or not manifest_path.is_file():
        logger.error("Native solver checksum manifest missing: %s", manifest_path)
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Native solver checksum manifest is unreadable: %s", exc)
        return False

    entry = manifest.get(_SOLVER_SOURCE)
    expected = entry.get("sha256") if isinstance(entry, dict) else entry
    if not isinstance(expected, str) or len(expected) != 64:
        logger.error("Native solver checksum manifest lacks a valid SHA-256 for %s", _SOLVER_SOURCE)
        return False

    try:
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
    except OSError as exc:
        logger.error("Native solver source could not be hashed: %s", exc)
        return False

    if not hmac.compare_digest(digest, expected.lower()):
        logger.error("Native solver source checksum mismatch: %s", src)
        return False
    return True


def _native_build_environment(out_dir: Path) -> dict[str, str]:
    """Return a minimal environment for the opt-in native build process."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "TMPDIR": str(out_dir),
        "LC_ALL": "C",
        "LANG": "C",
    }
    for key in ("SystemRoot", "WINDIR"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _native_build_compiler() -> str | None:
    """Return the absolute C++ compiler path admitted for native builds."""
    compiler = shutil.which(_NATIVE_BUILD_COMPILER)
    if compiler is None:
        logger.error("Native build compiler not found: %s", _NATIVE_BUILD_COMPILER)
        return None
    compiler_path = Path(compiler)
    if not compiler_path.is_absolute():
        logger.error("Native build compiler path is not admissible: %s", compiler)
        return None
    try:
        resolved_compiler = compiler_path.resolve(strict=True)
    except OSError as exc:
        logger.error("Native build compiler path cannot be resolved: %s", exc)
        return None
    if not resolved_compiler.is_file():
        logger.error("Native build compiler is not a regular file: %s", compiler)
        return None
    return str(resolved_compiler)


def _prepare_native_output_path(out_dir: Path, out: Path) -> Path | None:
    """Return a temporary output path after rejecting symlink build targets."""
    if out_dir.is_symlink():
        logger.error("Native build output directory must not be a symlink: %s", out_dir)
        return None
    if out.exists() and (out.is_symlink() or not out.is_file()):
        logger.error("Native build output path is not a regular file: %s", out)
        return None

    temp_out = out_dir / f".{out.name}.build.{os.getpid()}"
    if temp_out.exists() or temp_out.is_symlink():
        try:
            temp_out.unlink()
        except OSError as exc:
            logger.error("Native build temporary output path is not removable: %s", exc)
            return None
    return temp_out


def _release_native_solver(state: dict[str, Any]) -> None:
    """Release the native solver referenced by ``state`` exactly once."""
    solver_ptr = state.get("solver_ptr")
    if solver_ptr is None or not state.get("loaded"):
        state["solver_ptr"] = None
        return

    try:
        lib = state.get("lib")
        destroy_symbol = state.get("destroy_symbol")
        if lib is not None and destroy_symbol is not None:
            getattr(lib, str(destroy_symbol))(solver_ptr)
    except (OSError, AttributeError) as exc:
        logger.debug("C++ solver cleanup failed: %s", exc)
    state["solver_ptr"] = None


class HPCBridge:
    """Interface between Python and the compiled C++ Grad-Shafranov solver.

    Loads the shared library (``libscpn_solver.so`` / ``scpn_solver.dll``)
    at construction time.  If the library is not found the bridge
    gracefully degrades — :meth:`is_available` returns ``False`` and the
    caller falls back to Python.

    Parameters
    ----------
    lib_path : str, optional
        Explicit path to the shared library.  When *None* (default) the
        bridge searches trusted package-local locations only. ``SCPN_SOLVER_LIB``
        must be an absolute dynamic-library path and is accepted only when it
        resolves under the package-local solver directories, unless the operator
        also sets ``SCPN_ALLOW_EXTERNAL_SOLVER_LIB=1`` for a vetted external
        library.
    """

    def __init__(self, lib_path: str | None = None) -> None:
        self.lib: ctypes.CDLL | None = None
        self.solver_ptr = None
        self.loaded: bool = False
        self._destroy_symbol: str | None = None
        self._has_converged_api: bool = False
        self._has_boundary_api: bool = False

        lib_name = "scpn_solver.dll" if platform.system() == "Windows" else "libscpn_solver.so"
        env_path = os.environ.get("SCPN_SOLVER_LIB")
        lib_source = "explicit lib_path"
        if lib_path is None and env_path:
            lib_path = env_path
            lib_source = "SCPN_SOLVER_LIB"

        if lib_path is None:
            here = Path(__file__).resolve().parent
            candidates = [
                here / lib_name,
                here / "bin" / lib_name,
            ]
            for c in candidates:
                if c.exists():
                    lib_path = str(c)
                    break
            if lib_path is None:
                lib_path = str(here / lib_name)
            lib_source = "package-local default"

        self.lib_path = _validate_solver_library_path(str(lib_path), source=lib_source)

        try:
            self.lib = ctypes.CDLL(self.lib_path)
            self._setup_signatures()
            logger.info("Loaded C++ accelerator: %s", self.lib_path)
            self.loaded = True
            self._arm_cleanup_finalizer()
        except OSError as exc:
            logger.debug("C++ accelerator unavailable: %s", exc)

    def _arm_cleanup_finalizer(self) -> None:
        self._cleanup_state = {
            "lib": self.lib,
            "solver_ptr": self.solver_ptr,
            "loaded": self.loaded,
            "destroy_symbol": self._destroy_symbol,
        }
        self._finalizer = weakref.finalize(self, _release_native_solver, self._cleanup_state)

    def _sync_cleanup_state(self) -> None:
        if not hasattr(self, "_cleanup_state"):
            return
        self._cleanup_state.update(
            {
                "lib": self.lib,
                "solver_ptr": self.solver_ptr,
                "loaded": self.loaded,
                "destroy_symbol": self._destroy_symbol,
            }
        )

    def is_available(self) -> bool:
        """Return *True* if the compiled solver library was loaded."""
        return self.loaded

    def close(self) -> None:
        """Release the C++ solver instance, if one was created."""
        if hasattr(self, "_cleanup_state"):
            self._sync_cleanup_state()
            _release_native_solver(self._cleanup_state)
            self.solver_ptr = None
            if getattr(self, "_finalizer", None) is not None and self._finalizer.alive:
                self._finalizer.detach()
            return

        if self.solver_ptr is not None and self.loaded:
            try:
                if self.lib is not None and self._destroy_symbol is not None:
                    getattr(self.lib, self._destroy_symbol)(self.solver_ptr)
            except (OSError, AttributeError) as exc:
                logger.debug("C++ solver cleanup failed: %s", exc)
            self.solver_ptr = None

    def __enter__(self) -> "HPCBridge":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _setup_signatures(self) -> None:
        assert self.lib is not None
        self.lib.create_solver.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        self.lib.create_solver.restype = ctypes.c_void_p

        # void run_step(void* solver, double* j, double* psi, int size, int iter)
        self.lib.run_step.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
        ]

        # int run_step_converged(void* solver, const double* j, double* psi,
        #                        int size, int max_iter, double omega,
        #                        double tol, double* final_delta)
        if hasattr(self.lib, "run_step_converged"):
            self.lib.run_step_converged.argtypes = [
                ctypes.c_void_p,
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_double),
            ]
            self.lib.run_step_converged.restype = ctypes.c_int
            self._has_converged_api = True
        else:
            self._has_converged_api = False

        # void set_boundary_dirichlet(void* solver, double boundary_value)
        if hasattr(self.lib, "set_boundary_dirichlet"):
            self.lib.set_boundary_dirichlet.argtypes = [ctypes.c_void_p, ctypes.c_double]
            self.lib.set_boundary_dirichlet.restype = None
            self._has_boundary_api = True
        else:
            self._has_boundary_api = False

        # void destroy_solver(void* solver) or void delete_solver(void* solver)
        if hasattr(self.lib, "destroy_solver"):
            self.lib.destroy_solver.argtypes = [ctypes.c_void_p]
            self.lib.destroy_solver.restype = None
            self._destroy_symbol = "destroy_solver"
        elif hasattr(self.lib, "delete_solver"):
            self.lib.delete_solver.argtypes = [ctypes.c_void_p]
            self.lib.delete_solver.restype = None
            self._destroy_symbol = "delete_solver"
        else:
            self._destroy_symbol = None

    def initialize(
        self,
        nr: int,
        nz: int,
        r_range: tuple[float, float],
        z_range: tuple[float, float],
        boundary_value: float = 0.0,
    ) -> None:
        """Create the C++ solver instance for the given grid dimensions."""
        if not self.loaded or self.lib is None:
            return
        self.nr = nr
        self.nz = nz
        self.solver_ptr = self.lib.create_solver(nr, nz, r_range[0], r_range[1], z_range[0], z_range[1])
        self._sync_cleanup_state()
        self.set_boundary_dirichlet(boundary_value)

    def set_boundary_dirichlet(self, boundary_value: float = 0.0) -> None:
        """Set a fixed Dirichlet boundary value for psi edges, if supported."""
        if not self.loaded or self.solver_ptr is None or self.lib is None or not self._has_boundary_api:
            return
        self.lib.set_boundary_dirichlet(self.solver_ptr, float(boundary_value))

    def solve(
        self,
        j_phi: NDArray[np.float64],
        iterations: int = 100,
    ) -> NDArray[np.float64] | None:
        """Run the C++ solver for *iterations* sweeps.

        Returns *None* if the library is not loaded (caller should
        fall back to a Python solver).
        """
        prepared = self._prepare_inputs(j_phi)
        if prepared is None:
            return None
        _, expected_shape = prepared

        psi_out = np.zeros(expected_shape, dtype=np.float64)
        solved = self.solve_into(j_phi, psi_out, iterations=iterations)
        if solved is None:
            return None
        return solved

    def solve_into(
        self,
        j_phi: NDArray[np.float64],
        psi_out: NDArray[np.float64],
        iterations: int = 100,
    ) -> NDArray[np.float64] | None:
        """Run the C++ solver and write results into ``psi_out`` in-place."""
        prepared = self._prepare_inputs(j_phi)
        if prepared is None:
            return None
        j_input, expected_shape = prepared
        psi_target = _require_c_contiguous_f64(psi_out, expected_shape, "psi_out")
        assert self.lib is not None

        self.lib.run_step(
            self.solver_ptr,
            j_input,
            psi_target,
            int(j_input.size),
            int(iterations),
        )
        return psi_target

    def solve_until_converged(
        self,
        j_phi: NDArray[np.float64],
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        omega: float = 1.8,
    ) -> tuple[NDArray[np.float64], int, float] | None:
        """Run solver until convergence, if native API is available.

        Returns ``(psi, iterations_used, final_delta)``. If the library is
        unavailable or uninitialized, returns ``None``.
        """
        prepared = self._prepare_inputs(j_phi)
        if prepared is None:
            return None
        _, expected_shape = prepared

        psi_out = np.zeros(expected_shape, dtype=np.float64)
        converged = self.solve_until_converged_into(
            j_phi,
            psi_out,
            max_iterations=max_iterations,
            tolerance=tolerance,
            omega=omega,
        )
        if converged is None:
            return None
        iterations_used, final_delta = converged
        return psi_out, iterations_used, final_delta

    def solve_until_converged_into(
        self,
        j_phi: NDArray[np.float64],
        psi_out: NDArray[np.float64],
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        omega: float = 1.8,
    ) -> tuple[int, float] | None:
        """Run convergence API and write results into ``psi_out`` in-place."""
        prepared = self._prepare_inputs(j_phi)
        if prepared is None:
            return None
        j_input, expected_shape = prepared
        psi_target = _require_c_contiguous_f64(psi_out, expected_shape, "psi_out")
        max_iters, tol_safe, omega_safe = _sanitize_convergence_params(max_iterations, tolerance, omega)

        assert self.lib is not None
        if not self._has_converged_api:
            self.lib.run_step(
                self.solver_ptr,
                j_input,
                psi_target,
                int(j_input.size),
                int(max_iters),
            )
            return int(max_iters), float("nan")

        final_delta = ctypes.c_double(0.0)
        iterations_used = int(
            self.lib.run_step_converged(
                self.solver_ptr,
                j_input,
                psi_target,
                int(j_input.size),
                int(max_iters),
                float(omega_safe),
                float(tol_safe),
                ctypes.byref(final_delta),
            )
        )
        return iterations_used, float(final_delta.value)

    def _prepare_inputs(self, j_phi: NDArray[np.float64]) -> tuple[NDArray[np.float64], tuple[int, int]] | None:
        if not self.loaded or self.solver_ptr is None:
            return None

        j_input = _as_contiguous_f64(j_phi)
        if j_input.ndim != 2:
            raise ValueError(f"j_phi must be a 2D array, received ndim={j_input.ndim}")
        if j_input.size == 0:
            raise ValueError("j_phi must be non-empty")
        if not np.all(np.isfinite(j_input)):
            raise ValueError("j_phi must contain only finite values")
        expected_shape = (
            getattr(self, "nz", j_input.shape[0]),
            getattr(self, "nr", j_input.shape[-1]),
        )
        if tuple(j_input.shape) != tuple(expected_shape):
            raise ValueError(f"j_phi shape mismatch: expected {expected_shape}, received {tuple(j_input.shape)}")
        return j_input, expected_shape


def compile_cpp() -> str | None:
    """Compile the C++ solver from source.

    Looks for ``solver.cpp`` in the same directory as this module and
    invokes ``g++`` to produce a shared library.

    Returns
    -------
    str or None
        Path to the compiled library, or *None* on failure.
    """
    if os.environ.get("SCPN_ALLOW_NATIVE_BUILD") != "1":
        logger.warning("Native build disabled. Set SCPN_ALLOW_NATIVE_BUILD=1 to enable.")
        return None

    compiler = _native_build_compiler()
    if compiler is None:
        return None

    logger.info("Compiling C++ solver kernel…")
    script_dir = Path(__file__).resolve().parent
    src = script_dir / _SOLVER_SOURCE
    manifest = script_dir / _SOLVER_MANIFEST
    if not _verify_solver_source(src, manifest):
        return None

    out_dir = script_dir / "bin"
    out_dir.mkdir(exist_ok=True)

    if platform.system() == "Windows":
        out = out_dir / "scpn_solver.dll"
        temp_out = _prepare_native_output_path(out_dir, out)
        if temp_out is None:
            return None
        cmd = [compiler, "-shared", "-o", str(temp_out), str(src), "-O3", "-fstack-protector-strong"]
    else:
        out = out_dir / "libscpn_solver.so"
        temp_out = _prepare_native_output_path(out_dir, out)
        if temp_out is None:
            return None
        cmd = [
            compiler,
            "-shared",
            "-fPIC",
            "-o",
            str(temp_out),
            str(src),
            "-O3",
            "-fstack-protector-strong",
            "-mtune=generic",
        ]
        if platform.system() == "Linux":
            cmd.extend(["-Wl,-z,relro", "-Wl,-z,now"])

    logger.info("Executing: %s", " ".join(cmd))
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=str(script_dir),
            env=_native_build_environment(out_dir),
            timeout=_NATIVE_BUILD_TIMEOUT_S,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.error("Compilation failed: %s", exc)
        try:
            temp_out.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_exc:
            logger.debug("Native build temporary cleanup failed: %s", cleanup_exc)
        return None

    if temp_out.is_symlink() or not temp_out.is_file():
        logger.error("Native build did not produce a regular shared library: %s", temp_out)
        return None
    try:
        os.replace(temp_out, out)
    except OSError as exc:
        logger.error("Native build output publication failed: %s", exc)
        return None

    logger.info("Compilation succeeded: %s", out)
    return str(out)


if __name__ == "__main__":
    # Test sequence
    lib_file = compile_cpp()

    if lib_file:
        bridge = HPCBridge(lib_file)

        # Test Grid
        N = 100
        bridge.initialize(N, N, (2.0, 10.0), (-5.0, 5.0))

        # Test current profile
        J = np.random.rand(N, N)

        # Run
        Psi = bridge.solve(J, iterations=500)
        if Psi is not None:
            print(f"Max Flux: {np.max(Psi)}")
