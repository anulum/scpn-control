# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FUSION-owned IDA equilibrium adapter
"""Expose the FUSION free-boundary equilibrium contract at the CONTROL boundary.

The Grad-Shafranov mathematics remains owned by :mod:`scpn_fusion`.  This module
validates a bounded SI request, evaluates compact profile coefficients with the
upstream B-spline basis, calls the upstream implicit-differentiation solver, and
binds the result to the exact upstream source bytes used at runtime.  It does not
contain a second free-boundary solver or grant facility, PCS, or safety admission.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

IDA_ADAPTER_SCHEMA = "scpn-control.ida-equilibrium-adapter.v1"
IDA_ADAPTER_STATUS = "research_only_blocked"
MIN_GRID_POINTS = 49
MAX_GRID_POINTS = 257
MAX_COILS = 64
MAX_PROFILE_COEFFICIENTS = 64
_SOLVER_NAME = "scpn_fusion.core.jax_free_boundary_gs_implicit.solve_free_boundary_gs_implicit"
_CLAIM_FIELDS = (
    "control_admission",
    "facility_validation",
    "pcs_deployment",
    "scientific_validation",
    "safety_admission",
)
_BLOCKERS = (
    "control_admission_evidence_not_bound",
    "external_same_case_validation_not_bound",
    "facility_validation_not_bound",
    "pcs_and_safety_programmes_not_bound",
    "upstream_source_commit_not_runtime_attested",
)


class IDAEquilibriumAdapterError(RuntimeError):
    """Raised when the bounded FUSION equilibrium facade cannot fail closed."""


FloatArray = NDArray[np.float64]


class _Backend(Protocol):
    version: str
    jax_version: str
    jaxlib_version: str
    solver_file_sha256: str
    profile_basis_file_sha256: str
    x64_enabled: bool
    asarray: Callable[[FloatArray], object]
    design_matrix: Callable[[FloatArray, int, int], object]
    evaluate_profile: Callable[[object, object], object]
    solve: Callable[[object, object, object, object, object, object, object, object, int, float], object]
    vjp: Callable[
        [Callable[[object, object, object], object], object, object, object],
        tuple[object, Callable[[object], tuple[object, object, object]]],
    ]


def _normalise_vector(value: object, *, field: str) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be coercible to a finite float64 vector") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{field} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain only finite values")
    frozen = np.array(array, dtype=np.float64, order="C", copy=True)
    frozen.flags.writeable = False
    return frozen


def _require_strictly_increasing(value: FloatArray, *, field: str) -> None:
    if np.any(np.diff(value) <= 0.0):
        raise ValueError(f"{field} must be strictly increasing")


def _require_uniform(value: FloatArray, *, field: str) -> None:
    spacing = np.diff(value)
    if not np.allclose(spacing, spacing[0], rtol=1.0e-12, atol=1.0e-15):
        raise ValueError(f"{field} must be uniformly spaced")


@dataclass(frozen=True, slots=True)
class IDAEquilibriumRequest:
    """Bounded SI input contract for a FUSION-owned free-boundary solve.

    Parameters
    ----------
    r_grid_m, z_grid_m
        Uniform one-dimensional cylindrical grid coordinates in metres.
    coil_r_m, coil_z_m, coil_current_a
        PF-coil positions in metres and currents in amperes.
    psin_knots
        Strictly increasing normalised-flux samples spanning exactly ``[0, 1]``.
    pprime_coefficients_pa_per_wb
        Compact B-spline coefficients for ``dp/dpsi`` in pascals per weber.
    ffprime_coefficients_t2_m2_per_wb
        Compact B-spline coefficients for ``F dF/dpsi`` in tesla squared metre
        squared per weber.
    profile_degree
        Fixed B-spline degree used by the upstream profile basis.
    n_picard, picard_omega
        Bounded upstream implicit-solver iteration count and relaxation factor.
    """

    r_grid_m: FloatArray
    z_grid_m: FloatArray
    coil_r_m: FloatArray
    coil_z_m: FloatArray
    coil_current_a: FloatArray
    psin_knots: FloatArray
    pprime_coefficients_pa_per_wb: FloatArray
    ffprime_coefficients_t2_m2_per_wb: FloatArray
    profile_degree: int = 3
    n_picard: int = 40
    picard_omega: float = 0.6

    def __post_init__(self) -> None:
        """Normalise mutable caller arrays and enforce the bounded IDA contract."""
        array_fields = (
            "r_grid_m",
            "z_grid_m",
            "coil_r_m",
            "coil_z_m",
            "coil_current_a",
            "psin_knots",
            "pprime_coefficients_pa_per_wb",
            "ffprime_coefficients_t2_m2_per_wb",
        )
        for field in array_fields:
            object.__setattr__(self, field, _normalise_vector(getattr(self, field), field=field))

        if isinstance(self.profile_degree, bool) or not isinstance(self.profile_degree, int):
            raise ValueError("profile_degree must be an integer")
        if isinstance(self.n_picard, bool) or not isinstance(self.n_picard, int):
            raise ValueError("n_picard must be an integer")
        if self.profile_degree < 1 or self.profile_degree > 5:
            raise ValueError("profile_degree must be between 1 and 5")
        if self.n_picard < 1 or self.n_picard > 256:
            raise ValueError("n_picard must be between 1 and 256")
        if isinstance(self.picard_omega, bool):
            raise ValueError("picard_omega must be finite and in (0, 1]")
        try:
            omega = float(self.picard_omega)
        except (TypeError, ValueError) as exc:
            raise ValueError("picard_omega must be finite and in (0, 1]") from exc
        if not np.isfinite(omega) or not 0.0 < omega <= 1.0:
            raise ValueError("picard_omega must be finite and in (0, 1]")
        object.__setattr__(self, "picard_omega", omega)

        for field in ("r_grid_m", "z_grid_m"):
            grid = cast(FloatArray, getattr(self, field))
            if not MIN_GRID_POINTS <= grid.size <= MAX_GRID_POINTS:
                raise ValueError(f"{field} must contain between {MIN_GRID_POINTS} and {MAX_GRID_POINTS} points")
            _require_strictly_increasing(grid, field=field)
            _require_uniform(grid, field=field)
        if np.any(self.r_grid_m <= 0.0):
            raise ValueError("r_grid_m must be strictly positive")

        coil_count = self.coil_current_a.size
        if self.coil_r_m.size != coil_count or self.coil_z_m.size != coil_count:
            raise ValueError("coil position and current vectors must have identical lengths")
        if coil_count > MAX_COILS:
            raise ValueError(f"coil vectors must contain at most {MAX_COILS} entries")
        if np.any(self.coil_r_m <= 0.0):
            raise ValueError("coil_r_m must be strictly positive")

        _require_strictly_increasing(self.psin_knots, field="psin_knots")
        if self.psin_knots[0] != 0.0 or self.psin_knots[-1] != 1.0:
            raise ValueError("psin_knots must span exactly [0, 1]")
        for field in ("pprime_coefficients_pa_per_wb", "ffprime_coefficients_t2_m2_per_wb"):
            coefficients = cast(FloatArray, getattr(self, field))
            minimum = self.profile_degree + 1
            if not minimum <= coefficients.size <= MAX_PROFILE_COEFFICIENTS:
                raise ValueError(f"{field} must contain between {minimum} and {MAX_PROFILE_COEFFICIENTS} values")


@dataclass(frozen=True, slots=True)
class IDAEquilibriumEvidence:
    """Digest-bound provenance and fail-closed claim state for one adapter call."""

    schema_version: str
    status: str
    upstream_version: str
    jax_version: str
    jaxlib_version: str
    upstream_solver: str
    solver_file_sha256: str
    profile_basis_file_sha256: str
    request_sha256: str
    psi_sha256: str
    grid_shape: tuple[int, int]
    differentiated_inputs: tuple[str, ...]
    blockers: tuple[str, ...]
    claim_boundary: tuple[tuple[str, bool], ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable evidence record."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "upstream_version": self.upstream_version,
            "jax_version": self.jax_version,
            "jaxlib_version": self.jaxlib_version,
            "upstream_solver": self.upstream_solver,
            "solver_file_sha256": self.solver_file_sha256,
            "profile_basis_file_sha256": self.profile_basis_file_sha256,
            "request_sha256": self.request_sha256,
            "psi_sha256": self.psi_sha256,
            "grid_shape": list(self.grid_shape),
            "differentiated_inputs": list(self.differentiated_inputs),
            "blockers": list(self.blockers),
            "claim_boundary": dict(self.claim_boundary),
        }


@dataclass(frozen=True, slots=True)
class IDAEquilibriumResult:
    """Poloidal-flux solution and its bounded CONTROL evidence."""

    psi_wb: FloatArray
    evidence: IDAEquilibriumEvidence


@dataclass(frozen=True, slots=True)
class IDAEquilibriumVJPResult:
    """Vector-Jacobian product through the FUSION equilibrium solve."""

    equilibrium: IDAEquilibriumResult
    coil_current_gradient: FloatArray
    pprime_coefficient_gradient: FloatArray
    ffprime_coefficient_gradient: FloatArray
    cotangent_sha256: str


def _array_digest(value: FloatArray) -> str:
    canonical = np.asarray(value, dtype="<f8", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _request_digest(request: IDAEquilibriumRequest) -> str:
    fields: dict[str, object] = {
        "profile_degree": request.profile_degree,
        "n_picard": request.n_picard,
        "picard_omega": request.picard_omega,
    }
    units = {
        "r_grid_m": "m",
        "z_grid_m": "m",
        "coil_r_m": "m",
        "coil_z_m": "m",
        "coil_current_a": "A",
        "psin_knots": "1",
        "pprime_coefficients_pa_per_wb": "Pa/Wb",
        "ffprime_coefficients_t2_m2_per_wb": "T^2 m^2/Wb",
    }
    for name, unit in units.items():
        value = cast(FloatArray, getattr(request, name))
        fields[name] = {"shape": list(value.shape), "unit": unit, "value_sha256": _array_digest(value)}
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_major(version: str) -> int:
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)\.([0-9]+)(?:[A-Za-z0-9.+-]*)?", version)
    if match is None:
        raise IDAEquilibriumAdapterError("scpn-fusion version is not a semantic numeric version")
    return int(match.group(1))


def _source_file_sha256(module: object, *, label: str) -> str:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        raise IDAEquilibriumAdapterError(f"{label} does not expose a source file")
    path = Path(raw_path).resolve()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise IDAEquilibriumAdapterError(f"cannot read {label} source bytes: {exc}") from exc
    if not raw:
        raise IDAEquilibriumAdapterError(f"{label} source file is empty")
    return hashlib.sha256(raw).hexdigest()


class _LoadedBackend:
    """Runtime binding to the FUSION 4.x JAX implementation."""

    def __init__(self) -> None:
        try:
            package = importlib.import_module("scpn_fusion")
            solver_module = importlib.import_module("scpn_fusion.core.jax_free_boundary_gs_implicit")
            basis_module = importlib.import_module("scpn_fusion.core.jax_profile_basis")
            jax_module = importlib.import_module("jax")
            jaxlib_module = importlib.import_module("jaxlib")
            jnp_module = importlib.import_module("jax.numpy")
        except ImportError as exc:
            raise IDAEquilibriumAdapterError("IDA equilibrium adapter requires scpn-fusion 4.x with JAX") from exc

        self.version = str(getattr(package, "__version__", ""))
        self.jax_version = str(getattr(jax_module, "__version__", ""))
        self.jaxlib_version = str(getattr(jaxlib_module, "__version__", ""))
        if _parse_major(self.version) != 4:
            raise IDAEquilibriumAdapterError(
                f"IDA equilibrium adapter requires scpn-fusion 4.x, observed {self.version!r}"
            )
        config = getattr(jax_module, "config", None)
        self.x64_enabled = bool(getattr(config, "x64_enabled", False))
        self.solver_file_sha256 = _source_file_sha256(solver_module, label="FUSION implicit solver")
        self.profile_basis_file_sha256 = _source_file_sha256(basis_module, label="FUSION profile basis")
        self._asarray = cast(Callable[[object], object], vars(jnp_module)["asarray"])
        self._design_matrix = cast(Callable[[object, int, int], object], vars(basis_module)["bspline_design_matrix"])
        self._evaluate_profile = cast(Callable[[object, object], object], vars(basis_module)["evaluate_profile"])
        self._solve = cast(Callable[..., object], vars(solver_module)["solve_free_boundary_gs_implicit"])
        self._vjp = cast(
            Callable[..., tuple[object, Callable[[object], tuple[object, object, object]]]],
            vars(jax_module)["vjp"],
        )

    def asarray(self, value: FloatArray) -> object:
        """Convert a validated NumPy vector to an upstream JAX array."""
        return self._asarray(value)

    def design_matrix(self, query: FloatArray, n_coeff: int, degree: int) -> object:
        """Build the fixed upstream B-spline design matrix."""
        return self._design_matrix(query, n_coeff, degree)

    def evaluate_profile(self, coefficients: object, design_matrix: object) -> object:
        """Evaluate compact coefficients on the requested normalised-flux grid."""
        return self._evaluate_profile(coefficients, design_matrix)

    def solve(
        self,
        coil_current: object,
        pprime_values: object,
        ffprime_values: object,
        r_grid: object,
        z_grid: object,
        coil_r: object,
        coil_z: object,
        psin_knots: object,
        n_picard: int,
        picard_omega: float,
    ) -> object:
        """Call the FUSION implicit free-boundary solver with smooth-axis AD."""
        return self._solve(
            coil_current,
            pprime_values,
            ffprime_values,
            r_grid,
            z_grid,
            coil_r,
            coil_z,
            psin_knots,
            n_picard,
            picard_omega,
            4.0e-7 * np.pi,
            True,
        )

    def vjp(
        self,
        function: Callable[[object, object, object], object],
        coil_current: object,
        pprime_coefficients: object,
        ffprime_coefficients: object,
    ) -> tuple[object, Callable[[object], tuple[object, object, object]]]:
        """Build the upstream JAX vector-Jacobian product closure."""
        return self._vjp(function, coil_current, pprime_coefficients, ffprime_coefficients)


def _require_backend(backend: _Backend) -> None:
    if _parse_major(backend.version) != 4:
        raise IDAEquilibriumAdapterError(
            f"IDA equilibrium adapter requires scpn-fusion 4.x, observed {backend.version!r}"
        )
    if not backend.x64_enabled:
        raise IDAEquilibriumAdapterError("IDA equilibrium adapter requires JAX FP64 mode")
    if not backend.jax_version or not backend.jaxlib_version:
        raise IDAEquilibriumAdapterError("IDA equilibrium adapter requires JAX and jaxlib runtime versions")
    for field in ("solver_file_sha256", "profile_basis_file_sha256"):
        value = getattr(backend, field)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise IDAEquilibriumAdapterError(f"{field} must be a lowercase SHA-256 digest")


def _model(
    request: IDAEquilibriumRequest,
    backend: _Backend,
) -> Callable[[object, object, object], object]:
    pprime_design = backend.design_matrix(
        request.psin_knots,
        request.pprime_coefficients_pa_per_wb.size,
        request.profile_degree,
    )
    ffprime_design = backend.design_matrix(
        request.psin_knots,
        request.ffprime_coefficients_t2_m2_per_wb.size,
        request.profile_degree,
    )
    r_grid = backend.asarray(request.r_grid_m)
    z_grid = backend.asarray(request.z_grid_m)
    coil_r = backend.asarray(request.coil_r_m)
    coil_z = backend.asarray(request.coil_z_m)
    psin_knots = backend.asarray(request.psin_knots)

    def solve(coil_current: object, pprime_coefficients: object, ffprime_coefficients: object) -> object:
        pprime_values = backend.evaluate_profile(pprime_coefficients, pprime_design)
        ffprime_values = backend.evaluate_profile(ffprime_coefficients, ffprime_design)
        return backend.solve(
            coil_current,
            pprime_values,
            ffprime_values,
            r_grid,
            z_grid,
            coil_r,
            coil_z,
            psin_knots,
            request.n_picard,
            request.picard_omega,
        )

    return solve


def _normalise_output(value: object, *, shape: tuple[int, ...], field: str) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise IDAEquilibriumAdapterError(f"{field} is not a float64 array") from exc
    if array.shape != shape:
        raise IDAEquilibriumAdapterError(f"{field} shape {array.shape} does not match {shape}")
    if not np.all(np.isfinite(array)):
        raise IDAEquilibriumAdapterError(f"{field} contains non-finite values")
    frozen = np.array(array, dtype=np.float64, order="C", copy=True)
    frozen.flags.writeable = False
    return frozen


def _result(request: IDAEquilibriumRequest, backend: _Backend, value: object) -> IDAEquilibriumResult:
    psi = _normalise_output(value, shape=(request.z_grid_m.size, request.r_grid_m.size), field="psi_wb")
    evidence = IDAEquilibriumEvidence(
        schema_version=IDA_ADAPTER_SCHEMA,
        status=IDA_ADAPTER_STATUS,
        upstream_version=backend.version,
        jax_version=backend.jax_version,
        jaxlib_version=backend.jaxlib_version,
        upstream_solver=_SOLVER_NAME,
        solver_file_sha256=backend.solver_file_sha256,
        profile_basis_file_sha256=backend.profile_basis_file_sha256,
        request_sha256=_request_digest(request),
        psi_sha256=_array_digest(psi),
        grid_shape=(psi.shape[0], psi.shape[1]),
        differentiated_inputs=("coil_current_a", "pprime_coefficients_pa_per_wb", "ffprime_coefficients_t2_m2_per_wb"),
        blockers=_BLOCKERS,
        claim_boundary=tuple((field, False) for field in _CLAIM_FIELDS),
    )
    return IDAEquilibriumResult(psi_wb=psi, evidence=evidence)


def _solve_with_backend(request: IDAEquilibriumRequest, backend: _Backend) -> IDAEquilibriumResult:
    _require_backend(backend)
    model = _model(request, backend)
    value = model(
        backend.asarray(request.coil_current_a),
        backend.asarray(request.pprime_coefficients_pa_per_wb),
        backend.asarray(request.ffprime_coefficients_t2_m2_per_wb),
    )
    return _result(request, backend, value)


def solve_ida_equilibrium(request: IDAEquilibriumRequest) -> IDAEquilibriumResult:
    """Solve the bounded IDA equilibrium request through FUSION 4.x.

    Parameters
    ----------
    request
        Validated compact-profile, coil, and grid contract in SI units.

    Returns
    -------
    IDAEquilibriumResult
        Read-only poloidal flux in webers plus digest-bound blocked evidence.

    Raises
    ------
    IDAEquilibriumAdapterError
        If the optional upstream package, FP64 mode, source provenance, solve,
        or returned field cannot be admitted.
    """
    return _solve_with_backend(request, cast(_Backend, _LoadedBackend()))


def _vjp_with_backend(
    request: IDAEquilibriumRequest,
    cotangent: object,
    backend: _Backend,
) -> IDAEquilibriumVJPResult:
    _require_backend(backend)
    cotangent_array = _normalise_output(
        cotangent,
        shape=(request.z_grid_m.size, request.r_grid_m.size),
        field="cotangent",
    )
    model = _model(request, backend)
    value, pullback = backend.vjp(
        model,
        backend.asarray(request.coil_current_a),
        backend.asarray(request.pprime_coefficients_pa_per_wb),
        backend.asarray(request.ffprime_coefficients_t2_m2_per_wb),
    )
    gradients = pullback(backend.asarray(cotangent_array))
    if len(gradients) != 3:
        raise IDAEquilibriumAdapterError("upstream VJP must return three differentiated inputs")
    coil_gradient = _normalise_output(gradients[0], shape=request.coil_current_a.shape, field="coil_current_gradient")
    pprime_gradient = _normalise_output(
        gradients[1],
        shape=request.pprime_coefficients_pa_per_wb.shape,
        field="pprime_coefficient_gradient",
    )
    ffprime_gradient = _normalise_output(
        gradients[2],
        shape=request.ffprime_coefficients_t2_m2_per_wb.shape,
        field="ffprime_coefficient_gradient",
    )
    return IDAEquilibriumVJPResult(
        equilibrium=_result(request, backend, value),
        coil_current_gradient=coil_gradient,
        pprime_coefficient_gradient=pprime_gradient,
        ffprime_coefficient_gradient=ffprime_gradient,
        cotangent_sha256=_array_digest(cotangent_array),
    )


def ida_equilibrium_vjp(request: IDAEquilibriumRequest, cotangent: object) -> IDAEquilibriumVJPResult:
    """Apply a flux-field cotangent to the FUSION implicit equilibrium Jacobian.

    Parameters
    ----------
    request
        Validated SI equilibrium request.
    cotangent
        Finite ``(NZ, NR)`` derivative of a scalar IDA objective with respect to
        poloidal flux in inverse webers.

    Returns
    -------
    IDAEquilibriumVJPResult
        Gradients with respect to coil currents and compact p-prime/FF-prime
        coefficients, together with the solved equilibrium and blocked evidence.

    Raises
    ------
    IDAEquilibriumAdapterError
        If the upstream VJP or any result fails shape, finite-value, provenance,
        version, or FP64 admission.
    """
    return _vjp_with_backend(request, cotangent, cast(_Backend, _LoadedBackend()))


__all__ = [
    "IDA_ADAPTER_SCHEMA",
    "IDAEquilibriumAdapterError",
    "IDAEquilibriumEvidence",
    "IDAEquilibriumRequest",
    "IDAEquilibriumResult",
    "IDAEquilibriumVJPResult",
    "ida_equilibrium_vjp",
    "solve_ida_equilibrium",
]
