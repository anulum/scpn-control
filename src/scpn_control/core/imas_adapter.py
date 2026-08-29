# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — backend-neutral equilibrium-data contract and IDS adapters.

"""Validated equilibrium snapshots with real IMAS-Python and OMAS adapters.

The solver-facing representation uses COCOS 1: poloidal flux per radian and
``(Z, R)`` array orientation. IMAS Data Dictionary v3 uses COCOS 11 and v4
uses COCOS 17; both store rectangular 2-D fields in ``(R, Z)`` orientation.
Conversions in this module therefore apply the required flux factor/sign and
transpose explicitly. No facility endpoint, credential, or physical default is
owned by this module.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np

from scpn_control._typing import FloatArray

_SOLVER_COCOS = 1
_RECTANGULAR_GRID_IDENTIFIER = 1
_TWO_PI = 2.0 * np.pi
_LEGACY_REMOVAL_VERSION = "0.25.0"

BackendName = Literal["memory", "kernel", "geqdsk", "imas-python", "omas"]


class EquilibriumDataError(ValueError):
    """Raised when equilibrium data violate the neutral contract."""


class EquilibriumBackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested optional data backend is absent."""


class EquilibriumBackendError(RuntimeError):
    """Raised when a backend rejects or cannot decode equilibrium data."""


def _finite_scalar(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EquilibriumDataError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(result):
        raise EquilibriumDataError(f"{name} must be a finite scalar")
    return result


def _readonly_vector(name: str, value: object, *, positive: bool = False) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise EquilibriumDataError(f"{name} must be a one-dimensional array with at least two points")
    if not np.all(np.isfinite(array)):
        raise EquilibriumDataError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise EquilibriumDataError(f"{name} must contain only positive values")
    if np.any(np.diff(array) <= 0.0):
        raise EquilibriumDataError(f"{name} must be strictly increasing")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _readonly_field(name: str, value: object, shape: tuple[int, int]) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise EquilibriumDataError(f"{name} shape {array.shape} must equal solver (Z, R) shape {shape}")
    if not np.all(np.isfinite(array)):
        raise EquilibriumDataError(f"{name} must contain only finite values")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class EquilibriumSnapshot:
    """Backend-neutral rectangular equilibrium slice in solver convention.

    Arrays use SI units and solver orientation: ``r_m`` and ``z_m`` are
    strictly increasing vectors, while ``psi_wb_per_rad`` and optional
    ``j_phi_a_per_m2`` have shape ``(z_m.size, r_m.size)``. The snapshot is
    always COCOS 1; adapters own conversion to backend conventions.

    Attributes
    ----------
    r_m, z_m
        Strictly increasing radial and vertical grid coordinates in metres.
    psi_wb_per_rad
        Poloidal-flux field in webers per radian with shape ``(Z, R)``.
    j_phi_a_per_m2
        Toroidal current-density field in A/m² with shape ``(Z, R)``, or
        ``None`` when the source format does not carry that field.
    plasma_current_a
        Signed total plasma current in amperes.
    vacuum_toroidal_field_t
        Signed vacuum toroidal field in tesla; zero is invalid.
    vacuum_field_reference_radius_m
        Positive reference radius for the vacuum toroidal field in metres.
    time_s
        Slice time in seconds.
    source_backend, source
        Backend category and non-empty provenance identifier.
    data_dictionary_version
        Source Data Dictionary version, when applicable.
    cocos
        Solver convention identifier. This contract accepts only COCOS 1.
    """

    r_m: FloatArray
    z_m: FloatArray
    psi_wb_per_rad: FloatArray
    j_phi_a_per_m2: FloatArray | None
    plasma_current_a: float
    vacuum_toroidal_field_t: float
    vacuum_field_reference_radius_m: float
    time_s: float
    source_backend: BackendName
    source: str
    data_dictionary_version: str | None = None
    cocos: int = _SOLVER_COCOS

    def __post_init__(self) -> None:
        """Validate, copy, and freeze every numerical value."""
        r_m = _readonly_vector("r_m", self.r_m, positive=True)
        z_m = _readonly_vector("z_m", self.z_m)
        shape = (z_m.size, r_m.size)
        psi = _readonly_field("psi_wb_per_rad", self.psi_wb_per_rad, shape)
        current = None
        if self.j_phi_a_per_m2 is not None:
            current = _readonly_field("j_phi_a_per_m2", self.j_phi_a_per_m2, shape)

        plasma_current = _finite_scalar("plasma_current_a", self.plasma_current_a)
        vacuum_field = _finite_scalar("vacuum_toroidal_field_t", self.vacuum_toroidal_field_t)
        reference_radius = _finite_scalar("vacuum_field_reference_radius_m", self.vacuum_field_reference_radius_m)
        time_s = _finite_scalar("time_s", self.time_s)
        if vacuum_field == 0.0:
            raise EquilibriumDataError("vacuum_toroidal_field_t must be non-zero")
        if reference_radius <= 0.0:
            raise EquilibriumDataError("vacuum_field_reference_radius_m must be positive")
        if self.cocos != _SOLVER_COCOS:
            raise EquilibriumDataError("EquilibriumSnapshot must use solver COCOS 1")
        if self.source_backend not in {"memory", "kernel", "geqdsk", "imas-python", "omas"}:
            raise EquilibriumDataError(f"unsupported source_backend: {self.source_backend!r}")
        if not isinstance(self.source, str) or not self.source.strip():
            raise EquilibriumDataError("source must be a non-empty provenance string")
        if self.data_dictionary_version is not None and (
            not isinstance(self.data_dictionary_version, str) or not self.data_dictionary_version.strip()
        ):
            raise EquilibriumDataError("data_dictionary_version must be a non-empty string when provided")

        object.__setattr__(self, "r_m", r_m)
        object.__setattr__(self, "z_m", z_m)
        object.__setattr__(self, "psi_wb_per_rad", psi)
        object.__setattr__(self, "j_phi_a_per_m2", current)
        object.__setattr__(self, "plasma_current_a", plasma_current)
        object.__setattr__(self, "vacuum_toroidal_field_t", vacuum_field)
        object.__setattr__(self, "vacuum_field_reference_radius_m", reference_radius)
        object.__setattr__(self, "time_s", time_s)


def _load_backend(module_name: str, extra_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise EquilibriumBackendUnavailableError(
            f"{module_name} backend is unavailable; install scpn-control[{extra_name}]"
        ) from exc


def _dd_major(dd_version: str) -> int:
    try:
        major = int(dd_version.split(".", maxsplit=1)[0])
    except (AttributeError, ValueError) as exc:
        raise EquilibriumDataError(f"invalid Data Dictionary version: {dd_version!r}") from exc
    if major not in (3, 4):
        raise EquilibriumDataError(f"unsupported IMAS Data Dictionary major version {major}; expected 3 or 4")
    return major


def _dd_cocos(dd_version: str) -> int:
    return 11 if _dd_major(dd_version) == 3 else 17


def _solver_to_dd_psi_factor(dd_version: str) -> float:
    return _TWO_PI if _dd_cocos(dd_version) == 11 else -_TWO_PI


def _validate_index(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EquilibriumDataError(f"{name} must be a non-negative integer")
    return value


def _required_kernel_scalar(config: Mapping[str, Any], section: str, key: str) -> float:
    nested = config.get(section)
    if not isinstance(nested, Mapping) or key not in nested:
        raise EquilibriumDataError(f"kernel config must declare {section}.{key}")
    return _finite_scalar(f"kernel config {section}.{key}", nested[key])


def snapshot_from_kernel(kernel: Any, *, time_s: float = 0.0) -> EquilibriumSnapshot:
    """Extract a strict neutral snapshot from a solved FusionKernel-like object.

    Physical metadata must be present in the kernel configuration. The adapter
    never substitutes ITER defaults for an unknown machine.

    Parameters
    ----------
    kernel
        Solved object exposing ``R``, ``Z``, ``Psi``, ``J_phi``, and a ``cfg``
        mapping with ``physics.plasma_current_target``, ``physics.B0``, and
        ``dimensions.R0``.
    time_s
        Slice time in seconds.

    Returns
    -------
    EquilibriumSnapshot
        Validated immutable solver-state snapshot.

    Raises
    ------
    EquilibriumDataError
        If arrays or explicit machine metadata are absent or invalid.
    """
    config = getattr(kernel, "cfg", None)
    if not isinstance(config, Mapping):
        raise EquilibriumDataError("kernel.cfg must be a mapping with explicit physical metadata")
    try:
        r_m = kernel.R
        z_m = kernel.Z
        psi = kernel.Psi
        current = kernel.J_phi
    except AttributeError as exc:
        raise EquilibriumDataError("kernel must expose R, Z, Psi, and J_phi arrays") from exc
    return EquilibriumSnapshot(
        r_m=r_m,
        z_m=z_m,
        psi_wb_per_rad=psi,
        j_phi_a_per_m2=current,
        plasma_current_a=_required_kernel_scalar(config, "physics", "plasma_current_target"),
        vacuum_toroidal_field_t=_required_kernel_scalar(config, "physics", "B0"),
        vacuum_field_reference_radius_m=_required_kernel_scalar(config, "dimensions", "R0"),
        time_s=time_s,
        source_backend="kernel",
        source=f"kernel:{type(kernel).__module__}.{type(kernel).__qualname__}",
    )


def snapshot_to_kernel_state(snapshot: EquilibriumSnapshot) -> dict[str, Any]:
    """Return defensive arrays and scalars for a FusionKernel state import.

    Parameters
    ----------
    snapshot
        Validated neutral equilibrium carrying a 2-D current-density field.

    Returns
    -------
    dict[str, Any]
        Writable ``R``, ``Z``, ``Psi``, and ``J_phi`` copies plus ``ip``,
        ``b0``, and ``r0`` scalars.

    Raises
    ------
    EquilibriumDataError
        If current density is explicitly absent.
    """
    if snapshot.j_phi_a_per_m2 is None:
        raise EquilibriumDataError("kernel state requires a toroidal current-density field")
    return {
        "R": snapshot.r_m.copy(),
        "Z": snapshot.z_m.copy(),
        "Psi": snapshot.psi_wb_per_rad.copy(),
        "J_phi": snapshot.j_phi_a_per_m2.copy(),
        "ip": snapshot.plasma_current_a,
        "b0": snapshot.vacuum_toroidal_field_t,
        "r0": snapshot.vacuum_field_reference_radius_m,
    }


def snapshot_from_geqdsk(filepath: str | Path) -> EquilibriumSnapshot:
    """Read a GEQDSK equilibrium without fabricating an absent 2-D current map.

    Parameters
    ----------
    filepath
        Path to a GEQDSK file accepted by :func:`read_geqdsk`.

    Returns
    -------
    EquilibriumSnapshot
        COCOS-1-style repository snapshot with ``j_phi_a_per_m2=None`` and
        resolved file provenance.

    Raises
    ------
    EquilibriumDataError
        If decoded grids, fields, or machine scalars violate the neutral
        contract.
    OSError
        If the file cannot be read.
    """
    from scpn_control.core.eqdsk import read_geqdsk

    path = Path(filepath)
    geqdsk = read_geqdsk(str(path))
    return EquilibriumSnapshot(
        r_m=geqdsk.r,
        z_m=geqdsk.z,
        psi_wb_per_rad=geqdsk.psirz,
        j_phi_a_per_m2=None,
        plasma_current_a=geqdsk.current,
        vacuum_toroidal_field_t=geqdsk.bcentr,
        vacuum_field_reference_radius_m=geqdsk.rcentr,
        time_s=0.0,
        source_backend="geqdsk",
        source=f"geqdsk:{path.resolve()}",
    )


def _imas_ids_version(equilibrium: Any) -> str | None:
    version = getattr(equilibrium, "_dd_version", None)
    return str(version) if version is not None else None


def _imas_current_field(profile: Any, dd_version: str) -> str:
    field = "j_tor" if _dd_major(dd_version) == 3 else "j_phi"
    if not hasattr(profile, field):
        raise EquilibriumDataError(f"equilibrium profile does not expose required {field} field")
    return field


def _replace_or_append(values: object, index: int, value: float, name: str) -> FloatArray:
    current = np.asarray(values, dtype=np.float64)
    if index > current.size:
        raise EquilibriumDataError(f"{name} cannot create a gap before index {index}")
    if index == current.size:
        return np.append(current, value)
    result = current.copy()
    result[index] = value
    return result


def export_imas_equilibrium(
    snapshot: EquilibriumSnapshot,
    *,
    dd_version: str,
    equilibrium: Any | None = None,
    time_index: int = 0,
) -> Any:
    """Export a snapshot to a real IMAS-Python equilibrium IDS.

    ``dd_version`` is mandatory because DD v3 and v4 use different COCOS and
    current-density leaf names. Existing IDS objects may be updated or appended
    sequentially; gaps and DD-version mismatches fail closed.

    Parameters
    ----------
    snapshot
        Neutral COCOS-1 equilibrium to encode.
    dd_version
        Exact IMAS Data Dictionary version. Major versions 3 and 4 are
        supported.
    equilibrium
        Existing real IMAS-Python equilibrium IDS to update, or ``None`` to
        create one through ``imas.IDSFactory``.
    time_index
        Existing slice index to overwrite or next sequential index to append.

    Returns
    -------
    Any
        Validated real IMAS-Python equilibrium IDS.

    Raises
    ------
    EquilibriumBackendUnavailableError
        If the ``imas`` optional dependency is not installed.
    EquilibriumDataError
        If versions, indices, reference radius, or snapshot data conflict.
    EquilibriumBackendError
        If IMAS-Python rejects the populated IDS.

    Notes
    -----
    DD3 output uses COCOS 11 and ``j_tor``; DD4 output uses COCOS 17 and
    ``j_phi``. Rectangular fields are transposed from solver ``(Z, R)`` to IDS
    ``(R, Z)`` orientation and poloidal flux receives the corresponding signed
    ``2π`` conversion.
    """
    index = _validate_index("time_index", time_index)
    _dd_major(dd_version)
    imas = _load_backend("imas", "imas")
    if equilibrium is None:
        equilibrium = imas.IDSFactory(dd_version).equilibrium()
    actual_version = _imas_ids_version(equilibrium)
    if actual_version is not None and actual_version != dd_version:
        raise EquilibriumDataError(
            f"declared Data Dictionary version {dd_version} does not match IDS version {actual_version}"
        )

    existing_slices = len(equilibrium.time_slice)
    if index > existing_slices:
        raise EquilibriumDataError(f"time_index cannot create a gap before index {index}")
    if index == existing_slices:
        equilibrium.time_slice.resize(existing_slices + 1)

    existing_r0 = _finite_scalar("vacuum_toroidal_field.r0", equilibrium.vacuum_toroidal_field.r0)
    if existing_r0 != -9.0e40 and not np.isclose(
        existing_r0, snapshot.vacuum_field_reference_radius_m, rtol=0.0, atol=0.0
    ):
        raise EquilibriumDataError("vacuum-field reference radius cannot vary between time slices")

    equilibrium.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    equilibrium.ids_properties.comment = "SCPN-CONTROL validated COCOS-1 equilibrium snapshot"
    equilibrium.time = _replace_or_append(equilibrium.time, index, snapshot.time_s, "equilibrium.time")
    equilibrium.vacuum_toroidal_field.r0 = snapshot.vacuum_field_reference_radius_m
    equilibrium.vacuum_toroidal_field.b0 = _replace_or_append(
        equilibrium.vacuum_toroidal_field.b0,
        index,
        snapshot.vacuum_toroidal_field_t,
        "equilibrium.vacuum_toroidal_field.b0",
    )

    time_slice = equilibrium.time_slice[index]
    time_slice.time = snapshot.time_s
    time_slice.global_quantities.ip = snapshot.plasma_current_a
    if len(time_slice.profiles_2d) == 0:
        time_slice.profiles_2d.resize(1)
    profile = time_slice.profiles_2d[0]
    profile.grid_type.index = _RECTANGULAR_GRID_IDENTIFIER
    profile.grid_type.name = "rectangular"
    profile.grid_type.description = "Cylindrical R,Z rectangular grid"
    profile.grid.dim1 = snapshot.r_m
    profile.grid.dim2 = snapshot.z_m
    profile.psi = snapshot.psi_wb_per_rad.T * _solver_to_dd_psi_factor(dd_version)
    current_field = _imas_current_field(profile, dd_version)
    if snapshot.j_phi_a_per_m2 is not None:
        setattr(profile, current_field, snapshot.j_phi_a_per_m2.T)
    else:
        setattr(profile, current_field, np.empty((0, 0), dtype=np.float64))

    try:
        equilibrium.validate()
    except Exception as exc:
        raise EquilibriumBackendError(f"IMAS-Python rejected exported equilibrium: {exc}") from exc
    return equilibrium


def import_imas_equilibrium(
    equilibrium: Any,
    *,
    dd_version: str,
    time_index: int = 0,
    profiles_2d_index: int = 0,
    require_current_density: bool = True,
    source: str | None = None,
) -> EquilibriumSnapshot:
    """Import a real IMAS-Python IDS into the neutral solver convention.

    Parameters
    ----------
    equilibrium
        Real IMAS-Python equilibrium IDS.
    dd_version
        Exact Data Dictionary version used to interpret the IDS.
    time_index, profiles_2d_index
        Non-negative slice and rectangular-profile indices.
    require_current_density
        Reject an empty DD current leaf when ``True``; preserve it as ``None``
        when ``False``.
    source
        Optional caller-supplied provenance string.

    Returns
    -------
    EquilibriumSnapshot
        Immutable SI snapshot in COCOS 1 and ``(Z, R)`` orientation.

    Raises
    ------
    EquilibriumDataError
        If the requested schema, indices, metadata, or arrays are inconsistent.
    """
    index = _validate_index("time_index", time_index)
    profile_index = _validate_index("profiles_2d_index", profiles_2d_index)
    _dd_major(dd_version)
    actual_version = _imas_ids_version(equilibrium)
    if actual_version is not None and actual_version != dd_version:
        raise EquilibriumDataError(
            f"declared Data Dictionary version {dd_version} does not match IDS version {actual_version}"
        )
    try:
        if index >= len(equilibrium.time_slice):
            raise EquilibriumDataError(f"time_index {index} is outside equilibrium.time_slice")
        time_slice = equilibrium.time_slice[index]
        if profile_index >= len(time_slice.profiles_2d):
            raise EquilibriumDataError(f"profiles_2d_index {profile_index} is outside profiles_2d")
        times = np.asarray(equilibrium.time, dtype=np.float64)
        b0_values = np.asarray(equilibrium.vacuum_toroidal_field.b0, dtype=np.float64)
        if index >= times.size or index >= b0_values.size:
            raise EquilibriumDataError("equilibrium time and b0 arrays must cover the selected time slice")
        time_s = _finite_scalar("time_slice.time", time_slice.time)
        if not np.isclose(times[index], time_s, rtol=0.0, atol=1.0e-12):
            raise EquilibriumDataError("time_slice time does not match the top-level equilibrium time array")
        profile = time_slice.profiles_2d[profile_index]
        field_name = _imas_current_field(profile, dd_version)
        raw_current = np.asarray(getattr(profile, field_name), dtype=np.float64)
        current: FloatArray | None
        if raw_current.size == 0:
            if require_current_density:
                raise EquilibriumDataError(f"required {field_name} current-density field is empty")
            current = None
        else:
            current = np.asarray(raw_current.T, dtype=np.float64)
        return EquilibriumSnapshot(
            r_m=np.asarray(profile.grid.dim1, dtype=np.float64),
            z_m=np.asarray(profile.grid.dim2, dtype=np.float64),
            psi_wb_per_rad=np.asarray(profile.psi, dtype=np.float64).T / _solver_to_dd_psi_factor(dd_version),
            j_phi_a_per_m2=current,
            plasma_current_a=time_slice.global_quantities.ip,
            vacuum_toroidal_field_t=b0_values[index],
            vacuum_field_reference_radius_m=equilibrium.vacuum_toroidal_field.r0,
            time_s=time_s,
            source_backend="imas-python",
            source=source or f"imas-python:equilibrium:0:{index}",
            data_dictionary_version=dd_version,
        )
    except EquilibriumDataError:
        raise
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise EquilibriumDataError(f"invalid IMAS-Python equilibrium structure: {exc}") from exc


def write_imas_entry(entry: Any, snapshot: EquilibriumSnapshot, *, occurrence: int = 0) -> None:
    """Write one snapshot through a caller-owned, already-open IMAS DBEntry.

    Parameters
    ----------
    entry
        Open real ``imas.DBEntry``. Its URI, mode, credentials, Data Dictionary
        version, and lifecycle remain caller-owned.
    snapshot
        Neutral equilibrium to export and put.
    occurrence
        Non-negative equilibrium IDS occurrence.

    Raises
    ------
    EquilibriumBackendUnavailableError
        If IMAS-Python is unavailable.
    EquilibriumDataError
        If the occurrence or snapshot contract is invalid.
    EquilibriumBackendError
        If IDS validation or DBEntry write fails.
    """
    occurrence = _validate_index("occurrence", occurrence)
    try:
        dd_version = str(entry.dd_version)
        equilibrium = export_imas_equilibrium(snapshot, dd_version=dd_version)
        entry.put(equilibrium, occurrence=occurrence)
    except (EquilibriumDataError, EquilibriumBackendError):
        raise
    except Exception as exc:
        raise EquilibriumBackendError(f"IMAS DBEntry write failed: {exc}") from exc


def read_imas_entry(
    entry: Any,
    *,
    occurrence: int = 0,
    time_index: int = 0,
    profiles_2d_index: int = 0,
    autoconvert: bool = False,
) -> EquilibriumSnapshot:
    """Read one equilibrium slice through a caller-owned, open IMAS DBEntry.

    Automatic conversion is disabled by default so a DD-major transition is a
    caller-visible decision rather than an implicit reinterpretation.

    Parameters
    ----------
    entry
        Open caller-owned real ``imas.DBEntry``.
    occurrence
        Non-negative equilibrium IDS occurrence.
    time_index, profiles_2d_index
        Non-negative time-slice and rectangular-profile indices.
    autoconvert
        Pass explicit IMAS automatic Data Dictionary conversion authority.

    Returns
    -------
    EquilibriumSnapshot
        Validated neutral snapshot with DBEntry URI provenance.

    Raises
    ------
    EquilibriumDataError
        If indices or decoded equilibrium data violate the contract.
    EquilibriumBackendError
        If the DBEntry read fails.
    """
    occurrence = _validate_index("occurrence", occurrence)
    try:
        dd_version = str(entry.dd_version)
        equilibrium = entry.get("equilibrium", occurrence=occurrence, autoconvert=autoconvert)
        source = f"imas-python:{entry.uri}:equilibrium:{occurrence}:{time_index}"
        return import_imas_equilibrium(
            equilibrium,
            dd_version=dd_version,
            time_index=time_index,
            profiles_2d_index=profiles_2d_index,
            source=source,
        )
    except (EquilibriumDataError, EquilibriumBackendError):
        raise
    except Exception as exc:
        raise EquilibriumBackendError(f"IMAS DBEntry read failed: {exc}") from exc


def export_omas_equilibrium(
    snapshot: EquilibriumSnapshot,
    *,
    ods: Any | None = None,
    time_index: int = 0,
    imas_version: str = "3.41.0",
) -> Any:
    """Export a snapshot to a real OMAS ODS with COCOS conversion enabled.

    Parameters
    ----------
    snapshot
        Neutral COCOS-1 equilibrium to encode.
    ods
        Existing real OMAS ODS to update, or ``None`` to create one.
    time_index
        Existing slice index to overwrite or next sequential index to append.
    imas_version
        Exact bundled OMAS Data Dictionary v3 version.

    Returns
    -------
    Any
        Real OMAS ODS configured for solver-side COCOS-1 I/O.

    Raises
    ------
    EquilibriumBackendUnavailableError
        If OMAS is unavailable.
    EquilibriumDataError
        If the index or Data Dictionary contract is invalid.
    EquilibriumBackendError
        If OMAS rejects an assignment or conversion.

    Notes
    -----
    OMAS 0.x is deliberately limited to its bundled DD3 schema. Use the
    IMAS-Python adapter for DD4.
    """
    index = _validate_index("time_index", time_index)
    if _dd_major(imas_version) != 3:
        raise EquilibriumDataError("OMAS 0.x adapter supports bundled Data Dictionary v3 schemas only")
    omas = _load_backend("omas", "omas")
    if ods is None:
        ods = omas.ODS(imas_version=imas_version, cocos=11, cocosio=_SOLVER_COCOS, consistency_check=True)
    if str(ods.imas_version) != imas_version:
        raise EquilibriumDataError(
            f"declared Data Dictionary version {imas_version} does not match ODS version {ods.imas_version}"
        )
    try:
        with omas.omas_environment(ods, cocosio=_SOLVER_COCOS):
            ods["equilibrium.vacuum_toroidal_field.r0"] = snapshot.vacuum_field_reference_radius_m
            ods.set_time_array("equilibrium.time", index, snapshot.time_s)
            ods.set_time_array("equilibrium.vacuum_toroidal_field.b0", index, snapshot.vacuum_toroidal_field_t)
            prefix = f"equilibrium.time_slice.{index}"
            ods[f"{prefix}.time"] = snapshot.time_s
            ods[f"{prefix}.global_quantities.ip"] = snapshot.plasma_current_a
            profile = f"{prefix}.profiles_2d.0"
            ods[f"{profile}.grid_type.index"] = _RECTANGULAR_GRID_IDENTIFIER
            ods[f"{profile}.grid_type.name"] = "rectangular"
            ods[f"{profile}.grid_type.description"] = "Cylindrical R,Z rectangular grid"
            ods[f"{profile}.grid.dim1"] = snapshot.r_m
            ods[f"{profile}.grid.dim2"] = snapshot.z_m
            ods[f"{profile}.psi"] = snapshot.psi_wb_per_rad.T
            current_path = f"{profile}.j_tor"
            if snapshot.j_phi_a_per_m2 is not None:
                ods[current_path] = snapshot.j_phi_a_per_m2.T
            elif current_path in ods:
                del ods[current_path]
    except Exception as exc:
        raise EquilibriumBackendError(f"OMAS rejected exported equilibrium: {exc}") from exc
    return ods


def import_omas_equilibrium(
    ods: Any,
    *,
    time_index: int = 0,
    profiles_2d_index: int = 0,
    require_current_density: bool = True,
    source: str | None = None,
) -> EquilibriumSnapshot:
    """Import an OMAS ODS into the neutral COCOS-1 solver convention.

    Parameters
    ----------
    ods
        Real OMAS ODS using its bundled Data Dictionary v3 schema.
    time_index, profiles_2d_index
        Non-negative equilibrium slice and rectangular-profile indices.
    require_current_density
        Reject absent ``j_tor`` when ``True``; preserve it as ``None`` when
        ``False``.
    source
        Optional caller-supplied provenance string.

    Returns
    -------
    EquilibriumSnapshot
        Immutable SI snapshot in COCOS 1 and ``(Z, R)`` orientation.

    Raises
    ------
    EquilibriumBackendUnavailableError
        If OMAS is unavailable.
    EquilibriumDataError
        If schema, indices, time metadata, or required fields are invalid.
    """
    index = _validate_index("time_index", time_index)
    profile_index = _validate_index("profiles_2d_index", profiles_2d_index)
    omas = _load_backend("omas", "omas")
    imas_version = str(ods.imas_version)
    if _dd_major(imas_version) != 3:
        raise EquilibriumDataError("OMAS 0.x adapter supports bundled Data Dictionary v3 schemas only")
    prefix = f"equilibrium.time_slice.{index}"
    profile = f"{prefix}.profiles_2d.{profile_index}"
    try:
        with omas.omas_environment(ods, cocosio=_SOLVER_COCOS):
            top_time = _finite_scalar("equilibrium.time", ods["equilibrium.time"][index])
            slice_time = _finite_scalar("time_slice.time", ods[f"{prefix}.time"])
            if not np.isclose(top_time, slice_time, rtol=0.0, atol=1.0e-12):
                raise EquilibriumDataError("time_slice time does not match the top-level equilibrium time array")
            current_path = f"{profile}.j_tor"
            current: FloatArray | None
            if current_path not in ods:
                if require_current_density:
                    raise EquilibriumDataError("required j_tor current-density field is absent")
                current = None
            else:
                current = np.asarray(ods[current_path], dtype=np.float64).T
            return EquilibriumSnapshot(
                r_m=np.asarray(ods[f"{profile}.grid.dim1"], dtype=np.float64),
                z_m=np.asarray(ods[f"{profile}.grid.dim2"], dtype=np.float64),
                psi_wb_per_rad=np.asarray(ods[f"{profile}.psi"], dtype=np.float64).T,
                j_phi_a_per_m2=current,
                plasma_current_a=ods[f"{prefix}.global_quantities.ip"],
                vacuum_toroidal_field_t=ods["equilibrium.vacuum_toroidal_field.b0"][index],
                vacuum_field_reference_radius_m=ods["equilibrium.vacuum_toroidal_field.r0"],
                time_s=slice_time,
                source_backend="omas",
                source=source or f"omas:equilibrium:0:{index}",
                data_dictionary_version=imas_version,
            )
    except EquilibriumDataError:
        raise
    except (AttributeError, IndexError, KeyError, LookupError, TypeError, ValueError) as exc:
        raise EquilibriumDataError(f"invalid OMAS equilibrium structure: {exc}") from exc


def _warn_legacy(name: str, replacement: str) -> None:
    warnings.warn(
        f"{name} is deprecated; use {replacement}; removal is scheduled for {_LEGACY_REMOVAL_VERSION}",
        DeprecationWarning,
        stacklevel=3,
    )


class EquilibriumIDS(EquilibriumSnapshot):
    """Deprecated field-name-compatible facade for :class:`EquilibriumSnapshot`."""

    __slots__ = ()

    def __init__(
        self,
        r: object,
        z: object,
        psi: object,
        j_tor: object,
        ip: float,
        b0: float,
        r0: float,
        time: float = 0.0,
    ) -> None:
        _warn_legacy("EquilibriumIDS", "EquilibriumSnapshot")
        super().__init__(
            r_m=np.asarray(r, dtype=np.float64),
            z_m=np.asarray(z, dtype=np.float64),
            psi_wb_per_rad=np.asarray(psi, dtype=np.float64),
            j_phi_a_per_m2=np.asarray(j_tor, dtype=np.float64),
            plasma_current_a=ip,
            vacuum_toroidal_field_t=b0,
            vacuum_field_reference_radius_m=r0,
            time_s=time,
            source_backend="memory",
            source="legacy:EquilibriumIDS",
        )

    @property
    def r(self) -> FloatArray:
        """Deprecated alias for ``r_m``."""
        return self.r_m

    @property
    def z(self) -> FloatArray:
        """Deprecated alias for ``z_m``."""
        return self.z_m

    @property
    def psi(self) -> FloatArray:
        """Deprecated alias for ``psi_wb_per_rad``."""
        return self.psi_wb_per_rad

    @property
    def j_tor(self) -> FloatArray:
        """Deprecated alias for ``j_phi_a_per_m2``."""
        assert self.j_phi_a_per_m2 is not None
        return self.j_phi_a_per_m2

    @property
    def ip(self) -> float:
        """Deprecated alias for ``plasma_current_a``."""
        return self.plasma_current_a

    @property
    def b0(self) -> float:
        """Deprecated alias for ``vacuum_toroidal_field_t``."""
        return self.vacuum_toroidal_field_t

    @property
    def r0(self) -> float:
        """Deprecated alias for ``vacuum_field_reference_radius_m``."""
        return self.vacuum_field_reference_radius_m

    @property
    def time(self) -> float:
        """Deprecated alias for ``time_s``."""
        return self.time_s


def _legacy_snapshot(snapshot: EquilibriumSnapshot) -> EquilibriumIDS:
    assert snapshot.j_phi_a_per_m2 is not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return EquilibriumIDS(
            r=snapshot.r_m,
            z=snapshot.z_m,
            psi=snapshot.psi_wb_per_rad,
            j_tor=snapshot.j_phi_a_per_m2,
            ip=snapshot.plasma_current_a,
            b0=snapshot.vacuum_toroidal_field_t,
            r0=snapshot.vacuum_field_reference_radius_m,
            time=snapshot.time_s,
        )


def from_kernel(kernel: Any, time: float = 0.0) -> EquilibriumIDS:
    """Forward a legacy kernel conversion to :func:`snapshot_from_kernel`."""
    _warn_legacy("from_kernel", "snapshot_from_kernel")
    return _legacy_snapshot(snapshot_from_kernel(kernel, time_s=time))


def to_kernel_arrays(ids: EquilibriumSnapshot) -> dict[str, Any]:
    """Forward a legacy array conversion to :func:`snapshot_to_kernel_state`."""
    _warn_legacy("to_kernel_arrays", "snapshot_to_kernel_state")
    return snapshot_to_kernel_state(ids)


def to_omas(ids: EquilibriumSnapshot) -> Any:
    """Forward a legacy ODS export to :func:`export_omas_equilibrium`."""
    _warn_legacy("to_omas", "export_omas_equilibrium")
    return export_omas_equilibrium(ids)


def from_omas(ods: Any, time_index: int = 0) -> EquilibriumIDS:
    """Forward a legacy ODS import to :func:`import_omas_equilibrium`."""
    _warn_legacy("from_omas", "import_omas_equilibrium")
    return _legacy_snapshot(import_omas_equilibrium(ods, time_index=time_index))


def from_geqdsk(filepath: str) -> EquilibriumSnapshot:
    """Forward a legacy file import to :func:`snapshot_from_geqdsk`.

    A neutral snapshot is returned because GEQDSK does not carry the 2-D
    current-density field required by the legacy ``EquilibriumIDS`` facade.
    """
    _warn_legacy("from_geqdsk", "snapshot_from_geqdsk")
    return snapshot_from_geqdsk(filepath)
