# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — real IMAS-Python equilibrium-adapter contract tests.

"""Exercise the neutral equilibrium contract against real IMAS-Python IDS objects."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

imas = pytest.importorskip(
    "imas",
    reason="real IMAS-Python tests execute in the Ubuntu 3.12 fusion-data CI lane",
)

from scpn_control.core.imas_adapter import (
    EquilibriumBackendError,
    EquilibriumBackendUnavailableError,
    EquilibriumDataError,
    EquilibriumSnapshot,
    export_imas_equilibrium,
    import_imas_equilibrium,
    read_imas_entry,
    write_imas_entry,
)


def _snapshot(*, time_s: float = 1.25, **updates: object) -> EquilibriumSnapshot:
    r_m = np.array([1.4, 1.8, 2.2, 2.6])
    z_m = np.array([-0.8, 0.1, 0.9])
    rr, zz = np.meshgrid(r_m, z_m)
    values: dict[str, object] = {
        "r_m": r_m,
        "z_m": z_m,
        "psi_wb_per_rad": 0.12 * rr**2 - 0.07 * zz**2 + 0.03 * rr * zz,
        "j_phi_a_per_m2": 8.0e5 - 1.2e5 * rr + 2.5e4 * zz,
        "plasma_current_a": 1.15e6,
        "vacuum_toroidal_field_t": -2.15,
        "vacuum_field_reference_radius_m": 1.72,
        "time_s": time_s,
        "source_backend": "memory",
        "source": "tests/non-square-signed-equilibrium",
    }
    values.update(updates)
    return EquilibriumSnapshot(**cast(Any, values))


@pytest.mark.parametrize(
    ("dd_version", "cocos", "current_field", "psi_factor"),
    [
        ("3.41.0", 11, "j_tor", 2.0 * np.pi),
        ("4.1.1", 17, "j_phi", -2.0 * np.pi),
    ],
)
def test_real_imas_object_round_trip_across_dd_majors(
    dd_version: str,
    cocos: int,
    current_field: str,
    psi_factor: float,
) -> None:
    """Round-trip real DD3 and DD4 IDS objects with exact COCOS transforms."""
    snapshot = _snapshot()
    equilibrium = export_imas_equilibrium(snapshot, dd_version=dd_version)

    equilibrium.validate()
    profile = equilibrium.time_slice[0].profiles_2d[0]
    assert equilibrium.vacuum_toroidal_field.r0 == snapshot.vacuum_field_reference_radius_m
    np.testing.assert_allclose(equilibrium.vacuum_toroidal_field.b0, [snapshot.vacuum_toroidal_field_t])
    np.testing.assert_allclose(profile.grid.dim1, snapshot.r_m)
    np.testing.assert_allclose(profile.grid.dim2, snapshot.z_m)
    np.testing.assert_allclose(profile.psi, snapshot.psi_wb_per_rad.T * psi_factor)
    np.testing.assert_allclose(getattr(profile, current_field), snapshot.j_phi_a_per_m2.T)

    recovered = import_imas_equilibrium(equilibrium, dd_version=dd_version)
    assert recovered.cocos == 1
    assert recovered.source_backend == "imas-python"
    assert recovered.data_dictionary_version == dd_version
    np.testing.assert_allclose(recovered.r_m, snapshot.r_m)
    np.testing.assert_allclose(recovered.z_m, snapshot.z_m)
    np.testing.assert_allclose(recovered.psi_wb_per_rad, snapshot.psi_wb_per_rad)
    np.testing.assert_allclose(recovered.j_phi_a_per_m2, snapshot.j_phi_a_per_m2)
    assert recovered.plasma_current_a == snapshot.plasma_current_a
    assert recovered.vacuum_toroidal_field_t == snapshot.vacuum_toroidal_field_t
    assert recovered.vacuum_field_reference_radius_m == snapshot.vacuum_field_reference_radius_m
    assert recovered.time_s == snapshot.time_s
    assert cocos in (11, 17)


def test_real_imas_multiple_time_slices_preserve_indexed_b0_and_time() -> None:
    """Preserve indexed time and vacuum field across multiple IDS slices."""
    first = _snapshot(time_s=0.25)
    second = _snapshot(time_s=0.75)
    equilibrium = export_imas_equilibrium(first, dd_version="4.1.1")
    equilibrium = export_imas_equilibrium(second, equilibrium=equilibrium, time_index=1, dd_version="4.1.1")

    equilibrium.validate()
    np.testing.assert_allclose(equilibrium.time, [0.25, 0.75])
    np.testing.assert_allclose(equilibrium.vacuum_toroidal_field.b0, [-2.15, -2.15])
    assert import_imas_equilibrium(equilibrium, dd_version="4.1.1", time_index=1).time_s == 0.75


def test_real_netcdf_dbentry_round_trip(tmp_path: Path) -> None:
    """Round-trip a real nonzero-occurrence IMAS NetCDF DBEntry."""
    path = tmp_path / "equilibrium.nc"
    snapshot = _snapshot()

    with imas.DBEntry(str(path), "w", dd_version="4.1.1") as entry:
        write_imas_entry(entry, snapshot, occurrence=2)

    with imas.DBEntry(str(path), "r", dd_version="4.1.1") as entry:
        recovered = read_imas_entry(entry, occurrence=2)

    assert recovered.data_dictionary_version == "4.1.1"
    assert recovered.source == f"imas-python:{path}:equilibrium:2:0"
    np.testing.assert_allclose(recovered.psi_wb_per_rad, snapshot.psi_wb_per_rad)
    np.testing.assert_allclose(recovered.j_phi_a_per_m2, snapshot.j_phi_a_per_m2)
    assert recovered.vacuum_toroidal_field_t == snapshot.vacuum_toroidal_field_t


def test_imas_import_rejects_declared_version_mismatch() -> None:
    """Reject a declared Data Dictionary version that differs from the IDS."""
    equilibrium = export_imas_equilibrium(_snapshot(), dd_version="4.1.1")
    with pytest.raises(EquilibriumDataError, match="declared Data Dictionary version"):
        import_imas_equilibrium(equilibrium, dd_version="3.41.0")


def test_imas_import_rejects_inconsistent_top_level_time() -> None:
    """Reject disagreement between top-level and slice time."""
    equilibrium = export_imas_equilibrium(_snapshot(), dd_version="4.1.1")
    equilibrium.time = np.array([9.0])
    with pytest.raises(EquilibriumDataError, match="time_slice time"):
        import_imas_equilibrium(equilibrium, dd_version="4.1.1")


def test_imas_import_rejects_missing_current_density() -> None:
    """Reject an empty required DD4 current-density field."""
    equilibrium = export_imas_equilibrium(_snapshot(), dd_version="4.1.1")
    equilibrium.time_slice[0].profiles_2d[0].j_phi = np.empty((0, 0))
    with pytest.raises(EquilibriumDataError, match="j_phi"):
        import_imas_equilibrium(equilibrium, dd_version="4.1.1")


def test_imas_import_can_preserve_explicitly_absent_current_density() -> None:
    """Preserve absent current when the caller explicitly permits it."""
    equilibrium = export_imas_equilibrium(_snapshot(), dd_version="4.1.1")
    equilibrium.time_slice[0].profiles_2d[0].j_phi = np.empty((0, 0))
    recovered = import_imas_equilibrium(
        equilibrium,
        dd_version="4.1.1",
        require_current_density=False,
    )
    assert recovered.j_phi_a_per_m2 is None


def test_imas_rejects_unsupported_data_dictionary_major() -> None:
    """Reject unsupported IMAS Data Dictionary major versions."""
    with pytest.raises(EquilibriumDataError, match="major version"):
        export_imas_equilibrium(_snapshot(), dd_version="5.0.0")


@pytest.mark.parametrize("dd_version", ["not-a-version", None])
def test_imas_rejects_malformed_data_dictionary_version(dd_version: Any) -> None:
    """Reject malformed or absent Data Dictionary version declarations."""
    with pytest.raises(EquilibriumDataError, match="invalid Data Dictionary version"):
        export_imas_equilibrium(_snapshot(), dd_version=dd_version)


@pytest.mark.parametrize("index", [True, 1.5, -1])
def test_imas_rejects_invalid_indices(index: Any) -> None:
    """Reject boolean, fractional, and negative indices."""
    with pytest.raises(EquilibriumDataError, match="non-negative integer"):
        export_imas_equilibrium(_snapshot(), dd_version="4.1.1", time_index=index)


def test_imas_export_rejects_version_gap_and_reference_radius_change() -> None:
    """Reject version drift, sequence gaps, and time-varying reference radius."""
    equilibrium = export_imas_equilibrium(_snapshot(), dd_version="4.1.1")
    with pytest.raises(EquilibriumDataError, match="does not match IDS version"):
        export_imas_equilibrium(_snapshot(), equilibrium=equilibrium, dd_version="3.41.0")
    with pytest.raises(EquilibriumDataError, match="cannot create a gap"):
        export_imas_equilibrium(_snapshot(), equilibrium=equilibrium, dd_version="4.1.1", time_index=2)
    with pytest.raises(EquilibriumDataError, match="reference radius cannot vary"):
        export_imas_equilibrium(
            _snapshot(vacuum_field_reference_radius_m=1.73),
            equilibrium=equilibrium,
            dd_version="4.1.1",
            time_index=1,
        )

    malformed = export_imas_equilibrium(_snapshot(), dd_version="4.1.1")
    malformed.time = np.array([])
    malformed.vacuum_toroidal_field.b0 = np.array([])
    with pytest.raises(EquilibriumDataError, match="equilibrium.time cannot create a gap"):
        export_imas_equilibrium(
            _snapshot(time_s=3.0),
            equilibrium=malformed,
            dd_version="4.1.1",
            time_index=1,
        )


def test_imas_export_can_overwrite_existing_slice_and_omit_current() -> None:
    """Overwrite one slice and preserve an explicitly absent current field."""
    equilibrium = export_imas_equilibrium(_snapshot(), dd_version="4.1.1")
    replacement = _snapshot(time_s=2.5, j_phi_a_per_m2=None)
    equilibrium = export_imas_equilibrium(
        replacement,
        equilibrium=equilibrium,
        dd_version="4.1.1",
        time_index=0,
    )
    assert len(equilibrium.time_slice) == 1
    assert equilibrium.time_slice[0].time == 2.5
    assert equilibrium.time_slice[0].profiles_2d[0].j_phi.size == 0


def test_imas_import_rejects_missing_indices_and_time_arrays() -> None:
    """Reject missing slices, profiles, and indexed metadata arrays."""
    equilibrium = export_imas_equilibrium(_snapshot(), dd_version="4.1.1")
    with pytest.raises(EquilibriumDataError, match="outside equilibrium.time_slice"):
        import_imas_equilibrium(equilibrium, dd_version="4.1.1", time_index=1)
    with pytest.raises(EquilibriumDataError, match="outside profiles_2d"):
        import_imas_equilibrium(equilibrium, dd_version="4.1.1", profiles_2d_index=1)
    equilibrium.time = np.array([])
    with pytest.raises(EquilibriumDataError, match="time and b0 arrays"):
        import_imas_equilibrium(equilibrium, dd_version="4.1.1")


def test_imas_import_rejects_structures_without_current_schema_or_ids_shape() -> None:
    """Reject objects that do not satisfy the requested IMAS schema."""
    profile = SimpleNamespace(
        grid=SimpleNamespace(dim1=np.array([1.0, 2.0]), dim2=np.array([-1.0, 1.0])),
        psi=np.zeros((2, 2)),
    )
    time_slice = SimpleNamespace(
        time=0.0,
        profiles_2d=[profile],
        global_quantities=SimpleNamespace(ip=1.0),
    )
    equilibrium = SimpleNamespace(
        time_slice=[time_slice],
        time=np.array([0.0]),
        vacuum_toroidal_field=SimpleNamespace(b0=np.array([2.0]), r0=1.5),
    )
    with pytest.raises(EquilibriumDataError, match="required j_phi"):
        import_imas_equilibrium(equilibrium, dd_version="4.1.1")
    with pytest.raises(EquilibriumDataError, match="invalid IMAS-Python equilibrium structure"):
        import_imas_equilibrium(object(), dd_version="4.1.1")


def test_imas_export_translates_backend_validation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate a real IDS validation failure into a backend error."""
    equilibrium = export_imas_equilibrium(_snapshot(), dd_version="4.1.1")

    def reject(_self: object) -> None:
        raise RuntimeError("backend validation failed")

    monkeypatch.setattr(type(equilibrium), "validate", reject)
    with pytest.raises(EquilibriumBackendError, match="rejected exported equilibrium"):
        export_imas_equilibrium(
            _snapshot(),
            equilibrium=equilibrium,
            dd_version="4.1.1",
        )


def test_imas_backend_unavailable_error_is_descriptive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report the exact optional extra when IMAS-Python is unavailable."""
    real_import = importlib.import_module

    def unavailable(name: str, package: str | None = None) -> object:
        if name == "imas":
            raise ImportError("deliberately unavailable")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    with pytest.raises(EquilibriumBackendUnavailableError, match=r"scpn-control\[imas\]"):
        export_imas_equilibrium(_snapshot(), dd_version="4.1.1")


def test_imas_dbentry_translates_and_preserves_errors() -> None:
    """Preserve contract errors and translate external DBEntry failures."""

    class BadVersionEntry:
        dd_version = "5.0.0"

        def put(self, _equilibrium: object, *, occurrence: int) -> None:
            raise AssertionError(occurrence)

    with pytest.raises(EquilibriumDataError, match="major version"):
        write_imas_entry(BadVersionEntry(), _snapshot())

    class FailedWriteEntry:
        dd_version = "4.1.1"

        def put(self, _equilibrium: object, *, occurrence: int) -> None:
            raise OSError(f"write {occurrence} failed")

    with pytest.raises(EquilibriumBackendError, match="DBEntry write failed"):
        write_imas_entry(FailedWriteEntry(), _snapshot())

    class FailedReadEntry:
        dd_version = "4.1.1"
        uri = "memory://failed"

        def get(self, *_args: object, **_kwargs: object) -> object:
            raise OSError("read failed")

    with pytest.raises(EquilibriumBackendError, match="DBEntry read failed"):
        read_imas_entry(FailedReadEntry())

    class InvalidReadEntry(FailedReadEntry):
        def get(self, *_args: object, **_kwargs: object) -> object:
            return object()

    with pytest.raises(EquilibriumDataError, match="invalid IMAS-Python"):
        read_imas_entry(InvalidReadEntry())
