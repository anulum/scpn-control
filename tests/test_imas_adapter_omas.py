# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — real OMAS equilibrium-adapter contract tests.

"""Exercise the neutral equilibrium contract against real OMAS ODS objects."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

omas = pytest.importorskip(
    "omas",
    reason="real OMAS tests execute in the Ubuntu 3.12 fusion-data CI lane",
)
ODS = omas.ODS
omas_environment = omas.omas_environment

from scpn_control.core.imas_adapter import (
    EquilibriumBackendError,
    EquilibriumDataError,
    EquilibriumSnapshot,
    export_omas_equilibrium,
    import_omas_equilibrium,
    snapshot_to_kernel_state,
)


def _snapshot(*, time_s: float = 0.5, field_t: float = -2.3, **updates: object) -> EquilibriumSnapshot:
    r_m = np.array([1.1, 1.45, 1.9, 2.35])
    z_m = np.array([-0.9, -0.1, 0.8])
    rr, zz = np.meshgrid(r_m, z_m)
    values: dict[str, object] = {
        "r_m": r_m,
        "z_m": z_m,
        "psi_wb_per_rad": 0.08 * rr**2 - 0.04 * zz**2 + 0.02 * rr * zz,
        "j_phi_a_per_m2": -5.0e5 + 7.0e4 * rr - 3.0e4 * zz,
        "plasma_current_a": -9.5e5,
        "vacuum_toroidal_field_t": field_t,
        "vacuum_field_reference_radius_m": 1.68,
        "time_s": time_s,
        "source_backend": "memory",
        "source": "tests/real-omas-non-square",
    }
    values.update(updates)
    return EquilibriumSnapshot(**cast(Any, values))


def test_real_omas_round_trip_preserves_solver_convention() -> None:
    """Round-trip a real ODS while preserving the solver convention."""
    original = _snapshot()
    ods = export_omas_equilibrium(original)
    assert isinstance(ods, ODS)
    assert ods.imas_version == "3.41.0"
    assert ods.cocos == 11

    with omas_environment(ods, cocosio=11):
        stored_psi = np.asarray(ods["equilibrium.time_slice.0.profiles_2d.0.psi"])
    np.testing.assert_allclose(stored_psi, original.psi_wb_per_rad.T * (2.0 * np.pi))

    recovered = import_omas_equilibrium(ods)
    assert recovered.source_backend == "omas"
    assert recovered.data_dictionary_version == "3.41.0"
    assert recovered.cocos == 1
    np.testing.assert_allclose(recovered.r_m, original.r_m)
    np.testing.assert_allclose(recovered.z_m, original.z_m)
    np.testing.assert_allclose(recovered.psi_wb_per_rad, original.psi_wb_per_rad)
    np.testing.assert_allclose(recovered.j_phi_a_per_m2, original.j_phi_a_per_m2)
    assert recovered.plasma_current_a == original.plasma_current_a
    assert recovered.vacuum_toroidal_field_t == original.vacuum_toroidal_field_t
    assert recovered.vacuum_field_reference_radius_m == original.vacuum_field_reference_radius_m


def test_real_omas_multiple_time_slices_and_external_cocosio() -> None:
    """Preserve multiple slices when an external caller changes COCOS I/O."""
    first = _snapshot(time_s=0.2, field_t=-2.1)
    second = _snapshot(time_s=0.7, field_t=-2.4)
    ods = export_omas_equilibrium(first)
    ods = export_omas_equilibrium(second, ods=ods, time_index=1)
    ods.cocosio = 11

    recovered_first = import_omas_equilibrium(ods, time_index=0)
    recovered_second = import_omas_equilibrium(ods, time_index=1, source="omas:tests:second")
    assert recovered_first.time_s == 0.2
    assert recovered_first.vacuum_toroidal_field_t == -2.1
    assert recovered_second.time_s == 0.7
    assert recovered_second.vacuum_toroidal_field_t == -2.4
    assert recovered_second.source == "omas:tests:second"
    np.testing.assert_allclose(recovered_second.psi_wb_per_rad, second.psi_wb_per_rad)


def test_real_omas_to_kernel_state_preserves_orientation() -> None:
    """Preserve solver field orientation through ODS and kernel conversion."""
    original = _snapshot()
    state = snapshot_to_kernel_state(import_omas_equilibrium(export_omas_equilibrium(original)))
    assert state["Psi"].shape == (original.z_m.size, original.r_m.size)
    np.testing.assert_allclose(state["Psi"], original.psi_wb_per_rad)
    np.testing.assert_allclose(state["J_phi"], original.j_phi_a_per_m2)


def test_omas_import_can_preserve_explicitly_absent_current_density() -> None:
    """Distinguish permitted absent current from a required field failure."""
    original = _snapshot()
    ods = export_omas_equilibrium(original)
    del ods["equilibrium.time_slice.0.profiles_2d.0.j_tor"]
    recovered = import_omas_equilibrium(ods, require_current_density=False)
    assert recovered.j_phi_a_per_m2 is None
    with pytest.raises(EquilibriumDataError, match="j_tor"):
        import_omas_equilibrium(ods)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("time", "time_slice time"),
        ("version", "Data Dictionary v3"),
    ],
)
def test_omas_import_rejects_ambiguous_metadata(mutation: str, message: str) -> None:
    """Reject inconsistent time and unsupported ODS schema metadata."""
    ods = export_omas_equilibrium(_snapshot())
    if mutation == "time":
        ods["equilibrium.time_slice.0.time"] = 8.0
    else:
        ods.imas_version = "4.1.1"
    with pytest.raises(EquilibriumDataError, match=message):
        import_omas_equilibrium(ods)


def test_omas_export_rejects_version_mismatch_and_index_gap() -> None:
    """Reject ODS version mismatch and nonsequential slice insertion."""
    ods = export_omas_equilibrium(_snapshot())
    with pytest.raises(EquilibriumDataError, match="does not match ODS version"):
        export_omas_equilibrium(_snapshot(), ods=ods, imas_version="3.40.0")
    with pytest.raises(EquilibriumBackendError, match="OMAS rejected exported equilibrium"):
        export_omas_equilibrium(_snapshot(), ods=ods, time_index=3)


def test_omas_export_rejects_unsupported_schema_and_omits_absent_current() -> None:
    """Reject DD v4 and preserve an explicitly absent current-density field."""
    with pytest.raises(EquilibriumDataError, match="Data Dictionary v3"):
        export_omas_equilibrium(_snapshot(), imas_version="4.1.1")

    ods = export_omas_equilibrium(_snapshot(j_phi_a_per_m2=None))
    assert "equilibrium.time_slice.0.profiles_2d.0.j_tor" not in ods

    overwritten = export_omas_equilibrium(_snapshot())
    overwritten = export_omas_equilibrium(_snapshot(j_phi_a_per_m2=None), ods=overwritten)
    assert "equilibrium.time_slice.0.profiles_2d.0.j_tor" not in overwritten


def test_omas_import_rejects_missing_real_ods_structure() -> None:
    """Translate missing required ODS paths into the neutral data error."""
    with pytest.raises(EquilibriumDataError, match="invalid OMAS equilibrium structure"):
        import_omas_equilibrium(ODS())
