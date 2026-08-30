# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — neutral equilibrium-data and compatibility tests.

"""Verify the backend-neutral equilibrium contract and legacy transition API."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from scpn_control.core.eqdsk import GEqdsk, write_geqdsk
from scpn_control.core.imas_adapter import (
    EquilibriumBackendUnavailableError,
    EquilibriumDataError,
    EquilibriumIDS,
    EquilibriumSnapshot,
    export_omas_equilibrium,
    from_geqdsk,
    from_kernel,
    from_omas,
    snapshot_from_geqdsk,
    snapshot_from_kernel,
    snapshot_to_kernel_state,
    to_kernel_arrays,
    to_omas,
)


def _snapshot(**updates: object) -> EquilibriumSnapshot:
    r_m = np.array([1.0, 1.6, 2.4])
    z_m = np.array([-0.7, 0.2])
    defaults: dict[str, object] = {
        "r_m": r_m,
        "z_m": z_m,
        "psi_wb_per_rad": np.arange(6, dtype=np.float64).reshape(2, 3) / 10.0,
        "j_phi_a_per_m2": np.arange(6, dtype=np.float64).reshape(2, 3) * 1.0e4,
        "plasma_current_a": 1.2e6,
        "vacuum_toroidal_field_t": -2.2,
        "vacuum_field_reference_radius_m": 1.7,
        "time_s": 0.4,
        "source_backend": "memory",
        "source": "tests/neutral-snapshot",
    }
    defaults.update(updates)
    return EquilibriumSnapshot(**cast(Any, defaults))


def test_snapshot_copies_and_freezes_arrays() -> None:
    """Ensure snapshots own immutable copies of caller arrays."""
    r_m = np.array([1.0, 1.6, 2.4])
    snapshot = _snapshot(r_m=r_m)
    r_m[0] = 99.0
    assert snapshot.r_m[0] == 1.0
    assert not snapshot.r_m.flags.writeable
    assert not snapshot.psi_wb_per_rad.flags.writeable
    assert snapshot.cocos == 1


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"r_m": [[1.0, 2.0]]}, "one-dimensional"),
        ({"r_m": [1.0]}, "at least two"),
        ({"r_m": [1.0, np.inf]}, "finite"),
        ({"r_m": [0.0, 1.0]}, "positive"),
        ({"r_m": [1.0, 1.0]}, "strictly increasing"),
        ({"z_m": [0.2, -0.7]}, "strictly increasing"),
        ({"psi_wb_per_rad": np.zeros((3, 2))}, r"solver \(Z, R\) shape"),
        ({"psi_wb_per_rad": [[0.0, 1.0, 2.0], [3.0, np.nan, 5.0]]}, "finite"),
        ({"j_phi_a_per_m2": np.zeros((3, 2))}, r"solver \(Z, R\) shape"),
        ({"plasma_current_a": "bad"}, "finite scalar"),
        ({"vacuum_toroidal_field_t": np.inf}, "finite scalar"),
        ({"vacuum_toroidal_field_t": 0.0}, "non-zero"),
        ({"vacuum_field_reference_radius_m": -1.0}, "positive"),
        ({"time_s": np.nan}, "finite scalar"),
        ({"cocos": 11}, "COCOS 1"),
        ({"source_backend": "unknown"}, "unsupported source_backend"),
        ({"source": ""}, "non-empty provenance"),
        ({"data_dictionary_version": ""}, "non-empty string"),
    ],
)
def test_snapshot_rejects_invalid_domains(updates: dict[str, object], message: str) -> None:
    """Reject malformed grids, fields, scalars, convention, and provenance."""
    with pytest.raises(EquilibriumDataError, match=message):
        _snapshot(**updates)


def test_snapshot_without_current_cannot_enter_kernel() -> None:
    """Require current density when reconstructing a solver state."""
    snapshot = _snapshot(j_phi_a_per_m2=None)
    assert snapshot.j_phi_a_per_m2 is None
    with pytest.raises(EquilibriumDataError, match="requires a toroidal current-density"):
        snapshot_to_kernel_state(snapshot)


def test_kernel_state_is_defensive_and_complete() -> None:
    """Return complete writable copies for the solver boundary."""
    snapshot = _snapshot()
    state = snapshot_to_kernel_state(snapshot)
    state["Psi"][0, 0] = 99.0
    assert snapshot.psi_wb_per_rad[0, 0] != 99.0
    assert state["ip"] == snapshot.plasma_current_a
    assert state["b0"] == snapshot.vacuum_toroidal_field_t
    assert state["r0"] == snapshot.vacuum_field_reference_radius_m


def test_snapshot_from_kernel_requires_and_preserves_metadata() -> None:
    """Preserve explicit machine metadata and solver array orientation."""

    class Kernel:
        R = np.array([1.0, 2.0])
        Z = np.array([-1.0, 0.0, 1.0])
        Psi = np.arange(6, dtype=np.float64).reshape(3, 2)
        J_phi = np.ones((3, 2))
        cfg = {
            "physics": {"plasma_current_target": -9.0e5, "B0": -1.8},
            "dimensions": {"R0": 1.55},
        }

    snapshot = snapshot_from_kernel(Kernel(), time_s=0.3)
    assert snapshot.source_backend == "kernel"
    assert snapshot.plasma_current_a == -9.0e5
    assert snapshot.vacuum_toroidal_field_t == -1.8
    assert snapshot.vacuum_field_reference_radius_m == 1.55
    assert snapshot.time_s == 0.3


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        (type("NoConfig", (), {})(), "kernel.cfg"),
        (type("NoArrays", (), {"cfg": {"physics": {}, "dimensions": {}}})(), "must expose"),
        (
            type(
                "NoCurrent",
                (),
                {
                    "R": np.array([1.0, 2.0]),
                    "Z": np.array([0.0, 1.0]),
                    "Psi": np.zeros((2, 2)),
                    "cfg": {"physics": {}, "dimensions": {}},
                },
            )(),
            "must expose",
        ),
    ],
)
def test_snapshot_from_kernel_rejects_missing_contract(kernel: object, message: str) -> None:
    """Reject kernels that omit configuration or required arrays."""
    with pytest.raises(EquilibriumDataError, match=message):
        snapshot_from_kernel(kernel)


def test_snapshot_from_kernel_rejects_missing_physical_key() -> None:
    """Reject kernels that omit mandatory physical metadata."""

    class Kernel:
        R = np.array([1.0, 2.0])
        Z = np.array([0.0, 1.0])
        Psi = np.zeros((2, 2))
        J_phi = np.zeros((2, 2))
        cfg = {"physics": {"B0": 2.0}, "dimensions": {"R0": 1.5}}

    with pytest.raises(EquilibriumDataError, match="physics.plasma_current_target"):
        snapshot_from_kernel(Kernel())


def test_geqdsk_import_does_not_fabricate_current_density(tmp_path: Path) -> None:
    """Import GEQDSK without inventing an unavailable 2-D current field."""
    nr, nz = 5, 7
    geqdsk = GEqdsk(
        description="contract",
        nw=nr,
        nh=nz,
        rdim=2.0,
        zdim=3.0,
        rcentr=1.7,
        rleft=0.8,
        zmid=0.0,
        rmaxis=1.6,
        zmaxis=0.1,
        simag=-0.8,
        sibry=0.2,
        bcentr=-2.1,
        current=-8.0e5,
        fpol=np.linspace(3.0, 2.0, nr),
        pres=np.linspace(2.0e4, 0.0, nr),
        ffprime=np.linspace(-0.2, 0.0, nr),
        pprime=np.linspace(-2.0e4, 0.0, nr),
        qpsi=np.linspace(1.0, 4.0, nr),
        psirz=np.arange(nr * nz, dtype=np.float64).reshape(nz, nr) / 20.0,
    )
    path = tmp_path / "contract.geqdsk"
    write_geqdsk(geqdsk, str(path))
    snapshot = snapshot_from_geqdsk(path)
    assert snapshot.source_backend == "geqdsk"
    assert snapshot.j_phi_a_per_m2 is None
    assert snapshot.vacuum_toroidal_field_t == -2.1


def test_legacy_facade_warns_and_forwards_backend_neutral_data() -> None:
    """Keep backend-neutral deprecated names truthful without optional extras."""
    with pytest.warns(DeprecationWarning, match="EquilibriumIDS"):
        legacy = EquilibriumIDS(
            r=np.array([1.0, 2.0]),
            z=np.array([-1.0, 1.0]),
            psi=np.zeros((2, 2)),
            j_tor=np.ones((2, 2)),
            ip=1.0e6,
            b0=2.0,
            r0=1.5,
        )
    np.testing.assert_allclose(legacy.r, [1.0, 2.0])
    np.testing.assert_allclose(legacy.z, [-1.0, 1.0])
    np.testing.assert_allclose(legacy.j_tor, np.ones((2, 2)))
    assert legacy.ip == 1.0e6
    assert legacy.b0 == 2.0
    assert legacy.r0 == 1.5
    assert legacy.time == 0.0
    with pytest.warns(DeprecationWarning, match="to_kernel_arrays"):
        assert to_kernel_arrays(legacy)["J_phi"].shape == (2, 2)


def test_legacy_omas_facade_warns_and_forwards_real_omas() -> None:
    """Forward deprecated OMAS names when the optional real backend is present."""
    pytest.importorskip("omas", reason="real OMAS backend is an optional dependency")
    with pytest.warns(DeprecationWarning, match="EquilibriumIDS"):
        legacy = EquilibriumIDS(
            r=np.array([1.0, 2.0]),
            z=np.array([-1.0, 1.0]),
            psi=np.zeros((2, 2)),
            j_tor=np.ones((2, 2)),
            ip=1.0e6,
            b0=2.0,
            r0=1.5,
        )
    with pytest.warns(DeprecationWarning, match="to_omas"):
        ods = to_omas(legacy)
    with pytest.warns(DeprecationWarning, match="from_omas"):
        recovered = from_omas(ods)
    np.testing.assert_allclose(recovered.psi, legacy.psi)


def test_legacy_kernel_and_geqdsk_functions_warn(tmp_path: Path) -> None:
    """Warn for legacy kernel and GEQDSK entry points."""

    class Kernel:
        R = np.array([1.0, 2.0])
        Z = np.array([-1.0, 1.0])
        Psi = np.zeros((2, 2))
        J_phi = np.ones((2, 2))
        cfg = {"physics": {"plasma_current_target": 1.0e6, "B0": 2.0}, "dimensions": {"R0": 1.5}}

    with pytest.warns(DeprecationWarning, match="from_kernel"):
        assert from_kernel(Kernel()).ip == 1.0e6

    path = tmp_path / "empty.geqdsk"
    geqdsk = GEqdsk(
        nw=2,
        nh=2,
        rdim=1.0,
        zdim=1.0,
        rcentr=1.5,
        rleft=1.0,
        bcentr=2.0,
        current=1.0,
        fpol=np.ones(2),
        pres=np.zeros(2),
        ffprime=np.zeros(2),
        pprime=np.zeros(2),
        qpsi=np.ones(2),
        psirz=np.zeros((2, 2)),
    )
    write_geqdsk(geqdsk, str(path))
    with pytest.warns(DeprecationWarning, match="from_geqdsk"):
        assert from_geqdsk(str(path)).j_phi_a_per_m2 is None


def test_missing_backend_is_descriptive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report the exact optional extra when OMAS is unavailable."""
    real_import = importlib.import_module

    def unavailable(name: str, package: str | None = None) -> object:
        if name == "omas":
            raise ImportError("deliberately unavailable")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    with pytest.raises(EquilibriumBackendUnavailableError, match=r"scpn-control\[omas\]"):
        export_omas_equilibrium(_snapshot())
