# ──────────────────────────────────────────────────────────────────────
# SCPN Control — IMAS/OMAS Round-Trip Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Round-trip tests using real omas ODS objects (not mocks)."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.imas_adapter import (
    EquilibriumIDS,
    from_omas,
    to_kernel_arrays,
    to_omas,
)

try:
    from omas import ODS

    HAS_OMAS = True
except ImportError:
    HAS_OMAS = False

pytestmark = pytest.mark.skipif(not HAS_OMAS, reason="omas not installed")

NR, NZ = 33, 65
NRHO = 50
RNG = np.random.default_rng(0)


def _iter_like_ids() -> EquilibriumIDS:
    """Synthetic ITER-like equilibrium for round-trip tests."""
    r = np.linspace(4.0, 8.4, NR)
    z = np.linspace(-4.6, 4.6, NZ)
    rr, zz = np.meshgrid(r, z)
    # Solov'ev-ish flux: psi ~ (R-R0)^2 + Z^2
    psi = (rr - 6.2) ** 2 + (zz / 2.0) ** 2
    j_tor = -4.0 * np.ones_like(psi)
    return EquilibriumIDS(
        r=r,
        z=z,
        psi=psi,
        j_tor=j_tor,
        ip=15e6,
        b0=5.3,
        r0=6.2,
        time=100.0,
    )


# ── equilibrium round-trip via adapter ──────────────────────────────


class TestEquilibriumRoundTrip:
    """Export EquilibriumIDS → ODS → re-import and verify."""

    def test_psi_fidelity(self):
        original = _iter_like_ids()
        ods = to_omas(original)
        recovered = from_omas(ods)
        np.testing.assert_allclose(recovered.psi, original.psi)

    def test_grid_fidelity(self):
        original = _iter_like_ids()
        ods = to_omas(original)
        recovered = from_omas(ods)
        np.testing.assert_allclose(recovered.r, original.r)
        np.testing.assert_allclose(recovered.z, original.z)

    def test_j_tor_fidelity(self):
        original = _iter_like_ids()
        ods = to_omas(original)
        recovered = from_omas(ods)
        np.testing.assert_allclose(recovered.j_tor, original.j_tor)

    def test_scalar_fields(self):
        original = _iter_like_ids()
        ods = to_omas(original)
        recovered = from_omas(ods)
        assert recovered.ip == original.ip
        assert recovered.r0 == original.r0
        assert recovered.time == original.time

    def test_to_kernel_arrays_after_round_trip(self):
        """Adapter → ODS → adapter → kernel arrays: shapes and values survive."""
        original = _iter_like_ids()
        ods = to_omas(original)
        recovered = from_omas(ods)
        arrays = to_kernel_arrays(recovered)
        assert arrays["Psi"].shape == (NZ, NR)
        np.testing.assert_allclose(arrays["Psi"], original.psi)
        np.testing.assert_allclose(arrays["R"], original.r)
        np.testing.assert_allclose(arrays["Z"], original.z)


# ── core_profiles IDS round-trip (pure omas, not adapter) ──────────


class TestCoreProfilesRoundTrip:
    """Write plasma profiles into core_profiles IDS and read back."""

    @staticmethod
    def _write_profiles(ods: ODS) -> dict:
        rho = np.linspace(0.0, 1.0, NRHO)
        te = 10e3 * (1.0 - rho**2)  # eV, parabolic
        ti = 8e3 * (1.0 - rho**2)
        ne = 1e20 * (1.0 - 0.8 * rho**2)  # m^-3
        q = 1.0 + 2.0 * rho**2

        cp = ods["core_profiles"]["profiles_1d"][0]
        cp["grid"]["rho_tor_norm"] = rho
        cp["electrons"]["temperature"] = te
        cp["electrons"]["density"] = ne
        cp["ion"][0]["temperature"] = ti
        cp["q"] = q
        return {"rho": rho, "te": te, "ti": ti, "ne": ne, "q": q}

    def test_te_round_trip(self):
        ods = ODS()
        ref = self._write_profiles(ods)
        got = np.asarray(ods["core_profiles"]["profiles_1d"][0]["electrons"]["temperature"])
        np.testing.assert_allclose(got, ref["te"])

    def test_ti_round_trip(self):
        ods = ODS()
        ref = self._write_profiles(ods)
        got = np.asarray(ods["core_profiles"]["profiles_1d"][0]["ion"][0]["temperature"])
        np.testing.assert_allclose(got, ref["ti"])

    def test_ne_round_trip(self):
        ods = ODS()
        ref = self._write_profiles(ods)
        got = np.asarray(ods["core_profiles"]["profiles_1d"][0]["electrons"]["density"])
        np.testing.assert_allclose(got, ref["ne"])

    def test_q_round_trip(self):
        ods = ODS()
        ref = self._write_profiles(ods)
        got = np.asarray(ods["core_profiles"]["profiles_1d"][0]["q"])
        np.testing.assert_allclose(got, ref["q"])

    def test_rho_grid_round_trip(self):
        ods = ODS()
        ref = self._write_profiles(ods)
        got = np.asarray(ods["core_profiles"]["profiles_1d"][0]["grid"]["rho_tor_norm"])
        np.testing.assert_allclose(got, ref["rho"])


# ── equilibrium 1D profiles round-trip (pure omas) ─────────────────


class TestEquilibrium1DRoundTrip:
    """Write 1D equilibrium profiles (psi, pressure, q) and read back."""

    def test_pressure_and_q(self):
        ods = ODS()
        rho = np.linspace(0.0, 1.0, NRHO)
        psi_1d = np.linspace(-5.0, -1.0, NRHO)
        pressure = 1e5 * (1.0 - rho**2)
        q = 1.0 + 3.0 * rho**2

        eq1d = ods["equilibrium"]["time_slice"][0]["profiles_1d"]
        eq1d["psi"] = psi_1d
        eq1d["pressure"] = pressure
        eq1d["q"] = q

        np.testing.assert_allclose(
            np.asarray(ods["equilibrium"]["time_slice"][0]["profiles_1d"]["psi"]),
            psi_1d,
        )
        np.testing.assert_allclose(
            np.asarray(ods["equilibrium"]["time_slice"][0]["profiles_1d"]["pressure"]),
            pressure,
        )
        np.testing.assert_allclose(
            np.asarray(ods["equilibrium"]["time_slice"][0]["profiles_1d"]["q"]),
            q,
        )


# ── edge cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_ods_has_no_equilibrium_data(self):
        ods = ODS()
        assert "equilibrium" not in ods or len(ods["equilibrium"]["time_slice"]) == 0

    def test_single_cell_grid(self):
        """1x1 grid survives the adapter round-trip."""
        ids = EquilibriumIDS(
            r=np.array([6.2]),
            z=np.array([0.0]),
            psi=np.array([[0.5]]),
            j_tor=np.array([[1.0]]),
            ip=1e6,
            b0=5.0,
            r0=6.2,
            time=0.0,
        )
        ods = to_omas(ids)
        recovered = from_omas(ods)
        np.testing.assert_allclose(recovered.psi, ids.psi)
        assert recovered.ip == ids.ip

    def test_multiple_time_slices(self):
        """Write two time slices, read each back independently."""
        ods = ODS()
        for idx, t in enumerate([0.0, 1.0]):
            r = np.linspace(4.0, 8.0, 5)
            z = np.linspace(-3.0, 3.0, 7)
            eq = ods["equilibrium"]["time_slice"][idx]
            eq["time"] = t
            eq["global_quantities"]["ip"] = 15e6 + idx * 1e6
            eq["global_quantities"]["magnetic_axis"]["r"] = 6.2
            p2d = eq["profiles_2d"][0]
            p2d["grid"]["dim1"] = r
            p2d["grid"]["dim2"] = z
            psi = RNG.standard_normal((7, 5))
            p2d["psi"] = psi
            p2d["j_tor"] = np.zeros_like(psi)

        ids0 = from_omas(ods, time_index=0)
        ids1 = from_omas(ods, time_index=1)
        assert ids0.time == 0.0
        assert ids1.time == 1.0
        assert ids1.ip == 16e6

    def test_large_grid_performance(self):
        """256x512 grid round-trips without error."""
        nr, nz = 256, 512
        r = np.linspace(3.0, 9.0, nr)
        z = np.linspace(-6.0, 6.0, nz)
        psi = RNG.standard_normal((nz, nr))
        ids = EquilibriumIDS(
            r=r,
            z=z,
            psi=psi,
            j_tor=np.zeros_like(psi),
            ip=15e6,
            b0=5.3,
            r0=6.2,
        )
        ods = to_omas(ids)
        recovered = from_omas(ods)
        assert recovered.psi.shape == (nz, nr)
        np.testing.assert_allclose(recovered.psi, psi)


# ── solver state export/import ──────────────────────────────────────


class TestSolverInterop:
    """Verify adapter can bridge solver state ↔ IMAS format."""

    def test_kernel_arrays_to_omas_and_back(self):
        """Simulate solver output → EquilibriumIDS → ODS → EquilibriumIDS → kernel arrays."""
        ids = _iter_like_ids()
        arrays_before = to_kernel_arrays(ids)

        ods = to_omas(ids)
        ids_after = from_omas(ods)
        arrays_after = to_kernel_arrays(ids_after)

        for key in ("R", "Z", "Psi", "J_phi"):
            np.testing.assert_allclose(
                arrays_after[key],
                arrays_before[key],
                err_msg=f"Mismatch in {key}",
            )

    def test_omas_equilibrium_into_solver_init(self):
        """ODS equilibrium → adapter → dict suitable for kernel initialisation."""
        ods = ODS()
        r = np.linspace(4.0, 8.4, NR)
        z = np.linspace(-4.6, 4.6, NZ)
        rr, zz = np.meshgrid(r, z)
        psi = (rr - 6.2) ** 2 + (zz / 2.0) ** 2

        eq = ods["equilibrium"]["time_slice"][0]
        eq["time"] = 50.0
        eq["global_quantities"]["ip"] = 15e6
        eq["global_quantities"]["magnetic_axis"]["r"] = 6.2
        p2d = eq["profiles_2d"][0]
        p2d["grid"]["dim1"] = r
        p2d["grid"]["dim2"] = z
        p2d["psi"] = psi
        p2d["j_tor"] = -4.0 * np.ones_like(psi)

        ids = from_omas(ods)
        arrays = to_kernel_arrays(ids)

        assert arrays["R"].shape == (NR,)
        assert arrays["Z"].shape == (NZ,)
        assert arrays["Psi"].shape == (NZ, NR)
        assert arrays["ip"] == 15e6
        assert arrays["r0"] == 6.2
        np.testing.assert_allclose(arrays["Psi"], psi)
