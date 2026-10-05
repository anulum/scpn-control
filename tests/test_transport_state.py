# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Transport state tests
"""Exercise declared snapshots, atomic restoration and replay through the public solver."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.integrated_transport_solver import TransportSolver


class TestTransportState:
    """Snapshots cover every evolved field, including the momentum sub-solver."""

    @staticmethod
    def _seeded(config_file: Path) -> TransportSolver:
        """Build a multi-ion solver whose ion and electron temperatures differ."""
        ts = TransportSolver(str(config_file), multi_ion=True)
        ts.Ti = 5.0 * (1 - ts.rho**2)
        ts.Te = 3.0 * (1 - ts.rho**2)
        ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
        ts.n_D = 0.5 * ts.ne.copy()
        ts.n_T = 0.5 * ts.ne.copy()
        ts.n_He = np.zeros(ts.nr)
        ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
        ts.update_transport_model(50.0)
        return ts

    def test_capture_restore_round_trips_every_declared_field(self, config_file: Path) -> None:
        """Restoring a snapshot returns each declared field to its captured value."""
        ts = self._seeded(config_file)
        snapshot = ts.capture_evolution_state()
        assert set(snapshot) <= set(TransportSolver.EVOLUTION_STATE_FIELDS) | {TransportSolver.MOMENTUM_ROTATION_KEY}
        ts.evolve_profiles(0.01, 50.0)
        assert np.max(np.abs(ts.Ti - snapshot["Ti"])) > 0.0
        ts.restore_evolution_state(snapshot)
        for name, expected in snapshot.items():
            if name == TransportSolver.MOMENTUM_ROTATION_KEY:
                assert ts._momentum_solver is not None
                np.testing.assert_array_equal(ts._momentum_solver.omega_phi, expected)
                continue
            actual = getattr(ts, name)
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            else:
                assert actual == expected

    def test_restored_snapshot_is_independent_of_later_mutation(self, config_file: Path) -> None:
        """A snapshot survives repeated restores, which a trial sequence performs."""
        ts = self._seeded(config_file)
        snapshot = ts.capture_evolution_state()
        for _ in range(2):
            ts.restore_evolution_state(snapshot)
            ts.evolve_profiles(0.01, 50.0)
        ts.restore_evolution_state(snapshot)
        np.testing.assert_array_equal(ts.Ti, snapshot["Ti"])
        np.testing.assert_array_equal(ts.Te, snapshot["Te"])

    @pytest.mark.parametrize("multi_ion", [True, False])
    def test_a_step_mutates_nothing_outside_the_declared_state(self, config_file: Path, multi_ion: bool) -> None:
        """Every attribute a step changes is one the snapshot covers.

        The rollback is only as complete as its declaration. This compares the
        whole instance before and after a step, the momentum sub-solver
        included, so an evolved quantity added later without being declared
        fails here rather than leaking between Richardson trials.
        """
        ts = self._seeded(config_file)
        if not multi_ion:
            ts = TransportSolver(str(config_file), multi_ion=False)
            ts.Ti = 5.0 * (1 - ts.rho**2)
            ts.Te = 3.0 * (1 - ts.rho**2)
            ts.ne = 8.0 * (1 - ts.rho**2) ** 0.5
            ts.set_neoclassical(R0=6.2, a=2.0, B0=5.3)
            ts.update_transport_model(50.0)

        def frozen(instance: object) -> dict[str, object]:
            return {name: copy.deepcopy(value) for name, value in vars(instance).items()}

        def same(old: object, new: object) -> bool:
            if isinstance(old, np.ndarray) or isinstance(new, np.ndarray):
                return (
                    isinstance(old, np.ndarray)
                    and isinstance(new, np.ndarray)
                    and old.shape == new.shape
                    and bool(np.array_equal(old, new, equal_nan=True))
                )
            if isinstance(old, dict) and isinstance(new, dict):
                return set(old) == set(new) and all(same(old[key], new[key]) for key in old)
            if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
                return len(old) == len(new) and all(same(a, b) for a, b in zip(old, new))
            if hasattr(old, "__dict__") and hasattr(new, "__dict__") and type(old) is type(new):
                return same(vars(old), vars(new))
            return bool(old == new)

        def changed(before: dict[str, object], after: dict[str, object]) -> set[str]:
            return {
                name
                for name in set(before) | set(after)
                if name != "_momentum_solver" and not same(before.get(name), after.get(name))
            }

        outer_before, inner_before = frozen(ts), frozen(ts._momentum_solver)
        ts.evolve_profiles(0.01, 50.0)
        outer_changed = changed(outer_before, frozen(ts))
        inner_changed = changed(inner_before, frozen(ts._momentum_solver))

        assert outer_changed, "the step changed nothing, so the comparison proves nothing"
        assert outer_changed <= set(TransportSolver.EVOLUTION_STATE_FIELDS)
        assert inner_changed == {"omega_phi"}
        assert TransportSolver.MOMENTUM_ROTATION_KEY in ts.capture_evolution_state()

    def test_rollback_returns_the_rotation_the_sub_solver_owns(self, config_file: Path) -> None:
        """The next step after a restore starts from the captured rotation.

        The momentum sub-solver advances its own profile and the outer attribute
        is the array it last returned. Restoring the outer one alone left the
        accepted rotation 3.9 rad/s away from the two half steps on a profile of
        3.7 rad/s.
        """
        reference = self._seeded(config_file)
        reference.evolve_profiles(0.005, 50.0)
        reference.evolve_profiles(0.005, 50.0)

        trial = self._seeded(config_file)
        entry = trial.capture_evolution_state()
        trial.evolve_profiles(0.01, 50.0)
        assert trial._momentum_solver is not None
        assert np.max(np.abs(trial._momentum_solver.omega_phi - entry[TransportSolver.MOMENTUM_ROTATION_KEY])) > 0.0
        trial.restore_evolution_state(entry)
        np.testing.assert_array_equal(trial._momentum_solver.omega_phi, entry[TransportSolver.MOMENTUM_ROTATION_KEY])
        assert trial._momentum_solver.omega_phi is not entry[TransportSolver.MOMENTUM_ROTATION_KEY]
        trial.evolve_profiles(0.005, 50.0)
        trial.evolve_profiles(0.005, 50.0)
        np.testing.assert_array_equal(trial.omega_phi, reference.omega_phi)

    def test_rollback_returns_the_effective_charge(self, config_file: Path) -> None:
        """The effective charge a step recomputes is part of the snapshot."""
        ts = self._seeded(config_file)
        entry = ts.capture_evolution_state()
        assert entry["_Z_eff"] == ts._Z_eff
        ts.evolve_profiles(0.01, 50.0)
        assert ts._Z_eff != entry["_Z_eff"]
        ts.restore_evolution_state(entry)
        assert ts._Z_eff == entry["_Z_eff"]

    def test_a_rotation_snapshot_needs_a_momentum_solver_to_return_to(self, config_file: Path) -> None:
        """A snapshot with a rotation cannot be restored where no sub-solver exists."""
        ts = self._seeded(config_file)
        snapshot = ts.capture_evolution_state()
        bare = TransportSolver(str(config_file), multi_ion=True)
        assert bare._momentum_solver is None
        assert TransportSolver.MOMENTUM_ROTATION_KEY not in bare.capture_evolution_state()
        before = bare.capture_evolution_state()
        with pytest.raises(RuntimeError, match="no momentum solver"):
            bare.restore_evolution_state(snapshot)
        for name, expected in before.items():
            actual = getattr(bare, name)
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            else:
                assert actual == expected

    def test_restore_refuses_a_snapshot_from_another_contract(self, config_file: Path) -> None:
        """An unknown field means the snapshot cannot be trusted to be complete."""
        ts = self._seeded(config_file)
        snapshot = ts.capture_evolution_state()
        snapshot["chi_i"] = ts.chi_i.copy()
        with pytest.raises(KeyError, match="unknown evolution fields"):
            ts.restore_evolution_state(snapshot)

    def test_capture_omits_a_profile_removed_by_external_mutation(self, config_file: Path) -> None:
        """Snapshot creation leaves an absent field absent, as documented."""
        ts = TransportSolver(str(config_file))
        del ts.Ti
        snapshot = ts.capture_evolution_state()
        assert "Ti" not in snapshot
        ts.restore_evolution_state(snapshot)
        assert not hasattr(ts, "Ti")

    def test_unconfigured_solver_restores_without_momentum_state(self, config_file: Path) -> None:
        """A solver without a momentum sub-solver can replay its own state."""
        ts = TransportSolver(str(config_file))
        snapshot = ts.capture_evolution_state()
        assert TransportSolver.MOMENTUM_ROTATION_KEY not in snapshot
        ts.Ti[:] = 7.0
        ts.restore_evolution_state(snapshot)
        np.testing.assert_array_equal(ts.Ti, snapshot["Ti"])
