# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nonlinear MPC transport-model tuning error paths
"""Fail-closed and validator branches of the NMPC transport-model tuning surface.

Covers the rollout gradient-audit index parsing and finite-difference tolerance
guards in :mod:`scpn_control.control.nmpc_transport_tuning`, together with the
gradient-contract, shape, and bound checks on the coefficient, single-step
source-schedule, and multi-step source-rollout tuning entry points. The
happy-path and provenance tests live in ``test_nmpc_transport_tuning.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_control.control import nmpc_transport_tuning as tuning_mod
from scpn_control.control.nmpc_transport_tuning import (
    _audit_transport_rollout_source_gradients,
    _rollout_audit_indices,
)
from scpn_control.core import differentiable_transport as transport_mod

# ── rollout audit index parsing ───────────────────────────────────────


class TestRolloutAuditIndices:
    def test_default_indices_for_valid_shape(self):
        indices = _rollout_audit_indices((4, 4, 6), None)
        assert all(0 <= s < 4 and 0 <= c < 4 and 0 <= r < 6 for s, c, r in indices)
        assert len(set(indices)) == len(indices)

    def test_rejects_non_three_dimensional_shape(self):
        with pytest.raises(ValueError, match=r"shape \(n_steps, 4, n_rho\)"):
            _rollout_audit_indices((4, 6), None)

    def test_rejects_wrong_channel_or_radius_count(self):
        with pytest.raises(ValueError, match="n_rho >= 3"):
            _rollout_audit_indices((1, 4, 2), None)

    def test_rejects_non_iterable_sample_indices(self):
        with pytest.raises(ValueError, match="iterable of three-part indices"):
            _rollout_audit_indices((2, 4, 5), 7)

    def test_rejects_non_iterable_index_entry(self):
        with pytest.raises(ValueError, match="iterable three-part indices"):
            _rollout_audit_indices((2, 4, 5), [7])

    def test_rejects_index_with_wrong_arity(self):
        with pytest.raises(ValueError, match="three-part indices"):
            _rollout_audit_indices((2, 4, 5), [(0, 1)])

    def test_rejects_out_of_range_index(self):
        with pytest.raises(ValueError, match="out-of-range rollout source index"):
            _rollout_audit_indices((2, 4, 5), [(99, 0, 0)])

    def test_rejects_empty_index_set(self):
        with pytest.raises(ValueError, match="at least one index"):
            _rollout_audit_indices((2, 4, 5), [])


class TestRolloutAuditTolerances:
    def test_rejects_nonpositive_epsilon(self):
        dummy = np.zeros((1, 4, 3))
        with pytest.raises(ValueError, match="gradient_audit_epsilon must be positive"):
            _audit_transport_rollout_source_gradients(
                np.zeros((4, 3)),
                np.zeros((4, 3)),
                dummy,
                np.zeros((1, 4, 3)),
                np.linspace(0.1, 1.0, 3),
                1.0e-3,
                np.zeros(4),
                dummy,
                weights=None,
                epsilon=0.0,
                tolerance=1.0e-3,
                sample_indices=None,
            )

    def test_rejects_nonpositive_tolerance(self):
        dummy = np.zeros((1, 4, 3))
        with pytest.raises(ValueError, match="gradient_audit_tolerance must be positive"):
            _audit_transport_rollout_source_gradients(
                np.zeros((4, 3)),
                np.zeros((4, 3)),
                dummy,
                np.zeros((1, 4, 3)),
                np.linspace(0.1, 1.0, 3),
                1.0e-3,
                np.zeros(4),
                dummy,
                weights=None,
                epsilon=1.0e-5,
                tolerance=-1.0,
                sample_indices=None,
            )


# ── JAX transport tuning entry points ─────────────────────────────────


def _coeff_case() -> tuple[np.ndarray, ...]:
    rho = np.linspace(0.05, 1.0, 24)
    profiles = np.stack(
        [
            8.0 * np.exp(-((rho - 0.35) ** 2) / 0.03) + 0.2,
            6.0 * np.exp(-((rho - 0.42) ** 2) / 0.04) + 0.2,
            4.0 + 0.8 * (1.0 - rho**2),
            0.03 + 0.02 * np.exp(-((rho - 0.65) ** 2) / 0.02),
        ]
    )
    chi = np.stack([0.2 + 0.02 * rho, 0.16 + 0.02 * rho, 0.04 + 0.005 * rho, 0.012 + 0.001 * rho])
    sources = np.zeros_like(profiles)
    target = profiles.copy()
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    return profiles, chi, sources, target, rho, edge_values


class TestCoefficientTuningGuards:
    def test_rejects_nonpositive_learning_rate(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            tuning_mod.tune_transport_coefficients_for_tracking(
                profiles, chi, sources, target, rho, 1.0e-3, edge, learning_rate=0.0
            )

    def test_rejects_negative_chi_min(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="chi_min must be non-negative"):
            tuning_mod.tune_transport_coefficients_for_tracking(
                profiles, chi, sources, target, rho, 1.0e-3, edge, learning_rate=0.05, chi_min=-1.0
            )

    def test_rejects_nonpositive_max_fractional_update(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="max_fractional_update must be positive"):
            tuning_mod.tune_transport_coefficients_for_tracking(
                profiles,
                chi,
                sources,
                target,
                rho,
                1.0e-3,
                edge,
                learning_rate=0.05,
                max_fractional_update=0.0,
            )

    def test_rejects_non_two_dimensional_chi(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="chi must be a finite non-negative two-dimensional"):
            tuning_mod.tune_transport_coefficients_for_tracking(
                profiles, np.ones(4), sources, target, rho, 1.0e-3, edge, learning_rate=0.05
            )

    def test_rejects_gradient_shape_mismatch(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        monkeypatch.setattr(tuning_mod, "transport_loss_gradient", lambda *a, **k: (0.1, np.zeros((2, 2))))
        with pytest.raises(ValueError, match="transport gradient must be finite and match chi shape"):
            tuning_mod.tune_transport_coefficients_for_tracking(
                profiles, chi, sources, target, rho, 1.0e-3, edge, learning_rate=0.05
            )

    def test_unbounded_fractional_update_path_runs_without_clip(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        gradient = 0.01 * np.ones_like(chi)
        audit = transport_mod.TransportGradientAudit(
            loss=0.1,
            epsilon=1.0e-5,
            tolerance=5.0e-4,
            checked_indices=((0, 0),),
            chi_max_abs_error=1.0e-9,
            source_max_abs_error=1.0e-9,
            passed=True,
        )
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        monkeypatch.setattr(tuning_mod, "transport_loss_gradient", lambda *a, **k: (0.1, gradient))
        monkeypatch.setattr(tuning_mod, "assert_transport_parameter_gradients_consistent", lambda *a, **k: audit)

        result = tuning_mod.tune_transport_coefficients_for_tracking(
            profiles,
            chi,
            sources,
            target,
            rho,
            1.0e-3,
            edge,
            learning_rate=1.0,
            max_fractional_update=None,
        )
        np.testing.assert_allclose(result.updated_chi, np.maximum(0.0, chi - gradient))


class TestSourceScheduleTuningGuards:
    def test_rejects_nonpositive_learning_rate(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            tuning_mod.tune_transport_sources_for_tracking(
                profiles, chi, sources, target, rho, 1.0e-3, edge, learning_rate=0.0
            )

    def test_rejects_nonpositive_max_absolute_update(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="max_absolute_update must be positive"):
            tuning_mod.tune_transport_sources_for_tracking(
                profiles,
                chi,
                sources,
                target,
                rho,
                1.0e-3,
                edge,
                learning_rate=0.05,
                max_absolute_update=0.0,
            )

    def test_rejects_non_two_dimensional_sources(self, monkeypatch):
        profiles, chi, _sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="sources must be a finite two-dimensional"):
            tuning_mod.tune_transport_sources_for_tracking(
                profiles, chi, np.zeros(4), target, rho, 1.0e-3, edge, learning_rate=0.05
            )

    def test_rejects_non_dataclass_gradient_result(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        monkeypatch.setattr(tuning_mod, "transport_parameter_gradients", lambda *a, **k: object())
        with pytest.raises(ValueError, match="must return TransportParameterGradients"):
            tuning_mod.tune_transport_sources_for_tracking(
                profiles, chi, sources, target, rho, 1.0e-3, edge, learning_rate=0.05
            )

    def test_rejects_source_gradient_shape_mismatch(self, monkeypatch):
        profiles, chi, sources, target, rho, edge = _coeff_case()
        result = transport_mod.TransportParameterGradients(
            loss=0.1, chi_gradient=np.zeros_like(chi), source_gradient=np.zeros((2, 2))
        )
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        monkeypatch.setattr(tuning_mod, "transport_parameter_gradients", lambda *a, **k: result)
        with pytest.raises(ValueError, match="source gradient must be finite and match source shape"):
            tuning_mod.tune_transport_sources_for_tracking(
                profiles, chi, sources, target, rho, 1.0e-3, edge, learning_rate=0.05
            )


class TestSourceRolloutTuningGuards:
    def _args(self) -> dict[str, Any]:
        return {
            "initial_profiles": np.zeros((4, 5)),
            "chi": np.zeros((4, 5)),
            "source_sequence": np.zeros((3, 4, 5)),
            "target_history": np.zeros((3, 4, 5)),
            "rho": np.linspace(0.1, 1.0, 5),
            "dt": 1.0e-3,
            "edge_values": np.zeros(4),
        }

    def _call(self, args: dict[str, Any], **kw: Any) -> Any:
        return tuning_mod.tune_transport_source_rollout_for_tracking(
            args["initial_profiles"],
            args["chi"],
            args["source_sequence"],
            args["target_history"],
            args["rho"],
            args["dt"],
            args["edge_values"],
            **kw,
        )

    def test_rejects_nonpositive_learning_rate(self, monkeypatch):
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            self._call(self._args(), learning_rate=0.0)

    def test_rejects_nonpositive_max_absolute_update(self, monkeypatch):
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="max_absolute_update must be positive"):
            self._call(self._args(), learning_rate=0.05, max_absolute_update=0.0)

    def test_rejects_inverted_source_bounds(self, monkeypatch):
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        with pytest.raises(ValueError, match="source_min entries must be less than or equal"):
            self._call(self._args(), learning_rate=0.05, source_min=1.0, source_max=0.0)

    def test_rejects_non_dataclass_gradient_result(self, monkeypatch):
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        monkeypatch.setattr(tuning_mod, "transport_rollout_source_gradients", lambda *a, **k: object())
        with pytest.raises(ValueError, match="must return TransportRolloutSourceGradients"):
            self._call(self._args(), learning_rate=0.05)

    def test_rejects_source_gradient_shape_mismatch(self, monkeypatch):
        result = transport_mod.TransportRolloutSourceGradients(
            loss=0.1, source_gradient=np.zeros((2, 2)), final_profiles=np.zeros((4, 5))
        )
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        monkeypatch.setattr(tuning_mod, "transport_rollout_source_gradients", lambda *a, **k: result)
        with pytest.raises(ValueError, match="rollout source gradient must be finite and match"):
            self._call(self._args(), learning_rate=0.05)

    def test_rejects_final_profile_shape_mismatch(self, monkeypatch):
        result = transport_mod.TransportRolloutSourceGradients(
            loss=0.1, source_gradient=np.zeros((3, 4, 5)), final_profiles=np.zeros((2, 2))
        )
        monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
        monkeypatch.setattr(tuning_mod, "transport_rollout_source_gradients", lambda *a, **k: result)
        with pytest.raises(ValueError, match="rollout final profiles must be finite and match"):
            self._call(self._args(), learning_rate=0.05)
