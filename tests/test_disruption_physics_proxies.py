# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for disruption physics proxies leaf

"""Drive production disruption physics-proxy leaf contracts."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_control.control.disruption_physics_proxies as leaf
import scpn_control.control.disruption_predictor as owner


def test_owner_proxy_symbols_bind_to_leaf() -> None:
    """Owner re-exports are the production physics-proxy leaf callables."""
    assert owner.simulate_tearing_mode is leaf.simulate_tearing_mode
    assert owner.build_disruption_feature_vector is leaf.build_disruption_feature_vector
    assert owner.predict_disruption_risk is leaf.predict_disruption_risk
    assert owner.disruption_warning_time is leaf.disruption_warning_time


def test_tearing_mode_and_risk_pipeline() -> None:
    """Synthetic NTM trajectory yields a finite feature vector and risk score."""
    signal, label, ttd = leaf.simulate_tearing_mode(200, rng=np.random.default_rng(0))
    assert signal.shape == (200,)
    assert label in (0, 1)
    features = leaf.build_disruption_feature_vector(signal)
    assert features.ndim == 1
    assert np.all(np.isfinite(features))
    risk = leaf.predict_disruption_risk(signal)
    assert 0.0 <= risk <= 1.0
    with pytest.raises((ValueError, TypeError)):
        leaf.simulate_tearing_mode(10.5)  # type: ignore[arg-type]  # refused input under test


def test_tearing_modes_density_limit_and_vde() -> None:
    """Density-limit and VDE synthetic modes can trip disruption thresholds."""
    for mode in ("density_limit", "vde"):
        signal, label, ttd = leaf.simulate_tearing_mode(1000, mode=mode, rng=np.random.default_rng(0))
        assert label == 1
        assert ttd >= 0
        assert signal.size < 1000
        assert np.all(np.isfinite(signal))
    with pytest.raises(ValueError, match="Unknown disruption mode"):
        leaf.simulate_tearing_mode(50, mode="not_a_mode")


@pytest.mark.parametrize("mode", ["density_limit", "vde"])
def test_short_synthetic_trajectory_retains_safe_label(mode: str) -> None:
    """A short simulated trajectory ends without inventing a disruption."""
    signal, label, ttd = owner.simulate_tearing_mode(1, mode=mode, rng=np.random.default_rng(0))
    assert signal.shape == (1,)
    assert label == 0
    assert ttd == -1


def test_warning_time_and_empty_signal_guards() -> None:
    """Warning-time returns 0 without alarm; feature builder rejects empty signals."""
    quiet = np.ones(20, dtype=float) * 0.01
    assert leaf.disruption_warning_time(quiet, risk_threshold=0.99, dt=0.001) == 0.0
    with pytest.raises(ValueError):
        leaf.build_disruption_feature_vector(np.array([]))


def test_public_risk_rejects_finite_inputs_that_overflow_the_logit() -> None:
    """Large finite toroidal inputs must not silently saturate an invalid score."""
    with pytest.raises(ValueError, match="logit"):
        owner.predict_disruption_risk(
            np.ones(8),
            {"toroidal_n1_amp": 1.7e308, "toroidal_asymmetry_index": 0.0},
        )


@pytest.mark.parametrize(
    "field",
    ["toroidal_n1_amp", "toroidal_n2_amp", "toroidal_n3_amp", "toroidal_asymmetry_index", "toroidal_radial_spread"],
)
def test_public_risk_rejects_negative_toroidal_observables(field: str) -> None:
    """Negative amplitude, asymmetry or spread cannot lower the risk score."""
    with pytest.raises(ValueError, match="non-negative"):
        owner.predict_disruption_risk(np.ones(8), {field: -1.0})


def test_public_risk_rejects_unknown_or_null_toroidal_observables() -> None:
    """A misspelled key or null field cannot silently become zero."""
    with pytest.raises(ValueError, match="unknown"):
        owner.predict_disruption_risk(np.ones(8), {"toroidal_n1_amplitude": 0.8})
    with pytest.raises(ValueError, match="numeric"):
        owner.predict_disruption_risk(np.ones(8), {"toroidal_n1_amp": None})  # type: ignore[dict-item]  # refused input under test


@pytest.mark.parametrize(
    ("signal", "threshold", "dt"),
    [
        ([], 0.5, 0.001),
        ([0.1, float("nan")], 0.5, 0.001),
        ([0.1], float("nan"), 0.001),
        ([0.1], -0.1, 0.001),
        ([0.1], 1.1, 0.001),
        ([0.1], 0.5, -0.001),
        ([0.1], 0.5, 0.0),
    ],
)
def test_public_warning_time_rejects_invalid_inputs(signal: list[float], threshold: float, dt: float) -> None:
    """The public warning estimate refuses invalid signals and time semantics."""
    with pytest.raises(ValueError):
        owner.disruption_warning_time(signal, risk_threshold=threshold, dt=dt)


def test_public_warning_time_refuses_overflowed_duration() -> None:
    """Even a finite sampling interval can overflow the returned duration."""
    with pytest.raises(ValueError, match="warning time"):
        owner.disruption_warning_time([0.1, 0.1, 0.1], risk_threshold=0.0, dt=1e308)


@pytest.mark.parametrize("field", ["risk_threshold", "dt"])
@pytest.mark.parametrize("value", [True, np.True_, "0.5"])
def test_public_warning_time_refuses_non_numeric_units(field: str, value: object) -> None:
    """Boolean values cannot stand in for a risk threshold or sampling time."""
    with pytest.raises(ValueError, match="numeric"):
        if field == "risk_threshold":
            owner.disruption_warning_time([0.1], risk_threshold=value)  # type: ignore[arg-type]  # refused input under test
        else:
            owner.disruption_warning_time([0.1], dt=value)  # type: ignore[arg-type]  # refused input under test


def test_ntm_mode_reaches_locking_threshold() -> None:
    """Long NTM trajectories can cross the locked-mode amplitude threshold."""
    signal, label, ttd = leaf.simulate_tearing_mode(5000, mode="ntm", rng=np.random.default_rng(0))
    assert label == 1
    assert ttd >= 0
    assert signal.size < 5000  # early return on lock
