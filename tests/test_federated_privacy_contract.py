# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Federated privacy helper contract tests.
"""Public Gaussian update-noise helpers reject invalid numeric domains."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_control.control.federated_disruption import (
    DifferentialPrivacyConfig,
    compose_privacy_epsilon,
    differential_privacy_clip,
    gaussian_mechanism_epsilon,
)


@pytest.mark.parametrize("max_norm", [0.0, -1.0, float("inf"), True])
def test_clipping_requires_positive_finite_norm(max_norm: float) -> None:
    """A nonpositive or nonnumeric bound cannot reverse an update."""
    with pytest.raises(ValueError, match="max_norm"):
        differential_privacy_clip({"w": np.array([2.0])}, max_norm=max_norm, noise_sigma=0.0)


@pytest.mark.parametrize("noise_sigma", [-1.0, float("nan"), float("inf"), True])
def test_clipping_rejects_invalid_noise_scale(noise_sigma: float) -> None:
    """Noise has a finite nonnegative scale, including the deterministic zero case."""
    with pytest.raises(ValueError, match="noise_sigma"):
        differential_privacy_clip({"w": np.array([2.0])}, max_norm=1.0, noise_sigma=noise_sigma)


def test_clipping_preserves_direction_for_huge_finite_gradient() -> None:
    """Scaled norm computation must not turn a huge finite vector into zero."""
    clipped = differential_privacy_clip(
        {"w": np.array([1e200, 0.0])},
        max_norm=1.0,
        noise_sigma=0.0,
        rng=np.random.default_rng(1),
    )
    np.testing.assert_allclose(clipped["w"], [1.0, 0.0], atol=1e-12)


def test_clipping_rejects_nonfinite_gradient() -> None:
    """An invalid input cannot be reported as a bounded update."""
    with pytest.raises(ValueError, match="gradients.w"):
        differential_privacy_clip({"w": np.array([float("nan")])}, max_norm=1.0, noise_sigma=0.0)


def test_clipping_rejects_empty_update() -> None:
    """An empty update is not a bounded gradient vector."""
    with pytest.raises(ValueError, match="gradients"):
        differential_privacy_clip({}, max_norm=1.0, noise_sigma=0.0)


def test_clipping_rejects_nonfinite_sampled_noise() -> None:
    """A finite scale can still overflow on a real Gaussian sample."""
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="nonfinite noise"):
        differential_privacy_clip({"w": np.zeros(1)}, max_norm=1.0, noise_sigma=1e308, rng=np.random.default_rng(3))


def test_dp_config_rejects_coerced_numeric_fields() -> None:
    """String and fractional seed values cannot reach NumPy random generation."""
    malformed_norm: Any = "1.0"
    with pytest.raises(ValueError, match="max_update_norm"):
        DifferentialPrivacyConfig(max_update_norm=malformed_norm)
    malformed_seed: Any = 1.0
    with pytest.raises(ValueError, match="seed"):
        DifferentialPrivacyConfig(seed=malformed_seed)
    malformed_delta: Any = "1e-5"
    with pytest.raises(ValueError, match="delta"):
        DifferentialPrivacyConfig(delta=malformed_delta)
    with pytest.raises(ValueError, match="delta"):
        gaussian_mechanism_epsilon(1.0, malformed_delta)
    with pytest.raises(ValueError, match="max_update_norm"):
        DifferentialPrivacyConfig(max_update_norm=10**1000)
    with pytest.raises(ValueError, match="delta"):
        DifferentialPrivacyConfig(delta=10**1000)


def test_nominal_epsilon_must_remain_finite() -> None:
    """Subnormal noise and excessive composition cannot emit infinity."""
    with pytest.raises(ValueError, match="Gaussian epsilon"):
        gaussian_mechanism_epsilon(1e-320, 1e-5)
    with pytest.raises(ValueError, match="composed nominal epsilon"):
        compose_privacy_epsilon(1e-150, 1e-5, 10**200)
