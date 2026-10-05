# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption episode numeric admission tests
"""Public numeric-admission checks for bounded disruption episode proxies."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_control.control.disruption_contracts import (
    impurity_transport_response,
    mcnp_lite_tbr,
    post_disruption_halo_runaway,
    synthetic_disruption_signal,
)
from scpn_control.control.spi_mitigation import ShatteredPelletInjection


class _NonFiniteGenerator:
    """Inject an invalid RNG return at the public signal dependency boundary."""

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Return an invalid phase or amplitude draw."""
        return float("nan")

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: tuple[int, ...] = (1,)) -> NDArray[np.float64]:
        """Supply a finite noise vector so the invalid phase remains observable."""
        return np.zeros(size, dtype=np.float64)


def test_signal_refuses_nonfinite_disturbance_without_consuming_rng() -> None:
    """A bad scenario amplitude leaves the caller's random stream intact."""
    rng = np.random.default_rng(7)
    expected = np.random.default_rng(7).uniform()
    with pytest.raises(ValueError, match="disturbance"):
        synthetic_disruption_signal(rng=rng, disturbance=float("nan"))
    assert rng.uniform() == expected


def test_signal_refuses_nonfinite_rng_output() -> None:
    """A broken draw cannot publish a synthetic diagnostic trace."""
    with pytest.raises(ValueError, match="synthetic disruption signal"):
        synthetic_disruption_signal(rng=cast(np.random.Generator, _NonFiniteGenerator()), disturbance=0.5)


@pytest.mark.parametrize("window", [True, 5.5, 0])
def test_signal_refuses_unscorable_window(window: object) -> None:
    """Signal length must be a positive integer before any draw."""
    with pytest.raises(ValueError, match="window"):
        synthetic_disruption_signal(rng=np.random.default_rng(1), disturbance=0.5, window=cast(int, window))


def test_tbr_refuses_derived_overflow() -> None:
    """An extreme but finite baseline cannot publish infinite TBR."""
    with pytest.raises(ValueError, match="tbr"):
        mcnp_lite_tbr(base_tbr=1.0e308, li6_enrichment=1.0, be_multiplier_fraction=1.0, reflector_albedo=1.0)


@pytest.mark.parametrize("quantity,disturbance", [(float("nan"), 0.5), (0.5, float("nan")), (1.0e308, 0.5)])
def test_impurity_proxy_refuses_nonfinite_inputs_or_output(quantity: float, disturbance: float) -> None:
    """Gas transport must not publish invalid physical diagnostics."""
    with pytest.raises(ValueError, match="neon_quantity_mol|disturbance|impurity"):
        impurity_transport_response(
            neon_quantity_mol=quantity,
            argon_quantity_mol=0.2,
            xenon_quantity_mol=0.05,
            disturbance=disturbance,
            seed_shift=0,
        )


def test_impurity_proxy_refuses_nonfinite_estimator_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed SPI estimator cannot contaminate published proxy metrics."""
    monkeypatch.setattr(
        ShatteredPelletInjection,
        "estimate_z_eff_cocktail",
        staticmethod(lambda **_kwargs: float("nan")),
    )
    with pytest.raises(ValueError, match="impurity response"):
        impurity_transport_response(
            neon_quantity_mol=0.5,
            argon_quantity_mol=0.2,
            xenon_quantity_mol=0.05,
            disturbance=0.5,
            seed_shift=0,
        )


def test_impurity_proxy_refuses_unrepresentable_seed_metadata() -> None:
    """Integer seed metadata cannot silently round in the float result."""
    with pytest.raises(ValueError, match="seed_shift"):
        impurity_transport_response(
            neon_quantity_mol=0.5,
            argon_quantity_mol=0.2,
            xenon_quantity_mol=0.05,
            disturbance=0.5,
            seed_shift=2**53 + 1,
        )


@pytest.mark.parametrize(
    "current,tau,mitigation,zeff",
    [
        (float("nan"), 0.01, 0.5, 2.0),
        (15.0, -0.01, 0.5, 2.0),
        (15.0, 0.01, 1.5, 2.0),
        (15.0, 0.01, 0.5, float("nan")),
    ],
)
def test_halo_proxy_refuses_invalid_parameters(current: float, tau: float, mitigation: float, zeff: float) -> None:
    """The quench proxy admits only finite physical input conditions."""
    with pytest.raises(ValueError, match="pre_current_ma|tau_cq_s|mitigation_strength|zeff_eff"):
        post_disruption_halo_runaway(
            pre_current_ma=current,
            tau_cq_s=tau,
            disturbance=0.5,
            mitigation_strength=mitigation,
            zeff_eff=zeff,
        )


def test_halo_proxy_refuses_derived_current_quench_overflow() -> None:
    """A finite current can exceed representable quench derivatives."""
    with pytest.raises(ValueError, match="current-quench derivative"):
        post_disruption_halo_runaway(
            pre_current_ma=1.0e308,
            tau_cq_s=0.004,
            disturbance=0.5,
            mitigation_strength=0.5,
            zeff_eff=2.0,
        )


def test_halo_proxy_refuses_derived_response_overflow() -> None:
    """A finite disturbance must not publish infinite halo or beam current."""
    with pytest.raises(ValueError, match="percentile sample values must be finite"):
        post_disruption_halo_runaway(
            pre_current_ma=15.0,
            tau_cq_s=0.01,
            disturbance=1.0e308,
            mitigation_strength=0.5,
            zeff_eff=2.0,
        )
