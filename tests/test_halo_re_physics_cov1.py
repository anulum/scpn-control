# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Halo runaway physics COV-1 regressions.

"""COV-1 regressions for halo runaway numerical guard paths."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_control.control.halo_re_physics as halo


def test_dreicer_rate_rejects_corrupted_ratio_and_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dreicer generation fails closed when validated state is corrupted."""
    corrupted = halo.RunawayElectronModel()
    cast(Any, corrupted).n_e_free = float("inf")

    assert corrupted._dreicer_rate(E=1.0, T_e_keV=1.0) == 0.0

    model = halo.RunawayElectronModel()
    monkeypatch.setattr(cast(Any, halo).np, "exp", lambda _arg: float("inf"))

    assert model._dreicer_rate(E=model.E_D * 2.0, T_e_keV=model.T_e0) == 0.0


def test_runaway_growth_rates_reject_overflowed_corrupted_state() -> None:
    """Avalanche and momentum-space growth fail closed after state corruption."""
    model = halo.RunawayElectronModel()
    state = cast(Any, model)
    state.E_c = 0.5
    state.tau_av = 1e-300

    assert model._avalanche_rate(E=1.0, n_re=1e308) == 0.0
    assert model._momentum_space_growth(E=1.0, n_re=1e308) == 0.0


def test_momentum_space_growth_rejects_python_float_overflow() -> None:
    """Momentum-space growth catches overflow before finite checks can run."""
    model = halo.RunawayElectronModel()
    state = cast(Any, model)
    state.E_c = 1e-300
    state.tau_av = 1e-300

    assert model._momentum_space_growth(E=1.0, n_re=1e308) == 0.0


def test_relativistic_loss_rejects_overflowed_corrupted_population() -> None:
    """Relativistic loss fails closed when current conversion overflows."""
    model = halo.RunawayElectronModel(enable_relativistic_losses=True)

    assert model._relativistic_loss_rate(E=model.E_c * 2.0, n_re=1e308) == 0.0


def test_runaway_simulation_resets_nonfinite_loss_and_population(monkeypatch: pytest.MonkeyPatch) -> None:
    """The time-step loop resets non-finite loss rate and runaway density."""
    model = halo.RunawayElectronModel()
    cast(Any, model).n_e_free = 1e308
    monkeypatch.setattr(model, "_dreicer_rate", lambda _e, _t: 1e307)
    monkeypatch.setattr(model, "_avalanche_rate", lambda _e, _n: 0.0)
    monkeypatch.setattr(model, "_momentum_space_growth", lambda _e, _n: 0.0)
    monkeypatch.setattr(model, "_relativistic_loss_rate", lambda **_kwargs: 0.0)

    result = model.simulate(duration_s=10.0, dt_s=10.0, seed_re_fraction=1.0)

    assert 0.0 <= result.peak_re_current_ma <= 15.0


def test_runaway_simulation_resets_nonfinite_density_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    """The time-step loop resets non-finite density increments."""
    model = halo.RunawayElectronModel()
    monkeypatch.setattr(model, "_dreicer_rate", lambda _e, _t: float("inf"))
    monkeypatch.setattr(model, "_avalanche_rate", lambda _e, _n: 0.0)
    monkeypatch.setattr(model, "_momentum_space_growth", lambda _e, _n: 0.0)
    monkeypatch.setattr(model, "_relativistic_loss_rate", lambda **_kwargs: 0.0)

    result = model.simulate(duration_s=1e-5, dt_s=1e-5)

    assert result.peak_re_current_ma >= 0.0
