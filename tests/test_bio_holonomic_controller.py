# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Bio-holonomic controller tests

from __future__ import annotations

from typing import Any

import pytest

from scpn_control.control import bio_holonomic_controller as mod


def test_bio_telemetry_snapshot_is_immutable() -> None:
    snapshot = mod.BioTelemetrySnapshot(
        heart_rate_bpm=60.0,
        eeg_coherence_r=0.5,
        galvanic_skin_response=1.0,
    )

    with pytest.raises(AttributeError):
        snapshot.heart_rate_bpm = 80.0  # type: ignore[misc]


def test_bio_holonomic_controller_requires_sc_neurocore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "SC_NEUROCORE_HOLONOMIC_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="sc-neurocore holonomic adapters are required"):
        mod.BioHolonomicController()


def test_bio_holonomic_controller_maps_l4_l5_metrics_to_vibrana_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    l4_instances: list[_FakeL4Adapter] = []
    l5_instances: list[_FakeL5Adapter] = []

    class L4AdapterFactory(_FakeL4Adapter):
        def __init__(self, params: _FakeParams, seed: int) -> None:
            super().__init__(params, seed)
            l4_instances.append(self)

    class L5AdapterFactory(_FakeL5Adapter):
        def __init__(self, params: _FakeParams, seed: int) -> None:
            super().__init__(params, seed)
            l5_instances.append(self)

    monkeypatch.setattr(mod, "SC_NEUROCORE_HOLONOMIC_AVAILABLE", True)
    monkeypatch.setattr(mod, "L4_HolonomicParameters", _FakeParams, raising=False)
    monkeypatch.setattr(mod, "L5_HolonomicParameters", _FakeParams, raising=False)
    monkeypatch.setattr(mod, "L4_CellularAdapter", L4AdapterFactory)
    monkeypatch.setattr(mod, "L5_OrganismalAdapter", L5AdapterFactory)

    controller = mod.BioHolonomicController(dt_s=0.02, seed=7)
    action = controller.step(
        mod.BioTelemetrySnapshot(
            heart_rate_bpm=72.0,
            eeg_coherence_r=0.5,
            galvanic_skin_response=0.2,
        )
    )

    assert l4_instances[0].seed == 7
    assert l5_instances[0].seed == 8
    assert l4_instances[0].params.k_coupling == pytest.approx(0.45)
    assert l4_instances[0].last_dt_s == pytest.approx(0.02)
    assert l5_instances[0].last_dt_s == pytest.approx(0.02)
    assert l5_instances[0].last_inputs == {"phase_bits": [1, 0, 1]}
    assert action["actuator_vibrana_intensity"] == pytest.approx(0.375)
    assert action == {
        "l4_order_parameter": 0.81,
        "l4_avalanche_density": 0.13,
        "l5_hrv_coherence": 0.25,
        "l5_emotional_valence": -0.2,
        "actuator_vibrana_intensity": action["actuator_vibrana_intensity"],
    }


class _FakeParams:
    def __init__(self) -> None:
        self.k_coupling = 0.0


class _FakeL4Adapter:
    def __init__(self, params: _FakeParams, seed: int) -> None:
        self.params = params
        self.seed = seed
        self.last_dt_s = 0.0

    def step_jax(self, dt_s: float) -> dict[str, list[int]]:
        self.last_dt_s = dt_s
        return {"phase_bits": [1, 0, 1]}

    def get_metrics(self) -> dict[str, float]:
        return {
            "order_parameter": 0.81,
            "avalanche_density": 0.13,
        }


class _FakeL5Adapter:
    def __init__(self, _params: _FakeParams, seed: int) -> None:
        self.seed = seed
        self.last_dt_s = 0.0
        self.last_inputs: Any = None

    def step_jax(self, dt_s: float, *, inputs: Any) -> None:
        self.last_dt_s = dt_s
        self.last_inputs = inputs

    def get_metrics(self) -> dict[str, float]:
        return {
            "hrv_coherence_r5": 0.25,
            "emotional_valence": -0.2,
        }
