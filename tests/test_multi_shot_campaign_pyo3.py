# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Multi-shot campaign PyO3 parity tests.
from __future__ import annotations

import numpy as np
import pytest

rust = pytest.importorskip("scpn_control_rs", reason="optional PyO3 extension is not installed")


def _rust_orchestrator() -> object:
    if not hasattr(rust, "PyMultiShotCampaignOrchestrator"):
        pytest.skip("installed PyO3 extension was not rebuilt with PyMultiShotCampaignOrchestrator")
    spec = rust.PyPulsedScenarioSpec(
        100.0,
        2.0e6,
        0.01,
        0.002,
        1.0e3,
        2.0e6,
        1.0e3,
        40.0,
        0.95,
        20.0,
        1.0e3,
        0.0,
    )
    bank = rust.PyCapacitorBankSpec(100e-6, 100e-6, 0.5, 10_000.0, 20.0)
    return rust.PyMultiShotCampaignOrchestrator("campaign-a", spec, bank, True)


def test_pyo3_multi_shot_table_runs_complete_lifecycle() -> None:
    orchestrator = _rust_orchestrator()
    shot_ids = ["shot-001"]
    shot_index = np.zeros(8, dtype=np.uintp)
    t_s = np.array([0.0, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3], dtype=np.float64)
    plasma = np.array(
        [
            [0.0, 10.0, 0.02, 0.01, 0.0, 0.0],
            [2.5e6, 10.0, 0.02, 0.01, 0.0, 0.0],
            [2.5e6, 1200.0, 0.004, 0.001, 0.0, 0.0],
            [2.5e6, 1500.0, 0.004, 0.001, 3.0e6, 0.0],
            [0.0, 200.0, 0.02, 0.01, 0.0, 1500.0],
            [0.0, 120.0, 0.02, 0.01, 0.0, 0.0],
            [0.0, 40.0, 0.02, 0.01, 0.0, 0.0],
            [100.0, 15.0, 0.02, 0.01, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    bank = np.array(
        [
            [9800.0, 10_000.0, 200.0],
            [9800.0, 10_000.0, 200.0],
            [9800.0, 10_000.0, 200.0],
            [9800.0, 10_000.0, 200.0],
            [9800.0, 10_000.0, 200.0],
            [2000.0, 10_000.0, 20.0],
            [9700.0, 10_000.0, 180.0],
            [9800.0, 10_000.0, 200.0],
        ],
        dtype=np.float64,
    )
    report = orchestrator.run_table(shot_ids, shot_index, t_s, plasma, bank, np.array([5000.0]))

    assert report["shot_count"] == 1
    assert report["passed_count"] == 1
    assert report["shots"][0]["terminal_state"] == "idle"
