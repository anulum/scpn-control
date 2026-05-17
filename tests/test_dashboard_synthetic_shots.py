# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard synthetic shot tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Tests for runtime synthetic DIII-D shot generation used by the dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dashboard.synthetic_shots import DISRUPTION_TYPES, generate_synthetic_diiid_shot


def test_generate_synthetic_diiid_shot_returns_replay_ready_vectors() -> None:
    shot = generate_synthetic_diiid_shot(
        shot_id=123456,
        n_steps=256,
        disruption=True,
        disruption_type="locked_mode",
        seed=7,
    )

    time_s = shot["time_s"]
    assert isinstance(time_s, np.ndarray)
    assert time_s.shape == (256,)
    assert np.all(np.diff(time_s) > 0.0)
    assert shot["is_disruption"] is True
    assert shot["disruption_time_idx"] == 204
    assert shot["disruption_type"] == "locked_mode"

    for key in ("Ip_MA", "BT_T", "beta_N", "q95", "ne_1e19", "n1_amp", "locked_mode_amp"):
        values = shot[key]
        assert isinstance(values, np.ndarray)
        assert values.shape == time_s.shape
        assert np.all(np.isfinite(values))

    assert float(shot["Ip_MA"][-1]) < float(shot["Ip_MA"][200])
    assert float(np.max(shot["n1_amp"][150:204])) > float(np.max(shot["n1_amp"][:80]))


def test_generate_synthetic_diiid_shot_is_seed_deterministic() -> None:
    first = generate_synthetic_diiid_shot(seed=11, n_steps=192)
    second = generate_synthetic_diiid_shot(seed=11, n_steps=192)

    assert first.keys() == second.keys()
    for key, first_value in first.items():
        second_value = second[key]
        if isinstance(first_value, np.ndarray):
            np.testing.assert_allclose(first_value, second_value)
        else:
            assert first_value == second_value


def test_generate_synthetic_diiid_shot_safe_case_has_no_disruption_drop() -> None:
    shot = generate_synthetic_diiid_shot(n_steps=192, disruption=False, seed=5)

    assert shot["is_disruption"] is False
    assert shot["disruption_time_idx"] == 191
    assert shot["disruption_type"] == "safe"
    assert float(shot["Ip_MA"][-1]) == pytest.approx(1.5)
    assert float(np.max(shot["locked_mode_amp"][150:])) < 0.02


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shot_id": True}, "shot_id"),
        ({"n_steps": 127}, "n_steps"),
        ({"disruption_type": "unknown"}, "disruption_type"),
        ({"seed": True}, "seed"),
    ],
)
def test_generate_synthetic_diiid_shot_rejects_invalid_inputs(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_synthetic_diiid_shot(**kwargs)


def test_dashboard_runtime_does_not_import_test_generators() -> None:
    dashboard_source = Path("dashboard/control_dashboard.py").read_text(encoding="utf-8")

    assert "tests.mock_diiid" not in dashboard_source
    assert "Synthetic DIII-D" in dashboard_source
    assert set(DISRUPTION_TYPES) == {"hmode", "locked_mode", "density_limit", "vde", "beta_limit"}
