# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard reference shot tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Tests for safe reference-shot loading used by the dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dashboard.reference_shots import list_reference_shots, load_reference_shot


def _write_npz(path: Path, *, n: int = 8, include_time: bool = True) -> None:
    payload: dict[str, object] = {
        "Ip_MA": np.linspace(0.5, 1.5, n, dtype=np.float64),
        "beta_N": np.linspace(0.2, 2.0, n, dtype=np.float64),
        "q95": np.linspace(4.0, 3.2, n, dtype=np.float64),
        "ne_1e19": np.linspace(2.5, 5.0, n, dtype=np.float64),
        "is_disruption": np.bool_(True),
        "disruption_time_idx": np.int64(n - 2),
        "disruption_type": np.str_("locked_mode"),
    }
    if include_time:
        payload["time_s"] = np.linspace(0.0, 0.7, n, dtype=np.float64)
    np.savez_compressed(path, **payload)


def test_list_reference_shots_returns_sorted_npz_files(tmp_path: Path) -> None:
    _write_npz(tmp_path / "shot_b.npz")
    _write_npz(tmp_path / "shot_a.npz")
    (tmp_path / "notes.txt").write_text("not a shot", encoding="utf-8")

    assert [path.name for path in list_reference_shots(tmp_path)] == ["shot_a.npz", "shot_b.npz"]


def test_load_reference_shot_returns_replay_ready_scalars_and_arrays(tmp_path: Path) -> None:
    path = tmp_path / "shot_001.npz"
    _write_npz(path, n=10)

    shot = load_reference_shot(path)

    assert set(shot) >= {"time_s", "Ip_MA", "is_disruption", "disruption_time_idx", "disruption_type"}
    assert isinstance(shot["time_s"], np.ndarray)
    assert shot["time_s"].shape == (10,)
    assert shot["is_disruption"] is True
    assert shot["disruption_time_idx"] == 8
    assert shot["disruption_type"] == "locked_mode"


def test_load_reference_shot_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_reference_shot(tmp_path / "missing.npz")


def test_load_reference_shot_rejects_invalid_timeline(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    _write_npz(path, include_time=False)

    with pytest.raises(ValueError, match="time_s"):
        load_reference_shot(path)


def test_dashboard_reference_loader_does_not_enable_pickle() -> None:
    source = Path("dashboard/reference_shots.py").read_text(encoding="utf-8")
    dashboard_source = Path("dashboard/control_dashboard.py").read_text(encoding="utf-8")

    assert "allow_pickle=False" in source
    assert "allow_pickle=True" not in source
    assert "allow_pickle=True" not in dashboard_source
