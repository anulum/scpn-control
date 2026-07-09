# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Synthetic DIII-D Reference Fixture Tests

"""Validate synthetic DIII-D-like reference fixtures.

The repository fixtures intentionally exercise the same file and signal
contracts as DIII-D disruption-shot replays, but they are not facility data and
do not support real-shot validation claims. The tests preserve CI coverage for
manifested arrays, phase-monitor ingestion, and disruption-feature contracts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pytest

from scpn_control.phase.realtime_monitor import RealtimeMonitor

logger = logging.getLogger(__name__)

_SHOT_DIR = Path(__file__).resolve().parents[1] / "validation" / "reference_data" / "diiid" / "disruption_shots"

pytestmark = pytest.mark.skipif(
    not _SHOT_DIR.is_dir(),
    reason=f"DIII-D synthetic reference fixture directory not found: {_SHOT_DIR}",
)

ShotData: TypeAlias = dict[str, npt.NDArray[Any]]
NumericArray: TypeAlias = npt.NDArray[np.float64]

REQUIRED_KEYS = {
    "time_s",
    "Ip_MA",
    "BT_T",
    "beta_N",
    "q95",
    "ne_1e19",
    "n1_amp",
    "n2_amp",
    "locked_mode_amp",
    "dBdt_gauss_per_s",
    "vertical_position_m",
    "is_disruption",
    "disruption_time_idx",
    "disruption_type",
}

PHYSICAL_BOUNDS = {
    "Ip_MA": (-0.1, 3.0),
    "BT_T": (0.0, 3.0),
    "beta_N": (-1.0, 8.0),
    "q95": (0.5, 100.0),
    "ne_1e19": (0.0, 20.0),
}


def _all_shot_paths() -> list[Path]:
    """Return all synthetic DIII-D-like shot fixture paths."""
    return sorted(_SHOT_DIR.glob("shot_*.npz"))


def _load_shot(path: Path) -> ShotData:
    """Load one manifested reference fixture into an owned array mapping."""
    with np.load(path, allow_pickle=True) as data:
        return {key: cast(npt.NDArray[Any], data[key]) for key in data.files}


def _numeric(data: ShotData, key: str) -> NumericArray:
    """Return one fixture signal as a floating-point vector."""
    return np.asarray(data[key], dtype=np.float64)


def _scalar_bool(data: ShotData, key: str) -> bool:
    """Return one scalar boolean fixture field."""
    return bool(np.asarray(data[key]).item())


def _scalar_int(data: ShotData, key: str) -> int:
    """Return one scalar integer fixture field."""
    return int(np.asarray(data[key]).item())


def _scalar_str(data: ShotData, key: str) -> str:
    """Return one scalar string fixture field."""
    return str(np.asarray(data[key]).item())


class TestShotDataIntegrity:
    """Verify structure and plausibility of synthetic reference fixtures."""

    def test_shot_directory_not_empty(self) -> None:
        """The repository must carry the expected manifested fixture set."""
        shots = _all_shot_paths()
        assert len(shots) >= 16, f"Expected >=16 fixtures, found {len(shots)}"
        logger.info("Found %d DIII-D-like synthetic reference fixtures", len(shots))

    @pytest.mark.parametrize("shot_path", _all_shot_paths(), ids=lambda path: path.stem)
    def test_required_keys_present(self, shot_path: Path) -> None:
        """Every fixture must expose the replay-array contract."""
        data = _load_shot(shot_path)
        missing = REQUIRED_KEYS - set(data.keys())
        assert not missing, f"{shot_path.stem}: missing keys {missing}"

    @pytest.mark.parametrize("shot_path", _all_shot_paths(), ids=lambda path: path.stem)
    def test_timeseries_lengths_consistent(self, shot_path: Path) -> None:
        """Core time-series signals must share one timebase length."""
        data = _load_shot(shot_path)
        sample_count = len(data["time_s"])
        assert sample_count >= 100, f"{shot_path.stem}: only {sample_count} timesteps"
        for key in ("Ip_MA", "BT_T", "beta_N", "q95", "ne_1e19"):
            assert len(data[key]) == sample_count, f"{shot_path.stem}: {key} length mismatch"

    @pytest.mark.parametrize("shot_path", _all_shot_paths(), ids=lambda path: path.stem)
    def test_all_signals_finite(self, shot_path: Path) -> None:
        """Numeric fixture signals must be finite."""
        data = _load_shot(shot_path)
        for key in ("time_s", "Ip_MA", "BT_T", "beta_N", "q95", "ne_1e19"):
            assert np.all(np.isfinite(_numeric(data, key))), f"{shot_path.stem}: {key} has non-finite values"

    @pytest.mark.parametrize("shot_path", _all_shot_paths(), ids=lambda path: path.stem)
    def test_physical_ranges(self, shot_path: Path) -> None:
        """Synthetic signals must stay inside documented DIII-D-like bounds."""
        data = _load_shot(shot_path)
        for key, (lo, hi) in PHYSICAL_BOUNDS.items():
            signal = _numeric(data, key)
            assert np.all(signal >= lo) and np.all(signal <= hi), (
                f"{shot_path.stem}: {key} out of range [{lo}, {hi}], "
                f"actual [{float(np.min(signal)):.3f}, {float(np.max(signal)):.3f}]"
            )

    @pytest.mark.parametrize("shot_path", _all_shot_paths(), ids=lambda path: path.stem)
    def test_time_monotonic(self, shot_path: Path) -> None:
        """Synthetic fixture timebases must be strictly increasing."""
        data = _load_shot(shot_path)
        dt = np.diff(_numeric(data, "time_s"))
        assert np.all(dt > 0), f"{shot_path.stem}: time not strictly monotonic"


class TestDisruptionShotPairs:
    """Synthetic disrupted and safe fixture labels should be self-consistent."""

    def test_is_disruption_flag_matches_type(self) -> None:
        """``is_disruption`` should be true exactly when type is not ``safe``."""
        for path in _all_shot_paths():
            data = _load_shot(path)
            is_disruption = _scalar_bool(data, "is_disruption")
            disruption_type = _scalar_str(data, "disruption_type")
            if disruption_type == "safe":
                assert not is_disruption, f"{path.stem}: type='safe' but is_disruption=True"
            else:
                assert is_disruption, f"{path.stem}: type='{disruption_type}' but is_disruption=False"

    def test_safe_suffix_files_are_safe(self) -> None:
        """Files with a ``_safe`` suffix must carry a non-disrupted label."""
        for path in _all_shot_paths():
            if "_safe" not in path.stem:
                continue
            data = _load_shot(path)
            assert not _scalar_bool(data, "is_disruption"), f"{path.stem}: _safe file is disrupted"

    def test_disruption_type_valid(self) -> None:
        """Fixture disruption labels must stay inside the documented taxonomy."""
        valid_types = {
            "hmode",
            "vde",
            "beta_limit",
            "locked_mode",
            "density_limit",
            "tearing",
            "snowflake",
            "negdelta",
            "highbeta",
            "hybrid",
            "safe",
        }
        for path in _all_shot_paths():
            data = _load_shot(path)
            disruption_type = _scalar_str(data, "disruption_type")
            assert disruption_type in valid_types, f"{path.stem}: unknown disruption_type '{disruption_type}'"


def _disruption_shots() -> list[Path]:
    """Return synthetic fixture paths labelled as disrupted."""
    result: list[Path] = []
    for path in _all_shot_paths():
        if _scalar_bool(_load_shot(path), "is_disruption"):
            result.append(path)
    return result


def _safe_shots() -> list[Path]:
    """Return synthetic fixture paths labelled as safe."""
    result: list[Path] = []
    for path in _all_shot_paths():
        if not _scalar_bool(_load_shot(path), "is_disruption"):
            result.append(path)
    return result


class TestPhaseSyncWithReferenceFixtures:
    """Drive ``RealtimeMonitor`` with DIII-D-like beta_N fixture waveforms."""

    @pytest.mark.parametrize(
        "shot_path",
        _disruption_shots()[:5],
        ids=lambda path: path.stem,
    )
    def test_monitor_handles_disruption_fixture(self, shot_path: Path) -> None:
        """The phase-sync pipeline should ingest disruption-shaped fixtures."""
        data = _load_shot(shot_path)
        beta_n = _numeric(data, "beta_N")
        stride = max(1, len(beta_n) // 200)
        beta_sub = beta_n[::stride]

        monitor = RealtimeMonitor.from_paper27(
            L=4,
            N_per=20,
            dt=5e-3,
            zeta_uniform=0.5,
            psi_driver=0.0,
        )
        snap: dict[str, Any] = {}
        for value in beta_sub:
            monitor.psi_driver = float(np.clip(value - 2.0, -np.pi, np.pi))
            snap = monitor.tick()

        assert snap["tick"] == len(beta_sub)
        assert np.isfinite(snap["R_global"])
        logger.info(
            "%s: R_global=%.3f after %d steps",
            shot_path.stem,
            snap["R_global"],
            len(beta_sub),
        )

    @pytest.mark.parametrize(
        "shot_path",
        _safe_shots()[:3],
        ids=lambda path: path.stem,
    )
    def test_safe_fixtures_maintain_coherence(self, shot_path: Path) -> None:
        """Safe fixtures should yield finite phase evolution."""
        data = _load_shot(shot_path)
        beta_n = _numeric(data, "beta_N")
        stride = max(1, len(beta_n) // 200)
        beta_sub = beta_n[::stride]

        monitor = RealtimeMonitor.from_paper27(
            L=4,
            N_per=20,
            dt=5e-3,
            zeta_uniform=1.0,
            psi_driver=0.0,
        )
        r_history: list[float] = []
        for value in beta_sub:
            monitor.psi_driver = float(np.clip(value - 2.0, -np.pi, np.pi))
            snap = monitor.tick()
            r_history.append(float(snap["R_global"]))

        r_arr = np.array(r_history, dtype=np.float64)
        assert np.all(np.isfinite(r_arr))
        assert float(np.mean(r_arr[-20:])) > 0.01, f"{shot_path.stem}: R_global collapsed to zero"


class TestFeatureExtraction:
    """Extract disruption-predictor features from synthetic reference fixtures."""

    def test_locked_mode_precursor(self) -> None:
        """Locked-mode fixtures should show a precursor before disruption."""
        for path in _all_shot_paths():
            if "locked_mode" not in path.stem or "_safe" in path.stem:
                continue
            data = _load_shot(path)
            locked_mode = _numeric(data, "locked_mode_amp")
            disruption_index = _scalar_int(data, "disruption_time_idx")
            pre_window = locked_mode[max(0, disruption_index - 50) : disruption_index]
            baseline = locked_mode[: max(1, disruption_index - 100)]
            if len(pre_window) > 0 and len(baseline) > 0:
                assert float(np.max(pre_window)) > float(np.mean(baseline)), "Locked-mode precursor not visible"

    def test_vde_vertical_displacement(self) -> None:
        """VDE fixtures should show vertical-displacement growth."""
        for path in _all_shot_paths():
            if "vde" not in path.stem:
                continue
            data = _load_shot(path)
            vertical_position = _numeric(data, "vertical_position_m")
            disruption_index = _scalar_int(data, "disruption_time_idx")
            pre_std = float(np.std(vertical_position[:disruption_index]))
            post_excursion = float(np.max(np.abs(vertical_position[disruption_index:])))
            assert post_excursion > pre_std, "VDE vertical excursion not visible"

    def test_beta_limit_exceeds_threshold(self) -> None:
        """Beta-limit fixtures should show elevated beta_N before disruption."""
        for path in _all_shot_paths():
            if "beta_limit" not in path.stem:
                continue
            data = _load_shot(path)
            beta = _numeric(data, "beta_N")
            disruption_index = _scalar_int(data, "disruption_time_idx")
            pre_max = float(np.max(beta[:disruption_index]))
            assert pre_max > 1.5, f"Beta limit fixture max beta_N={pre_max:.2f} too low"
