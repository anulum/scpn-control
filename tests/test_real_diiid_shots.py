# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Real DIII-D Shot Validation Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Validation against real DIII-D disruption shot data (17 shots).

These tests load actual DIII-D shot .npz files and validate that:
  - Shot data has correct structure and physical ranges
  - Disruption predictor features extract cleanly from real signals
  - RealtimeMonitor phase sync pipeline handles real beta_N waveforms
  - Disruption/safe shot pairs show distinguishable dynamics
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from scpn_control.phase.realtime_monitor import RealtimeMonitor

logger = logging.getLogger(__name__)

_SHOT_DIR = Path(__file__).resolve().parents[1] / "validation" / "reference_data" / "diiid" / "disruption_shots"

pytestmark = pytest.mark.skipif(
    not _SHOT_DIR.is_dir(),
    reason=f"DIII-D disruption shot directory not found: {_SHOT_DIR}",
)

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

# Physical ranges for DIII-D (includes disruption transients):
# Ip can go slightly negative during current quench
# q95 spikes to >50 during locked-mode disruptions as Ip collapses
PHYSICAL_BOUNDS = {
    "Ip_MA": (-0.1, 3.0),
    "BT_T": (0.0, 3.0),
    "beta_N": (-1.0, 8.0),
    "q95": (0.5, 100.0),
    "ne_1e19": (0.0, 20.0),
}


def _all_shot_paths():
    """Yield all .npz files in the disruption_shots directory."""
    return sorted(_SHOT_DIR.glob("shot_*.npz"))


def _load_shot(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


class TestShotDataIntegrity:
    """Verify structure and physical plausibility of all 17 shots."""

    def test_shot_directory_not_empty(self):
        shots = _all_shot_paths()
        assert len(shots) >= 16, f"Expected >=16 shots, found {len(shots)}"
        logger.info("Found %d DIII-D disruption shots", len(shots))

    @pytest.mark.parametrize("shot_path", _all_shot_paths(), ids=lambda p: p.stem)
    def test_required_keys_present(self, shot_path: Path):
        data = _load_shot(shot_path)
        missing = REQUIRED_KEYS - set(data.keys())
        assert not missing, f"{shot_path.stem}: missing keys {missing}"

    @pytest.mark.parametrize("shot_path", _all_shot_paths(), ids=lambda p: p.stem)
    def test_timeseries_lengths_consistent(self, shot_path: Path):
        data = _load_shot(shot_path)
        n = len(data["time_s"])
        assert n >= 100, f"{shot_path.stem}: only {n} timesteps"
        for key in ("Ip_MA", "BT_T", "beta_N", "q95", "ne_1e19"):
            assert len(data[key]) == n, f"{shot_path.stem}: {key} length mismatch"

    @pytest.mark.parametrize("shot_path", _all_shot_paths(), ids=lambda p: p.stem)
    def test_all_signals_finite(self, shot_path: Path):
        data = _load_shot(shot_path)
        for key in ("time_s", "Ip_MA", "BT_T", "beta_N", "q95", "ne_1e19"):
            assert np.all(np.isfinite(data[key])), f"{shot_path.stem}: {key} has non-finite values"

    @pytest.mark.parametrize("shot_path", _all_shot_paths(), ids=lambda p: p.stem)
    def test_physical_ranges(self, shot_path: Path):
        data = _load_shot(shot_path)
        for key, (lo, hi) in PHYSICAL_BOUNDS.items():
            arr = data[key]
            assert np.all(arr >= lo) and np.all(arr <= hi), (
                f"{shot_path.stem}: {key} out of range [{lo}, {hi}], "
                f"actual [{float(np.min(arr)):.3f}, {float(np.max(arr)):.3f}]"
            )

    @pytest.mark.parametrize("shot_path", _all_shot_paths(), ids=lambda p: p.stem)
    def test_time_monotonic(self, shot_path: Path):
        data = _load_shot(shot_path)
        dt = np.diff(data["time_s"])
        assert np.all(dt > 0), f"{shot_path.stem}: time not strictly monotonic"


class TestDisruptionShotPairs:
    """Disruption and safe shot labeling should be self-consistent."""

    def test_is_disruption_flag_matches_type(self):
        """is_disruption should be True iff disruption_type is not 'safe'."""
        for path in _all_shot_paths():
            data = _load_shot(path)
            is_disruption = bool(data["is_disruption"])
            dtype = str(data["disruption_type"])
            if dtype == "safe":
                assert not is_disruption, f"{path.stem}: type='safe' but is_disruption=True"
            else:
                assert is_disruption, f"{path.stem}: type='{dtype}' but is_disruption=False"

    def test_safe_suffix_files_are_safe(self):
        """Files with _safe suffix should have is_disruption=False."""
        for path in _all_shot_paths():
            if "_safe" not in path.stem:
                continue
            data = _load_shot(path)
            assert not bool(data["is_disruption"]), f"{path.stem}: _safe file is disrupted"

    def test_disruption_type_valid(self):
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
            dtype = str(data["disruption_type"])
            assert dtype in valid_types, f"{path.stem}: unknown disruption_type '{dtype}'"


def _disruption_shots():
    """Return shot paths where is_disruption=True."""
    result = []
    for p in _all_shot_paths():
        data = np.load(p, allow_pickle=True)
        if bool(data["is_disruption"]):
            result.append(p)
    return result


def _safe_shots():
    """Return shot paths where is_disruption=False."""
    result = []
    for p in _all_shot_paths():
        data = np.load(p, allow_pickle=True)
        if not bool(data["is_disruption"]):
            result.append(p)
    return result


class TestPhaseSyncWithRealShots:
    """Drive RealtimeMonitor with real DIII-D beta_N waveforms."""

    @pytest.mark.parametrize(
        "shot_path",
        _disruption_shots()[:5],
        ids=lambda p: p.stem,
    )
    def test_monitor_handles_real_disruption(self, shot_path: Path):
        """Phase sync pipeline should not crash on real disruption dynamics."""
        data = _load_shot(shot_path)
        beta_n = data["beta_N"]
        stride = max(1, len(beta_n) // 200)
        beta_sub = beta_n[::stride]

        mon = RealtimeMonitor.from_paper27(
            L=4,
            N_per=20,
            dt=5e-3,
            zeta_uniform=0.5,
            psi_driver=0.0,
        )
        for val in beta_sub:
            mon.psi_driver = float(np.clip(val - 2.0, -np.pi, np.pi))
            snap = mon.tick()

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
        ids=lambda p: p.stem,
    )
    def test_safe_shots_maintain_coherence(self, shot_path: Path):
        """Safe (non-disrupted) shots should yield stable phase evolution."""
        data = _load_shot(shot_path)
        beta_n = data["beta_N"]
        stride = max(1, len(beta_n) // 200)
        beta_sub = beta_n[::stride]

        mon = RealtimeMonitor.from_paper27(
            L=4,
            N_per=20,
            dt=5e-3,
            zeta_uniform=1.0,
            psi_driver=0.0,
        )
        r_history = []
        for val in beta_sub:
            mon.psi_driver = float(np.clip(val - 2.0, -np.pi, np.pi))
            snap = mon.tick()
            r_history.append(snap["R_global"])

        r_arr = np.array(r_history)
        assert np.all(np.isfinite(r_arr))
        # Safe shots should not show catastrophic decoherence
        assert float(np.mean(r_arr[-20:])) > 0.01, f"{shot_path.stem}: R_global collapsed to zero in safe shot"


class TestFeatureExtraction:
    """Extract disruption predictor features from real shot data."""

    def test_locked_mode_precursor(self):
        """Locked mode amplitude should spike before disruption."""
        for path in _all_shot_paths():
            if "locked_mode" not in path.stem or "_safe" in path.stem:
                continue
            data = _load_shot(path)
            lm = data["locked_mode_amp"]
            dt_idx = int(data["disruption_time_idx"])
            # Pre-disruption window should have elevated locked mode
            pre_window = lm[max(0, dt_idx - 50) : dt_idx]
            baseline = lm[: max(1, dt_idx - 100)]
            if len(pre_window) > 0 and len(baseline) > 0:
                assert float(np.max(pre_window)) > float(np.mean(baseline)), "Locked mode precursor not visible"

    def test_vde_vertical_displacement(self):
        """VDE shots should show vertical displacement growth."""
        for path in _all_shot_paths():
            if "vde" not in path.stem:
                continue
            data = _load_shot(path)
            vpos = data["vertical_position_m"]
            dt_idx = int(data["disruption_time_idx"])
            # Post-disruption vertical displacement should exceed pre-disruption
            pre_std = float(np.std(vpos[:dt_idx]))
            post_excursion = float(np.max(np.abs(vpos[dt_idx:])))
            assert post_excursion > pre_std, "VDE vertical excursion not visible"

    def test_beta_limit_exceeds_threshold(self):
        """Beta-limit shots should show elevated beta_N."""
        for path in _all_shot_paths():
            if "beta_limit" not in path.stem:
                continue
            data = _load_shot(path)
            beta = data["beta_N"]
            dt_idx = int(data["disruption_time_idx"])
            pre_max = float(np.max(beta[:dt_idx]))
            # DIII-D beta limit typically around 2.5-3.5
            assert pre_max > 1.5, f"Beta limit shot max beta_N={pre_max:.2f} too low"
