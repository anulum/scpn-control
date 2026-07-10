# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for the FAIR-MAST disruption dataset builder
"""Offline tests for :mod:`validation.build_mast_disruption_dataset`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.real_data_manifest import load_real_data_manifest
from validation.build_mast_disruption_dataset import (
    MEASURED_CHANNELS,
    _sha256_json,
    build_dataset,
    build_shot_npz,
    derive_ip_quench_label,
    main,
)

_FIXED_TS = "2026-07-10T00:00:00+00:00"


def _time(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * 1.0e-3


def _channels(ip: np.ndarray) -> dict[str, np.ndarray]:
    n = ip.shape[0]
    rng = np.random.default_rng(0)
    channels = {name: rng.normal(0.0, 0.1, n) for name in MEASURED_CHANNELS}
    channels["time_s"] = _time(n)
    channels["Ip_MA"] = ip
    return channels


def _disruptive_ip(n: int = 300) -> np.ndarray:
    ip = np.ones(n, dtype=np.float64)
    ip[:100] = np.linspace(0.0, 1.0, 100)
    ip[250] = 0.5
    ip[251] = 0.05
    ip[252:] = 0.0
    return ip


# --------------------------------------------------------------------------- #
# derive_ip_quench_label — all four outcomes
# --------------------------------------------------------------------------- #
def test_label_current_quench() -> None:
    ip = _disruptive_ip()
    is_disruption, onset, kind = derive_ip_quench_label(ip, _time(ip.shape[0]))
    assert is_disruption is True
    assert kind == "current_quench"
    assert onset == 249


def test_label_no_current() -> None:
    ip = np.zeros(50)
    assert derive_ip_quench_label(ip, _time(50)) == (False, -1, "no_current")


def test_label_no_quench() -> None:
    ip = np.ones(50)
    assert derive_ip_quench_label(ip, _time(50)) == (False, -1, "no_quench")


def test_label_ignores_initial_ramp() -> None:
    # The ramp-up starts below the collapse threshold but must not be mistaken
    # for a quench; a shot that never collapses after flat-top is safe.
    ip = np.concatenate([np.linspace(0.0, 1.0, 150), np.ones(150)])
    assert derive_ip_quench_label(ip, _time(ip.shape[0])) == (False, -1, "no_quench")


def test_label_slow_rampdown() -> None:
    ip = np.concatenate([np.linspace(0.0, 1.0, 100), np.ones(100), np.linspace(1.0, 0.0, 100)])
    is_disruption, onset, kind = derive_ip_quench_label(ip, _time(ip.shape[0]))
    assert (is_disruption, onset, kind) == (False, -1, "slow_rampdown")


# --------------------------------------------------------------------------- #
# build_shot_npz
# --------------------------------------------------------------------------- #
def test_build_shot_npz_writes_labelled_shot(tmp_path: Path) -> None:
    record = build_shot_npz(123, _channels(_disruptive_ip()), out_dir=tmp_path, drop_fraction=0.8, quench_window_ms=5.0)
    assert record["label"] == 1
    assert record["disruption_type"] == "current_quench"
    assert len(record["checksum_sha256"]) == 64
    with np.load(tmp_path / "shot_123.npz", allow_pickle=False) as data:
        assert bool(data["is_disruption"]) is True
        assert set(MEASURED_CHANNELS).issubset(set(data.files))


def test_build_shot_npz_rejects_missing_channel(tmp_path: Path) -> None:
    channels = _channels(_disruptive_ip())
    del channels["q95"]
    with pytest.raises(ValueError, match="missing measured channels"):
        build_shot_npz(1, channels, out_dir=tmp_path, drop_fraction=0.8, quench_window_ms=5.0)


def test_build_shot_npz_rejects_wrong_shape(tmp_path: Path) -> None:
    channels = _channels(_disruptive_ip())
    channels["BT_T"] = np.ones(299)
    with pytest.raises(ValueError, match="must be 1-D"):
        build_shot_npz(1, channels, out_dir=tmp_path, drop_fraction=0.8, quench_window_ms=5.0)


def test_build_shot_npz_rejects_non_finite(tmp_path: Path) -> None:
    channels = _channels(_disruptive_ip())
    channels["ne_1e19"] = channels["ne_1e19"].copy()
    channels["ne_1e19"][5] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        build_shot_npz(1, channels, out_dir=tmp_path, drop_fraction=0.8, quench_window_ms=5.0)


# --------------------------------------------------------------------------- #
# build_dataset (end to end) + manifest validation
# --------------------------------------------------------------------------- #
def test_build_dataset_writes_manifest_report_and_verifies(tmp_path: Path) -> None:
    shots = [
        {"shot_id": 11766, "channels": _channels(_disruptive_ip())},
        {"shot_id": 11767, "channels": _channels(np.ones(300))},  # no_quench → safe
    ]
    report = build_dataset(
        shots,
        dataset_id="mast-disruption-pilot",
        out_dir=tmp_path,
        retrieved_at=_FIXED_TS,
        generated_at=_FIXED_TS,
    )
    assert report["schema_version"] == "scpn-control.mast-disruption-supervised-dataset.v1"
    assert report["status"] == "blocked"
    assert report["synthetic"] is False
    assert report["n_shots"] == 2
    assert report["n_disruptive"] == 1
    assert report["payload_sha256"] == _sha256_json({**report, "payload_sha256": None})
    # The emitted manifest is a valid, checksum-verifiable real-data manifest.
    manifest_path = tmp_path / "mast-disruption-pilot.manifest.json"
    manifest = load_real_data_manifest(manifest_path, verify_artifact=True)
    assert manifest.synthetic is False
    assert manifest.source.kind == "local_archive"
    assert len(manifest.artifacts) == 2


def test_main_writes_dataset(tmp_path: Path) -> None:
    channels_npz = tmp_path / "channels.npz"
    payload: dict[str, np.ndarray] = {"shot_ids": np.array([11766, 11767])}
    for shot_id, ip in ((11766, _disruptive_ip()), (11767, np.ones(300))):
        for name, array in _channels(ip).items():
            payload[f"{shot_id}:{name}"] = array
    np.savez(channels_npz, **payload)
    json_out = tmp_path / "out" / "dataset.json"
    exit_code = main(
        [
            "--channels-npz",
            str(channels_npz),
            "--dataset-id",
            "mast-disruption-pilot",
            "--out-dir",
            str(tmp_path / "out"),
            "--json-out",
            str(json_out),
            "--retrieved-at",
            _FIXED_TS,
            "--generated-at",
            _FIXED_TS,
        ]
    )
    assert exit_code == 0
    written = json.loads(json_out.read_text(encoding="utf-8"))
    assert written["n_disruptive"] == 1
    assert written["status"] == "blocked"
