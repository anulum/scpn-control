# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for the FAIR-MAST disruption evaluation harness
"""Offline tests for :mod:`validation.evaluate_mast_disruption`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from validation.evaluate_mast_disruption import (
    REQUIRED_ARRAY_CHANNELS,
    _sha256_json,
    build_report,
    load_shot,
    load_shots,
    main,
    render_markdown,
)

_FIXED_TS = "2026-07-10T00:00:00+00:00"


def _write_shot(path: Path, *, disruptive: bool, n: int = 200, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.0, 1.0, n)
    channels: dict[str, object] = {name: rng.normal(0.0, 0.1, n) for name in REQUIRED_ARRAY_CHANNELS}
    channels["time_s"] = np.arange(n, dtype=np.float64) * 1.0e-3
    if disruptive:
        # Rising precursor drives the heuristic before the labelled disruption.
        channels["dBdt_gauss_per_s"] = 5.0e4 * ramp
        channels["n1_amp"] = 4.0 * ramp
        channels["n2_amp"] = 2.0 * ramp
        channels["is_disruption"] = True
        channels["disruption_time_idx"] = n - 20
        channels["disruption_type"] = "locked_mode"
    else:
        channels["is_disruption"] = False
        channels["disruption_time_idx"] = -1
        channels["disruption_type"] = "safe"
    np.savez(path, **channels)


def _write_manifest(path: Path, *, synthetic: bool = True) -> None:
    payload = {
        "schema_version": "1.0",
        "dataset_id": "mast-disruption-test",
        "machine": "MAST",
        "shot": "unit-test",
        "synthetic": synthetic,
        "source": {
            "kind": "synthetic" if synthetic else "local_archive",
            "uri": "synthetic://unit-test" if synthetic else "file://archive",
            "access": "generated",
        },
        "signals": [
            {"name": "dBdt_gauss_per_s", "path": "dBdt", "units": "G/s", "timebase": "time_s"},
        ],
        "retrieved_at": None if synthetic else "2026-07-10T00:00:00+00:00",
        "licence": None if synthetic else "MIT",
        "synthetic_generator": "unit-test-generator" if synthetic else None,
        "synthetic_seed": 0 if synthetic else None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def campaign(tmp_path: Path) -> tuple[Path, Path]:
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    _write_shot(shots_dir / "shot_0001_locked.npz", disruptive=True, seed=1)
    _write_shot(shots_dir / "shot_0002_safe.npz", disruptive=False, seed=2)
    manifest = tmp_path / "campaign.manifest.json"
    _write_manifest(manifest, synthetic=True)
    return shots_dir, manifest


# --------------------------------------------------------------------------- #
# load_shot
# --------------------------------------------------------------------------- #
def test_load_shot_reads_valid_shot(tmp_path: Path) -> None:
    path = tmp_path / "shot.npz"
    _write_shot(path, disruptive=True)
    record = load_shot(path)
    assert record.label == 1
    assert record.disruption_type == "locked_mode"
    assert record.dbdt.shape == (200,)


def test_load_shot_rejects_missing_channel(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, time_s=np.arange(10, dtype=np.float64))
    with pytest.raises(ValueError, match="missing shot channels"):
        load_shot(path)


def test_load_shot_rejects_wrong_length(tmp_path: Path) -> None:
    path = tmp_path / "shape.npz"
    _write_shot(path, disruptive=False)
    good = dict(np.load(path, allow_pickle=False))
    good["Ip_MA"] = np.ones(199)  # one sample short of time_s
    np.savez(path, **good)
    with pytest.raises(ValueError, match="must be 1-D"):
        load_shot(path)


def test_load_shot_rejects_non_finite(tmp_path: Path) -> None:
    path = tmp_path / "nan.npz"
    _write_shot(path, disruptive=False)
    good = dict(np.load(path, allow_pickle=False))
    good["q95"] = good["q95"].copy()
    good["q95"][7] = np.inf
    np.savez(path, **good)
    with pytest.raises(ValueError, match="must be finite"):
        load_shot(path)


def test_load_shots_rejects_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no .npz shots"):
        load_shots(empty)


# --------------------------------------------------------------------------- #
# build_report + render_markdown
# --------------------------------------------------------------------------- #
def test_build_report_is_fail_closed_and_self_digested(campaign: tuple[Path, Path]) -> None:
    shots_dir, manifest = campaign
    shots = load_shots(shots_dir)
    report = build_report(
        shots,
        manifest_path=manifest,
        window_size=64,
        alarm_threshold=0.65,
        warning_ms=(10, 50),
        generated_at=_FIXED_TS,
    )
    assert report["schema_version"] == "scpn-control.mast-disruption-evaluation.v1"
    assert report["status"] == "blocked"
    assert report["admission_ready"] is False
    assert report["data_provenance"]["synthetic"] is True
    assert report["claim_boundary"]["public_claim_allowed"] is False
    assert 0.0 <= float(report["metrics"]["auc"]) <= 1.0
    # payload digest is a self-consistent canonical SHA-256.
    digest = report["payload_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    assert digest == _sha256_json({**report, "payload_sha256": None})


def test_build_report_is_deterministic(campaign: tuple[Path, Path]) -> None:
    shots_dir, manifest = campaign
    shots = load_shots(shots_dir)
    kwargs = {
        "manifest_path": manifest,
        "window_size": 64,
        "alarm_threshold": 0.65,
        "warning_ms": (10, 50),
        "generated_at": _FIXED_TS,
    }
    first = build_report(shots, **kwargs)
    second = build_report(shots, **kwargs)
    assert first["payload_sha256"] == second["payload_sha256"]


def test_render_markdown_summarises_report(campaign: tuple[Path, Path]) -> None:
    shots_dir, manifest = campaign
    report = build_report(
        load_shots(shots_dir),
        manifest_path=manifest,
        window_size=64,
        alarm_threshold=0.65,
        warning_ms=(10, 50),
        generated_at=_FIXED_TS,
    )
    markdown = render_markdown(report)
    assert "# FAIR-MAST Disruption Evaluation" in markdown
    assert "Warning-time recall" in markdown
    assert "AUC" in markdown


# --------------------------------------------------------------------------- #
# main (end to end)
# --------------------------------------------------------------------------- #
def test_main_writes_reports(campaign: tuple[Path, Path], tmp_path: Path) -> None:
    shots_dir, manifest = campaign
    json_out = tmp_path / "out" / "eval.json"
    md_out = tmp_path / "out" / "eval.md"
    exit_code = main(
        [
            "--shots-dir",
            str(shots_dir),
            "--manifest",
            str(manifest),
            "--json-out",
            str(json_out),
            "--report-out",
            str(md_out),
            "--window-size",
            "64",
            "--generated-at",
            _FIXED_TS,
        ]
    )
    assert exit_code == 0
    assert json_out.exists() and md_out.exists()
    written = json.loads(json_out.read_text(encoding="utf-8"))
    assert written["status"] == "blocked"
    assert written["metrics"]["n_disruptive"] == 1


def test_main_defaults_timestamp(campaign: tuple[Path, Path], tmp_path: Path) -> None:
    shots_dir, manifest = campaign
    exit_code = main(
        [
            "--shots-dir",
            str(shots_dir),
            "--manifest",
            str(manifest),
            "--json-out",
            str(tmp_path / "e.json"),
            "--report-out",
            str(tmp_path / "e.md"),
        ]
    )
    assert exit_code == 0
    written = json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))
    assert written["generated_at_utc"]  # a real timestamp was stamped
