# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Geometry-Neutral Replay Tests

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from scpn_control.scpn import (
    GEOMETRY_NEUTRAL_REPLAY_MANIFEST_SCHEMA_VERSION as PUBLIC_MANIFEST_SCHEMA_VERSION,
)
from scpn_control.scpn.geometry_neutral_replay import (
    GEOMETRY_NEUTRAL_REPLAY_MANIFEST_SCHEMA_VERSION,
    SCHEMA_VERSION,
    generate_report,
    render_markdown,
    validate_report,
)


def test_geometry_neutral_replay_is_deterministic_and_schema_valid() -> None:
    first = generate_report(steps=10, seed=123)
    second = generate_report(steps=10, seed=123)

    assert first == second
    validate_report(first)
    bench = first["geometry_neutral_replay"]
    assert bench["schema_version"] == SCHEMA_VERSION
    assert bench["magnetic_configuration"]["device_class"] == "stellarator"
    assert bench["replay"]["deterministic"] is True
    assert bench["metrics"]["max_abs_current_A"] <= bench["thresholds"]["max_abs_current_A"]
    assert bench["passes_thresholds"] is True
    assert bench["manifest"]["schema_version"] == GEOMETRY_NEUTRAL_REPLAY_MANIFEST_SCHEMA_VERSION
    assert PUBLIC_MANIFEST_SCHEMA_VERSION == GEOMETRY_NEUTRAL_REPLAY_MANIFEST_SCHEMA_VERSION
    assert bench["manifest"]["acceptance"]["passes_thresholds"] is True
    assert bench["manifest"]["provenance"]["latency_model"]


def test_geometry_neutral_replay_uses_non_tokamak_features() -> None:
    report = generate_report(steps=8, seed=7)
    frame = report["geometry_neutral_replay"]["scenario"]["initial_frame"]
    channel_names = {channel["name"] for channel in frame["channels"]}

    assert "fieldline_spread" in channel_names
    assert "effective_ripple" in channel_names
    assert "R_axis_m" not in channel_names
    assert "Z_axis_m" not in channel_names


def test_geometry_neutral_replay_manifest_rejects_trace_tampering() -> None:
    report = generate_report(steps=8, seed=11)
    tampered = deepcopy(report)
    tampered["geometry_neutral_replay"]["replay"]["trace"][0]["fieldline_spread"] += 0.01

    try:
        validate_report(tampered)
    except ValueError as exc:
        assert "manifest trace digest" in str(exc)
    else:
        raise AssertionError("tampered replay trace was accepted")


def test_geometry_neutral_replay_manifest_rejects_acceptance_tampering() -> None:
    report = generate_report(steps=8, seed=12)
    tampered = deepcopy(report)
    tampered["geometry_neutral_replay"]["manifest"]["acceptance"]["deterministic"] = False

    try:
        validate_report(tampered)
    except ValueError as exc:
        assert "manifest acceptance" in str(exc)
    else:
        raise AssertionError("tampered manifest acceptance was accepted")


def test_markdown_report_keeps_research_limitations_visible() -> None:
    report = generate_report(steps=8, seed=9)
    markdown = render_markdown(report)

    assert markdown.startswith("# Geometry-Neutral Stellarator Replay")
    assert "not a production PCS" in markdown
    assert "No external company data" in markdown


def test_module_cli_writes_report_files(tmp_path: Path) -> None:
    output_json = tmp_path / "geometry_neutral_replay.json"
    output_md = tmp_path / "geometry_neutral_replay.md"
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scpn_control.scpn.geometry_neutral_replay",
            "--steps",
            "9",
            "--seed",
            "77",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--strict",
        ],
        cwd=repo_root,
        env=os.environ | {"PYTHONPATH": str(repo_root / "src")},
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    validate_report(payload)
    assert output_md.read_text(encoding="utf-8").startswith("# Geometry-Neutral Stellarator Replay")
