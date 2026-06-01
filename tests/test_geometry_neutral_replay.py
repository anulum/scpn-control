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

import pytest

from scpn_control.scpn import (
    GEOMETRY_NEUTRAL_REPLAY_MANIFEST_SCHEMA_VERSION as PUBLIC_MANIFEST_SCHEMA_VERSION,
)
from scpn_control.scpn.geometry_neutral_replay import (
    DEFAULT_THRESHOLDS,
    GEOMETRY_NEUTRAL_REPLAY_EVIDENCE_SCHEMA_VERSION,
    GEOMETRY_NEUTRAL_REPLAY_MANIFEST_SCHEMA_VERSION,
    SCHEMA_VERSION,
    _build_manifest,
    assert_geometry_neutral_replay_claim_admissible,
    generate_report,
    geometry_neutral_replay_evidence,
    load_geometry_neutral_replay_evidence,
    render_geometry_neutral_markdown,
    render_markdown,
    save_geometry_neutral_replay_evidence,
    validate_geometry_neutral_report,
    validate_report,
)


def _manual_valid_report() -> dict[str, object]:
    scenario = {
        "name": "manual_geometry_neutral_replay",
        "magnetic_configuration": {
            "name": "manual_w7x_like",
            "device_class": "stellarator",
            "reference": "public synthetic W7-X-like reduced-order fixture",
        },
    }
    trace = [
        {"step": 0, "fieldline_spread": 0.025, "applied_current_A": 0.0, "latency_us": 120.0},
        {"step": 1, "fieldline_spread": 0.018, "applied_current_A": 500.0, "latency_us": 130.0},
    ]
    metrics = {
        "initial_fieldline_spread": 0.025,
        "final_fieldline_spread": 0.018,
        "improvement_fraction": 0.28,
        "max_abs_current_A": 500.0,
        "p95_latency_us": 130.0,
    }
    thresholds = dict(DEFAULT_THRESHOLDS)
    manifest = _build_manifest(
        scenario_payload=scenario,
        trace=trace,
        metrics=metrics,
        thresholds=thresholds,
        deterministic=True,
        passes_thresholds=True,
    )
    return {
        "geometry_neutral_replay": {
            "schema_version": SCHEMA_VERSION,
            "scenario": scenario,
            "replay": {
                "deterministic": True,
                "signature": "manual-signature",
                "trace": trace,
            },
            "magnetic_configuration": scenario["magnetic_configuration"],
            "metrics": metrics,
            "thresholds": thresholds,
            "passes_thresholds": True,
            "manifest": manifest,
            "limitations": [
                "This compact replay is not a production PCS.",
                "No external company data is used.",
            ],
        }
    }


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


def test_geometry_neutral_replay_rejects_non_integer_runtime_inputs() -> None:
    with pytest.raises(TypeError, match="steps"):
        generate_report(steps=True)

    with pytest.raises(TypeError, match="steps"):
        generate_report(steps=8.0)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="seed"):
        generate_report(seed=False)


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


def test_geometry_neutral_replay_rejects_threshold_tampering() -> None:
    report = generate_report(steps=8, seed=13)
    tampered = deepcopy(report)
    tampered["geometry_neutral_replay"]["thresholds"]["max_abs_current_A"] = 1.0

    with pytest.raises(ValueError, match="metrics do not satisfy"):
        validate_report(tampered)


def test_geometry_neutral_replay_rejects_blank_manifest_provenance() -> None:
    report = generate_report(steps=8, seed=14)
    tampered = deepcopy(report)
    tampered["geometry_neutral_replay"]["manifest"]["provenance"]["latency_model"] = " "

    with pytest.raises(ValueError, match="latency_model"):
        validate_report(tampered)


def test_geometry_neutral_replay_public_aliases_validate_and_render_manual_contract() -> None:
    report = _manual_valid_report()

    validate_geometry_neutral_report(report)
    markdown = render_geometry_neutral_markdown(report)

    assert markdown.startswith("# Geometry-Neutral Stellarator Replay")
    assert "Threshold pass: `YES`" in markdown


def test_geometry_neutral_replay_evidence_round_trips_bounded_report(tmp_path: Path) -> None:
    report = generate_report(steps=8, seed=41)
    evidence = geometry_neutral_replay_evidence(
        report,
        generated_utc="2026-05-31T00:00:00Z",
    )

    assert evidence.schema_version == GEOMETRY_NEUTRAL_REPLAY_EVIDENCE_SCHEMA_VERSION
    assert evidence.device_claim_allowed is False
    assert evidence.deterministic is True
    assert evidence.passes_thresholds is True
    assert evidence.replay_report_sha256
    assert evidence.scenario_digest == report["geometry_neutral_replay"]["manifest"]["scenario_digest"]
    with pytest.raises(ValueError, match="bounded-only"):
        assert_geometry_neutral_replay_claim_admissible(evidence)

    path = tmp_path / "geometry_neutral_replay_evidence.json"
    save_geometry_neutral_replay_evidence(evidence, path)
    assert load_geometry_neutral_replay_evidence(path) == evidence


def test_geometry_neutral_replay_evidence_rejects_device_claim_without_external_artefact() -> None:
    report = generate_report(steps=8, seed=42)

    with pytest.raises(ValueError, match="measured or benchmark"):
        geometry_neutral_replay_evidence(
            report,
            generated_utc="2026-05-31T00:00:00Z",
            device_claim_allowed=True,
        )


def test_geometry_neutral_replay_evidence_rejects_synthetic_device_claim_even_with_artefact() -> None:
    report = generate_report(steps=8, seed=43)

    with pytest.raises(ValueError, match="synthetic magnetic-configuration"):
        geometry_neutral_replay_evidence(
            report,
            generated_utc="2026-05-31T00:00:00Z",
            measured_or_benchmark_artefact_sha256="a" * 64,
            device_claim_allowed=True,
        )


def test_geometry_neutral_replay_evidence_rejects_tampering_and_duplicate_keys(tmp_path: Path) -> None:
    report = generate_report(steps=8, seed=44)
    evidence = geometry_neutral_replay_evidence(
        report,
        generated_utc="2026-05-31T00:00:00Z",
    )
    path = tmp_path / "geometry_neutral_replay_evidence.json"
    save_geometry_neutral_replay_evidence(evidence, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["p95_latency_us"] = 999999.0
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="payload_sha256"):
        load_geometry_neutral_replay_evidence(path)

    duplicate_path = tmp_path / "duplicate_geometry_neutral_replay_evidence.json"
    duplicate_path.write_text(
        (
            '{"schema_version":"'
            + GEOMETRY_NEUTRAL_REPLAY_EVIDENCE_SCHEMA_VERSION
            + '","schema_version":"'
            + GEOMETRY_NEUTRAL_REPLAY_EVIDENCE_SCHEMA_VERSION
            + '"}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_geometry_neutral_replay_evidence(duplicate_path)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda report: report.pop("geometry_neutral_replay"), "missing geometry_neutral_replay"),
        (
            lambda report: report["geometry_neutral_replay"].__setitem__("schema_version", "wrong"),
            "schema_version",
        ),
        (
            lambda report: report["geometry_neutral_replay"]["replay"].__setitem__("deterministic", False),
            "deterministic",
        ),
        (
            lambda report: report["geometry_neutral_replay"]["metrics"].__setitem__("p95_latency_us", float("nan")),
            "finite",
        ),
        (
            lambda report: report["geometry_neutral_replay"].__setitem__("passes_thresholds", False),
            "passes_thresholds",
        ),
        (
            lambda report: report["geometry_neutral_replay"]["manifest"].__setitem__("scenario_digest", "bad"),
            "scenario digest",
        ),
        (
            lambda report: report["geometry_neutral_replay"]["manifest"]["acceptance"].__setitem__(
                "passes_thresholds",
                False,
            ),
            "acceptance threshold",
        ),
        (
            lambda report: report["geometry_neutral_replay"]["manifest"].__setitem__("provenance", []),
            "provenance",
        ),
    ],
)
def test_geometry_neutral_replay_validator_rejects_manual_contract_drift(mutator, match: str) -> None:
    report = _manual_valid_report()
    mutator(report)

    with pytest.raises(ValueError, match=match):
        validate_report(report)
