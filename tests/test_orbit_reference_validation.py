# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Orbit reference validation tests

"""Exercise the persisted orbit declaration reader and actual report/API/CLI boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner

from scpn_control.cli import main as root_cli
from validation import validate_orbit_reference as validator
from validation.validate_orbit_reference import validate_orbit_reference


def _valid_orbit_reference_artifact() -> dict[str, object]:
    """Build a declaration-schema carrier; no referenced orbit computation or DOI is authenticated."""
    return {
        "schema_version": "1.0",
        "source": "documented_public_reference",
        "reference_doi": "10.1016/0021-9991(81)90038-4",
        "model_id": "guiding-centre-orbit-following",
        "model_version": "0.19.0",
        "reference_dataset_id": "goldston-first-orbit-reference-2026-05-20",
        "reference_artifact_sha256": "1" * 64,
        "reference_case_count": 12,
        "executed_at": "2026-05-20T02:00:00Z",
        "units": {
            "orbit_width": "m",
            "loss_fraction": "1",
            "energy": "keV",
            "magnetic_field": "T",
        },
        "metrics": {
            "banana_width_relative_error": 0.015,
            "first_orbit_loss_abs_error": 0.018,
            "passing_trapped_classification_accuracy": 0.96,
        },
        "tolerances": {
            "banana_width_relative_error": 0.03,
            "first_orbit_loss_abs_error": 0.03,
            "passing_trapped_classification_accuracy_min": 0.90,
        },
    }


def test_strict_orbit_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    """Required empty reference roots return an artifact-root finding."""
    report = validate_orbit_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no orbit reference artifacts found"


def test_orbit_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    """Well-formed public-reference metadata returns its declared case count."""
    artifact = tmp_path / "goldston_orbit_reference.json"
    artifact.write_text(json.dumps(_valid_orbit_reference_artifact()), encoding="utf-8")

    report = validate_orbit_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "documented_public_reference"
    assert report["entries"][0]["reference_case_count"] == 12


def test_orbit_gate_accepts_real_campaign_artifact(tmp_path: Path) -> None:
    """A policy-conforming campaign URI permits declared metadata without opening the referenced bytes."""
    payload = _valid_orbit_reference_artifact()
    payload["source"] = "real_orbit_campaign"
    payload.pop("reference_doi")
    payload["campaign_artifact_uri"] = "file:///validation/reports/orbit/alpha_orbits.h5"
    artifact = tmp_path / "real_orbit_campaign.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_orbit_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "real_orbit_campaign"


def test_orbit_gate_rejects_traversing_campaign_artifact_uri(tmp_path: Path) -> None:
    """Literal traversal in a campaign URI returns its authored location finding."""
    payload = _valid_orbit_reference_artifact()
    payload["source"] = "real_orbit_campaign"
    payload.pop("reference_doi")
    payload["campaign_artifact_uri"] = "https://example.invalid/validation/../alpha_orbits.h5"
    artifact = tmp_path / "bad_real_orbit_campaign.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_orbit_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "campaign_artifact_uri"
    assert "stable remote artifact path" in report["errors"][0]["error"]


def test_orbit_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    """The declared source label must identify a public reference or campaign."""
    payload = _valid_orbit_reference_artifact()
    payload["source"] = "synthetic"
    artifact = tmp_path / "synthetic_orbit_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_orbit_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_orbit_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    """An excessive declared loss error fails its declared bound."""
    payload = _valid_orbit_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["first_orbit_loss_abs_error"] = 0.08
    artifact = tmp_path / "bad_loss_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_orbit_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "first_orbit_loss_abs_error"


def test_orbit_gate_rejects_missing_unit_contract(tmp_path: Path) -> None:
    """All four reference unit labels are mandatory."""
    payload = _valid_orbit_reference_artifact()
    payload["units"] = {"orbit_width": "m"}
    artifact = tmp_path / "bad_units_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_orbit_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "units"


@pytest.mark.parametrize("content", [b"[]", b"null", b"{broken", b"\xff", b'{"secret-key":1,"secret-key":2}'])
def test_actual_decode_findings(tmp_path: Path, content: bytes) -> None:
    """Persisted root/JSON/encoding/duplicate failures return fixed findings without echoing malformed keys."""
    path = tmp_path / "bad.json"
    path.write_bytes(content)
    report = validate_orbit_reference(path, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    finding = report["errors"][0]
    assert finding["field"] == ("root" if content in {b"[]", b"null"} else "json")
    assert "secret-key" not in finding["error"]
    assert "decode" not in finding["error"] and "line 1" not in finding["error"]


def test_selection_and_unreadable_input(tmp_path: Path) -> None:
    """Optional/required absent roots, sorted immediate files and real unreadable entries keep selection semantics."""
    missing = tmp_path / "missing"
    assert validate_orbit_reference(missing)["status"] == "pass"
    assert validate_orbit_reference(missing, require_reference_artifacts=True)["status"] == "fail"
    for name in ["b.json", "a.json"]:
        (tmp_path / name).write_text(json.dumps(_valid_orbit_reference_artifact()))
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "ignored.json").write_text("null")
    report = validate_orbit_reference(tmp_path)
    assert [Path(entry["path"]).name for entry in report["entries"]] == ["a.json", "b.json"]
    # A directory selected by the public *.json pattern is a real IO failure.
    (tmp_path / "unreadable.json").mkdir()
    report = validate_orbit_reference(tmp_path)
    assert report["reference_artifacts"] == 2 and report["status"] == "fail"
    assert report["errors"][0]["error"] == "artifact must be readable UTF-8 JSON with unique keys"


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink", "directory-root"])
def test_input_aliases_are_preserved(tmp_path: Path, alias: str) -> None:
    """Public writer/argparse/registered Click all refuse selected input aliases before mutation."""
    path = tmp_path / "input.json"
    path.write_text(json.dumps(_valid_orbit_reference_artifact()))
    before = path.read_bytes()
    output = path
    root = path
    if alias in {"symlink", "hardlink"}:
        output = tmp_path / "output.json"
        if alias == "symlink":
            output.symlink_to(path)
        else:
            os.link(path, output)
    elif alias == "directory-root":
        root = tmp_path
        output = root
    report = validate_orbit_reference(root)
    with pytest.raises(ValueError, match="must not overwrite selected input"):
        validator.write_orbit_reference_report(report, output, artifact_root=root)
    assert validator.main(["--artifact-root", str(root), "--output-json", str(output)]) == 1
    click = CliRunner().invoke(
        root_cli, ["validate-orbit-reference", "--artifact-root", str(root), "--output-json", str(path)]
    )
    assert click.exit_code == 1
    assert "could not inspect artifacts or write report" in click.output
    assert "Traceback" not in click.output and path.read_bytes() == before


def test_actual_api_and_registered_cli_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The public writer, argparse and root-registered Click persist the same declaration report and preserve inputs."""
    path = tmp_path / "input.data"
    path.write_text(json.dumps(_valid_orbit_reference_artifact()))
    before = path.read_bytes()
    report = validate_orbit_reference(path, require_reference_artifacts=True)
    output = tmp_path / "reports" / "result.json"
    validator.write_orbit_reference_report(report, output, artifact_root=path)
    assert json.loads(output.read_text()) == report
    assert (
        validator.main(
            ["--artifact-root", str(path), "--require-reference-artifacts", "--json-out", "--output-json", str(output)]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == report
    for json_mode in [False, True]:
        args = ["validate-orbit-reference", "--artifact-root", str(path), "--output-json", str(output)]
        args.append("--require-reference-artifacts")
        if json_mode:
            args.append("--json-out")
        click = CliRunner().invoke(root_cli, args)
        assert click.exit_code == 0, click.output
        assert json.loads(output.read_text()) == report
        if json_mode:
            assert json.loads(click.output) == report
        else:
            assert "Orbit reference: pass reference_artifacts=1" in click.output
    assert path.read_bytes() == before


def test_actual_refusal_and_parser_exits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Default/required/text paths and an actual blocked output parent retain documented fixed refusal/parser exits."""
    assert validator.main([]) == 0
    capsys.readouterr()
    default_click = CliRunner().invoke(root_cli, ["validate-orbit-reference"])
    assert default_click.exit_code == 0
    missing = tmp_path / "missing"
    assert validator.main(["--artifact-root", str(missing), "--require-reference-artifacts"]) == 1
    assert "no orbit reference artifacts found" in capsys.readouterr().err
    click = CliRunner().invoke(
        root_cli, ["validate-orbit-reference", "--artifact-root", str(missing), "--require-reference-artifacts"]
    )
    assert click.exit_code == 1 and "no orbit reference artifacts found" in click.output
    blocked = tmp_path / "blocked"
    blocked.write_text("keep")
    assert validator.main(["--artifact-root", str(missing), "--output-json", str(blocked / "result.json")]) == 1
    assert capsys.readouterr().err == "Orbit reference FAILED: could not inspect artifacts or write report\n"
    for args, code in [(["--help"], 0), (["--unknown-option"], 2)]:
        with pytest.raises(SystemExit) as error:
            validator.main(args)
        assert error.value.code == code


def test_actual_cold_script(tmp_path: Path) -> None:
    """The actual script imports and persists from an unrelated cwd without inherited PYTHONPATH."""
    path = tmp_path / "input.json"
    path.write_text(json.dumps(_valid_orbit_reference_artifact()))
    output = tmp_path / "saved" / "result.json"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(validator.__file__).resolve()),
            "--artifact-root",
            str(path),
            "--json-out",
            "--output-json",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == json.loads(output.read_text())
    assert json.loads(result.stdout)["reference_artifacts"] == 1
