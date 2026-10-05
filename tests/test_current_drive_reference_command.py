# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Current-drive public reader and command boundaries

"""Exercise actual declaration decoding, persistence and registered/script commands without current_drive computation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner
from current_drive_declaration_fixtures import declaration

from scpn_control.cli import main as root_cli
from validation import validate_current_drive_reference as validator


def test_registered_help_preserved() -> None:
    """Detailed API documentation does not alter the existing public command help and option labels."""
    result = CliRunner().invoke(root_cli, ["validate-current-drive-reference", "--help"])
    assert result.exit_code == 0
    assert "Inspect local reference declarations without fetching or recomputing evidence." in result.stdout
    assert "--output-json FILE" in result.stdout and "--require-reference-artifacts" in result.stdout


@pytest.mark.parametrize(
    ("raw", "finding"),
    [
        (b'{"PRIVATE_MEMBER_SENTINEL":1,"PRIVATE_MEMBER_SENTINEL":2}', "contains duplicate JSON keys"),
        (b"\xff", "is not UTF-8"),
        (b"{", "is not valid JSON"),
        (b"[" * 1200, "is not valid JSON"),
        (b"NaN", "contains non-finite JSON numbers"),
        (b"[NaN]", "contains non-finite JSON numbers"),
        (b'{"unused":Infinity}', "contains non-finite JSON numbers"),
        (b'{"unused":-Infinity}', "contains non-finite JSON numbers"),
        (b'{"unused":1e999}', "contains non-finite JSON numbers"),
        (b'{"unused":1e-999}', "contains underflowed JSON numbers"),
        (b'{"unused":-1e-999}', "contains underflowed JSON numbers"),
    ],
)
def test_fixed_public_decode_findings(tmp_path: Path, raw: bytes, finding: str) -> None:
    """Decoder failures at every depth give fixed findings without raw keys, exception details or input mutation."""
    source = tmp_path / "declaration.json"
    source.write_bytes(raw)
    report = validator.validate_current_drive_reference(source, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == (
        "reference artifact must contain valid UTF-8 JSON"
        if finding in {"is not UTF-8", "is not valid JSON"}
        else "reference artifact " + finding
    )
    result = CliRunner().invoke(
        root_cli,
        [
            "validate-current-drive-reference",
            "--artifact-root",
            str(source),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )
    assert result.exit_code == 1
    observed = json.loads(result.stdout)
    assert observed == report
    assert "PRIVATE_MEMBER_SENTINEL" not in result.output and "Traceback" not in result.output
    assert source.read_bytes() == raw


def test_underflow_cannot_be_sealed_as_zero(tmp_path: Path) -> None:
    """A body hash computed over zero cannot admit a nonzero decimal that binary64 collapses to zero."""
    source = tmp_path / "declaration.json"
    source.write_text(json.dumps(dict(declaration(), unused=0.0)).replace('"unused": 0.0', '"unused": 1e-400'))
    report = validator.validate_current_drive_reference(source)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "reference artifact contains underflowed JSON numbers"


def test_actual_selection_and_io_refusal(tmp_path: Path) -> None:
    """Selection stays sorted/immediate, counts distinct code/run declarations and returns fixed real filesystem failures."""
    missing = tmp_path / "missing"
    assert validator.validate_current_drive_reference(missing)["status"] == "pass"
    assert validator.validate_current_drive_reference(missing, require_reference_artifacts=True)["status"] == "fail"
    for name in ["b.json", "a.json"]:
        (tmp_path / name).write_text(json.dumps(dict(declaration(), model_id=name)))
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ignored.json").write_bytes(b"null")
    (tmp_path / "unselected.txt").write_bytes(b"null")
    report = validator.validate_current_drive_reference(tmp_path, require_reference_artifacts=True)
    assert report["status"] == "pass" and report["reference_artifacts"] == 2
    assert [Path(entry["path"]).name for entry in report["entries"]] == ["a.json", "b.json"]
    (tmp_path / "broken.json").symlink_to(tmp_path / "PRIVATE_MISSING_SENTINEL")
    report = validator.validate_current_drive_reference(tmp_path, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 2
    assert report["errors"][0]["error"] == "could not read reference artifact"
    assert "PRIVATE_MISSING_SENTINEL" not in json.dumps(report)


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink", "directory-root"])
def test_selected_input_aliases_preserved(tmp_path: Path, capsys: pytest.CaptureFixture[str], alias: str) -> None:
    """Public writer and both command entry points refuse selected direct/resolved/hardlink/root aliases before writing."""
    source = tmp_path / "declaration.json"
    source.write_text(json.dumps(declaration()))
    original = source.read_bytes()
    root = tmp_path
    output = source
    if alias in {"symlink", "hardlink"}:
        output = tmp_path / "result.txt"
        if alias == "symlink":
            output.symlink_to(source)
        else:
            output.hardlink_to(source)
    elif alias == "directory-root":
        output = root
    report = validator.validate_current_drive_reference(root)
    with pytest.raises(ValueError, match="must not overwrite selected input"):
        validator.write_current_drive_reference_report(report, output, artifact_root=root)
    argv = ["--artifact-root", str(root), "--output-json", str(output)]
    assert validator.main(argv) == 2
    assert capsys.readouterr().err.strip() == "could not write current-drive reference report"
    # Click refuses a directory output as usage before entering the command.
    result = CliRunner().invoke(root_cli, ["validate-current-drive-reference", *argv])
    assert result.exit_code == (2 if alias == "directory-root" else 1)
    assert result.stdout == "" and "Traceback" not in result.output
    assert source.read_bytes() == original


@pytest.mark.parametrize("failure", ["parent-file", "nul"])
def test_fixed_operational_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str], failure: str) -> None:
    """Real output path failures propagate at API and yield fixed script/root operational or usage refusals."""
    blocker = tmp_path / "PRIVATE_OUTPUT_SENTINEL"
    blocker.write_bytes(b"preserved")
    output = str(blocker / "report.json") if failure == "parent-file" else "PRIVATE_OUTPUT_SENTINEL\0report.json"
    root = tmp_path / "absent"
    report = validator.validate_current_drive_reference(root)
    with pytest.raises((OSError, ValueError)):
        validator.write_current_drive_reference_report(report, output, artifact_root=root)
    argv = ["--artifact-root", str(root), "--output-json", output]
    assert validator.main(argv) == 2
    assert capsys.readouterr().err.strip() == "could not write current-drive reference report"
    result = CliRunner().invoke(root_cli, ["validate-current-drive-reference", *argv])
    assert result.exit_code == (2 if failure == "nul" else 1) and result.stdout == ""
    if failure == "nul":
        assert "report output path must not contain NUL" in result.stderr
    else:
        assert result.stderr.strip() == "Error: could not write current-drive reference report"
    assert "PRIVATE_OUTPUT_SENTINEL" not in result.output and blocker.read_bytes() == b"preserved"


@pytest.mark.parametrize(("valid", "json_mode"), [(True, True), (True, False), (False, True), (False, False)])
def test_public_reports_match(tmp_path: Path, capsys: pytest.CaptureFixture[str], valid: bool, json_mode: bool) -> None:
    """Public API, script and root persist the same pass/fail report for regular files regardless of input suffix."""
    source = tmp_path / "declaration.data"
    source.write_text(json.dumps(declaration() if valid else None))
    original = source.read_bytes()
    output = tmp_path / "reports" / "result.json"
    report = validator.validate_current_drive_reference(source, require_reference_artifacts=True)
    argv = ["--artifact-root", str(source), "--require-reference-artifacts", "--output-json", str(output)]
    if json_mode:
        argv.append("--json-out")
    validator.write_current_drive_reference_report(report, output, artifact_root=source)
    assert validator.main(argv) == (0 if valid else 1)
    captured = capsys.readouterr()
    result = CliRunner().invoke(root_cli, ["validate-current-drive-reference", *argv])
    assert result.exit_code == (0 if valid else 1)
    persisted = json.loads(output.read_text())
    assert output.read_bytes().endswith(b"\n")
    if json_mode:
        assert json.loads(result.stdout) == persisted
        script_report = json.loads(captured.out)
        assert report == persisted == script_report
    else:
        assert "Current-drive reference: " + report["status"] in result.stdout
        assert ("ERROR " in result.stderr) is (not valid)
    assert source.read_bytes() == original


def test_actual_cold_script(tmp_path: Path) -> None:
    """Launch the real script from outside the checkout with PYTHONPATH absent and verify required refusal/help/usage."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    script = Path(validator.__file__)
    for args, expected in [
        (["--artifact-root", str(tmp_path), "--json-out"], 0),
        (["--artifact-root", str(tmp_path), "--require-reference-artifacts", "--json-out"], 1),
        (["--help"], 0),
        (["--unknown"], 2),
    ]:
        result = subprocess.run(
            [sys.executable, str(script), *args], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
        )
        assert result.returncode == expected and "Traceback" not in result.stderr
        if "--json-out" in args:
            assert json.loads(result.stdout)["reference_artifacts"] == 0


def test_cold_without_site_packages(tmp_path: Path) -> None:
    """The stdlib-only script executes from outside the checkout with site packages and PYTHONPATH absent."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    p = subprocess.run(
        [sys.executable, "-S", validator.__file__, "--artifact-root", str(tmp_path), "--json-out"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 0 and json.loads(p.stdout)["reference_artifacts"] == 0


def test_zero_tokens_and_serialization_refusal(tmp_path: Path) -> None:
    """Exact decimal zeros remain admissible; public persistence refuses nonfinite report values before replacing output."""
    source = tmp_path / "declaration.json"
    for token in ["0e-999", "-0.00e-999"]:
        source.write_text(json.dumps(dict(declaration(), unused=0.0)).replace('"unused": 0.0', '"unused": ' + token))
        assert validator.validate_current_drive_reference(source)["status"] == "pass"
    output = tmp_path / "result.data"
    output.write_bytes(b"old unrelated output")
    report = validator.validate_current_drive_reference(source)
    validator.write_current_drive_reference_report(report, output, artifact_root=source)
    assert json.loads(output.read_text()) == report
    preserved = output.read_bytes()
    with pytest.raises(ValueError):
        validator.write_current_drive_reference_report(dict(report, unused=float("nan")), output, artifact_root=source)
    assert output.read_bytes() == preserved


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"total_power_W": 1.0, "rho_points": 2.5, "rho_min": 0.1, "rho_max": 1.5},
        {"total_power_W": 1.0, "rho_points": 2.5, "rho_min": 0.5, "rho_max": 0.1},
    ],
)
def test_declared_source_metadata_refuses_bad_domains(tmp_path: Path, metadata: object) -> None:
    """Real public declaration inspection preserves finite positive and ordered normalized source grid rules."""
    source = tmp_path / "reference.json"
    source.write_text(json.dumps(dict(declaration(), source_metadata=metadata)))
    report = validator.validate_current_drive_reference(source)
    assert report["status"] == "fail" and any(e["field"] == "source_metadata" for e in report["errors"])


@pytest.mark.parametrize(
    "source", ["documented_public_reference", "measured_deposition_replay", "fokker_planck_benchmark"]
)
def test_declared_source_identities_remain_metadata_only(tmp_path: Path, source: str) -> None:
    """Original declared DOI/shot/code branches return only schema identity without authenticating external evidence."""
    payload = dict(
        declaration(),
        source=source,
        reference_doi="DECLARATION_ONLY_TEST",
        shot_id="DECLARATION_ONLY_TEST",
        diagnostic_uri="DECLARATION_ONLY_TEST",
    )
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload))
    report = validator.validate_current_drive_reference(path)
    assert report["status"] == "pass" and report["reference_artifacts"] == 1
    assert set(report["entries"][0]) == {
        "path",
        "source",
        "model_id",
        "model_version",
        "reference_dataset_id",
        "reference_case_count",
    }
    assert "external_claim_allowed" not in report
