# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Persisted current-drive reference inspection contracts

"""Exercise actual persisted bounded evidence and public reference refusal paths.

Copies remain bounded reports, including when fields are corrupted to probe a
particular refusal. These tests never fabricate an admitted external reference.
The canonical reference directory currently supplies no positive reference case.
"""

from __future__ import annotations

import doctest
import hashlib
import json
import pydoc
import subprocess
import sys
from html import unescape
from pathlib import Path
from typing import Any

import pytest

from validation import validate_current_drive_reference as reference


@pytest.fixture
def actual_bounded_report() -> dict[str, Any]:
    """Read the real persisted analytic report; its external claim remains refused."""
    report = json.loads((reference.ROOT / "validation/reports/current_drive_claims.json").read_bytes())
    assert isinstance(report, dict)
    assert report["external_claim_allowed"] is False
    assert report["claim_status"] == "bounded_current_drive_evidence"
    return report


def _persist(root: Path, report: dict[str, Any]) -> Path:
    """Write an isolated copy of the real corpus, retaining its bounded identity."""
    path = root / "current_drive.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _source_cli(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the actual stdlib script without site packages against explicit inputs."""
    return subprocess.run(
        [
            sys.executable,
            "-S",
            str(reference.ROOT / "validation/validate_current_drive_reference.py"),
            "--artifact-root",
            str(path),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )


def _installed_cli(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the registered command through the installed package's real CLI graph."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scpn_control.cli",
            "validate-current-drive-reference",
            "--artifact-root",
            str(path),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_real_bounded_corpus_is_refused_by_api_and_both_commands(
    tmp_path: Path, actual_bounded_report: dict[str, Any]
) -> None:
    """Both runtime commands and the API refuse bounded evidence as external reference metadata."""
    path = _persist(tmp_path, actual_bounded_report)
    expected = reference.validate_current_drive_reference(path, require_reference_artifacts=True)
    assert expected["status"] == "fail" and expected["reference_artifacts"] == 0
    assert any(error["field"] == "schema_version" for error in expected["errors"])
    for run in (_source_cli, _installed_cli):
        result = run(path, "--require-reference-artifacts", "--json-out")
        assert result.returncode == 1 and result.stderr == ""
        assert json.loads(result.stdout) == expected


@pytest.mark.parametrize("required", [False, True])
def test_absence_has_explicit_optional_and_required_semantics(tmp_path: Path, required: bool) -> None:
    """Absence is optional pass or explicit failure, never positive reference evidence."""
    root = tmp_path / "absent"
    report = reference.validate_current_drive_reference(root, require_reference_artifacts=required)
    assert report["status"] == ("fail" if required else "pass")
    assert report["reference_artifacts"] == 0 and report["entries"] == []
    result = _source_cli(root, "--json-out", *(["--require-reference-artifacts"] if required else []))
    assert result.returncode == int(required) and json.loads(result.stdout) == report


@pytest.mark.parametrize("source", [[], {}, False])
def test_wrong_source_types_are_field_refusals(
    tmp_path: Path, actual_bounded_report: dict[str, Any], source: object
) -> None:
    """Unhashable and boolean source fields are refused without a structural traceback."""
    actual_bounded_report["source"] = source
    path = _persist(tmp_path, actual_bounded_report)
    expected = reference.validate_current_drive_reference(path)
    assert any(error["field"] == "source" for error in expected["errors"])
    result = _source_cli(path, "--json-out")
    assert result.returncode == 1 and result.stderr == "" and json.loads(result.stdout) == expected


@pytest.mark.parametrize("code", [[], {}, False, "unknown-code"])
def test_external_code_types_are_field_refusals(
    tmp_path: Path, actual_bounded_report: dict[str, Any], code: object
) -> None:
    """A declared external-code source cannot crash membership checking on a JSON container."""
    actual_bounded_report.update(source="ray_tracing_benchmark", external_code=code)
    path = _persist(tmp_path, actual_bounded_report)
    expected = reference.validate_current_drive_reference(path)
    assert any(error["field"] == "external_code" for error in expected["errors"])
    result = _source_cli(path, "--json-out")
    assert result.returncode == 1 and result.stderr == "" and json.loads(result.stdout) == expected


@pytest.mark.parametrize("metric", [10**400, True, -1, "wrong-type", None])
def test_invalid_metric_scalars_are_refused(
    tmp_path: Path, actual_bounded_report: dict[str, Any], metric: object
) -> None:
    """Even decoder-valid huge integers cannot escape the finite-number refusal path."""
    field = "total_power_relative_error"
    actual_bounded_report.update(
        metrics={field: metric}, tolerances={field: actual_bounded_report["total_power_relative_tolerance"]}
    )
    path = _persist(tmp_path, actual_bounded_report)
    report = reference.validate_current_drive_reference(path)
    assert any(
        error["field"] == field and error["error"] == "metric must be finite and non-negative"
        for error in report["errors"]
    )
    result = _source_cli(path, "--json-out")
    assert result.returncode == 1 and result.stderr == "" and json.loads(result.stdout) == report


@pytest.mark.parametrize(
    "raw",
    [
        b'{"UNAUTHORED_INPUT_MARKER":1,"UNAUTHORED_INPUT_MARKER":2}',
        b'{"metadata":{"UNAUTHORED_INPUT_MARKER":1,"UNAUTHORED_INPUT_MARKER":2}}',
        b'{"metadata":NaN}',
        b'{"metadata":Infinity}',
        b'{"metadata":-Infinity}',
        b'{"metadata":1e400}',
        b'{"metadata":-1e400}',
        b'{"metadata":"\xff"}',
        b'{"metadata":',
        b'{"metadata":' + b"9" * 5000 + b"}",
        pytest.param(
            b'{"metadata":'
            + b"[" * max(20000, sys.getrecursionlimit() + 100)
            + b"0"
            + b"]" * max(20000, sys.getrecursionlimit() + 100)
            + b"}",
            id="decoder-depth-limit",
        ),
    ],
)
def test_decoder_refusals_use_authored_text(tmp_path: Path, raw: bytes) -> None:
    """Ambiguous, nonfinite, non-UTF8 and malformed JSON never expose decoder or key text."""
    path = tmp_path / "reference.json"
    path.write_bytes(raw)
    report = reference.validate_current_drive_reference(path)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert len(report["errors"]) == 1 and report["errors"][0]["field"] == "json"
    assert report["errors"][0]["error"] in {
        "reference artifact contains duplicate JSON keys",
        "reference artifact contains non-finite JSON numbers",
        "reference artifact must contain valid UTF-8 JSON",
    }
    result = _source_cli(path, "--json-out")
    assert result.returncode == 1 and result.stderr == "" and json.loads(result.stdout) == report
    assert "UNAUTHORED_INPUT_MARKER" not in result.stdout
    installed = _installed_cli(path, "--json-out")
    assert installed.returncode == 1 and installed.stderr == "" and json.loads(installed.stdout) == report


def test_discovery_is_sorted_nonrecursive_and_file_suffix_independent(
    tmp_path: Path, actual_bounded_report: dict[str, Any]
) -> None:
    """Actual file copies define the directory traversal and explicit-file contract."""
    original = _persist(tmp_path, actual_bounded_report)
    original.rename(tmp_path / "z.json")
    (tmp_path / "a.json").write_bytes((tmp_path / "z.json").read_bytes())
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ignored.json").write_bytes(b"not JSON")
    (tmp_path / "directory.json").mkdir()
    report = reference.validate_current_drive_reference(tmp_path)
    paths = list(dict.fromkeys(error["path"] for error in report["errors"]))
    assert paths == [str(tmp_path / name) for name in ("a.json", "directory.json", "z.json")]
    assert any(error["error"] == "could not read reference artifact" for error in report["errors"])
    renamed = tmp_path / "explicit.data"
    renamed.write_bytes((tmp_path / "z.json").read_bytes())
    assert reference.validate_current_drive_reference(renamed)["errors"]


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("parent_failure", [False, True])
def test_report_output_filesystem_refusal_is_fixed(tmp_path: Path, installed: bool, parent_failure: bool) -> None:
    """Real directory and blocked-parent output failures have no traceback or OS exception text."""
    if parent_failure:
        parent = tmp_path / "file_parent"
        parent.write_bytes(b"occupied")
        output = parent / "report.json"
    else:
        output = tmp_path / "directory_output"
        output.mkdir()
    result = (_installed_cli if installed else _source_cli)(
        tmp_path / "absent", "--output-json", str(output), "--json-out"
    )
    assert result.stdout == ""
    if installed and not parent_failure:
        assert result.returncode == 2
        # The command line library quotes the path as a Python literal, which
        # doubles the backslashes of a Windows path.
        assert result.stderr.endswith(
            f"Error: Invalid value for '--output-json': File {str(output)!r} is a directory.\n"
        )
    else:
        assert result.returncode == (1 if installed else 2)
        assert result.stderr == ("Error: " if installed else "") + "could not write current-drive reference report\n"
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("installed", [False, True])
def test_report_writes_same_failed_report_before_emission(
    tmp_path: Path, actual_bounded_report: dict[str, Any], installed: bool
) -> None:
    """Failed inspections can persist diagnostics through the real commands without admission."""
    path = _persist(tmp_path, actual_bounded_report)
    output = tmp_path / "new_parent" / "report.json"
    result = (_installed_cli if installed else _source_cli)(path, "--json-out", "--output-json", str(output))
    assert result.returncode == 1 and result.stderr == ""
    assert output.read_text() == result.stdout
    assert json.loads(result.stdout) == reference.validate_current_drive_reference(path)


def test_native_examples_and_actual_pydoc_render(tmp_path: Path) -> None:
    """The real bounded-corpus doctest and owning-language HTML expose the same limited contract."""
    result = doctest.testmod(reference)
    assert result.failed == 0 and result.attempted == 3
    html = pydoc.HTMLDoc().docmodule(reference)
    target = tmp_path / "current_drive_reference.html"
    target.write_text(html, encoding="utf-8")
    rendered = unescape(html).replace("\xa0", " ")
    assert "validate_current_drive_reference" in rendered and "digest recomputation" in rendered


def test_public_main_text_and_json_modes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The direct public Python entrypoint has the same optional/required text and JSON decisions."""
    assert reference.main(["--artifact-root", str(tmp_path)]) == 0
    assert capsys.readouterr().out == "Current-drive reference: pass reference_artifacts=0\n"
    assert reference.main(["--artifact-root", str(tmp_path), "--require-reference-artifacts"]) == 1
    output = capsys.readouterr()
    assert output.out == "Current-drive reference: fail reference_artifacts=0\n"
    assert "no current-drive reference artifacts found" in output.err
    assert reference.main(["--artifact-root", str(tmp_path), "--json-out"]) == 0
    assert json.loads(capsys.readouterr().out)["reference_artifacts"] == 0


def test_public_main_report_writes_and_filesystem_refusals(
    tmp_path: Path, actual_bounded_report: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """The public Python CLI entrypoint persists failures and maps both filesystem failure sites."""
    path = _persist(tmp_path, actual_bounded_report)
    output = tmp_path / "reports" / "report.json"
    assert reference.main(["--artifact-root", str(path), "--output-json", str(output), "--json-out"]) == 1
    assert capsys.readouterr().out == output.read_text()
    assert reference.main(["--artifact-root", str(path)]) == 1
    assert "schema_version" not in capsys.readouterr().out
    directory = tmp_path / "output_directory"
    directory.mkdir()
    blocked_parent = tmp_path / "blocked_parent"
    blocked_parent.write_bytes(b"occupied")
    for target in (directory, blocked_parent / "report.json"):
        assert reference.main(["--artifact-root", str(path), "--output-json", str(target), "--json-out"]) == 2
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == "could not write current-drive reference report\n"


@pytest.mark.parametrize("digest", ["", "not-a-digest", "f" * 64 + "\n"])
def test_reference_digest_spelling_is_exact(tmp_path: Path, actual_bounded_report: dict[str, Any], digest: str) -> None:
    """Corrupted declared digests, including a trailing LF, cannot pass hex-format validation."""
    actual_bounded_report["reference_artifact_sha256"] = digest
    path = _persist(tmp_path, actual_bounded_report)
    report = reference.validate_current_drive_reference(path)
    assert report["status"] == "fail"
    assert any(error["field"] == "reference_artifact_sha256" for error in report["errors"])
    result = _source_cli(path, "--json-out")
    assert result.returncode == 1 and result.stderr == "" and json.loads(result.stdout) == report


def test_json_array_wrapping_actual_report_is_not_a_reference_object(
    tmp_path: Path, actual_bounded_report: dict[str, Any]
) -> None:
    """Wrapping real bounded evidence in a JSON array is a root-shape refusal."""
    path = tmp_path / "wrapped.json"
    path.write_text(json.dumps([actual_bounded_report]), encoding="utf-8")
    report = reference.validate_current_drive_reference(path)
    assert report["errors"] == [{"path": str(path), "field": "root", "error": "artifact root must be an object"}]
    result = _source_cli(path, "--json-out")
    assert result.returncode == 1 and result.stderr == "" and json.loads(result.stdout) == report


@pytest.mark.parametrize(
    "source,field",
    [
        ("documented_public_reference", "reference"),
        ("measured_deposition_replay", "shot_id"),
        ("measured_deposition_replay", "diagnostic_uri"),
        ("fokker_planck_benchmark", "external_code"),
    ],
)
def test_provenance_declarations_require_their_source_fields(
    tmp_path: Path, actual_bounded_report: dict[str, Any], source: str, field: str
) -> None:
    """Relabelling the real bounded corpus cannot substitute for source-specific provenance fields."""
    actual_bounded_report["source"] = source
    report = reference.validate_current_drive_reference(_persist(tmp_path, actual_bounded_report))
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize("tolerance", [None, [], True, 0.0, -1.0, 10**400])
def test_zero_declared_metric_still_requires_positive_finite_tolerance(
    tmp_path: Path, actual_bounded_report: dict[str, Any], tolerance: object
) -> None:
    """Corrupting a copied metric to zero does not bypass tolerance shape or finite bounds."""
    field = "total_power_relative_error"
    actual_bounded_report["metrics"] = {field: actual_bounded_report["rho_min"]}
    actual_bounded_report["tolerances"] = (
        tolerance if tolerance is None or isinstance(tolerance, list) else {field: tolerance}
    )
    report = reference.validate_current_drive_reference(_persist(tmp_path, actual_bounded_report))
    assert report["status"] == "fail"
    assert any(
        error["field"] == ("tolerances" if tolerance is None or isinstance(tolerance, list) else field)
        for error in report["errors"]
    )


def test_using_absorbed_power_as_an_error_exceeds_declared_tolerance(
    tmp_path: Path, actual_bounded_report: dict[str, Any]
) -> None:
    """A mistaken copied field cannot replace the declared relative error inside its tolerance."""
    field = "total_power_relative_error"
    actual_bounded_report.update(
        metrics={field: actual_bounded_report["total_absorbed_power_W"]},
        tolerances={field: actual_bounded_report["total_power_relative_tolerance"]},
    )
    report = reference.validate_current_drive_reference(_persist(tmp_path, actual_bounded_report))
    assert report["status"] == "fail"
    assert any(
        error["field"] == field and error["error"] == "metric exceeds declared tolerance" for error in report["errors"]
    )


def test_actual_zero_inner_radius_is_outside_current_positive_metadata_contract(
    tmp_path: Path, actual_bounded_report: dict[str, Any]
) -> None:
    """The real analytic grid's zero inner radius remains refused by the existing reference metadata rule."""
    actual_bounded_report["source_metadata"] = {
        "total_power_W": actual_bounded_report["total_absorbed_power_W"],
        "rho_points": actual_bounded_report["profile_points"],
        "rho_min": actual_bounded_report["rho_min"],
        "rho_max": actual_bounded_report["rho_max"],
    }
    report = reference.validate_current_drive_reference(_persist(tmp_path, actual_bounded_report))
    assert report["status"] == "fail"
    assert any(error["field"] == "source_metadata" for error in report["errors"])


def test_named_code_and_uri_are_only_declarations(tmp_path: Path, actual_bounded_report: dict[str, Any]) -> None:
    """Permitted external-code metadata does not turn the bounded persisted corpus into reference evidence."""
    actual_bounded_report.update(
        source="ray_tracing_benchmark",
        external_code="TORBEAM",
        reference_artifact_uri=str(reference.ROOT / "validation/reports/current_drive_claims.json"),
    )
    path = _persist(tmp_path, actual_bounded_report)
    report = reference.validate_current_drive_reference(path)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert not any(error["field"] in {"external_code", "reference_artifact_uri"} for error in report["errors"])
    result = _installed_cli(path, "--json-out")
    assert result.returncode == 1 and result.stderr == "" and json.loads(result.stdout) == report


def test_declared_hash_of_actual_bounded_artifact_remains_format_only(
    tmp_path: Path, actual_bounded_report: dict[str, Any]
) -> None:
    """Even the real corpus digest cannot turn its bounded report into external reference evidence."""
    source = reference.ROOT / "validation/reports/current_drive_claims.json"
    actual_bounded_report["reference_artifact_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    path = _persist(tmp_path, actual_bounded_report)
    report = reference.validate_current_drive_reference(path)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert not any(error["field"] == "reference_artifact_sha256" for error in report["errors"])
    result = _installed_cli(path, "--json-out")
    assert result.returncode == 1 and result.stderr == "" and json.loads(result.stdout) == report
