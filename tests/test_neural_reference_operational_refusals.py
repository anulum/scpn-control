# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural reference operational refusals

"""Exercise real decoder and filesystem failures through public neural APIs and both CLIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from test_neural_equilibrium_reference_validation import _valid_pefit_reference_artifact

from scpn_control.cli import main as root_cli
from validation import validate_neural_equilibrium_reference as validator


@pytest.mark.parametrize(
    ("raw", "finding"),
    [
        (b'{"PRIVATE_INPUT_SENTINEL":1,"PRIVATE_INPUT_SENTINEL":2}', "duplicate JSON keys"),
        (b"\xff", "not UTF-8"),
        (b"{", "not valid JSON"),
        (b"[" * 1200, "not valid JSON"),
        (b'{"unused":NaN}', "non-finite JSON numbers"),
        (b"NaN", "non-finite JSON numbers"),
        (b"[NaN]", "non-finite JSON numbers"),
        (b'{"unused":Infinity}', "non-finite JSON numbers"),
        (b'{"unused":-Infinity}', "non-finite JSON numbers"),
        (b'{"unused":1e999}', "non-finite JSON numbers"),
        (b'{"unused":1e-999}', "underflowed JSON numbers"),
        (b'{"unused":-1e-999}', "underflowed JSON numbers"),
    ],
)
def test_fixed_public_decode_findings(tmp_path: Path, raw: bytes, finding: str) -> None:
    """Private duplicate keys and decoder exception details stay out of reports and root output."""
    path = tmp_path / "declaration.json"
    path.write_bytes(raw)
    report = validator.validate_neural_equilibrium_reference(path, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert (
        report["errors"][0]["error"]
        == "reference artifact " + ("contains " if "numbers" in finding or "keys" in finding else "is ") + finding
    )
    result = CliRunner().invoke(
        root_cli,
        [
            "validate-neural-equilibrium-reference",
            "--artifact-root",
            str(path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )
    assert result.exit_code == 1 and json.loads(result.stdout) == report
    assert "PRIVATE_INPUT_SENTINEL" not in result.output and "Traceback" not in result.output
    assert path.read_bytes() == raw


@pytest.mark.parametrize("number", [0.0, -0.0, 1.25, 5e-324])
def test_representable_unused_numbers_remain_admitted(tmp_path: Path, number: float) -> None:
    """Zero, signed zero, normal numbers and representable subnormals retain original declaration behavior."""
    payload: dict[str, Any] = _valid_pefit_reference_artifact()
    payload["unused"] = number
    payload["payload_sha256"] = validator.canonical_artifact_sha256(payload)
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(payload))
    report = validator.validate_neural_equilibrium_reference(path, require_reference_artifacts=True)
    assert report["status"] == "pass" and report["reference_artifacts"] == 1
    assert report["public_claims"]["predictive_equilibrium_claim_admitted"] is False


def test_nonzero_decimal_underflow_cannot_pass_body_consistency(tmp_path: Path) -> None:
    """An author hash over decoded zero cannot conceal a different nonzero decimal supplied in JSON."""
    payload: dict[str, Any] = _valid_pefit_reference_artifact()
    payload["unused"] = 0.0
    payload["payload_sha256"] = validator.canonical_artifact_sha256(payload)
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(payload).replace('"unused": 0.0', '"unused": 1e-400'))
    report = validator.validate_neural_equilibrium_reference(path, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "reference artifact contains underflowed JSON numbers"


def test_real_selected_unreadable_entry_has_fixed_finding(tmp_path: Path) -> None:
    """A selected dangling JSON symlink yields an authored IO finding without interpolating the OS exception."""
    path = tmp_path / "broken.json"
    path.symlink_to(tmp_path / "PRIVATE_MISSING_SENTINEL")
    report = validator.validate_neural_equilibrium_reference(tmp_path, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "could not read reference artifact"
    assert "PRIVATE_MISSING_SENTINEL" not in json.dumps(report)


@pytest.mark.parametrize("failure", ["parent-file", "nul"])
def test_operational_refusal_does_not_disclose_output_exception(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    """Real filesystem and path-value failures use fixed refusal through API/script/root boundaries."""
    root = tmp_path / "absent"
    blocker = tmp_path / "PRIVATE_OUTPUT_SENTINEL"
    blocker.write_bytes(b"preserved")
    output = str(blocker / "report.json") if failure == "parent-file" else "PRIVATE_OUTPUT_SENTINEL\0report.json"
    report = validator.validate_neural_equilibrium_reference(root)
    with pytest.raises((OSError, ValueError)):
        validator.write_neural_equilibrium_reference_report(report, output, artifact_root=root)
    argv = ["--artifact-root", str(root), "--output-json", output]
    assert validator.main(argv) == 1
    captured = capsys.readouterr()
    refusal = "Neural equilibrium reference FAILED: could not inspect artifacts or write report"
    assert captured.err.strip() == refusal and captured.out == ""
    result = CliRunner().invoke(root_cli, ["validate-neural-equilibrium-reference", *argv])
    if failure == "nul":
        assert result.exit_code == 2 and result.stdout == ""
        assert "report output path must not contain NUL" in result.stderr
    else:
        assert result.exit_code == 1 and result.stdout == "" and result.stderr.strip() == "Error: " + refusal
    assert "PRIVATE_OUTPUT_SENTINEL" not in result.output
    assert blocker.read_bytes() == b"preserved"


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_authored_alias_refusal_has_explicit_type(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], alias: str
) -> None:
    """Authored alias refusals preserve existing ValueError and CLI text without exposing arbitrary exceptions."""
    source = tmp_path / "declaration.json"
    source.write_text(json.dumps(_valid_pefit_reference_artifact()))
    original = source.read_bytes()
    output = source if alias == "same" else tmp_path / "alias.txt"
    if alias == "symlink":
        output.symlink_to(source)
    elif alias == "hardlink":
        output.hardlink_to(source)
    report = validator.validate_neural_equilibrium_reference(source)
    with pytest.raises(validator.NeuralReferenceReportRefusal, match="must not overwrite"):
        validator.write_neural_equilibrium_reference_report(report, output, artifact_root=tmp_path)
    argv = ["--artifact-root", str(source), "--output-json", str(output)]
    assert validator.main(argv) == 1
    assert (
        capsys.readouterr().err.strip()
        == "Neural equilibrium reference FAILED: neural reference report output must not overwrite selected input"
    )
    result = CliRunner().invoke(root_cli, ["validate-neural-equilibrium-reference", *argv])
    assert result.exit_code == 1 and "must not overwrite" in result.stderr and result.stdout == ""
    assert source.read_bytes() == original
