# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural reference public declaration and persistence contracts.

"""Exercise actual declaration JSON and API/CLI behavior, without authenticated/executed physics claims."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from test_neural_equilibrium_reference_validation import _valid_pefit_reference_artifact

from scpn_control.cli import main as root_cli
from scpn_control.cli_reference_validators import validate_neural_equilibrium_reference_command
from validation import validate_neural_equilibrium_reference as validator


@pytest.fixture
def declaration_file(tmp_path: Path) -> Path:
    """Persist the original metadata fixture; declared P-EFIT/arrays/weights remain unexecuted and unauthenticated."""
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(_valid_pefit_reference_artifact()), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("field", "value", "finding"),
    [
        ("source", [], "source"),
        ("payload_sha256", "é" * 64, "payload_sha256"),
        ("payload_sha256", None, "payload_sha256"),
        ("schema_version", "wrong", "schema_version"),
        ("model_version", " ", "model_version"),
        ("target_schema", [], "target_schema"),
        ("grid_shape", None, "grid_shape"),
        ("grid_shape", [2], "grid_shape"),
        ("grid_shape", [True, 3], "grid_shape"),
        ("grid_shape", [0, 3], "grid_shape"),
        ("units", [], "units"),
        ("reference_equilibria_count", True, "reference_equilibria_count"),
        ("reference_equilibria_count", 0, "reference_equilibria_count"),
        ("metrics", [], "metrics"),
        ("tolerances", [], "tolerances"),
        ("reference_artifact_uri", None, "reference_artifact_uri"),
        ("reference_artifact_uri", "nul\0name", "reference_artifact_uri"),
        ("reference_artifact_uri", "/absolute/reference.npz", "reference_artifact_uri"),
        ("reference_artifact_sha256", "a" * 64 + "\n", "reference_artifact_sha256"),
    ],
)
def test_public_declaration_field_refusals(declaration_file: Path, field: str, value: object, finding: str) -> None:
    """Persist malformed declaration fields and inspect authored findings instead of private preconditions."""
    payload: dict[str, Any] = json.loads(declaration_file.read_text())
    payload[field] = value
    if field != "payload_sha256":
        payload["payload_sha256"] = validator.canonical_artifact_sha256(payload)
    declaration_file.write_text(json.dumps(payload), encoding="utf-8")
    report = validator.validate_neural_equilibrium_reference(declaration_file, require_reference_artifacts=True)
    assert report["status"] == "fail"
    assert any(error["field"] == finding for error in report["errors"])
    assert report["public_claims"]["predictive_equilibrium_claim_admitted"] is False


@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("number", [-1, True, "1", 10**400, float("inf")])
def test_public_numeric_refusals(declaration_file: Path, block: str, number: object) -> None:
    """Declared errors/tolerances refuse wrong types, signs, nonfinite and unrepresentable integers without overflow."""
    payload: dict[str, Any] = json.loads(declaration_file.read_text())
    payload[block]["psi_rmse_Wb"] = number
    payload["payload_sha256"] = "bad" if isinstance(number, float) else validator.canonical_artifact_sha256(payload)
    declaration_file.write_text(json.dumps(payload), encoding="utf-8")
    report = validator.validate_neural_equilibrium_reference(declaration_file, require_reference_artifacts=True)
    assert report["status"] == "fail"
    assert any(error["field"] == "psi_rmse_Wb" for error in report["errors"])


@pytest.mark.parametrize("reference", [None, "", "doi:10.example/declaration", "https://example.invalid/declared"])
def test_public_reference_declaration_limits(declaration_file: Path, reference: str | None) -> None:
    """Public-reference declarations require only nonblank metadata; no URL/DOI fetch or physics admission occurs."""
    payload: dict[str, Any] = json.loads(declaration_file.read_text())
    payload["source"] = "documented_public_reference"
    payload["reference_doi"] = reference
    payload["reference_artifact_uri"] = "https://example.invalid/declared-array"
    payload["payload_sha256"] = validator.canonical_artifact_sha256(payload)
    declaration_file.write_text(json.dumps(payload), encoding="utf-8")
    report = validator.validate_neural_equilibrium_reference(declaration_file, require_reference_artifacts=True)
    assert report["status"] == ("pass" if reference else "fail")
    assert report["public_claims"]["predictive_equilibrium_claim_admitted"] is False


@pytest.mark.parametrize("raw", [b"[]", b"{", b' {"duplicate":1,"duplicate":2}', b"\xff"])
def test_public_decode_and_object_refusals(declaration_file: Path, raw: bytes) -> None:
    """Malformed UTF-8/JSON/objects become report errors with no reference declaration admitted."""
    declaration_file.write_bytes(raw)
    report = validator.validate_neural_equilibrium_reference(declaration_file, require_reference_artifacts=True)
    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"]


def test_exact_captured_bytes_and_report_digest(declaration_file: Path) -> None:
    """CRLF input hashes exact raw bytes; canonical payload/report consistency remains distinct from byte spelling."""
    original = declaration_file.read_bytes()
    raw = original.replace(b", ", b",\r\n")
    declaration_file.write_bytes(raw)
    report = validator.validate_neural_equilibrium_reference(declaration_file, require_reference_artifacts=True)
    assert report["status"] == "pass"
    assert report["entries"][0]["artifact_file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert declaration_file.read_bytes() == raw
    assert report["public_claims"]["predictive_equilibrium_claim_admitted"] is False
    payload = dict(report)
    digest = payload["payload_sha256"]
    payload["payload_sha256"] = None
    assert (
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
        == digest
    )


def test_canonical_checksum_contract() -> None:
    """Public canonical hashing ignores its own field/order, preserves caller state and refuses nonfinite JSON."""
    payload: dict[str, object] = {"z": 3, "payload_sha256": "ignored"}
    assert validator.canonical_artifact_sha256(payload) == validator.canonical_artifact_sha256({"z": 3})
    assert payload == {"z": 3, "payload_sha256": "ignored"}
    with pytest.raises(ValueError):
        validator.canonical_artifact_sha256({"z": float("nan")})


def test_absent_required_inputs_and_real_read_failure(tmp_path: Path) -> None:
    """Required missing roots and dangling selected JSON links fail; optional absent roots remain diagnostic-only passes."""
    missing = tmp_path / "missing"
    assert validator.validate_neural_equilibrium_reference(missing)["status"] == "pass"
    assert (
        validator.validate_neural_equilibrium_reference(missing, require_reference_artifacts=True)["status"] == "fail"
    )
    broken = tmp_path / "broken.json"
    broken.symlink_to(missing)
    report = validator.validate_neural_equilibrium_reference(tmp_path, require_reference_artifacts=True)
    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "json"


@pytest.mark.parametrize("alias_kind", ["same", "symlink", "hardlink"])
def test_public_writer_protects_selected_input_aliases(declaration_file: Path, tmp_path: Path, alias_kind: str) -> None:
    """Shared API and both CLI entry points refuse aliases without changing selected source bytes."""
    original = declaration_file.read_bytes()
    report = validator.validate_neural_equilibrium_reference(declaration_file)
    output = declaration_file if alias_kind == "same" else tmp_path / "alias.md"
    if alias_kind == "symlink":
        output.symlink_to(declaration_file)
    elif alias_kind == "hardlink":
        output.hardlink_to(declaration_file)
    with pytest.raises(ValueError, match="must not overwrite"):
        validator.write_neural_equilibrium_reference_report(report, output, artifact_root=tmp_path)
    assert validator.main(["--artifact-root", str(declaration_file), "--output-json", str(output)]) == 1
    result = CliRunner().invoke(
        validate_neural_equilibrium_reference_command,
        ["--artifact-root", str(declaration_file), "--output-json", str(output)],
    )
    assert result.exit_code == 1
    assert "must not overwrite" in result.output
    assert declaration_file.read_bytes() == original


def test_shared_writer_and_cli_reports(
    declaration_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Real API/argparse/Click outputs preserve the selected report, schema, byte digest and diagnostic-only claim."""
    output = tmp_path / "out" / "report.json"
    report = validator.validate_neural_equilibrium_reference(declaration_file)
    validator.write_neural_equilibrium_reference_report(report, output, artifact_root=declaration_file)
    assert json.loads(output.read_text()) == report
    assert validator.main(["--artifact-root", str(declaration_file), "--json-out", "--output-json", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == report
    assert validator.main(["--artifact-root", str(declaration_file)]) == 0
    assert "reference_artifacts=1" in capsys.readouterr().out
    result = CliRunner().invoke(
        validate_neural_equilibrium_reference_command, ["--artifact-root", str(declaration_file), "--json-out"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == report
    result = CliRunner().invoke(
        validate_neural_equilibrium_reference_command, ["--artifact-root", str(declaration_file)]
    )
    assert result.exit_code == 0 and "reference_artifacts=1" in result.output
    result = CliRunner().invoke(
        root_cli,
        [
            "validate-neural-equilibrium-reference",
            "--artifact-root",
            str(declaration_file),
            "--json-out",
            "--output-json",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == report
    assert json.loads(output.read_text()) == report


def test_real_cli_failure_summary_default_and_persistence(
    declaration_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Default/required invalid declarations and blocked output parent use authored nonzero entry-point behavior."""
    declaration_file.write_text("[]", encoding="utf-8")
    args = ["--artifact-root", str(declaration_file), "--require-reference-artifacts"]
    assert validator.main(args) == 1
    assert "ERROR" in capsys.readouterr().err
    result = CliRunner().invoke(validate_neural_equilibrium_reference_command, args)
    assert result.exit_code == 1 and "ERROR" in result.output
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")
    assert validator.main([*args, "--output-json", str(blocker / "report.json")]) == 1
    assert "FAILED" in capsys.readouterr().err
    assert validator.main([]) == 0
    assert "reference_artifacts=0" in capsys.readouterr().out
    result = CliRunner().invoke(validate_neural_equilibrium_reference_command, [])
    assert result.exit_code == 0 and "reference_artifacts=0" in result.output
    for argv, code in [(["--help"], 0), (["--unknown"], 2)]:
        with pytest.raises(SystemExit) as failure:
            validator.main(argv)
        assert failure.value.code == code


def test_real_cold_cli(declaration_file: Path, tmp_path: Path) -> None:
    """Invoke the actual validator script from another cwd and inspect the persisted declaration report."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(validator.__file__).resolve()),
            "--artifact-root",
            str(declaration_file),
            "--json-out",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["reference_artifacts"] == 1
