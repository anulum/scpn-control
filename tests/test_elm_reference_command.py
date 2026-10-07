# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — ELM reference validation tests

"""Exercise persisted ELM declarations through actual public reader and CLI/report boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest
from click.testing import CliRunner
from test_elm_reference_domains import declaration

from scpn_control.cli import main as root_cli
from validation import validate_elm_reference as validator
from validation.validate_elm_reference import validate_elm_reference


@pytest.mark.parametrize("content", [b"[]", b"null", b"{broken", b"\xff", b'{"secret-key":1,"secret-key":2}'])
def test_actual_decode_findings(tmp_path: Path, content: bytes) -> None:
    """Persisted root/JSON/encoding/duplicate failures return fixed findings without echoing malformed keys."""
    path = tmp_path / "bad.json"
    path.write_bytes(content)
    report = validate_elm_reference(path, require_reference_artifacts=True)
    assert report["status"] == "fail" and report["reference_artifacts"] == 0
    finding = report["errors"][0]
    assert finding["field"] == ("root" if content in {b"[]", b"null"} else "json")
    assert "secret-key" not in finding["error"]
    assert "decode" not in finding["error"] and "line 1" not in finding["error"]


def test_selection_and_unreadable_input(tmp_path: Path) -> None:
    """Optional/required absent roots, sorted immediate files and real unreadable entries keep selection semantics."""
    missing = tmp_path / "missing"
    assert validate_elm_reference(missing)["status"] == "pass"
    assert validate_elm_reference(missing, require_reference_artifacts=True)["status"] == "fail"
    for name in ["b.json", "a.json"]:
        (tmp_path / name).write_text(json.dumps(declaration()))
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "ignored.json").write_text("null")
    report = validate_elm_reference(tmp_path)
    assert [Path(entry["path"]).name for entry in report["entries"]] == ["a.json", "b.json"]
    # A directory selected by the public *.json pattern is a real IO failure.
    (tmp_path / "unreadable.json").mkdir()
    report = validate_elm_reference(tmp_path)
    assert report["reference_artifacts"] == 2 and report["status"] == "fail"
    assert report["errors"][0]["error"] == "artifact must be readable UTF-8 JSON with unique keys"


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink", "directory-root"])
def test_input_aliases_are_preserved(tmp_path: Path, alias: str) -> None:
    """Public writer/argparse/registered Click all refuse selected input aliases before mutation."""
    path = tmp_path / "input.json"
    path.write_text(json.dumps(declaration()))
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
    report = validate_elm_reference(root)
    with pytest.raises(ValueError, match="must not overwrite selected input"):
        validator.write_elm_reference_report(report, output, artifact_root=root)
    assert validator.main(["--artifact-root", str(root), "--output-json", str(output)]) == 1
    click = CliRunner().invoke(
        root_cli, ["validate-elm-reference", "--artifact-root", str(root), "--output-json", str(path)]
    )
    assert click.exit_code == 1
    assert "could not inspect artifacts or write report" in click.output
    assert "Traceback" not in click.output and path.read_bytes() == before


def test_actual_api_and_registered_cli_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The public writer, argparse and root-registered Click persist the same declaration report and preserve inputs."""
    path = tmp_path / "input.data"
    path.write_text(json.dumps(declaration()))
    before = path.read_bytes()
    report = validate_elm_reference(path, require_reference_artifacts=True)
    output = tmp_path / "reports" / "result.json"
    validator.write_elm_reference_report(report, output, artifact_root=path)
    assert json.loads(output.read_text()) == report
    assert (
        validator.main(
            ["--artifact-root", str(path), "--require-reference-artifacts", "--json-out", "--output-json", str(output)]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == report
    for json_mode in [False, True]:
        args = ["validate-elm-reference", "--artifact-root", str(path), "--output-json", str(output)]
        args.append("--require-reference-artifacts")
        if json_mode:
            args.append("--json-out")
        click = CliRunner().invoke(root_cli, args)
        assert click.exit_code == 0, click.output
        assert json.loads(output.read_text()) == report
        if json_mode:
            assert json.loads(click.output) == report
        else:
            assert "ELM reference: pass reference_artifacts=1" in click.output
    assert path.read_bytes() == before


def test_actual_refusal_and_parser_exits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Default/required/text paths and an actual blocked output parent retain documented fixed refusal/parser exits."""
    assert validator.main([]) == 0
    capsys.readouterr()
    default_click = CliRunner().invoke(root_cli, ["validate-elm-reference"])
    assert default_click.exit_code == 0
    missing = tmp_path / "missing"
    assert validator.main(["--artifact-root", str(missing), "--require-reference-artifacts"]) == 1
    assert "no ELM reference artifacts found" in capsys.readouterr().err
    click = CliRunner().invoke(
        root_cli,
        ["validate-elm-reference", "--artifact-root", str(missing), "--require-reference-artifacts"],
    )
    assert click.exit_code == 1 and "no ELM reference artifacts found" in click.output
    blocked = tmp_path / "blocked"
    blocked.write_text("keep")
    assert validator.main(["--artifact-root", str(missing), "--output-json", str(blocked / "result.json")]) == 1
    assert capsys.readouterr().err == "ELM reference FAILED: could not inspect artifacts or write report\n"
    for args, code in [(["--help"], 0), (["--unknown-option"], 2)]:
        with pytest.raises(SystemExit) as error:
            validator.main(args)
        assert error.value.code == code


def test_actual_cold_script(tmp_path: Path) -> None:
    """The actual script imports and persists from an unrelated cwd without inherited PYTHONPATH."""
    path = tmp_path / "input.json"
    path.write_text(json.dumps(declaration()))
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


def test_actual_cyclic_output_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A real cyclic output symlink produces fixed operational refusal without overwriting a valid declaration."""
    path = tmp_path / "input.json"
    path.write_text(json.dumps(declaration()))
    before = path.read_bytes()
    output = tmp_path / "cycle.json"
    output.symlink_to(output.name)
    report = validate_elm_reference(path)
    with pytest.raises((RuntimeError, OSError)):
        validator.write_elm_reference_report(report, output, artifact_root=path)
    assert validator.main(["--artifact-root", str(path), "--output-json", str(output)]) == 1
    assert capsys.readouterr().err == "ELM reference FAILED: could not inspect artifacts or write report\n"
    result = CliRunner().invoke(
        root_cli, ["validate-elm-reference", "--artifact-root", str(path), "--output-json", str(output)]
    )
    assert result.exit_code == 1
    assert "could not inspect artifacts or write report" in result.output
    assert "loop" not in result.output.lower() and path.read_bytes() == before


def test_actual_source_checkout_without_editable_install(tmp_path: Path) -> None:
    """The real script boots the source checkout when dependency packages are available but editable .pth installation is disabled."""
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(declaration()))
    output = tmp_path / "report.json"
    dependencies = Path(sysconfig.get_paths()["purelib"])
    assert dependencies.is_dir()
    # -S bypasses all .pth setup; supply real installed dependencies only.
    # The public coverage startup preserves optional instrumentation in this
    # child because site's normal startup hook is deliberately disabled.
    program = (
        "import sys, runpy; sys.path.append(sys.argv[1]); "
        "import coverage; coverage.process_startup(); "
        "sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            program,
            str(dependencies),
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
    report = json.loads(result.stdout)
    assert report == json.loads(output.read_text())
    assert report["status"] == "pass" and report["reference_artifacts"] == 1
    assert set(report) == {"status", "root", "reference_artifacts", "require_reference_artifacts", "entries", "errors"}
