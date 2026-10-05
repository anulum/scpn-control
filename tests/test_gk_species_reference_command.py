# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK species reference validation tests

"""Exercise real species reader, protected report writer, registered CLI and cold script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner
from test_gk_species_reference_domains import declaration

from scpn_control.cli import main as root_cli
from validation import validate_gk_species_reference as validator


@pytest.mark.parametrize("alias", ["direct", "resolved", "symlink", "hardlink"])
def test_input_aliases(tmp_path: Path, alias: str) -> None:
    """Writer/argparse/registered Click refuse direct/resolved/symbolic/hardlink reference aliases before mutation."""
    path = tmp_path / "input.json"
    path.write_text(json.dumps(declaration()))
    before = path.read_bytes()
    output = path
    if alias == "resolved":
        parent = tmp_path / "nested"
        parent.mkdir()
        output = parent / ".." / path.name
    elif alias in {"symlink", "hardlink"}:
        output = tmp_path / "alias.json"
        if alias == "symlink":
            output.symlink_to(path)
        else:
            os.link(path, output)
    report = validator.validate_gk_species_reference(path)
    with pytest.raises(ValueError, match="must not overwrite selected input"):
        validator.write_gk_species_reference_report(report, output, reference_path=path)
    assert validator.main(["--reference-path", str(path), "--output-json", str(output)]) == 1
    result = CliRunner().invoke(
        root_cli, ["validate-gk-species-reference", "--reference-path", str(path), "--output-json", str(output)]
    )
    assert result.exit_code == 1 and "could not inspect reference or write report" in result.output
    assert "Traceback" not in result.output and path.read_bytes() == before


def test_actual_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Public writer, argparse and registered Click persist the same actual species/operator comparison report and preserve reference bytes."""
    path = tmp_path / "input.data"
    path.write_text(json.dumps(declaration()))
    before = path.read_bytes()
    report = validator.validate_gk_species_reference(path)
    output = tmp_path / "reports" / "result.json"
    validator.write_gk_species_reference_report(report, output, reference_path=path)
    assert json.loads(output.read_text()) == report
    assert validator.main(["--reference-path", str(path), "--json-out", "--output-json", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == report
    for json_mode in [False, True]:
        args = ["validate-gk-species-reference", "--reference-path", str(path), "--output-json", str(output)]
        if json_mode:
            args.append("--json-out")
        result = CliRunner().invoke(root_cli, args)
        assert result.exit_code == 0, result.output
        assert json.loads(output.read_text()) == report
        if json_mode:
            assert json.loads(result.output) == report
        else:
            assert "GK species reference: pass cases=4" in result.output
    assert path.read_bytes() == before


def test_actual_defaults_and_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Default repository cases pass; missing/unreadable references and blocked report parents produce findings or fixed operational refusal."""
    assert validator.main([]) == 0
    capsys.readouterr()
    assert CliRunner().invoke(root_cli, ["validate-gk-species-reference"]).exit_code == 0
    missing = tmp_path / "absent.json"
    assert validator.main(["--reference-path", str(missing)]) == 1
    assert "reference must be readable UTF-8 JSON" in capsys.readouterr().err
    result = CliRunner().invoke(root_cli, ["validate-gk-species-reference", "--reference-path", str(missing)])
    assert result.exit_code == 1 and "ERROR" in result.output
    blocked = tmp_path / "blocked"
    blocked.write_text("keep")
    assert validator.validate_gk_species_reference(blocked / "child.json")["status"] == "fail"
    assert validator.validate_gk_species_reference(tmp_path)["status"] == "fail"
    assert validator.main(["--reference-path", str(missing), "--output-json", str(blocked / "result.json")]) == 1
    assert capsys.readouterr().err == "GK species reference FAILED: could not inspect reference or write report\n"
    for args, expected in [(["--help"], 0), (["--unknown-option"], 2)]:
        with pytest.raises(SystemExit) as error:
            validator.main(args)
        assert error.value.code == expected


def test_actual_cyclic_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Real cyclic input becomes a JSON finding; cyclic output gives authored refusal without overwriting the actual reference."""
    path = tmp_path / "input.json"
    path.write_text(json.dumps(declaration()))
    before = path.read_bytes()
    cycle = tmp_path / "cycle.json"
    cycle.symlink_to(cycle.name)
    assert validator.validate_gk_species_reference(cycle)["status"] == "fail"
    report = validator.validate_gk_species_reference(path)
    with pytest.raises((OSError, RuntimeError)):
        validator.write_gk_species_reference_report(report, cycle, reference_path=path)
    assert validator.main(["--reference-path", str(path), "--output-json", str(cycle)]) == 1
    assert capsys.readouterr().err == "GK species reference FAILED: could not inspect reference or write report\n"
    result = CliRunner().invoke(
        root_cli, ["validate-gk-species-reference", "--reference-path", str(path), "--output-json", str(cycle)]
    )
    assert result.exit_code == 1 and "could not inspect reference or write report" in result.output
    assert "loop" not in result.output.lower() and path.read_bytes() == before


def test_actual_cold_script(tmp_path: Path) -> None:
    """Actual script loads helpers from unrelated cwd without inherited PYTHONPATH and writes the comparison report."""
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(declaration()))
    output = tmp_path / "result.json"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(validator.__file__).resolve()),
            "--reference-path",
            str(path),
            "--output-json",
            str(output),
            "--json-out",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == json.loads(output.read_text())
    assert json.loads(result.stdout)["status"] == "pass"


def test_actual_source_checkout_without_editable_install(tmp_path: Path) -> None:
    """The real script boots the source checkout when dependency packages are available but editable .pth installation is disabled."""
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(declaration()))
    output = tmp_path / "report.json"
    dependencies = (
        Path(sys.executable).parent.parent
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
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
            "--reference-path",
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
    assert report["status"] == "pass" and report["cases"] == 4
    assert report["full_fidelity_claim_admitted"] is False
