# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second analytic report publication tests
"""Exercise the real report writer and existing CLI with filesystem refusals."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from validation.validate_volt_second import build_evidence, main, validate_evidence_payload, validate_volt_second
from validation.volt_second_report import write_report


def test_report_pair_is_created_and_read_through_contract(tmp_path: Path) -> None:
    """Publish both actual files and validate the stored JSON declaration."""
    payload = build_evidence(validate_volt_second(), target_id="file-round-trip")
    destination = tmp_path / "nested/report.json"
    write_report(payload, destination)
    stored = json.loads(destination.read_text(encoding="utf-8"))
    assert stored == payload
    assert validate_evidence_payload(stored) is True
    assert "file-round-trip" in destination.with_suffix(".md").read_text(encoding="utf-8")
    assert sorted(path.name for path in destination.parent.iterdir()) == ["report.json", "report.md"]


@pytest.mark.parametrize("kind", ["directory", "alias"])
def test_report_refuses_nonregular_or_aliased_pair(tmp_path: Path, kind: str) -> None:
    """Preserve existing outputs when requested destinations are invalid."""
    payload = build_evidence(validate_volt_second(), target_id="refusal")
    destination = tmp_path / ("report.json" if kind == "directory" else "report.md")
    if kind == "directory":
        destination.mkdir()
    else:
        destination.write_bytes(b"retained")
    with pytest.raises(ValueError):
        write_report(payload, destination)
    assert destination.is_dir() if kind == "directory" else destination.read_bytes() == b"retained"


def test_cli_has_fixed_report_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Refuse a real nonregular output without exposing native exception text."""
    destination = tmp_path / "occupied"
    destination.mkdir()
    assert main(["--report", str(destination)]) == 2
    assert capsys.readouterr().err.strip() == "Volt-second report could not be published"
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("strict", [False, True])
def test_native_module_emits_actual_positive_and_negative_reports(tmp_path: Path, strict: bool) -> None:
    """Exercise native CLI reports with the real bounded model and tolerance."""
    import validation.validate_volt_second as module

    root = Path(module.__file__).resolve().parents[1]
    destination = tmp_path / "child.json"
    command = [sys.executable, "-m", "validation.validate_volt_second", "--json-out", "--report", str(destination)]
    if strict:
        command += ["--exact-tol", "1e-30"]
    completed = subprocess.run(
        command,
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": str(root) + os.pathsep + str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == int(strict), completed.stderr
    emitted = json.loads(completed.stdout)
    stored = json.loads(destination.read_text(encoding="utf-8"))
    assert emitted == stored
    assert validate_evidence_payload(stored) is (not strict)
    assert destination.with_suffix(".md").is_file()


def test_owned_source_copy_preserves_inputs_and_report_namespace(tmp_path: Path) -> None:
    """Run exact report source on an owned repository-shaped filesystem."""
    import validation.volt_second_report as current

    root = tmp_path / "repository"
    source = root / "validation/volt_second_report.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(Path(current.__file__).read_bytes())
    reference = root / "validation/reference_data/input.json"
    reference.parent.mkdir()
    reference.write_bytes(b"preserve reference")
    spec = importlib.util.spec_from_file_location("owned_volt_second_report", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = build_evidence(validate_volt_second(), target_id="owned-namespace")
    before = source.read_bytes()
    for forbidden in [source, reference]:
        with pytest.raises(ValueError):
            module.write_report(payload, forbidden)
    assert source.read_bytes() == before
    assert reference.read_bytes() == b"preserve reference"
    destination = root / "validation/reports/volt_second.json"
    module.write_report(payload, destination)
    assert validate_evidence_payload(json.loads(destination.read_text())) is True
    assert destination.with_suffix(".md").is_file()
