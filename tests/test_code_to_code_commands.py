# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual code-to-code script and imported-main command tests.

"""Observe real local benchmark commands with an unavailable optional provider."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
from campaign_command_observation import ROOT, campaign_command, observe_command


@pytest.mark.parametrize("entry", ["script", "imported-main"])
@pytest.mark.parametrize("with_torax", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_actual_code_to_code_command(tmp_path: Path, entry: str, with_torax: bool, strict: bool) -> None:
    """Execute the actual fixed scenario, persist both reports and retain blocked admission."""
    assert importlib.util.find_spec("torax") is None
    argv = campaign_command("validation.code_to_code_benchmark", entry)
    output = tmp_path / "outputs/report.json"
    markdown = tmp_path / "outputs/report.md"
    argv += ["--json-out", str(output), "--markdown-out", str(markdown)]
    if with_torax:
        argv += ["--with-torax"]
    if strict:
        argv += ["--require-external"]
    result = observe_command(tmp_path, argv, include_repository=entry == "imported-main")
    assert result.returncode == int(strict), result.stderr
    report = json.loads(output.read_text())
    local = report["benchmark"]["scpn_control"]
    assert report["schema_version"] == "scpn-control.code-to-code-benchmark.v3"
    assert len(local["rho"]) == len(local["Te_initial"]) == len(local["ne_final"]) == 50
    assert local["Te_initial"][0] == local["Ti_initial"][0] == 10.0
    assert local["Te_initial"][-1] == pytest.approx(0.5)
    assert local["ne_initial"][0] == 10.0 and local["ne_initial"][-1] == pytest.approx(1.0)
    assert local["dt"] == 0.01 and local["t_final"] == 1.0
    assert report["benchmark"]["comparison"] == {} and report["benchmark"]["torax"] is None
    external = report["external_reference"]
    assert external["admitted"] is False
    assert external["status"] == ("blocked" if with_torax else "not_requested")
    assert external["blocked_reasons"] == ["torax_not_available_or_failed" if with_torax else "torax_not_requested"]
    assert external["diagnostic_comparison_available"] is False
    payload = dict(report)
    del payload["payload_sha256"]
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert report["payload_sha256"] == expected
    text = markdown.read_text()
    assert report["payload_sha256"] in text and "External admitted: " in text
    assert report["claim_boundary"] in text
    assert not list(tmp_path.glob("scpn-c2c-*"))
    assert not (tmp_path / "validation/reports/_tmp_c2c_config.json").exists()


@pytest.mark.parametrize(("flag", "expected"), [("--help", 0), ("--unknown-option", 2)])
def test_parser_exits_before_any_report(tmp_path: Path, flag: str, expected: int) -> None:
    """Observe genuine argparse help/refusal before any solver output."""
    result = observe_command(tmp_path, campaign_command("validation.code_to_code_benchmark", "script") + [flag])
    assert result.returncode == expected
    assert not (tmp_path / "validation").exists()


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_output_alias_refusal_preserves_existing_bytes(tmp_path: Path, alias: str) -> None:
    """Refuse actual path aliases before solver execution or either overwrite."""
    output = tmp_path / "owner-output"
    output.write_bytes(b"owner bytes")
    other = output
    if alias == "symlink":
        other = tmp_path / "link"
        other.symlink_to(output)
    elif alias == "hardlink":
        other = tmp_path / "hardlink"
        os.link(output, other)
    argv = campaign_command("validation.code_to_code_benchmark", "script") + [
        "--json-out",
        str(output),
        "--markdown-out",
        str(other),
    ]
    result = observe_command(tmp_path, argv)
    assert result.returncode != 0 and "report output aliases a selected input" in result.stderr
    assert output.read_bytes() == b"owner bytes" and other.read_bytes() == b"owner bytes"
    assert not list(tmp_path.glob("scpn-c2c-*"))


def test_protected_source_and_persistent_custody_refuse_before_computation(tmp_path: Path) -> None:
    """Actual source aliases and unrecorded canonical reports cannot overwrite inputs."""
    source = ROOT / "validation/code_to_code_local.py"
    report = ROOT / "validation/reports/code_to_code_benchmark.json"
    pins = {p: p.read_bytes() for p in (source, report)}
    for index, destination in enumerate((source, report)):
        argv = campaign_command("validation.code_to_code_benchmark", "script") + [
            "--json-out",
            str(destination),
            "--markdown-out",
            str(tmp_path / f"summary-{index}.md"),
        ]
        result = observe_command(tmp_path, argv)
        assert result.returncode != 0
        assert "Running code-to-code benchmark" not in result.stdout
        assert not (tmp_path / f"summary-{index}.md").exists()
    assert all(p.read_bytes() == content for p, content in pins.items())
