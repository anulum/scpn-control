# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Source-header policy CLI tests.

"""Exercise the tracked source-header policy through real repository paths."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from tools import check_source_headers

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/check_source_headers.py"
POLICY = ROOT / "tools/source_header_policy.toml"


def _write_policy(tmp_path: Path, text: str) -> Path:
    policy_path = tmp_path / "source_header_policy.toml"
    policy_path.write_text(text, encoding="utf-8")
    return policy_path


def _policy_variant(tmp_path: Path, old: str, new: str) -> Path:
    source = POLICY.read_text(encoding="utf-8")
    assert old in source
    return _write_policy(tmp_path, source.replace(old, new, 1))


def test_live_repository_source_header_policy_passes() -> None:
    """Every tracked path is classified and every enforced header is exact."""
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--root", str(ROOT), "--policy", str(POLICY), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert completed.returncode == 0, payload["findings"][:10]
    assert payload["passed"] is True
    assert payload["classifications"].get("unclassified", 0) == 0


def test_policy_classifies_every_live_tracked_path() -> None:
    """A new file format cannot bypass enforcement or reviewed exemption."""
    policy = check_source_headers.load_policy(POLICY)
    dispositions = [check_source_headers.classify(path, policy)[0] for path in check_source_headers.tracked_paths(ROOT)]
    assert "unclassified" not in dispositions
    assert {"enforced", "exempt"}.issubset(dispositions)


def test_exact_headers_accept_directive_and_reject_identity_drift(tmp_path: Path) -> None:
    """The file-path gate accepts a shebang but rejects ASCII identity drift."""
    relative = Path("probe.py")
    header = check_source_headers.expected_header(relative, "Header policy probe.")
    (tmp_path / relative).write_text("#!/usr/bin/env python3\n" + "\n".join(header) + "\n", encoding="utf-8")
    assert check_source_headers.header_finding(tmp_path, relative) is None

    drifted = (tmp_path / relative).read_text(encoding="utf-8").replace("Šotek", "Sotek", 1)
    (tmp_path / relative).write_text(drifted, encoding="utf-8")
    finding = check_source_headers.header_finding(tmp_path, relative)
    assert finding is not None
    assert finding.category == "header_mismatch"


@pytest.mark.parametrize("relative", [Path("probe.lean"), Path("probe.html"), Path("probe.rs")])
def test_format_native_headers_are_accepted(tmp_path: Path, relative: Path) -> None:
    """Lean, HTML, and slash-comment source headers retain native syntax."""
    header = check_source_headers.expected_header(relative, "Format-native policy probe.")
    (tmp_path / relative).write_text("\n".join(header) + "\n", encoding="utf-8")
    assert check_source_headers.header_finding(tmp_path, relative) is None


@pytest.mark.parametrize(
    ("relative", "mutation"),
    [
        (Path("probe.lean"), lambda lines: []),
        (Path("probe.lean"), lambda lines: ["BROKEN", *lines[1:]]),
        (Path("probe.lean"), lambda lines: [*lines[:7], "wrong purpose", lines[8]]),
        (Path("probe.lean"), lambda lines: [*lines[:7], "SCPN Control — ", lines[8]]),
        (Path("probe.lean"), lambda lines: [*lines[:8], "BROKEN"]),
        (Path("probe.py"), lambda lines: []),
        (Path("probe.py"), lambda lines: ["BROKEN", *lines[1:]]),
        (Path("probe.py"), lambda lines: [*lines[:6], "# wrong purpose"]),
        (Path("probe.py"), lambda lines: [*lines[:6], "# SCPN Control — "]),
        (Path("probe.html"), lambda lines: [*lines[:6], "<!-- SCPN Control — purpose"]),
    ],
)
def test_malformed_native_headers_are_rejected(
    tmp_path: Path,
    relative: Path,
    mutation: Callable[[list[str]], list[str]],
) -> None:
    """Every semantic component of a format-native header is mandatory."""
    lines = check_source_headers.expected_header(relative, "Format-native policy probe.")
    malformed = mutation(lines)
    (tmp_path / relative).write_text("\n".join(malformed) + "\n", encoding="utf-8")
    finding = check_source_headers.header_finding(tmp_path, relative)
    assert finding is not None
    assert finding.category == "header_mismatch"


def test_non_utf8_enforced_source_is_rejected(tmp_path: Path) -> None:
    """An enforced source file cannot bypass the contract with binary bytes."""
    relative = Path("probe.py")
    (tmp_path / relative).write_bytes(b"\xff\xfe")
    finding = check_source_headers.header_finding(tmp_path, relative)
    assert finding is not None
    assert finding.category == "non_utf8_enforced"


@pytest.mark.parametrize(
    ("policy_text", "message"),
    [
        ('schema = "wrong"\n[enforced]\nsuffixes = []\nnames = []\n', "unsupported"),
        ('schema = "scpn-control.source-header-policy.v1"\n', "[enforced]"),
        (
            'schema = "scpn-control.source-header-policy.v1"\n'
            'exemptions = ["invalid"]\n[enforced]\nsuffixes = []\nnames = []\n',
            "TOML table",
        ),
        (
            'schema = "scpn-control.source-header-policy.v1"\n'
            "[enforced]\nsuffixes = []\nnames = []\n"
            '[[exemptions]]\ncategory = ""\nreason = "A sufficiently specific reviewed reason."\n',
            "category",
        ),
        (
            'schema = "scpn-control.source-header-policy.v1"\n'
            "[enforced]\nsuffixes = []\nnames = []\n"
            '[[exemptions]]\ncategory = "probe"\nreason = "too short"\n',
            "specific reason",
        ),
        (
            'schema = "scpn-control.source-header-policy.v1"\n'
            '[enforced]\nsuffixes = [".py"]\nnames = []\n'
            '[[exemptions]]\ncategory = "probe"\nreason = "A sufficiently specific reviewed reason."\n'
            'suffixes = [".PY"]\n',
            "overlapping",
        ),
        (
            'schema = "scpn-control.source-header-policy.v1"\n'
            '[enforced]\nsuffixes = []\nnames = ["Makefile"]\n'
            '[[exemptions]]\ncategory = "probe"\nreason = "A sufficiently specific reviewed reason."\n'
            'names = ["Makefile"]\n',
            "overlapping",
        ),
    ],
)
def test_invalid_policy_contracts_fail_closed(tmp_path: Path, policy_text: str, message: str) -> None:
    """Malformed, vague, and overlapping policy entries are rejected."""
    with pytest.raises(ValueError, match=message):
        check_source_headers.load_policy(_write_policy(tmp_path, policy_text))


def test_live_audit_reports_real_unclassified_format(tmp_path: Path) -> None:
    """Removing a reviewed live suffix makes its tracked files fail closed."""
    policy_path = _policy_variant(tmp_path, '".png", ".pub", ".svg"', '".png", ".svg"')
    result = check_source_headers.audit(ROOT, policy_path)
    assert result["passed"] is False
    assert result["classifications"]["unclassified"] > 0
    assert any(finding["category"] == "unclassified" for finding in result["findings"])


def test_live_audit_reports_real_header_mismatch(tmp_path: Path) -> None:
    """Moving tracked prose into enforced scope exposes its missing source header."""
    source = POLICY.read_text(encoding="utf-8")
    source = source.replace('  ".js",\n', '  ".js",\n  ".md",\n', 1)
    source = source.replace('suffixes = [".bib", ".md", ".tex"]', 'suffixes = [".bib", ".tex"]', 1)
    result = check_source_headers.audit(ROOT, _write_policy(tmp_path, source))
    assert result["passed"] is False
    assert any(finding["category"] == "header_mismatch" for finding in result["findings"])


def test_cli_modes_report_pass_failure_and_policy_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The public entry point exposes stable success, finding, JSON, and error modes."""
    assert check_source_headers.main(["--root", str(ROOT), "--policy", str(POLICY)]) == 0
    assert capsys.readouterr().out == "Source-header policy passed\n"

    assert check_source_headers.main(["--root", str(ROOT), "--policy", str(POLICY), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True

    unclassified = _policy_variant(tmp_path, '".png", ".pub", ".svg"', '".png", ".svg"')
    assert check_source_headers.main(["--root", str(ROOT), "--policy", str(unclassified)]) == 1
    assert ": unclassified: no policy entry" in capsys.readouterr().err

    invalid = _write_policy(tmp_path, 'schema = "wrong"\n')
    assert check_source_headers.main(["--root", str(ROOT), "--policy", str(invalid)]) == 2
    assert "source-header policy error: unsupported" in capsys.readouterr().err


def test_help_is_side_effect_free(tmp_path: Path) -> None:
    """The CLI help path writes no report or repository artefact."""
    before = tuple(tmp_path.iterdir())
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "reviewed format exemptions" in completed.stdout
    assert tuple(tmp_path.iterdir()) == before
