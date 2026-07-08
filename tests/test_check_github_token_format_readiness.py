# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GitHub installation-token readiness guard tests.
"""Tests for the GitHub installation-token readiness guard."""

from __future__ import annotations

import subprocess
from pathlib import Path

from _pytest.capture import CaptureFixture

from tools.check_github_token_format_readiness import Finding, iter_scanned_files, main, scan_repository, scan_text


def _categories(text: str) -> set[str]:
    """Return finding categories emitted for ``text``."""

    return {finding.category for finding in scan_text("sample.py", text)}


def test_rejects_legacy_exact_ghs_regex() -> None:
    """Exact-width ``ghs_`` regexes are brittle."""

    text = r'pattern = r"ghs_[A-Za-z0-9]{36}"'

    assert "brittle-ghs-regex" in _categories(text)


def test_rejects_exact_installation_token_length_check() -> None:
    """Exact token-length checks are rejected."""

    text = "if len(installation_token) == 40:\n    pass\n"

    assert "fixed-token-length" in _categories(text)


def test_rejects_small_token_storage_column() -> None:
    """Small SQL token columns are rejected."""

    text = "github_installation_token VARCHAR(255) NOT NULL"

    assert "small-token-storage" in _categories(text)


def test_rejects_installation_token_endpoint_without_override_header() -> None:
    """Installation-token requests must carry the stateless override header."""

    text = 'requests.post("https://api.github.com/app/installations/1/access_tokens")'

    assert "missing-stateless-token-override-header" in _categories(text)


def test_accepts_github_recommended_opaque_pattern_and_override_header() -> None:
    """Opaque token handling with the override header is accepted."""

    text = (
        r'pattern = r"ghs_[A-Za-z0-9\._]{36,}"'
        "\n"
        'requests.post("https://api.github.com/app/installations/1/access_tokens", '
        'headers={"X-GitHub-Stateless-S2S-Token": "enabled"})\n'
        "if installation_token.startswith('ghs_'):\n"
        "    pass\n"
    )

    assert scan_text("sample.py", text) == []


def _git(repo: Path, *args: str) -> None:
    """Run a Git command in ``repo``."""

    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _tracked_repo(tmp_path: Path) -> Path:
    """Create a temporary Git repository with one initial commit."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("ready\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _track(repo: Path, relative_path: str, content: bytes) -> None:
    """Write and track ``relative_path`` in ``repo``."""

    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    _git(repo, "add", relative_path)
    _git(repo, "commit", "-m", f"add {relative_path}")


def test_iter_scanned_files_uses_tracked_text_files_and_skips_private_paths(tmp_path: Path) -> None:
    """Only tracked text/workflow paths outside skipped prefixes are scanned."""

    repo = _tracked_repo(tmp_path)
    _track(repo, "src/app.py", b"installation_token = 'opaque'\n")
    _track(repo, "docs/internal/private.md", b"len(installation_token) == 40\n")
    _track(repo, "tests/fixture.py", b"len(installation_token) == 40\n")
    _track(repo, ".github/workflows/ci", b"len(installation_token) == 40\n")

    scanned = {path.relative_to(repo).as_posix() for path in iter_scanned_files(repo)}

    assert "README.md" in scanned
    assert "src/app.py" in scanned
    assert ".github/workflows/ci" in scanned
    assert "docs/internal/private.md" not in scanned
    assert "tests/fixture.py" not in scanned


def test_scan_repository_skips_binary_payloads(tmp_path: Path) -> None:
    """Undecodable tracked files are skipped instead of crashing the guard."""

    repo = _tracked_repo(tmp_path)
    _track(repo, "schema.json", b"\xff\xfe\x00")

    assert scan_repository(repo) == []


def test_scan_repository_reports_repository_relative_paths(tmp_path: Path) -> None:
    """Repository findings use stable relative paths."""

    repo = _tracked_repo(tmp_path)
    _track(repo, "src/app.py", b"if len(installation_token) == 40:\n    pass\n")

    assert scan_repository(repo) == [
        Finding(
            path="src/app.py",
            line=1,
            category="fixed-token-length",
            detail="len(installation_token) == 40",
        )
    ]


def test_main_passes_for_clean_repository(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """The CLI exits zero when no format-coupled token handling is present."""

    repo = _tracked_repo(tmp_path)

    assert main(["--repo", str(repo)]) == 0
    assert "PASS:" in capsys.readouterr().out


def test_main_fails_and_prints_findings(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """The CLI exits non-zero and prints exact findings for unsafe handling."""

    repo = _tracked_repo(tmp_path)
    _track(repo, "src/app.py", b"github_installation_token VARCHAR(255) NOT NULL\n")

    assert main(["--repo", str(repo)]) == 1
    output = capsys.readouterr().out
    assert "FAIL:" in output
    assert "src/app.py:1: small-token-storage" in output


def test_module_entrypoint_uses_main() -> None:
    """The module keeps the standard ``python file.py`` entrypoint."""

    script = Path("tools/check_github_token_format_readiness.py")
    assert script.read_text(encoding="utf-8").rstrip().endswith("raise SystemExit(main())")
