# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public-surface claim hygiene guard tests.
"""Tests for the public-surface claim hygiene guard."""

from __future__ import annotations

import subprocess
from pathlib import Path

from _pytest.capture import CaptureFixture

from tools.check_public_surface_hygiene import Finding, iter_scanned_files, main, scan_repository, scan_text


def _categories(text: str) -> set[str]:
    """Return finding categories emitted for ``text``."""

    return {finding.category for finding in scan_text("sample.md", text)}


def test_rejects_bare_world_class_claim() -> None:
    """A bare world-class claim is rejected on outward surfaces."""

    assert "world-class" in _categories("This is a world-class controller.\n")


def test_rejects_bare_sota_abbreviation() -> None:
    """A bare SOTA claim is rejected on outward surfaces."""

    assert "SOTA" in _categories("The benchmark proves SOTA performance.\n")


def test_rejects_crown_jewel_label() -> None:
    """Product-ranking labels are rejected on outward surfaces."""

    assert "crown jewel" in _categories("The optional backend is the crown jewel.\n")


def test_rejects_unsupported_uniqueness_claim() -> None:
    """Unsupported exclusivity wording is rejected on outward surfaces."""

    assert "unsupported uniqueness" in _categories("This path does not exist elsewhere for fusion.\n")


def test_rejects_stale_notebook_output_path() -> None:
    """The public tutorials must use the repository's artifact directory."""

    assert "stale notebook output path" in _categories("--output-dir artefacts/notebook-exec\n")


def test_accepts_current_notebook_output_path() -> None:
    """The public tutorials may write executed notebooks under artifacts."""

    assert scan_text("docs/tutorials.md", "--output-dir artifacts/notebook-exec\n") == []


def test_accepts_bounded_negative_state_of_the_art_language() -> None:
    """Bounded negative comparison language is allowed."""

    assert scan_text("sample.md", "Current evidence is below the published state of the art.\n") == []


def test_accepts_candidate_maturity_language() -> None:
    """Candidate or maturity labels do not claim achieved superiority."""

    assert scan_text("sample.md", "The claim remains a SOTA-candidate ledger entry.\n") == []
    assert scan_text("sample.md", "Internal target: SOTA grade evidence before release.\n") == []


def test_rejects_public_changelog_internal_ai_profile() -> None:
    """The public changelog must not expose internal AI profile names."""

    assert scan_text("CHANGELOG.md", "uses the default director-ai profile\n") == [
        Finding(
            path="CHANGELOG.md",
            line=1,
            category="public changelog internal AI profile",
            detail="uses the default director-ai profile",
        )
    ]


def test_rejects_public_changelog_workstation_details() -> None:
    """The public changelog must not expose local workstation details."""

    assert scan_text("docs/changelog.md", "training runs on this workstation\n") == [
        Finding(
            path="docs/changelog.md",
            line=1,
            category="public changelog internal workstation detail",
            detail="training runs on this workstation",
        )
    ]


def test_rejects_public_changelog_facility_gateway_details() -> None:
    """The public changelog must not expose internal facility-gateway details."""

    assert scan_text("CHANGELOG.md", "allowlisting for facility gateways\n") == [
        Finding(
            path="CHANGELOG.md",
            line=1,
            category="public changelog facility gateway detail",
            detail="allowlisting for facility gateways",
        )
    ]


def test_allows_bounded_public_changelog_compute_host_language() -> None:
    """The public changelog may describe admitted compute-host evidence."""

    assert scan_text("CHANGELOG.md", "training must run on an admitted compute host\n") == []


def test_rejects_public_pricing_bank_identifiers() -> None:
    """The public pricing page must not publish bank account coordinates."""

    assert scan_text("docs/pricing.md", "IBAN CH14 8080 8002 1898 7544 1 / BIC RAIFCH22\n") == [
        Finding(
            path="docs/pricing.md",
            line=1,
            category="public payment bank account detail",
            detail="IBAN CH14 8080 8002 1898 7544 1 / BIC RAIFCH22",
        )
    ]


def test_rejects_public_readme_crypto_addresses() -> None:
    """The public README must not publish crypto settlement addresses."""

    assert scan_text("README.md", "ETH 0xd9b07F617bEff4aC9CAdC2a13Dd631B1980905FF\n") == [
        Finding(
            path="README.md",
            line=1,
            category="public payment crypto address",
            detail="ETH 0xd9b07F617bEff4aC9CAdC2a13Dd631B1980905FF",
        )
    ]


def test_allows_invoice_or_portal_payment_flow() -> None:
    """Public payment docs may route settlement through an invoice flow."""

    assert (
        scan_text(
            "docs/pricing.md",
            "Request a written invoice; bank coordinates are issued only on the invoice or customer portal.\n",
        )
        == []
    )


def test_rejects_rendered_markdown_legal_header() -> None:
    """Rendered Markdown must open with user-facing content."""

    assert scan_text("docs/index.md", "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->\n# Title\n") == [
        Finding(
            path="docs/index.md",
            line=1,
            category="rendered markdown legal header",
            detail="<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        )
    ]


def test_rejects_bare_rendered_markdown_legal_header() -> None:
    """Bare legal metadata at the top of Markdown is rejected."""

    assert scan_text("docs/index.md", "SPDX-License-Identifier: AGPL-3.0-or-later\n# Title\n") == [
        Finding(
            path="docs/index.md",
            line=1,
            category="rendered markdown legal header",
            detail="SPDX-License-Identifier: AGPL-3.0-or-later",
        )
    ]


def test_allows_source_file_headers() -> None:
    """Source-code SPDX headers remain valid outside rendered Markdown."""

    assert scan_text("module.py", "# SPDX-License-Identifier: AGPL-3.0-or-later\n") == []


def test_allows_markdown_that_opens_after_blank_line() -> None:
    """A blank line before content is not a legal-header finding."""

    assert scan_text("docs/index.md", "\n# Title\n") == []


def test_allows_empty_markdown() -> None:
    """Empty tracked Markdown does not produce a header finding."""

    assert scan_text("docs/empty.md", "") == []


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


def test_iter_scanned_files_skips_internal_and_guard_fixture_paths(tmp_path: Path) -> None:
    """The repository scan excludes internal surfaces and guard fixtures."""

    repo = _tracked_repo(tmp_path)
    _track(repo, "docs/index.md", b"public\n")
    _track(repo, "docs/internal/plan.md", b"world-class internal target\n")
    _track(repo, "tools/check_public_surface_hygiene.py", b"BANNED = 'world-class'\n")
    _track(repo, "tests/test_public_surface_hygiene.py", b"assert 'world-class'\n")

    scanned = {path.relative_to(repo).as_posix() for path in iter_scanned_files(repo)}

    assert "README.md" in scanned
    assert "docs/index.md" in scanned
    assert "docs/internal/plan.md" not in scanned
    assert "tools/check_public_surface_hygiene.py" not in scanned
    assert "tests/test_public_surface_hygiene.py" not in scanned


def test_scan_repository_reports_relative_paths(tmp_path: Path) -> None:
    """Repository findings use stable relative paths."""

    repo = _tracked_repo(tmp_path)
    _track(repo, "docs/index.md", b"Best-in-class control claim\n")

    assert scan_repository(repo) == [
        Finding(
            path="docs/index.md",
            line=1,
            category="best-in-class",
            detail="Best-in-class control claim",
        )
    ]


def test_scan_repository_reports_rendered_markdown_header(tmp_path: Path) -> None:
    """The repository scan rejects top legal blocks in tracked Markdown."""

    repo = _tracked_repo(tmp_path)
    _track(repo, "docs/index.md", b"<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->\n# Public docs\n")

    assert scan_repository(repo) == [
        Finding(
            path="docs/index.md",
            line=1,
            category="rendered markdown legal header",
            detail="<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        )
    ]


def test_scan_repository_skips_binary_payloads(tmp_path: Path) -> None:
    """Undecodable tracked files are skipped instead of crashing the guard."""

    repo = _tracked_repo(tmp_path)
    _track(repo, "docs/blob.md", b"\xff\xfe\x00")

    assert scan_repository(repo) == []


def test_main_passes_for_clean_repository(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """The CLI exits zero for clean public surfaces."""

    repo = _tracked_repo(tmp_path)

    assert main(["--repo", str(repo)]) == 0
    assert "PASS:" in capsys.readouterr().out


def test_main_fails_and_prints_findings(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """The CLI prints exact findings for unsafe outward claims."""

    repo = _tracked_repo(tmp_path)
    _track(repo, "docs/index.md", b"Groundbreaking safety claim\n")

    assert main(["--repo", str(repo)]) == 1
    output = capsys.readouterr().out
    assert "FAIL:" in output
    assert "docs/index.md:1: groundbreaking" in output


def test_module_entrypoint_uses_main() -> None:
    """The module keeps the standard ``python file.py`` entrypoint."""

    script = Path("tools/check_public_surface_hygiene.py")
    assert script.read_text(encoding="utf-8").rstrip().endswith("raise SystemExit(main())")
