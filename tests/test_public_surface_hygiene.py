# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public-surface claim hygiene guard tests.
"""Tests for the public-surface claim hygiene guard."""

from __future__ import annotations

import doctest
import os
import pydoc
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture

from tools import check_public_surface_hygiene as guard
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


@pytest.mark.parametrize(
    "claim",
    (
        "The controller has authority over plasma modes.",
        "This is the entry point a real control loop would call.",
        "This maps directly to SNN or PID output amplitude.",
        "This is the complete real-time monitoring loop.",
    ),
)
def test_rejects_unidentified_reactor_phase_control_claims(claim: str) -> None:
    """Example oscillator models cannot claim an unidentified reactor loop."""
    assert "unidentified reactor phase-control claim" in _categories(claim)


def test_rejects_public_operational_task_list() -> None:
    """Unchecked work items belong only on private operational surfaces."""
    assert "public operational task" in _categories("- [ ] Run the next validation campaign.\n")


def test_rejects_public_operational_headings_and_internal_ids() -> None:
    """Public prose must not expose prioritisation or internal task identity."""
    assert "public operational heading" in _categories("## Current priority order\n")
    assert "public operational heading" in _categories("## Current priorities\n")
    assert "public operational heading" in _categories("## Prioritisation\n")
    assert "internal task identifier" in _categories("Execute CONTROL-AUD-001 next.\n")
    assert "internal task identifier" in _categories("Closed CTL-G07 in this release.\n")
    assert "internal task identifier" in _categories("Completed L2F-90c(a).\n")
    assert "internal task identifier" in _categories("Closed SYS-AUDIT-02-MYPY1.\n")
    assert "internal task identifier" in _categories("Waiver WCG-5e is active.\n")
    assert "internal task identifier" in _categories("Continue with R3-S4.\n")
    assert "internal task identifier" in _categories("Resolve U-015 next.\n")


@pytest.mark.parametrize(
    "identifier",
    (
        "BL-19",
        "bl_19",
        "QWC-04",
        "qwc_04",
        "WS-12",
        "WS-alpha",
        "WS_12A",
    ),
)
def test_rejects_generic_internal_queue_and_workstream_identifiers(identifier: str) -> None:
    """Generic queue/workstream labels remain private coordination metadata."""
    findings = scan_text("src/runtime_identity.txt", f"owner={identifier}\n")

    assert {finding.category for finding in findings} == {"internal queue or workstream identifier"}


@pytest.mark.parametrize(
    "path",
    (
        "src/bl_19_controller.py",
        "validation/qwc-04.json",
        "docs/WS_12.md",
    ),
)
def test_rejects_generic_internal_identifiers_in_tracked_paths(path: str) -> None:
    """Internal queue/workstream labels are invalid as tracked path names."""
    assert scan_text(path, "descriptive content\n") == [
        Finding(
            path=path,
            line=0,
            category="internal queue or workstream identifier",
            detail=path,
        )
    ]


@pytest.mark.parametrize(
    "identifier",
    (
        "CONTROL-TORAX-CONSUMER-001",
        "CONTROL-PHASE-REACTOR-LOOP-001",
        "CONTROL-REACTOR-PLUGIN-PROTOCOL-001",
    ),
)
def test_rejects_internal_control_lane_identifiers(identifier: str) -> None:
    """CONTROL queue identities must not become public contract names."""
    findings = scan_text("docs/control/runtime.md", f"contract={identifier}\n")

    assert {finding.category for finding in findings} == {"internal CONTROL lane identifier"}


def test_rejects_internal_lif_lane_identifier() -> None:
    """The private LIF handoff lane must not become a runtime identity."""
    findings = scan_text("validation/lif_state.json", '{"profile": "LIF-FF-02"}\n')

    assert {finding.category for finding in findings} == {"internal LIF lane identifier"}


@pytest.mark.parametrize(
    ("path", "category"),
    (
        ("validation/CONTROL-TORAX-CONSUMER-001.json", "internal CONTROL lane identifier"),
        ("tests/LIF-FF-02.json", "internal LIF lane identifier"),
    ),
)
def test_rejects_internal_lane_identifiers_in_tracked_paths(path: str, category: str) -> None:
    """Private CONTROL and LIF lane labels cannot become artifact paths."""
    assert scan_text(path, "descriptive content\n") == [Finding(path=path, line=0, category=category, detail=path)]


@pytest.mark.parametrize(
    ("path", "content"),
    (
        (
            "src/scpn_control/phase/ws_phase_stream.py",
            'WEBSOCKET_EVIDENCE = "scpn-control.websocket-runtime-evidence.v1"\n',
        ),
        (
            "examples/streamlit_ws_client.py",
            'ws_url = "ws://localhost:8765"\n',
        ),
        ("examples/client.py", '_parser.add_argument("--ws-url")\n'),
        ("validation/host.json", '"host_id": "ws-aud002"\n'),
        ("validation/report.json", '"id": "ws-nmpc-hardening-host"\n'),
    ),
)
def test_accepts_descriptive_websocket_identifiers(path: str, content: str) -> None:
    """Established WebSocket abbreviations are domain names, not task codes."""
    assert scan_text(path, content) == []


def test_rejects_internal_ids_on_source_config_and_workflow_surfaces() -> None:
    """Internal identifier families are governed outside Markdown and JSON."""
    source = scan_text("module.py", 'VALIDATION = "L2F-22"\n')
    config = scan_text("pyproject.toml", "# SYS-AUDIT-02-MYPY1\n")
    workflow = scan_text(".github/workflows/ci.yml", "# WCG-5e\n")

    assert {finding.category for finding in source} == {"internal task identifier"}
    assert {finding.category for finding in config} == {"internal task identifier"}
    assert {finding.category for finding in workflow} == {"internal task identifier"}


def test_rejects_internal_coverage_campaign_identifier_in_content_and_path() -> None:
    """Coverage-campaign labels are rejected from prose and tracked filenames."""
    content = scan_text("tests/test_solver_edges.py", "Legacy COV-1 coverage campaign.\n")
    path = scan_text("tests/test_solver_cov1.py", "Descriptive test content.\n")

    assert {finding.category for finding in content} == {"internal coverage campaign identifier"}
    assert path == [
        Finding(
            path="tests/test_solver_cov1.py",
            line=0,
            category="internal coverage campaign identifier",
            detail="tests/test_solver_cov1.py",
        )
    ]


def test_rejects_internal_safety_campaign_identifiers() -> None:
    """Safety behaviour must be described without internal campaign labels."""
    for identifier in ("SS-14", "SS-14 b", "SS-12/F13", "CF-5", "SP-4", "LOCK-4"):
        findings = scan_text("tests/test_safety_edges.py", f"Legacy {identifier} boundary.\n")
        assert {finding.category for finding in findings} == {"internal safety campaign identifier"}


def test_rejects_internal_pulsed_control_campaign_identifiers() -> None:
    """Pulsed-control surfaces must describe responsibility, not campaign stage."""
    for identifier in ("CON-C", "CON-C.1", "CON-C.7"):
        findings = scan_text("docs/control/runtime_admission.md", f"Legacy {identifier} stage.\n")
        assert {finding.category for finding in findings} == {"internal pulsed-control campaign identifier"}


def test_rejects_internal_parity_campaign_identifier() -> None:
    """Parity evidence must retain a descriptive source reference."""
    findings = scan_text("validation/parity.json", '"raw_reference": "PARITY-1"\n')
    assert {finding.category for finding in findings} == {"internal parity campaign identifier"}


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("artifact=gai02_torax_hybrid", "internal controller workstream identifier"),
        ("artifact=gdep01_digital_twin", "internal controller workstream identifier"),
        ("artifact=gneu03_fueling", "internal controller workstream identifier"),
        ("CONTROL-F841-REVIEW", "internal implementation-review identifier"),
        ("INT-7", "internal integration-stage identifier"),
        ("WP-PY3", "internal polyglot work-package identifier"),
        ("test_reproduce_o002", "internal resolved-finding identifier"),
    ],
)
def test_rejects_residual_internal_identifiers(text: str, category: str) -> None:
    """Residual local workstream labels remain forbidden on public surfaces."""
    findings = scan_text("src/runtime_identity.txt", text)
    assert {finding.category for finding in findings} == {category}


def test_rejects_private_operational_path_reference() -> None:
    """Outward surfaces must not link into ignored planning trees."""
    assert "private operational path" in _categories("See docs/internal/TODO.md for status.\n")
    assert "private operational path" in _categories("Evidence is in .coordination/sessions/demo.md.\n")


def test_rejects_operational_fields_in_public_json() -> None:
    """Public metadata declares claim requirements rather than work actions."""
    findings = scan_text("validation/public.json", '{\n  "required_actions": ["run campaign"]\n}\n')
    assert {finding.category for finding in findings} == {"public unresolved execution plan"}


def test_allows_reviewed_tutorial_navigation_heading() -> None:
    """Tutorial navigation may use a reviewed Next Steps heading."""
    assert scan_text("docs/tutorials/first_steps.md", "## Next Steps\n\nInstall the package.\n") == []


def test_allows_github_contribution_template_checkboxes() -> None:
    """Pull-request templates may contain contributor-facing checkboxes."""
    assert scan_text(".github/PULL_REQUEST_TEMPLATE.md", "- [ ] I ran focused tests.\n") == []


def test_allows_code_examples_that_look_like_operational_headings() -> None:
    """Markdown code fences may contain comments that resemble headings."""
    text = "```python\n# Apply gains for next step\n```\n"
    assert scan_text("docs/example.md", text) == []


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


def test_rejects_validation_markdown_legal_header_producer() -> None:
    """Validation code must not regenerate a forbidden public preamble."""
    source = 'lines = ["<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->", "# Report"]\n'
    assert scan_text("validation/render_report.py", source) == [
        Finding(
            path="validation/render_report.py",
            line=1,
            category="rendered Markdown legal-header producer",
            detail=source.strip(),
        )
    ]


def test_allows_validation_source_code_header() -> None:
    """The producer guard does not reject the Python file's own legal header."""
    source = "# SPDX-License-Identifier: AGPL-3.0-or-later\n"
    assert scan_text("validation/render_report.py", source) == []


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
    """Index an actual maintained guide in a private repository without making commits."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    shutil.copy2(guard.REPO_ROOT / "docs/development.md", repo / "README.md")
    _git(repo, "add", "README.md")
    return repo


def _track(repo: Path, relative_path: str, content: bytes) -> None:
    """Write and index the requested worktree payload without a commit."""
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    _git(repo, "add", relative_path)


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


def test_iter_scanned_files_skips_tracked_worktree_deletions(tmp_path: Path) -> None:
    """A not-yet-staged file removal does not make the worktree scan crash."""
    repo = _tracked_repo(tmp_path)
    (repo / "README.md").unlink()

    assert list(iter_scanned_files(repo)) == []


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


def test_scan_repository_rejects_internal_code_and_keeps_websocket_path(tmp_path: Path) -> None:
    """The tracked-repository surface distinguishes workstream codes from WebSocket names."""
    repo = _tracked_repo(tmp_path)
    _track(repo, "src/ws_phase_stream.py", b"WEBSOCKET_SCHEME = 'ws'\n")
    _track(repo, "validation/control_contract.json", b'{"contract": "CONTROL-TORAX-CONSUMER-001"}\n')

    assert scan_repository(repo) == [
        Finding(
            path="validation/control_contract.json",
            line=1,
            category="internal CONTROL lane identifier",
            detail='{"contract": "CONTROL-TORAX-CONSUMER-001"}',
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
    """The real stdlib-only script exposes argparse help without importing package dependencies."""
    result = _cli(guard.REPO_ROOT, "--help")
    assert result.returncode == 0 and result.stderr == "" and "--repo" in result.stdout


def _cli(
    repo: Path, *args: str, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the actual source command with caller-controlled directory and executable environment."""
    return subprocess.run(
        [
            sys.executable,
            "-S",
            str(guard.REPO_ROOT / "tools/check_public_surface_hygiene.py"),
            "--repo",
            str(repo),
            *args,
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )


_CONTROL_CHARACTER_NAME = pytest.mark.skipif(
    os.name == "nt", reason="a file name cannot contain a control character on Windows"
)


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param("docs/release\nnotes.md", marks=_CONTROL_CHARACTER_NAME),
        "docs/résumé.md",
        pytest.param("docs/a\tb.md", marks=_CONTROL_CHARACTER_NAME),
    ],
)
def test_actual_index_path_spellings_reach_api_and_cli(tmp_path: Path, filename: str) -> None:
    """Git-quoted Unicode, newline and tab paths cannot hide claims in real copied documentation."""
    repo = _tracked_repo(tmp_path)
    content = (guard.REPO_ROOT / "docs/development.md").read_bytes() + b"\nGroundbreaking control claim.\n"
    _track(repo, filename, content)
    assert repo / filename in list(iter_scanned_files(repo))
    findings = scan_repository(repo)
    assert len(findings) == 1 and findings[0].path == filename and findings[0].category == "groundbreaking"
    result = _cli(repo)
    assert result.returncode == 1 and result.stderr == "" and "groundbreaking" in result.stdout


def test_index_selection_reads_unstaged_worktree_bytes(tmp_path: Path) -> None:
    """A real index excludes untracked declarations and reads later worktree bytes rather than its staged blob."""
    repo = _tracked_repo(tmp_path)
    content = (guard.REPO_ROOT / "docs/development.md").read_bytes()
    _track(repo, "docs/guide.md", content)
    (repo / "docs/guide.md").write_bytes(content + b"\nGroundbreaking control claim.\n")
    (repo / "docs/untracked.md").write_bytes(content + b"\nRevolutionary claim.\n")
    findings = scan_repository(repo)
    assert [(f.path, f.category) for f in findings] == [("docs/guide.md", "groundbreaking")]
    result = _cli(repo)
    assert result.returncode == 1 and "guide.md" in result.stdout and "untracked.md" not in result.stdout


def test_relative_subdirectory_is_its_actual_git_scope(tmp_path: Path) -> None:
    """The API preserves relative spelling while Git subdirectory enumeration confines the selected index scope."""
    repo = _tracked_repo(tmp_path)
    _track(repo, "docs/guide.md", (guard.REPO_ROOT / "docs/development.md").read_bytes())
    _track(repo, "outside.md", b"Groundbreaking control claim.\n")
    assert scan_repository(repo / "docs") == []
    relative = Path(os.path.relpath(repo / "docs", Path.cwd()))
    assert list(iter_scanned_files(relative)) == [relative / "guide.md"]
    assert scan_repository(relative) == []
    result = _cli(Path("docs"), cwd=repo)
    assert result.returncode == 0 and result.stderr == "" and "inspected tracked UTF-8" in result.stdout


def test_indexed_symlink_reads_actual_external_target(tmp_path: Path) -> None:
    """Indexed symlinks follow actual regular-file targets without claiming repository containment."""
    repo = _tracked_repo(tmp_path)
    target = tmp_path / "outside-guide.md"
    target.write_bytes((guard.REPO_ROOT / "docs/development.md").read_bytes() + b"\nGroundbreaking control claim.\n")
    link = repo / "docs/linked.md"
    link.parent.mkdir()
    link.symlink_to(target)
    _git(repo, "add", "--", "docs/linked.md")
    findings = scan_repository(repo)
    assert len(findings) == 1 and findings[0].path == "docs/linked.md" and findings[0].category == "groundbreaking"
    result = _cli(repo)
    assert result.returncode == 1 and result.stderr == "" and "docs/linked.md" in result.stdout


def test_non_repository_and_missing_git_have_authored_refusals(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """Actual Git exit and executable lookup failures return fixed public refusals without interpreter text."""
    with pytest.raises(guard.PublicSurfaceScanError, match="^could not enumerate tracked public files$"):
        scan_repository(tmp_path)
    assert main(["--repo", str(tmp_path)]) == 2
    assert capsys.readouterr().out == "FAIL: could not enumerate tracked public files\n"
    result = _cli(tmp_path)
    assert result.returncode == 2 and result.stderr == ""
    assert result.stdout == "FAIL: could not enumerate tracked public files\n"
    repo = _tracked_repo(tmp_path)
    empty_path = tmp_path / "no_executables"
    empty_path.mkdir()
    result = _cli(repo, env=dict(os.environ, PATH=str(empty_path)))
    assert (
        result.returncode == 2
        and result.stderr == ""
        and result.stdout == "FAIL: could not enumerate tracked public files\n"
    )


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows does not enforce POSIX permission bits, so the path stays accessible",
)
def test_actual_read_denial_refuses_api_and_cli(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """A tracked regular file with denied read permissions aborts inspection without partial success or an OS traceback."""
    repo = _tracked_repo(tmp_path)
    target = repo / "README.md"
    mode = stat.S_IMODE(target.stat().st_mode)
    target.chmod(0)
    try:
        with pytest.raises(guard.PublicSurfaceScanError, match="^could not read tracked public text$"):
            scan_repository(repo)
        assert main(["--repo", str(repo)]) == 2
        assert capsys.readouterr().out == "FAIL: could not read tracked public text\n"
        result = _cli(repo)
        assert (
            result.returncode == 2
            and result.stderr == ""
            and result.stdout == "FAIL: could not read tracked public text\n"
        )
    finally:
        target.chmod(mode)


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows does not enforce POSIX permission bits, so the path stays accessible",
)
def test_actual_parent_search_denial_is_an_inspection_refusal(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """An indexed file inside a genuinely inaccessible directory cannot produce success or leak a stat traceback."""
    repo = _tracked_repo(tmp_path)
    _track(repo, "docs/guide.md", (guard.REPO_ROOT / "docs/development.md").read_bytes())
    parent = repo / "docs"
    mode = stat.S_IMODE(parent.stat().st_mode)
    parent.chmod(0)
    try:
        with pytest.raises(guard.PublicSurfaceScanError, match="^could not inspect tracked public path$"):
            scan_repository(repo)
        assert main(["--repo", str(repo)]) == 2
        assert capsys.readouterr().out == "FAIL: could not inspect tracked public path\n"
        result = _cli(repo)
        assert (
            result.returncode == 2
            and result.stderr == ""
            and result.stdout == "FAIL: could not inspect tracked public path\n"
        )
    finally:
        parent.chmod(mode)


def test_looping_root_is_fixed_cli_resolution_refusal(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """A real root symlink loop is refused by both command entrypoints before Git runs."""
    root = tmp_path / "loop"
    root.symlink_to(root)
    assert main(["--repo", str(root)]) == 2
    assert capsys.readouterr().out == "FAIL: could not resolve public surface repository\n"
    result = _cli(root)
    assert (
        result.returncode == 2
        and result.stderr == ""
        and result.stdout == "FAIL: could not resolve public surface repository\n"
    )


def test_native_actual_document_example_and_rendering(tmp_path: Path) -> None:
    """Execute native documentation against the actual maintained guide and render the owning module with pydoc."""
    result = doctest.testmod(guard)
    assert result.failed == 0 and result.attempted == 2
    html = pydoc.HTMLDoc().docmodule(guard)
    (tmp_path / "public_surface_hygiene.html").write_text(html, encoding="utf-8")
    assert "scan_repository" in html and "scan_text" in html and "PublicSurfaceScanError" in html


@pytest.mark.parametrize("path", ["validation/physics_traceability.json", "docs/physics_traceability.md"])
def test_actual_traceability_artifacts_keep_review_paths_private(path: str) -> None:
    """The real public registry and generated report expose public evidence references without private audit paths."""
    assert scan_text(path, (guard.REPO_ROOT / path).read_text(encoding="utf-8")) == []
