# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — tests for the Studio offline sealing guard.
"""Tests for the Studio offline sealing guard."""

from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture

import tools.check_studio_offline_sealing as guard


def test_policy_file_scope_covers_ci_studio_docs_and_tools() -> None:
    """The guard scans the surfaces that can wire Studio sealing custody."""
    assert guard.is_policy_file(".github/workflows/ci.yml")
    assert guard.is_policy_file("src/scpn_control/studio/sealed_claim.py")
    assert guard.is_policy_file("studio-web/README.md")
    assert guard.is_policy_file("docs/development.md")
    assert guard.is_policy_file("tools/preflight.py")
    assert guard.is_policy_file("README.md")
    assert not guard.is_policy_file("tests/test_studio_offline_sealing.py")


@pytest.mark.parametrize(
    "name",
    [
        "STUDIO_SIGNING_KEY",
        "SCPN_STUDIO_SEALING_PRIVATE_KEY",
        "HUB_PUBLICATION_SEAL_JWK",
        "TRANSPARENCY_LOG_SIGNING_SECRET",
    ],
)
def test_forbidden_secret_names_identify_offline_custody_keys(name: str) -> None:
    """Signing/sealing secret names are forbidden in CI and deploy policy."""
    assert guard.is_forbidden_studio_secret_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "CODECOV_TOKEN",
        "SCPN_CONTROL_STUDIO_DEPLOY_KEY",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "PUBLIC_MANIFEST_SHA256",
    ],
)
def test_secret_name_guard_allows_unrelated_and_deploy_credentials(name: str) -> None:
    """The offline-sealing guard does not block non-signing operational secrets."""
    assert not guard.is_forbidden_studio_secret_name(name)


@pytest.mark.parametrize(
    "path",
    [
        "ops/studio-signing.key",
        "studio-web/sealing/private.pem",
        "docs/transparency-log-signing.jwk",
        "publication-seal/private.pkcs8",
    ],
)
def test_forbidden_key_paths_identify_signing_material(path: str) -> None:
    """Tracked signing/sealing private-key-like paths fail closed."""
    assert guard.is_forbidden_key_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "studio-web/deploy/scpn-control-studio-ci-deploy.pub",
        "studio-web/deploy/scpn-control-studio-ci-deploy.key",
        "docs/reference.pem.txt",
        "tools/check_studio_offline_sealing.py",
    ],
)
def test_forbidden_key_paths_allow_deploy_and_non_key_surfaces(path: str) -> None:
    """Deploy credentials are covered by the deploy-key lane, not sealing custody."""
    assert not guard.is_forbidden_key_path(path)


def test_validate_secret_references_rejects_workflow_secret_reference() -> None:
    """Workflow references to Studio signing secrets are rejected."""
    secret_name = "STUDIO_" + "SIGNING_KEY"
    violations = guard.validate_secret_references(".github/workflows/ci.yml", f"key: ${{{{ secrets.{secret_name} }}}}")

    assert violations == [f".github/workflows/ci.yml: forbidden Studio sealing secret reference: secrets.{secret_name}"]


def test_validate_secret_references_rejects_workflow_env_assignment() -> None:
    """Workflow environment names cannot reserve Studio sealing-key custody."""
    secret_name = "SCPN_STUDIO_" + "SEALING_PRIVATE_KEY"
    violations = guard.validate_secret_references(".github/workflows/ci.yml", f"env:\n  {secret_name}: offline\n")

    assert violations == [f".github/workflows/ci.yml: forbidden Studio sealing environment name: {secret_name}"]


def test_validate_secret_references_ignores_docs_env_examples() -> None:
    """Documentation can mention env-style names without wiring them into CI."""
    secret_name = "STUDIO_" + "SIGNING_KEY"
    assert guard.validate_secret_references("docs/development.md", f"`{secret_name}` remains offline.") == []


def test_validate_policy_files_rejects_private_key_blocks(tmp_path: Path) -> None:
    """Private-key blocks are never allowed in tracked Studio policy surfaces."""
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("-----BEGIN OPENSSH " + "PRIVATE " + "KEY-----\nredacted\n", encoding="utf-8")

    violations = guard.validate_policy_files([".github/workflows/ci.yml"], tmp_path)

    assert violations == [".github/workflows/ci.yml: private-key block is forbidden on Studio sealing policy surfaces"]


def test_validate_policy_files_skips_binary_policy_files(tmp_path: Path) -> None:
    """Binary policy files are skipped instead of decoded lossy."""
    binary = tmp_path / "studio-web" / "asset.bin"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\xff\xfe\x00")

    assert guard.validate_policy_files(["studio-web/asset.bin"], tmp_path) == []


def test_validate_policy_files_rejects_key_like_paths_without_reading(tmp_path: Path) -> None:
    """Tracked signing-key paths fail even if the file is absent."""
    violations = guard.validate_policy_files(["ops/studio-signing.key"], tmp_path)

    assert violations == ["ops/studio-signing.key: tracked Studio sealing key-like path is forbidden"]


def test_tracked_files_reads_git_index() -> None:
    """The guard inspects the real repository index."""
    paths = guard.tracked_files()

    assert "tools/preflight.py" in paths


def test_main_passes_for_current_repo(capsys: CaptureFixture[str]) -> None:
    """The command-line guard passes against the current repository."""
    assert guard.main() == 0
    assert "PASS: Studio evidence sealing remains offline" in capsys.readouterr().out


def test_main_reports_violations(monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    """The command-line guard reports violations with a failing exit code."""
    monkeypatch.setattr(guard, "tracked_files", lambda: ["ops/studio-signing.key"])

    assert guard.main() == 1
    output = capsys.readouterr().out
    assert "FAIL: Studio evidence sealing must remain keeper-offline" in output
    assert "ops/studio-signing.key" in output


def test_main_reports_git_failures(monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    """The command-line guard reports repository-read failures with a failing exit code."""

    def fail_tracked_files() -> list[str]:
        raise OSError("git unavailable")

    monkeypatch.setattr(guard, "tracked_files", fail_tracked_files)

    assert guard.main() == 1
    assert "FAIL: git unavailable" in capsys.readouterr().out
