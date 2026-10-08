# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — tests for the Studio offline sealing guard.
"""Tests for the Studio offline sealing guard."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pytest import CaptureFixture

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
    assert "PASS: Studio offline-sealing lexical checks passed" in capsys.readouterr().out


def test_main_reports_violation_count_without_secret_adjacent_details(tmp_path: Path) -> None:
    """The CLI fails without copying secret-adjacent diagnostics into logs."""
    secret_name = "STUDIO_" + "SIGNING_KEY"
    root = _repository(tmp_path, {".github/workflows/ci.yml": f"env:\n  {secret_name}: placeholder\n".encode()})
    result = _run_guard(root)
    assert result.returncode == 1
    output = result.stdout
    assert "FAIL: Studio evidence sealing must remain keeper-offline" in output
    assert "1 policy violation(s) detected" in output
    assert secret_name not in output


def test_main_reports_git_failures(tmp_path: Path) -> None:
    """The command-line guard reports repository-read failures with a failing exit code."""
    root = _repository(tmp_path, {}, indexed=False)
    result = _run_guard(root)
    assert result.returncode == 1
    assert result.stdout == "FAIL: Studio sealing policy files could not be inspected\n"
    assert not result.stderr


def _repository(tmp_path: Path, files: dict[str, bytes], *, indexed: bool = True) -> Path:
    """Create a real isolated index and a byte-identical standalone guard.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Owned temporary parent for the source and Git fixture.
    files : dict of str to bytes
        Placeholder policy inputs. Forbidden key paths remain index-only so
        their names are portable without reading or creating key material.
    indexed : bool, optional
        Whether to initialise Git; false exercises an actual repository failure.

    Returns
    -------
    pathlib.Path
        Natural script-relative repository root for cold CLI and public API calls.
    """
    root = tmp_path / "repository"
    script = root / "tools/check_studio_offline_sealing.py"
    script.parent.mkdir(parents=True)
    script.write_bytes((guard.ROOT / "tools/check_studio_offline_sealing.py").read_bytes())
    if not indexed:
        return root
    subprocess.run(
        ["git", "-c", "init.defaultBranch=main", "init", "--quiet", str(root)], check=True, capture_output=True
    )
    subprocess.run(["git", "-C", str(root), "config", "core.quotePath", "true"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "config", "core.protectNTFS", "false"], check=True, capture_output=True)
    for name, payload in files.items():
        if name.endswith(".key"):
            result = subprocess.run(
                ["git", "-C", str(root), "hash-object", "-w", "--stdin"],
                input=payload,
                check=True,
                capture_output=True,
            )
            entry = b"100644 " + result.stdout.strip() + b"\t" + os.fsencode(name) + b"\0"
            subprocess.run(
                ["git", "-C", str(root), "update-index", "-z", "--index-info"],
                input=entry,
                check=True,
                capture_output=True,
            )
        else:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            subprocess.run(["git", "-C", str(root), "add", "--force", "--", name], check=True, capture_output=True)
    return root


def _run_guard(root: Path) -> subprocess.CompletedProcess[str]:
    """Run unchanged production bytes without inheriting an ancestor Git index.

    Parameters
    ----------
    root : pathlib.Path
        Owned source-identical repository fixture.

    Returns
    -------
    subprocess.CompletedProcess of str
        Actual CLI status and captured stdout/stderr.
    """
    return subprocess.run(
        [sys.executable, str(root / "tools/check_studio_offline_sealing.py")],
        cwd=root,
        env={**os.environ, "GIT_CEILING_DIRECTORIES": str(root.parent), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


@pytest.mark.parametrize(
    "name", ["tools/studio-signing-π.key", "tools/studio-signing\n.key", "tools/studio-signing\t.key"]
)
def test_actual_git_names_and_forbidden_paths_remain_lossless(tmp_path: Path, name: str) -> None:
    """Public APIs and cold CLI refuse literal indexed names without opening key paths."""
    root = _repository(tmp_path, {name: b"placeholder, no key material"})
    assert guard.tracked_files(root) == [name]
    assert guard.validate_policy_files(guard.tracked_files(root), root) == [
        f"{name}: tracked Studio sealing key-like path is forbidden"
    ]
    assert not (root / name).exists()
    result = _run_guard(root)
    assert result.returncode == 1
    assert "1 policy violation(s)" in result.stdout
    assert name not in result.stdout
    assert not result.stderr


@pytest.mark.parametrize("name", ["STUDIO_DEPLOY_SIGNING_PRIVATE_KEY", "HUB_DEPLOY_SEALING_KEY", "DEPLOY_SEAL_SECRET"])
def test_actual_workflow_signing_roles_override_deploy_labels(tmp_path: Path, name: str) -> None:
    """Explicit signing and sealing roles refuse through both public validators and CLI."""
    content = f"env:\n  {name}: ${{{{ secrets.{name} }}}}\n".encode()
    root = _repository(tmp_path, {".github/workflows/ci.yml": content})
    assert guard.validate_policy_files(guard.tracked_files(root), root)
    result = _run_guard(root)
    assert result.returncode == 1
    assert name not in result.stdout
    assert (root / ".github/workflows/ci.yml").read_bytes() == content


@pytest.mark.parametrize("name", ["tools/deploy/studio-signing.key", "studio-web/deploy/sealing.key"])
def test_explicit_signing_paths_override_deploy_directories(tmp_path: Path, name: str) -> None:
    """Indexed signing paths stay forbidden below deploy without accessing their contents."""
    root = _repository(tmp_path, {name: b"placeholder, no key material"})
    assert _run_guard(root).returncode == 1
    assert not (root / name).exists()


@pytest.mark.parametrize(
    "name", [".github/workflows/ci.yml", "tools/config.json", "docs/CNAME", ".github/workflows/custom"]
)
def test_invalid_utf8_text_policy_refuses(tmp_path: Path, name: str) -> None:
    """Known textual policy and workflow scopes cannot silently skip malformed UTF-8."""
    payload = b"\xffSTUDIO_SIGNING_KEY: placeholder\n"
    root = _repository(tmp_path, {name: payload})
    result = _run_guard(root)
    assert result.returncode == 1
    assert "1 policy violation(s)" in result.stdout
    assert not result.stderr
    assert (root / name).read_bytes() == payload


def test_actual_bom_workflow_cannot_hide_an_environment_name(tmp_path: Path) -> None:
    """A valid leading UTF-8 BOM preserves the lexical signing-name refusal."""
    payload = b"\xef\xbb\xbfSTUDIO_SIGNING_KEY: placeholder\n"
    root = _repository(tmp_path, {".github/workflows/ci.yml": payload})
    assert guard.validate_secret_references(".github/workflows/ci.yml", payload.decode("utf-8"))
    assert _run_guard(root).returncode == 1


@pytest.mark.parametrize("name", ["docs/asset.bin", "docs/asset.png", "docs/asset.unknown"])
def test_actual_opaque_assets_and_deploy_only_workflow_pass(tmp_path: Path, name: str) -> None:
    """Existing binary allowance and transport-only private names remain valid public inputs."""
    content = b"env:\n  STUDIO_DEPLOY_PRIVATE_KEY: ${{ secrets.STUDIO_DEPLOY_PRIVATE_KEY }}\n"
    root = _repository(tmp_path, {name: b"\xff\xfe\x00", ".github/workflows/ci.yml": content})
    assert guard.validate_policy_files(guard.tracked_files(root), root) == []
    result = _run_guard(root)
    assert result.returncode == 0
    assert not result.stderr
    assert (root / name).read_bytes() == b"\xff\xfe\x00"


def test_actual_directory_read_failure_hides_native_path(tmp_path: Path) -> None:
    """A real indexed file changed to a directory returns fixed native refusal stdout."""
    root = _repository(tmp_path, {".github/workflows/private.yml": b"name: placeholder\n"})
    path = root / ".github/workflows/private.yml"
    path.unlink()
    path.mkdir()
    result = _run_guard(root)
    assert result.returncode == 1
    assert result.stdout == "FAIL: Studio sealing policy files could not be inspected\n"
    assert not result.stderr
    assert path.is_dir()


@pytest.mark.parametrize(
    "reference",
    [
        "secrets['STUDIO_SIGNING_KEY']",
        'secrets["STUDIO_SIGNING_KEY"]',
        "secrets [ 'STUDIO_SIGNING_KEY' ]",
        "secrets . STUDIO_SIGNING_KEY",
    ],
)
def test_actual_literal_secret_access_forms_refuse(tmp_path: Path, reference: str) -> None:
    """Dot and literal quoted-index references have the same public policy refusal."""
    content = ("key: ${{ " + reference + " }}\n").encode()
    root = _repository(tmp_path, {".github/workflows/ci.yml": content})
    assert guard.validate_secret_references(".github/workflows/ci.yml", content.decode())
    result = _run_guard(root)
    assert result.returncode == 1
    assert "STUDIO_SIGNING_KEY" not in result.stdout
    assert not result.stderr


def test_actual_quoted_deploy_reference_remains_allowed(tmp_path: Path) -> None:
    """A literal index does not change the deploy-only transport allowance."""
    content = b"key: ${{ secrets['STUDIO_DEPLOY_PRIVATE_KEY'] }}\n"
    root = _repository(tmp_path, {".github/workflows/ci.yml": content})
    assert guard.validate_policy_files(guard.tracked_files(root), root) == []
    assert _run_guard(root).returncode == 0


@pytest.mark.parametrize(
    "reference",
    [
        "secrets[env.SELECTED_KEY]",
        "secrets['STUDIO' + '_SIGNING_KEY']",
        "secrets['STUDIO_SIGNING_KEY'",
        "secrets['STUDIO_SIGNING_KEY\"]",
    ],
)
def test_unclassifiable_secret_index_refuses(tmp_path: Path, reference: str) -> None:
    """Dynamic and malformed lookups cannot bypass lexical key-name classification."""
    content = ("key: ${{ " + reference + " }}\n").encode()
    root = _repository(tmp_path, {".github/workflows/ci.yml": content})
    assert any("must name a literal key" in v for v in guard.validate_policy_files(guard.tracked_files(root), root))
    result = _run_guard(root)
    assert result.returncode == 1
    assert reference not in result.stdout
    assert not result.stderr


@pytest.mark.parametrize(
    "content",
    [
        "env:\n  'STUDIO_SIGNING_KEY': placeholder\n",
        'env:\n  "STUDIO_SIGNING_KEY": placeholder\n',
        "env: {STUDIO_SIGNING_KEY: placeholder}\n",
        'env: {"STUDIO_SIGNING_KEY": placeholder}\n',
    ],
)
def test_actual_quoted_and_flow_environment_names_refuse(tmp_path: Path, content: str) -> None:
    """Quoted and flow-mapping workflow keys cannot evade the same signing-name policy."""
    root = _repository(tmp_path, {".github/workflows/ci.yml": content.encode()})
    assert guard.validate_policy_files(guard.tracked_files(root), root)
    result = _run_guard(root)
    assert result.returncode == 1
    assert "STUDIO_SIGNING_KEY" not in result.stdout
    assert not result.stderr
