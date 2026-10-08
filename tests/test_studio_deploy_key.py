# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — tests for the Studio deploy public-key guard.
"""Tests for the Studio deploy public-key guard."""

from __future__ import annotations

import base64
import hashlib
import subprocess
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture

import tools.check_studio_deploy_key as guard


@pytest.mark.parametrize("fault", ["truncated", "trailing", "algorithm", "key-size"])
def test_actual_public_key_wire_corruption_is_refused(tmp_path: Path, fault: str) -> None:
    """Reject structural mutations of the tracked public blob through both public APIs."""
    key_type, encoded, comment = guard.PUBLIC_KEY.read_text(encoding="utf-8").split()
    body = base64.b64decode(encoded)
    if fault == "truncated":
        body = body[:-1]
    elif fault == "trailing":
        body += b"\x00"
    elif fault == "algorithm":
        body = body[:4] + b"ssh-ed25518" + body[15:]
    else:
        body = body[:15] + (31).to_bytes(4, "big") + body[19:]
    line = f"{key_type} {base64.b64encode(body).decode('ascii')} {comment}\n"
    path = tmp_path / "mutated-public.pub"
    path.write_text(line, encoding="utf-8")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="32-byte ssh-ed25519"):
        guard.parse_public_key(line)
    with pytest.raises(ValueError, match="32-byte ssh-ed25519"):
        guard.validate_public_key(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("filename", ["clé.key", "clé.KEY", "ID_ED25519", "id_ecdsa"])
def test_actual_unicode_index_name_cannot_hide_a_key_like_path(tmp_path: Path, filename: str) -> None:
    """Inspect real Git names without opening any private-key-like content."""
    payload = b"public regression carrier, not a credential\n"
    (tmp_path / filename).write_bytes(payload)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    excludes = tmp_path / "empty-excludes"
    excludes.write_bytes(b"")
    subprocess.run(["git", "-c", f"core.excludesFile={excludes}", "add", "--", filename], cwd=tmp_path, check=True)
    paths = guard.tracked_files(tmp_path)
    assert paths == [filename]
    with pytest.raises(ValueError, match="private key-like tracked path"):
        guard.validate_tracked_files(paths)
    assert (tmp_path / filename).read_bytes() == payload


def test_parse_public_key_accepts_generated_key() -> None:
    """The tracked public key is a single ed25519 key for the CONTROL deploy lane."""
    key_type, decoded, comment = guard.parse_public_key(guard.PUBLIC_KEY.read_text(encoding="utf-8"))

    assert key_type == "ssh-ed25519"
    assert decoded
    assert comment == guard.EXPECTED_COMMENT


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ("ssh-ed25519 AAAA", "type, base64 body, and comment"),
        ("ssh-rsa AAAA comment", "ssh-ed25519"),
        ("ssh-ed25519 AAAA wrong-comment", "public key comment"),
        ("ssh-ed25519 !!! scpn-control-studio-ci-deploy-2026-07-08", "valid base64"),
    ],
)
def test_parse_public_key_rejects_invalid_lines(line: str, match: str) -> None:
    """Malformed, wrong-type, wrong-comment, and invalid-base64 keys fail closed."""
    with pytest.raises(ValueError, match=match):
        guard.parse_public_key(line)


def test_validate_tracked_files_accepts_public_key_paths() -> None:
    """Public key and ordinary source paths are allowed in git."""
    guard.validate_tracked_files(
        [
            "studio-web/deploy/scpn-control-studio-ci-deploy.pub",
            "tools/check_studio_deploy_key.py",
        ]
    )


@pytest.mark.parametrize(
    "path",
    [
        "studio-web/deploy/scpn-control-studio-ci-deploy_ed25519",
        "studio-web/deploy/id_ed25519",
        "studio-web/deploy/prod.pem",
        "studio-web/deploy/prod.key",
    ],
)
def test_validate_tracked_files_rejects_private_key_like_paths(path: str) -> None:
    """Private key-like paths must never be tracked."""
    with pytest.raises(ValueError, match="private key-like tracked path"):
        guard.validate_tracked_files([path])


def test_validate_public_key_reads_supplied_path(tmp_path: Path) -> None:
    """The file-level validator accepts a copied valid public key."""
    key = tmp_path / "deploy.pub"
    key.write_text(guard.PUBLIC_KEY.read_text(encoding="utf-8"), encoding="utf-8")

    guard.validate_public_key(key)


def test_validate_deploy_workflow_accepts_current_ci() -> None:
    """The CI workflow deploys the built Studio remote through the jailed lane."""
    guard.validate_deploy_workflow()


def test_validate_deploy_workflow_rejects_missing_deploy_contract(tmp_path: Path) -> None:
    """Deploy workflow drift must fail closed before CI silently stops publishing."""
    workflow = tmp_path / "ci.yml"
    workflow.write_text("name: CI\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Studio deploy workflow missing marker"):
        guard.validate_deploy_workflow(workflow)


def test_tracked_files_reads_git_index() -> None:
    """The guard inspects the real repository index."""
    paths = guard.tracked_files()

    assert "tools/preflight.py" in paths


def test_main_passes_for_current_repo(capsys: CaptureFixture[str]) -> None:
    """The command-line guard passes against the current repository."""
    assert guard.main() == 0
    assert "PASS: Studio deploy key and CI deploy workflow" in capsys.readouterr().out


def test_main_reports_validation_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    """The command-line guard reports validation failures with a failing exit code."""

    def fail_public_key() -> None:
        raise ValueError("bad public key")

    monkeypatch.setattr(guard, "validate_public_key", fail_public_key)

    assert guard.main() == 1
    assert capsys.readouterr().out == "FAIL: Studio deploy inputs could not be inspected\n"
