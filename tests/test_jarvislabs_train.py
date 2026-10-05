# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JarvisLabs real refusal tests
"""Exercise real JarvisLabs API/CLI refusals without any provider request."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scpn_control.benchmark_records import CAMPAIGN_ENV
from tools import jarvislabs_train as workflow


def _protected_hashes() -> dict[str, str | None]:
    """Capture actual upload and canonical result bytes around refusal calls."""
    result: dict[str, str | None] = {}
    for name in (*workflow.UPLOAD_FILES, *workflow.DOWNLOAD_FILES):
        path = workflow.REPO_ROOT / name
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return result


def test_actual_missing_token_api_and_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing environment credentials fail through both public entry points."""
    monkeypatch.delenv("JARVISLABS_TOKEN", raising=False)
    before = _protected_hashes()
    assert workflow.main() == 1
    env = dict(os.environ)
    env.pop("JARVISLABS_TOKEN", None)
    process = subprocess.run(
        [sys.executable, str(workflow.REPO_ROOT / "tools/jarvislabs_train.py")],
        cwd=workflow.REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert process.returncode == 1
    assert process.stdout == "Set JARVISLABS_TOKEN environment variable\n"
    assert process.stderr == ""
    assert _protected_hashes() == before


def test_actual_local_preflight_and_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Actual files, custody input and absent optional import refuse before cloud."""
    root = tmp_path / "checkout"
    for name in workflow.UPLOAD_FILES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workflow.REPO_ROOT / name, target)
    (root / "weights").mkdir()
    monkeypatch.setattr(workflow, "REPO_ROOT", root)
    monkeypatch.setenv("JARVISLABS_TOKEN", "local-refusal-input")
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    assert workflow.main() == 1
    assert capsys.readouterr().out == "JarvisLabs training workflow failed; results are not accepted.\n"
    existing = root / workflow.DOWNLOAD_FILES[0]
    existing.write_bytes(b"caller-owned-artifact")
    monkeypatch.setenv(CAMPAIGN_ENV, "local-refusal-campaign")
    assert workflow.main() == 1
    assert existing.read_bytes() == b"caller-owned-artifact"
    existing.unlink()
    if importlib.util.find_spec("jlclient") is None:
        with pytest.raises(ModuleNotFoundError):
            workflow.setup_jarvislabs("local-refusal-input")
        assert workflow.main() == 1
    assert not any((root / name).exists() for name in workflow.DOWNLOAD_FILES)
    output = capsys.readouterr().out
    assert "local-refusal-input" not in output
    assert "Traceback" not in output and "ModuleNotFoundError" not in output


def test_actual_lifecycle_input_refusals(capsys: pytest.CaptureFixture[str]) -> None:
    """Invalid caller inputs require no optional SDK and cannot confirm cleanup."""
    for token in ("", " ", "\t"):
        with pytest.raises(ValueError, match="nonempty JarvisLabs token"):
            workflow.setup_jarvislabs(token)
    for timeout in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite and positive"):
            workflow.wait_for_ready(object(), timeout)
    with pytest.raises(RuntimeError, match="instance identity"):
        workflow.wait_for_ready(object())
    assert workflow.destroy_instance(object()) is False
    assert capsys.readouterr().out == (
        "JarvisLabs destruction was not confirmed; check the instance in the provider dashboard.\n"
    )


def test_actual_candidate_directory_api_cli_and_help(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Actual candidate CLI refuses missing parents/occupied outputs before SDK calls."""
    monkeypatch.setenv("JARVISLABS_TOKEN", "local-refusal-input")
    before = _protected_hashes()
    absent = tmp_path / "absent"
    assert workflow.main(["--output-dir", str(absent)]) == 1
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    retained = candidate / "ppo_tokamak.zip"
    retained.write_bytes(b"caller-owned-candidate")
    assert workflow.main(["--output-dir", str(candidate)]) == 1
    assert retained.read_bytes() == b"caller-owned-candidate"
    assert not absent.exists()
    env = {**os.environ, "JARVISLABS_TOKEN": "local-refusal-input"}
    for arguments, code in ((["--help"], 0), (["--output-dir", str(candidate)], 1)):
        process = subprocess.run(
            [sys.executable, str(workflow.REPO_ROOT / "tools/jarvislabs_train.py"), *arguments],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert process.returncode == code and "Traceback" not in process.stderr
        if code == 0:
            assert "--output-dir" in process.stdout
        else:
            assert process.stdout == "JarvisLabs training workflow failed; results are not accepted.\n"
        assert "local-refusal-input" not in process.stdout + process.stderr
    assert _protected_hashes() == before
    assert "local-refusal-input" not in capsys.readouterr().out
