# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Preflight Release Evidence Tests

"""Regression tests for the local preflight release-evidence contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = REPO_ROOT / "tools" / "preflight.py"


def _load_preflight_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scpn_control_preflight", PREFLIGHT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_runs_top_level_release_evidence_gate() -> None:
    """Local preflight must exercise release-evidence artifact admission."""
    preflight = _load_preflight_module()

    gates = {name: command for name, command, _cwd in preflight.GATES}

    assert gates["release-evidence"] == [
        preflight._PY,
        "-m",
        "scpn_control.cli",
        "validate-release-evidence",
    ]
    assert callable(preflight.run_release_evidence_gate)


def test_release_evidence_gate_is_not_skipped_by_no_tests_or_no_rust() -> None:
    """Evidence admission is not a test or Rust-only gate, so fast preflight keeps it."""
    preflight = _load_preflight_module()

    assert "release-evidence" not in preflight.TEST_GATES
    assert "release-evidence" not in preflight.RUST_GATES


def test_preflight_runs_runtime_wiring_gate() -> None:
    """Local preflight must fail closed on source modules with no repository wiring."""
    preflight = _load_preflight_module()

    gates = {name: command for name, command, _cwd in preflight.GATES}

    assert gates["runtime-wiring"] == [preflight._PY, "tools/check_runtime_wiring.py"]
    assert "runtime-wiring" not in preflight.TEST_GATES
    assert "runtime-wiring" not in preflight.RUST_GATES


def test_preflight_runs_coverage_pragma_gate() -> None:
    """Local preflight must reject unreasoned source coverage exclusions."""
    preflight = _load_preflight_module()

    gates = {name: command for name, command, _cwd in preflight.GATES}

    assert gates["coverage-pragmas"] == [preflight._PY, "tools/check_coverage_pragmas.py"]
    assert "coverage-pragmas" not in preflight.TEST_GATES
    assert "coverage-pragmas" not in preflight.RUST_GATES


def test_preflight_runs_studio_offline_sealing_gate() -> None:
    """Local preflight must keep Studio signing custody out of CI and deploy surfaces."""
    preflight = _load_preflight_module()

    gates = {name: command for name, command, _cwd in preflight.GATES}

    assert gates["studio-offline-sealing"] == [preflight._PY, "tools/check_studio_offline_sealing.py"]
    assert "studio-offline-sealing" not in preflight.TEST_GATES
    assert "studio-offline-sealing" not in preflight.RUST_GATES


def test_preflight_subprocess_env_prefers_source_tree() -> None:
    """Local preflight imports the checked-out source tree before installed packages."""
    preflight = _load_preflight_module()

    pythonpath = preflight._subprocess_env()["PYTHONPATH"].split(preflight.os.pathsep)

    assert pythonpath[0] == str(REPO_ROOT / "src")
    assert pythonpath[1] == str(REPO_ROOT)


def test_release_evidence_gate_generates_and_admits_temporary_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Local preflight validates generated release evidence before passing the gate."""
    preflight = _load_preflight_module()
    calls: list[list[str]] = []

    class DummyHandle:
        def __enter__(self) -> "DummyHandle":
            return self

        def __exit__(self, *_exc_info: object) -> Literal[False]:
            return False

        def write(self, _data: str) -> None:
            return None

    class DummyTemporaryDirectory:
        def __init__(self, prefix: str) -> None:
            assert prefix == "scpn-control-release-evidence-"

        def __enter__(self) -> str:
            tmp_path.mkdir(exist_ok=True)
            return str(tmp_path)

        def __exit__(self, *_exc_info: object) -> Literal[False]:
            return False

    class DummyCompletedProcess:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_open(_self: Path, *_args: object, **_kwargs: object) -> DummyHandle:
        return DummyHandle()

    def fake_run(cmd: list[str], **_kwargs: Any) -> DummyCompletedProcess:
        calls.append(cmd)
        return DummyCompletedProcess()

    monkeypatch.setattr(preflight.tempfile, "TemporaryDirectory", DummyTemporaryDirectory)
    monkeypatch.setattr(preflight.Path, "open", fake_open)
    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    assert preflight.run_release_evidence_gate() is True
    assert calls[0] == [preflight._PY, "-m", "scpn_control.cli", "validate", "--json-out"]
    assert calls[1][:4] == [preflight._PY, "-m", "scpn_control.cli", "validate-release-evidence"]
    assert calls[1][-1] == "--json-out"
