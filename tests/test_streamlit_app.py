# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Streamlit Cloud entry point tests.
"""Tests for the repository-root Streamlit Cloud adapter."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
import streamlit_app

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_cloud_args_default_to_embedded_server() -> None:
    """Streamlit Cloud gets an embedded server when no app arguments are set."""

    assert streamlit_app.streamlit_cloud_args(()) == ["--embedded"]


def test_streamlit_cloud_args_preserve_explicit_arguments() -> None:
    """Operator-supplied dashboard arguments are forwarded unchanged."""

    assert streamlit_app.streamlit_cloud_args(("--ws-url", "ws://example.invalid:8765")) == [
        "--ws-url",
        "ws://example.invalid:8765",
    ]


def test_streamlit_app_entrypoint_delegates_to_existing_ws_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The root entry point executes the existing dashboard script."""

    calls: list[tuple[str, str | None, list[str]]] = []

    def fake_run_path(path_name: str, *, run_name: str | None = None) -> dict[str, object]:
        calls.append((path_name, run_name, sys.argv[:]))
        return {}

    monkeypatch.setattr(runpy, "run_path", fake_run_path)
    original_argv = ["streamlit_app.py", "--marker"]
    monkeypatch.setattr(sys, "argv", original_argv[:])

    streamlit_app.main(("--embedded", "--zeta", "0.7"))

    assert calls == [
        (
            str(ROOT / "examples" / "streamlit_ws_client.py"),
            "__main__",
            [str(ROOT / "examples" / "streamlit_ws_client.py"), "--embedded", "--zeta", "0.7"],
        )
    ]
    assert sys.argv == original_argv


def test_streamlit_app_entrypoint_forwards_cli_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI arguments after the root script are forwarded when ``argv`` is omitted."""

    delegated_argv: list[str] = []

    def fake_run_path(_path_name: str, *, run_name: str | None = None) -> dict[str, object]:
        assert run_name == "__main__"
        delegated_argv.extend(sys.argv[1:])
        return {}

    monkeypatch.setattr(runpy, "run_path", fake_run_path)
    monkeypatch.setattr(sys, "argv", ["streamlit_app.py", "--ws-url", "ws://localhost:9000"])

    streamlit_app.main()

    assert delegated_argv == ["--ws-url", "ws://localhost:9000"]


def test_readme_streamlit_cloud_entrypoint_exists() -> None:
    """The README Streamlit Cloud instruction references an existing file."""

    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "streamlit_app.py" in readme
    assert (ROOT / "streamlit_app.py").is_file()
