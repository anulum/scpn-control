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


REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = REPO_ROOT / "tools" / "preflight.py"


def _load_preflight_module():
    spec = importlib.util.spec_from_file_location("scpn_control_preflight", PREFLIGHT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_runs_top_level_release_evidence_gate():
    """Local preflight must exercise the same release gate as manual validation."""
    preflight = _load_preflight_module()

    gates = {name: command for name, command, _cwd in preflight.GATES}

    assert gates["release-evidence"] == [
        preflight._PY,
        "-m",
        "scpn_control.cli",
        "validate",
        "--json-out",
    ]


def test_release_evidence_gate_is_not_skipped_by_no_tests_or_no_rust():
    """Evidence admission is not a test or Rust-only gate, so fast preflight keeps it."""
    preflight = _load_preflight_module()

    assert "release-evidence" not in preflight.TEST_GATES
    assert "release-evidence" not in preflight.RUST_GATES


def test_preflight_subprocess_env_prefers_source_tree():
    """Local preflight imports the checked-out source tree before installed packages."""
    preflight = _load_preflight_module()

    pythonpath = preflight._subprocess_env()["PYTHONPATH"].split(preflight.os.pathsep)

    assert pythonpath[0] == str(REPO_ROOT / "src")
    assert pythonpath[1] == str(REPO_ROOT)
