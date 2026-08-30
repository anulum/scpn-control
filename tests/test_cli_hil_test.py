# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI hil-test command tests
"""Tests for the ``hil-test`` CLI command (reference-shot HIL campaign).

Covers the happy path, empty/nonexistent shot directories, and the size-audited
NPZ loader. Each hostile or corrupt archive fails closed without aborting the
campaign.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner() -> CliRunner:
    """Provide an isolated Click runner for HIL command tests."""
    return CliRunner()


def test_hil_test_nonexistent_dir(runner: CliRunner) -> None:
    """Require the HIL command to fail for an absent shots directory."""
    result = runner.invoke(main, ["hil-test", "--shots-dir", "nonexistent_dir_12345", "--json-out"])
    assert result.exit_code != 0


def test_hil_test_with_mock_shots(runner: CliRunner, tmp_path: Path) -> None:
    """Process deterministic NPZ shot fixtures through the HIL command."""
    rng = np.random.default_rng(0)
    for name in ("shot_001", "shot_002"):
        np.savez(
            tmp_path / f"{name}.npz",
            psi=rng.standard_normal((10, 10)),
            ip=np.array([15e6]),
        )
    result = runner.invoke(main, ["hil-test", "--shots-dir", str(tmp_path), "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["n_shots"] == 2
    assert len(data["shots"]) == 2
    assert data["shots"][0]["status"] == "loaded"
    assert "psi" in data["shots"][0]["keys"]


def test_hil_test_text_output(runner: CliRunner, tmp_path: Path) -> None:
    """Render the accepted-shot summary in human-readable form."""
    np.savez(tmp_path / "s42.npz", plasma=np.zeros(5))
    result = runner.invoke(main, ["hil-test", "--shots-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "1 shots" in result.output
    assert "s42" in result.output


def test_hil_test_empty_dir(runner: CliRunner, tmp_path: Path) -> None:
    """Report an empty shot directory without failing the campaign."""
    result = runner.invoke(main, ["hil-test", "--shots-dir", str(tmp_path), "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["n_shots"] == 0


def test_hil_test_rejects_a_corrupt_npz_without_crashing(runner: CliRunner, tmp_path: Path) -> None:
    # A hostile/corrupt .npz must be recorded per file, not crash the whole
    # campaign; the load is size-audited and each file fails closed.
    """Record a corrupt NPZ as rejected without aborting the HIL campaign."""
    np.savez(tmp_path / "good.npz", ip=np.array([15e6]))
    (tmp_path / "hostile.npz").write_bytes(b"PK\x03\x04 not a valid zip archive")

    result = runner.invoke(main, ["hil-test", "--shots-dir", str(tmp_path), "--json-out"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    by_shot = {row["shot"]: row for row in data["shots"]}
    assert by_shot["good"]["status"] == "loaded"
    assert by_shot["hostile"]["status"] == "rejected"
    assert "error" in by_shot["hostile"]


def test_hil_test_text_output_reports_rejected_shot(runner: CliRunner, tmp_path: Path) -> None:
    """Include rejected-shot accounting in the human-readable HIL summary."""
    np.savez(tmp_path / "good.npz", ip=np.array([15e6]))
    (tmp_path / "hostile.npz").write_bytes(b"PK\x03\x04 not a valid zip archive")

    result = runner.invoke(main, ["hil-test", "--shots-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "good: 1 arrays" in result.output
    assert "hostile: rejected" in result.output
