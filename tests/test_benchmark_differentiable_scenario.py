# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable scenario benchmark producer tests.
"""Regression tests for append-safe differentiable benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scpn_control.core.differentiable_scenario as differentiable_scenario
from validation import benchmark_differentiable_scenario as benchmark


def test_help_does_not_write_default_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Argument discovery cannot execute or overwrite the canonical evidence."""
    writes: list[tuple[Path, str]] = []
    monkeypatch.setattr(Path, "write_text", lambda self, data, **_: writes.append((self, data)))

    with pytest.raises(SystemExit) as exc_info:
        benchmark.main(["--help"])

    assert exc_info.value.code == 0
    assert writes == []


def test_blocked_run_writes_only_requested_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A JAX-blocked run honors explicit append-safe destinations."""
    json_out = tmp_path / "refresh" / "evidence.json"
    md_out = tmp_path / "refresh" / "evidence.md"
    monkeypatch.setattr(differentiable_scenario, "has_jax", lambda: False)

    assert benchmark.main(["--json-out", str(json_out), "--md-out", str(md_out)]) == 0

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert "JAX is required" in payload["reason"]
    assert "Status: `blocked`" in md_out.read_text(encoding="utf-8")


def test_producer_command_quotes_output_paths() -> None:
    """Recorded commands remain reproducible when destinations contain spaces."""
    command = benchmark._producer_command(Path("artifacts/a b.json"), Path("artifacts/a b.md"))

    assert "'artifacts/a b.json'" in command
    assert "'artifacts/a b.md'" in command
