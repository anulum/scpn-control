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
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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


def test_profiles_and_fixture_have_consistent_shapes() -> None:
    """The deterministic campaign fixture keeps every coupled grid aligned."""
    rho = np.linspace(0.05, 1.0, 16)

    assert benchmark._profiles(rho).shape == (4, 16)
    fixture = benchmark._scenario_fixture()

    assert len(fixture) == 10
    assert fixture[1].shape == (4, 16)
    assert fixture[3].shape == (3, 4, 16)
    assert fixture[4].shape == (3, 4, 16)


def test_loadavg_is_explicit_when_supported_or_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Host-load capture has observable success and unavailable branches."""
    monkeypatch.setattr(os, "getloadavg", lambda: (1, 2, 3))
    assert benchmark._loadavg() == (1.0, 2.0, 3.0)

    def unavailable() -> tuple[float, float, float]:
        raise OSError("load average unavailable")

    monkeypatch.setattr(os, "getloadavg", unavailable)
    assert benchmark._loadavg() is None


@dataclass(frozen=True)
class _Metadata:
    backend: str = "jax"
    n_rho: int = 2
    n_steps: int = 1
    flux_grid_shape: tuple[int, int] = (2, 2)
    gradient_tolerance: float = 5.0e-4


@dataclass(frozen=True)
class _Audit:
    passed: bool = True


@dataclass(frozen=True)
class _Readiness:
    campaign_sha256: str = "a" * 64
    gradient_audit_sha256: str = "b" * 64
    latency_p95_ms: float = 2.0
    claim_admissible: bool = False
    blocked_reasons: tuple[str, ...] = ("physics_traceability",)


def test_passing_run_writes_only_requested_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The passing branch records bounded evidence at explicit destinations."""
    json_out = tmp_path / "new" / "evidence.json"
    md_out = tmp_path / "new" / "evidence.md"
    fixture = tuple(np.ones((1,), dtype=float) for _ in range(10))
    ticks = iter(float(index) for index in range(10))
    monkeypatch.setattr(differentiable_scenario, "has_jax", lambda: True)
    monkeypatch.setattr(
        differentiable_scenario,
        "assert_differentiable_scenario_gradient_consistent",
        lambda *_args, **_kwargs: _Audit(),
    )
    monkeypatch.setattr(
        differentiable_scenario,
        "scenario_campaign_metadata",
        lambda *_args, **_kwargs: _Metadata(),
    )
    monkeypatch.setattr(
        differentiable_scenario,
        "differentiable_scenario_readiness_evidence",
        lambda *_args, **_kwargs: _Readiness(),
    )
    monkeypatch.setattr(benchmark, "_scenario_fixture", lambda: fixture)
    monkeypatch.setattr(benchmark, "_loadavg", lambda: (1.0, 2.0, 3.0))
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

    assert benchmark.main(["--json-out", str(json_out), "--md-out", str(md_out)]) == 0

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["benchmark_context"]["timed_runs"] == 5
    assert str(json_out) in payload["benchmark_context"]["command"]
    markdown = md_out.read_text(encoding="utf-8")
    assert "Status: `pass`" in markdown
    assert "Claim admissible: `False`" in markdown


def test_markdown_tolerates_malformed_optional_readiness_context(tmp_path: Path) -> None:
    """A blocked report remains renderable when optional readiness is not an object."""
    path = tmp_path / "blocked.md"
    payload: dict[str, Any] = benchmark._blocked_payload("blocked for test")
    payload["readiness"] = "not-an-object"

    benchmark._write_markdown(payload, path)

    assert "Reason: blocked for test" in path.read_text(encoding="utf-8")
