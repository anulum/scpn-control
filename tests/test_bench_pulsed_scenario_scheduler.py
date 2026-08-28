# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Pulsed-scenario scheduler benchmark tests.
"""Tests for the Python and native-Rust scheduler benchmark contract."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest import CaptureFixture, MonkeyPatch

from benchmarks import bench_pulsed_scenario_scheduler as bench


def test_python_campaign_preserves_all_eight_states() -> None:
    """The benchmark fixture traverses the canonical lifecycle exactly once."""
    assert bench._run_python_campaign() == bench.EXPECTED_STATES


def test_run_without_rust_is_explicitly_unadmitted() -> None:
    """A Python-only run remains useful but cannot claim parity or promotion."""
    payload = bench.run(iterations=2, warmup=1, include_rust=False)

    assert payload["python_result"]["stats"]["samples"] == 2
    assert payload["rust_result"] is None
    assert payload["rust_command"] is None
    assert payload["parity_passed"] is False
    assert payload["orientation_only"] is True
    assert payload["public_claim_allowed"] is False
    digest = payload.pop("payload_sha256")
    payload["payload_sha256"] = ""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert digest == hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    ("iterations", "warmup", "message"),
    [(0, 0, "iterations"), (1, -1, "warmup")],
)
def test_run_rejects_invalid_sample_counts(iterations: int, warmup: int, message: str) -> None:
    """Empty measurement and negative warm-up domains fail closed."""
    with pytest.raises(ValueError, match=message):
        bench.run(iterations=iterations, warmup=warmup, include_rust=False)


def _rust_payload(states: object = bench.EXPECTED_STATES) -> dict[str, object]:
    return {
        "last_states": states,
        "stats": {
            "samples": 2,
            "mean_us": 1.0,
            "median_us": 1.0,
            "p95_us": 1.1,
            "p99_us": 1.2,
            "min_us": 0.9,
            "max_us": 1.3,
        },
    }


def test_rust_runner_validates_json_and_state_parity(monkeypatch: MonkeyPatch) -> None:
    """The native subprocess must emit an object with the exact state trace."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(_rust_payload())),
    )
    assert bench._run_rust(2, 1)["last_states"] == bench.EXPECTED_STATES

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="[]"),
    )
    with pytest.raises(ValueError, match="JSON object"):
        bench._run_rust(2, 1)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(_rust_payload(["idle"]))),
    )
    with pytest.raises(ValueError, match="eight-state campaign"):
        bench._run_rust(2, 1)


def test_side_by_side_run_admits_only_trace_parity(monkeypatch: MonkeyPatch) -> None:
    """Matching Python and Rust states pass parity without enabling claims."""
    monkeypatch.setattr(bench, "_run_rust", lambda iterations, warmup: _rust_payload())
    payload = bench.run(iterations=2, warmup=0)

    assert payload["parity_passed"] is True
    assert payload["rust_command"][-2:] == ["--warmup", "0"]
    assert payload["scientific_admission"] is False
    assert payload["production_admission"] is False


def test_markdown_renders_python_only_and_side_by_side(tmp_path: Path) -> None:
    """Markdown labels each measured implementation without overclaiming."""
    payload = bench.run(iterations=1, warmup=0, include_rust=False)
    python_only = tmp_path / "python.md"
    bench._write_markdown(python_only, payload)
    assert "| Python |" in python_only.read_text(encoding="utf-8")
    assert "| Rust native |" not in python_only.read_text(encoding="utf-8")

    payload["rust_result"] = _rust_payload()
    side_by_side = tmp_path / "side-by-side.md"
    bench._write_markdown(side_by_side, payload)
    rendered = side_by_side.read_text(encoding="utf-8")
    assert "| Python |" in rendered
    assert "| Rust native |" in rendered
    assert "Public claim allowed: `False`" in rendered


def test_cli_writes_requested_outputs(tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    """The CLI writes JSON and Markdown while preserving the stdout payload."""
    json_out = tmp_path / "report.json"
    md_out = tmp_path / "report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_pulsed_scenario_scheduler.py",
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--skip-rust",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
    )

    bench.main()

    assert json.loads(json_out.read_text(encoding="utf-8"))["orientation_only"] is True
    assert "Pulsed-Scenario Scheduler Benchmark" in md_out.read_text(encoding="utf-8")
    assert '"schema_version": "scpn-control.pulsed-scenario-scheduler-benchmark.v1"' in capsys.readouterr().out


def test_cli_can_emit_stdout_only(monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    """Output files are optional for interactive orientation runs."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench_pulsed_scenario_scheduler.py", "--iterations", "1", "--warmup", "0", "--skip-rust"],
    )

    bench.main()

    assert '"orientation_only": true' in capsys.readouterr().out


def test_host_helpers_fail_soft(monkeypatch: MonkeyPatch) -> None:
    """Unavailable load and affinity metadata remain explicit nulls."""
    monkeypatch.setattr(os, "getloadavg", lambda: (_ for _ in ()).throw(OSError("missing")))
    assert bench._loadavg() is None

    monkeypatch.delattr(os, "getloadavg", raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert bench._loadavg() is None
    assert bench._affinity() is None


def test_python_campaign_detects_contract_drift(monkeypatch: MonkeyPatch) -> None:
    """A broken benchmark fixture cannot emit plausible timing evidence."""
    monkeypatch.setattr(bench, "EXPECTED_STATES", ["idle"])
    with pytest.raises(RuntimeError, match="canonical eight-state campaign"):
        bench._run_python_campaign()
