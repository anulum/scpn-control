# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Python/Rust capacitor-bank state benchmark producer.
"""The capacitor-bank benchmark times both languages with one statistic definition.

The Python timings run in process with tiny iteration counts. The Rust side is
exercised through a stand-in ``cargo`` that checks the generated harness
project, so no release build runs here.
"""

from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import validation.benchmark_capacitor_bank_state as bench
from scpn_control.benchmark_records import CAMPAIGN_ENV


def _argv(json_out: Path, *extra: str) -> list[str]:
    """Return a command line with tiny iteration counts writing to ``json_out``."""
    return [
        "benchmark_capacitor_bank_state.py",
        "--json-out",
        str(json_out),
        "--step-iterations",
        "5",
        "--free-response-iterations",
        "4",
        "--discharge-iterations",
        "3",
        "--warmup",
        "200",
        *extra,
    ]


def test_main_writes_python_timings_and_records_a_missing_cargo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without cargo the report keeps the Python timings, marks Rust skipped and names its JSON."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    json_out = tmp_path / "out" / "capacitor.json"
    monkeypatch.setattr(sys, "argv", _argv(json_out))

    bench.main()

    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["benchmarks"]["rust"] == {"skipped": True, "reason": "cargo not found"}
    python = report["benchmarks"]["python"]
    assert [python[name]["n"] for name in ("free_response_ns", "step_ns", "discharge_10_step_ns")] == [4, 5, 3]
    assert report["parameters"] == {
        "step_iterations": 5,
        "free_response_iterations": 4,
        "discharge_iterations": 3,
        "warmup": 200,
    }
    assert report["host"]["rustc"] is None and report["host"]["cargo"] is None
    assert report["finished_unix_ns"] >= report["started_unix_ns"]
    markdown = json_out.with_suffix(".md").read_text(encoding="utf-8")
    assert "# Capacitor-Bank State Model Benchmark" in markdown.splitlines()[:3]
    assert f"- JSON evidence: `{json_out.as_posix()}`" in markdown


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--step-iterations", "0", "benchmark iterations must be positive"),
        ("--free-response-iterations", "-1", "benchmark iterations must be positive"),
        ("--discharge-iterations", "0", "benchmark iterations must be positive"),
        ("--warmup", "-1", "warmup must be non-negative"),
    ],
)
def test_main_refuses_nonpositive_iterations_and_negative_warmup(
    option: str, value: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid counts stop the run before any timing or report write."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    json_out = tmp_path / "capacitor.json"
    monkeypatch.setattr(sys, "argv", _argv(json_out, option, value))
    with pytest.raises(SystemExit, match=f"^{message}$"):
        bench.main()
    assert not json_out.exists()


def test_rust_harness_project_depends_on_the_checked_in_crate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The temporary project builds the embedded harness against the crate path with the given counts."""
    seen: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Record the generated project and answer with one JSON line."""
        cwd = Path(kwargs["cwd"])
        seen.update(cmd=cmd, cwd=cwd, timeout=kwargs["timeout"])
        seen["manifest"] = (cwd / "Cargo.toml").read_text(encoding="utf-8")
        seen["source"] = (cwd / "src" / "main.rs").read_text(encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout='{"step_ns": {"n": 7}}\n', stderr="")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/cargo")
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = bench._rust_benchmarks(step_iterations=7, free_response_iterations=8, discharge_iterations=9, warmup=10)

    assert result == {"step_ns": {"n": 7}}
    assert seen["cmd"] == ["cargo", "run", "--release", "--quiet", "--", "7", "8", "9", "10"]
    assert f'control-control = {{ path = "{bench.RUST_CRATE}" }}' in seen["manifest"]
    assert seen["source"] == bench._RUST_BENCH_SOURCE
    assert seen["timeout"] == 180
    assert not seen["cwd"].exists()


def test_rust_harness_median_matches_the_python_definition() -> None:
    """The embedded Rust statistics average the middle pair for an even count, as ``_stats`` does."""
    assert "(sorted[n / 2 - 1] + sorted[n / 2]) as f64 / 2.0" in bench._RUST_BENCH_SOURCE
    assert "((n as f64 * 0.95) as usize).min(n - 1)" in bench._RUST_BENCH_SOURCE


def test_stats_median_averages_the_middle_pair_and_ranks_the_tails() -> None:
    """Even counts average the middle pair; tail percentiles take the capped nearest rank."""
    assert bench._stats([4, 1, 3, 2]) == {
        "n": 4.0,
        "mean": 2.5,
        "median": 2.5,
        "p95": 4.0,
        "p99": 4.0,
        "min": 1.0,
        "max": 4.0,
    }
    assert bench._stats([5, 1, 3])["median"] == 3
    assert bench._stats(list(range(1, 101)))["p95"] == 96.0


def test_measure_ns_calls_once_per_sample() -> None:
    """Each sample is one call and a non-negative duration."""
    calls: list[int] = []
    samples = bench._measure_ns(3, lambda: calls.append(1))
    assert len(samples) == len(calls) == 3
    assert all(sample >= 0 for sample in samples)


def test_host_probes_report_absence_as_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing affinity, governor and tool probes are recorded as None, present ones as text."""
    governor = tmp_path / "scaling_governor"
    monkeypatch.setattr(bench, "Path", lambda _path: governor)
    assert bench._governor() is None
    governor.write_text("performance\n", encoding="utf-8")
    assert bench._governor() == "performance"

    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert bench._affinity() is None


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    [("rustc 1.80.0\n", "", "rustc 1.80.0"), ("", "cargo 1.80.0\n", "cargo 1.80.0"), ("", "", None)],
)
def test_command_version_takes_the_first_nonempty_stream(
    stdout: str, stderr: str, expected: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A version probe prefers standard output, falls back to standard error, and is None when silent."""
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Answer the probe with the parametrised streams."""
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert bench._command_version(["rustc", "--version"]) == expected


def test_script_entry_refuses_the_committed_report_without_a_campaign(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as a script with its default output, the benchmark stops at the recorded-campaign guard."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    monkeypatch.setattr(sys, "argv", ["benchmark_capacitor_bank_state.py"])
    before = bench.DEFAULT_JSON.read_bytes() if bench.DEFAULT_JSON.exists() else None
    with pytest.raises(RuntimeError, match="requires tools/run_recorded_benchmark.py"):
        runpy.run_path(str(bench.__file__), run_name="__main__")
    assert (bench.DEFAULT_JSON.read_bytes() if bench.DEFAULT_JSON.exists() else None) == before
