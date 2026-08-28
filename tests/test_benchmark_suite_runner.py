# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — benchmark suite runner tests
"""Tests for the polyglot benchmark suite report runner."""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import pytest

import tools.run_benchmark_suite as rbs
import validation.benchmark_capacitor_bank_state as capacitor_bank_state_benchmark
from tools.run_benchmark_suite import (
    BENCHMARKS,
    REPORT_SCHEMA,
    _affinity,
    _cpu_model,
    _git_commit,
    _language_metrics,
    _loadavg,
    _peak_rss_mb,
    _rust_release_profile,
    main,
    run_suite,
)


def test_language_metrics_maps_percentiles_and_derives_throughput() -> None:
    """Map harness latency fields and derive inverse-mean throughput."""
    stats = {"median_us": 12.5, "p95_us": 18.0, "p99_us": 25.0, "mean_us": 10.0}
    metrics = _language_metrics(stats)
    assert metrics["p50_us"] == 12.5
    assert metrics["p95_us"] == 18.0
    assert metrics["p99_us"] == 25.0
    # throughput is the inverse of the mean latency: 1e6 / 10 us = 1e5 ops/s.
    assert metrics["throughput_ops_s"] == pytest.approx(1.0e5)


def test_language_metrics_zero_mean_yields_zero_throughput() -> None:
    """Avoid division by zero when a harness reports zero mean latency."""
    metrics = _language_metrics({"median_us": 0.0, "p95_us": 0.0, "p99_us": 0.0, "mean_us": 0.0})
    assert metrics["throughput_ops_s"] == 0.0


def test_registry_contains_capacitor_bank_discharge() -> None:
    """Keep the canonical capacitor-bank benchmark registered."""
    assert "capacitor_bank_discharge" in BENCHMARKS
    assert callable(BENCHMARKS["capacitor_bank_discharge"])


def test_rust_release_profile_is_read_from_workspace_manifest() -> None:
    """Report the committed Rust release profile rather than inferred flags."""
    profile = _rust_release_profile()
    # The committed workspace pins an optimised release profile; the runner must
    # record it from the manifest rather than inventing flags.
    assert profile.get("lto") == "fat"
    assert profile.get("opt-level") == 3
    assert profile.get("codegen-units") == 1


# ── provenance helpers ────────────────────────────────────────────────


def test_ensure_repo_on_path_inserts_missing_entries() -> None:
    """Add each standalone-run import root exactly once."""
    fake_path: list[str] = []
    rbs._ensure_repo_on_path(fake_path)
    assert str(rbs.REPO_ROOT) in fake_path
    assert str(rbs.REPO_ROOT / "src") in fake_path
    # Idempotent: a second call must not duplicate entries.
    rbs._ensure_repo_on_path(fake_path)
    assert fake_path.count(str(rbs.REPO_ROOT)) == 1


def test_provenance_helpers_return_expected_types() -> None:
    """Return serialisable host-provenance values."""
    assert isinstance(_cpu_model(), str) and _cpu_model()
    assert _affinity() is None or isinstance(_affinity(), list)
    assert _loadavg() is None or isinstance(_loadavg(), list)
    assert isinstance(_git_commit(), str) and _git_commit()
    assert isinstance(_peak_rss_mb(), float)


def test_cpu_model_falls_back_when_cpuinfo_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the platform CPU description if procfs cannot be read."""

    class _BoomPath:
        def __init__(self, *_args: object) -> None: ...

        def read_text(self, *_args: object, **_kwargs: object) -> str:
            raise OSError("no /proc")

    monkeypatch.setattr(rbs, "Path", _BoomPath)
    monkeypatch.setattr(platform, "processor", lambda: "fallback-cpu")
    assert _cpu_model() == "fallback-cpu"


def test_cpu_model_falls_back_when_cpuinfo_has_no_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the platform description when procfs lacks a model-name field."""

    class _NoModelPath:
        def __init__(self, *_args: object) -> None:
            pass

        def read_text(self, *_args: object, **_kwargs: object) -> str:
            return "processor: 0\n"

    monkeypatch.setattr(rbs, "Path", _NoModelPath)
    monkeypatch.setattr(platform, "processor", lambda: "fallback-cpu")
    assert _cpu_model() == "fallback-cpu"


def test_affinity_returns_none_without_sched_getaffinity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Represent unavailable process affinity explicitly."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert _affinity() is None


def test_loadavg_returns_none_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Represent unavailable system load averages explicitly."""

    def _raise() -> list[float]:
        raise OSError("no loadavg")

    # getloadavg is absent on Windows, so allow creating the attribute there.
    monkeypatch.setattr(os, "getloadavg", _raise, raising=False)
    assert _loadavg() is None


@pytest.mark.parametrize(
    "benchmark_module",
    [
        rbs._load_control_benchmark_module("bench_aer_observation.py"),
        rbs._load_control_benchmark_module("bench_capacitor_bank_energy.py"),
        rbs._load_control_benchmark_module("bench_multi_shot_campaign.py"),
        rbs._load_control_benchmark_module("bench_pulsed_mpc_adapter.py"),
        capacitor_bank_state_benchmark,
    ],
)
def test_benchmark_loadavg_handles_windows_absence(monkeypatch: pytest.MonkeyPatch, benchmark_module: Any) -> None:
    """Represent the Windows absence of ``os.getloadavg`` explicitly."""
    monkeypatch.delattr(os, "getloadavg", raising=False)
    assert benchmark_module._loadavg() is None


def test_git_commit_falls_back_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Represent an unavailable Git executable without fabricating a digest."""

    def _raise(*_args: object, **_kwargs: object) -> object:
        raise OSError("no git")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert _git_commit() == "unknown"


def test_rust_release_profile_falls_back_when_manifest_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return an empty release profile when the workspace manifest is absent."""
    monkeypatch.setattr(rbs, "RUST_CARGO", Path("/nonexistent/Cargo.toml"))
    assert _rust_release_profile() == {}


# ── capacitor benchmark adapter ───────────────────────────────────────


def _deterministic_test_measurement(*, steps: int, warmup: int, discharge_steps: int, dt_s: float) -> dict[str, Any]:
    stats = {"median_us": 12.0, "p95_us": 14.0, "p99_us": 20.0, "mean_us": 10.0}
    return {
        "languages": {
            "python": {"stats": stats},
            "rust": {"stats": {"median_us": 0.2, "p95_us": 0.3, "p99_us": 0.5, "mean_us": 0.15}},
            "cross_language_parity": {"max_relative_difference": 1.4e-16},
            "rust_speedup_vs_python": 66.0,
        }
    }


def _deterministic_python_only_test_measurement(**kwargs: object) -> dict[str, Any]:
    return {
        "languages": {
            "python": {"stats": {"median_us": 12.0, "p95_us": 14.0, "p99_us": 20.0, "mean_us": 10.0}},
            "rust": None,
            "cross_language_parity": None,
        }
    }


def test_capacitor_bank_discharge_normalises_both_languages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normalise polyglot harness stats even when FUSION shadows ``benchmarks``."""
    bench = rbs._load_control_benchmark_module("bench_capacitor_bank_energy.py")

    monkeypatch.setattr(bench, "_measure", _deterministic_test_measurement)
    result = rbs._capacitor_bank_discharge(steps=5, warmup=1)
    assert result["rust_available"] is True
    assert set(result["languages"]) == {"python", "rust"}
    assert result["languages"]["rust"]["p50_us"] == 0.2


def test_capacitor_bank_discharge_handles_absent_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """Python-only harness output remains admissible without a Rust backend."""
    bench = rbs._load_control_benchmark_module("bench_capacitor_bank_energy.py")

    monkeypatch.setattr(bench, "_measure", _deterministic_python_only_test_measurement)
    result = rbs._capacitor_bank_discharge(steps=5, warmup=1)
    assert result["rust_available"] is False
    assert "rust" not in result["languages"]


def test_benchmark_module_loader_rejects_missing_import_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing file-loader specification fails with the exact harness path."""
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *_args, **_kwargs: None)
    with pytest.raises(ImportError, match="cannot load CONTROL benchmark harness"):
        rbs._load_control_benchmark_module("not-present.py")


# ── run_suite + main ──────────────────────────────────────────────────


def test_run_suite_assembles_a_valid_report() -> None:
    """Assemble a self-digesting report with explicit backend provenance."""
    report = run_suite(
        names=["capacitor_bank_discharge"],
        steps=2,
        warmup=1,
        evidence_class="local_regression",
        generated_utc="2026-06-16T00:00:00Z",
    )
    assert report["schema_version"] == REPORT_SCHEMA
    assert report["provenance"]["rust_backend"] in {"present", "absent"}
    assert report["benchmarks"]["capacitor_bank_discharge"]["languages"]["python"]["p50_us"] > 0.0
    # payload digest is self-consistent
    digest = report.pop("payload_sha256")
    assert digest == rbs._payload_digest(report)


def test_main_writes_report(tmp_path: Path) -> None:
    """Write a report only to an explicitly requested temporary path."""
    report_out = tmp_path / "r.json"
    rc = main(
        [
            "--benchmarks",
            "capacitor_bank_discharge",
            "--steps",
            "2",
            "--warmup",
            "1",
            "--json-out",
            str(report_out),
        ]
    )
    assert rc == 0
    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert report["schema_version"] == REPORT_SCHEMA


def test_main_rejects_implicit_baseline_update(tmp_path: Path) -> None:
    """Benchmark execution has no option that can update a baseline."""
    baseline_out = tmp_path / "baseline.json"
    with pytest.raises(SystemExit):
        main(["--write-baseline", str(baseline_out)])
    assert not baseline_out.exists()


def test_main_prints_report_when_no_output_path(capsys: pytest.CaptureFixture[str]) -> None:
    """Print a report when no persistent destination is requested."""
    rc = main(["--benchmarks", "capacitor_bank_discharge", "--steps", "2", "--warmup", "1"])
    assert rc == 0
    assert REPORT_SCHEMA in capsys.readouterr().out


def test_main_rejects_unknown_benchmark() -> None:
    """Fail closed when a benchmark name is not registered."""
    with pytest.raises(SystemExit):
        main(["--benchmarks", "no_such_bench"])
