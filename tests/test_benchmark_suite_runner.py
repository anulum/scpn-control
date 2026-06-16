# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — benchmark suite runner tests

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.run_benchmark_suite as rbs
from tools.benchmark_regression_gate import (
    canonical_metrics_digest,
    verify_baseline_integrity,
)
from tools.run_benchmark_suite import (
    BASELINE_SCHEMA,
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
    report_to_baseline,
    run_suite,
)


def _synthetic_report() -> dict:
    return {
        "schema_version": REPORT_SCHEMA,
        "generated_utc": "2026-06-15T21:00:00Z",
        "evidence_class": "local_regression",
        "production_claim_allowed": False,
        "provenance": {"commit": "feedface", "cpu_model": "test-cpu"},
        "settings": {"steps": 10, "warmup": 2},
        "benchmarks": {
            "capacitor_bank_discharge": {
                "languages": {
                    "python": {"p50_us": 1300.0, "p95_us": 1400.0, "p99_us": 1500.0, "throughput_ops_s": 760.0},
                    "rust": {"p50_us": 15.0, "p95_us": 17.0, "p99_us": 28.0, "throughput_ops_s": 65000.0},
                },
                "cross_language_parity": {"max_relative_difference": 1.4e-16},
            }
        },
        "payload_sha256": "unused-by-baseline-conversion",
    }


def test_language_metrics_maps_percentiles_and_derives_throughput() -> None:
    stats = {"median_us": 12.5, "p95_us": 18.0, "p99_us": 25.0, "mean_us": 10.0}
    metrics = _language_metrics(stats)
    assert metrics["p50_us"] == 12.5
    assert metrics["p95_us"] == 18.0
    assert metrics["p99_us"] == 25.0
    # throughput is the inverse of the mean latency: 1e6 / 10 us = 1e5 ops/s.
    assert metrics["throughput_ops_s"] == pytest.approx(1.0e5)


def test_language_metrics_zero_mean_yields_zero_throughput() -> None:
    metrics = _language_metrics({"median_us": 0.0, "p95_us": 0.0, "p99_us": 0.0, "mean_us": 0.0})
    assert metrics["throughput_ops_s"] == 0.0


def test_registry_contains_capacitor_bank_discharge() -> None:
    assert "capacitor_bank_discharge" in BENCHMARKS
    assert callable(BENCHMARKS["capacitor_bank_discharge"])


def test_rust_release_profile_is_read_from_workspace_manifest() -> None:
    profile = _rust_release_profile()
    # The committed workspace pins an optimised release profile; the runner must
    # record it from the manifest rather than inventing flags.
    assert profile.get("lto") == "fat"
    assert profile.get("opt-level") == 3
    assert profile.get("codegen-units") == 1


def test_report_to_baseline_sets_schema_commit_and_digest() -> None:
    baseline = report_to_baseline(_synthetic_report(), suite="capacitor_bank")
    assert baseline["schema_version"] == BASELINE_SCHEMA
    assert baseline["suite"] == "capacitor_bank"
    assert baseline["baseline_commit"] == "feedface"
    assert baseline["baseline_sha256"] == canonical_metrics_digest(baseline["benchmarks"])


def test_report_to_baseline_output_passes_gate_integrity_check() -> None:
    # The runner and the gate must agree on the tamper digest; a baseline the
    # runner writes must validate cleanly in the gate.
    baseline = report_to_baseline(_synthetic_report(), suite="capacitor_bank")
    assert verify_baseline_integrity(baseline) == []


# ── provenance helpers ────────────────────────────────────────────────


def test_ensure_repo_on_path_inserts_missing_entries() -> None:
    fake_path: list[str] = []
    rbs._ensure_repo_on_path(fake_path)
    assert str(rbs.REPO_ROOT) in fake_path
    assert str(rbs.REPO_ROOT / "src") in fake_path
    # Idempotent: a second call must not duplicate entries.
    rbs._ensure_repo_on_path(fake_path)
    assert fake_path.count(str(rbs.REPO_ROOT)) == 1


def test_provenance_helpers_return_expected_types() -> None:
    assert isinstance(_cpu_model(), str) and _cpu_model()
    assert _affinity() is None or isinstance(_affinity(), list)
    assert _loadavg() is None or isinstance(_loadavg(), list)
    assert isinstance(_git_commit(), str) and _git_commit()
    assert isinstance(_peak_rss_mb(), float)


def test_cpu_model_falls_back_when_cpuinfo_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BoomPath:
        def __init__(self, *_args: object) -> None: ...

        def read_text(self, *_args: object, **_kwargs: object) -> str:
            raise OSError("no /proc")

    monkeypatch.setattr(rbs, "Path", _BoomPath)
    monkeypatch.setattr(rbs.platform, "processor", lambda: "fallback-cpu")
    assert _cpu_model() == "fallback-cpu"


def test_affinity_returns_none_without_sched_getaffinity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(rbs.os, "sched_getaffinity", raising=False)
    assert _affinity() is None


def test_loadavg_returns_none_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> list[float]:
        raise OSError("no loadavg")

    monkeypatch.setattr(rbs.os, "getloadavg", _raise)
    assert _loadavg() is None


def test_git_commit_falls_back_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_args: object, **_kwargs: object) -> object:
        raise OSError("no git")

    monkeypatch.setattr(rbs.subprocess, "run", _raise)
    assert _git_commit() == "unknown"


def test_rust_release_profile_falls_back_when_manifest_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rbs, "RUST_CARGO", Path("/nonexistent/Cargo.toml"))
    assert _rust_release_profile() == {}


# ── capacitor benchmark adapter ───────────────────────────────────────


def _fake_measure(*, steps: int, warmup: int, discharge_steps: int, dt_s: float) -> dict:
    stats = {"median_us": 12.0, "p95_us": 14.0, "p99_us": 20.0, "mean_us": 10.0}
    return {
        "languages": {
            "python": {"stats": stats},
            "rust": {"stats": {"median_us": 0.2, "p95_us": 0.3, "p99_us": 0.5, "mean_us": 0.15}},
            "cross_language_parity": {"max_relative_difference": 1.4e-16},
            "rust_speedup_vs_python": 66.0,
        }
    }


def _fake_measure_python_only(**kwargs: object) -> dict:
    return {
        "languages": {
            "python": {"stats": {"median_us": 12.0, "p95_us": 14.0, "p99_us": 20.0, "mean_us": 10.0}},
            "rust": None,
            "cross_language_parity": None,
        }
    }


def test_capacitor_bank_discharge_normalises_both_languages(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.bench_capacitor_bank_energy as bench

    monkeypatch.setattr(bench, "_measure", _fake_measure)
    result = rbs._capacitor_bank_discharge(steps=5, warmup=1)
    assert result["rust_available"] is True
    assert set(result["languages"]) == {"python", "rust"}
    assert result["languages"]["rust"]["p50_us"] == 0.2


def test_capacitor_bank_discharge_handles_absent_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.bench_capacitor_bank_energy as bench

    monkeypatch.setattr(bench, "_measure", _fake_measure_python_only)
    result = rbs._capacitor_bank_discharge(steps=5, warmup=1)
    assert result["rust_available"] is False
    assert "rust" not in result["languages"]


# ── run_suite + main ──────────────────────────────────────────────────


def _fake_bench(steps: int, warmup: int) -> dict:
    return {
        "languages": {"python": {"p50_us": 12.0, "p95_us": 14.0, "p99_us": 20.0, "throughput_ops_s": 1.0e5}},
        "cross_language_parity": None,
        "rust_available": False,
    }


def test_run_suite_assembles_a_valid_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(rbs.BENCHMARKS, "capacitor_bank_discharge", _fake_bench)
    report = run_suite(
        names=["capacitor_bank_discharge"],
        steps=5,
        warmup=1,
        evidence_class="local_regression",
        generated_utc="2026-06-16T00:00:00Z",
    )
    assert report["schema_version"] == REPORT_SCHEMA
    assert report["provenance"]["rust_backend"] == "absent"
    assert report["benchmarks"]["capacitor_bank_discharge"]["languages"]["python"]["p50_us"] == 12.0
    # payload digest is self-consistent
    digest = report.pop("payload_sha256")
    assert digest == rbs._payload_digest(report)


def test_main_writes_report_and_baseline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(rbs.BENCHMARKS, "capacitor_bank_discharge", _fake_bench)
    report_out = tmp_path / "r.json"
    baseline_out = tmp_path / "b.json"
    rc = main(
        [
            "--benchmarks",
            "capacitor_bank_discharge",
            "--steps",
            "5",
            "--warmup",
            "1",
            "--json-out",
            str(report_out),
            "--write-baseline",
            str(baseline_out),
        ]
    )
    assert rc == 0
    report = json.loads(report_out.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_out.read_text(encoding="utf-8"))
    assert report["schema_version"] == REPORT_SCHEMA
    assert baseline["schema_version"] == BASELINE_SCHEMA
    assert verify_baseline_integrity(baseline) == []


def test_main_prints_report_when_no_output_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(rbs.BENCHMARKS, "capacitor_bank_discharge", _fake_bench)
    rc = main(["--benchmarks", "capacitor_bank_discharge", "--steps", "5", "--warmup", "1"])
    assert rc == 0
    assert REPORT_SCHEMA in capsys.readouterr().out


def test_main_rejects_unknown_benchmark() -> None:
    with pytest.raises(SystemExit):
        main(["--benchmarks", "no_such_bench"])
