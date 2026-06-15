# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — benchmark suite runner tests

from __future__ import annotations

import pytest

from tools.benchmark_regression_gate import (
    canonical_metrics_digest,
    verify_baseline_integrity,
)
from tools.run_benchmark_suite import (
    BASELINE_SCHEMA,
    BENCHMARKS,
    REPORT_SCHEMA,
    _language_metrics,
    _rust_release_profile,
    report_to_baseline,
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
