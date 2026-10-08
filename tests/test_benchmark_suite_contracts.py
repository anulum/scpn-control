# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark suite settings, adapter and publication contracts.
"""Exercise public settings and declared adapter records without claiming timings."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools import run_benchmark_suite as suite
from tools.benchmark_suite_metrics import BenchmarkSuiteRefusal, checked_adapter_result
from tools.run_benchmark_suite import _language_metrics

ROOT = Path(__file__).resolve().parents[1]
DECLARED: dict[str, object] = {
    "languages": {"python": {"p50_us": 12.0, "p95_us": 14.0, "p99_us": 20.0, "throughput_ops_s": 100000.0}},
    "rust_available": False,
    "cross_language_parity": None,
}


@pytest.mark.parametrize(
    ("names", "steps", "warmup", "label", "utc"),
    [
        ([], 2, 0, "local_regression", "2026-06-16T00:00:00Z"),
        (["capacitor_bank_discharge"] * 2, 2, 0, "local_regression", "2026-06-16T00:00:00Z"),
        (["unknown"], 2, 0, "local_regression", "2026-06-16T00:00:00Z"),
        (["capacitor_bank_discharge"], 0, 0, "local_regression", "2026-06-16T00:00:00Z"),
        (["capacitor_bank_discharge"], -1, 0, "local_regression", "2026-06-16T00:00:00Z"),
        (["capacitor_bank_discharge"], True, 0, "local_regression", "2026-06-16T00:00:00Z"),
        (["capacitor_bank_discharge"], 2, -1, "local_regression", "2026-06-16T00:00:00Z"),
        (["capacitor_bank_discharge"], 2, False, "local_regression", "2026-06-16T00:00:00Z"),
        (["capacitor_bank_discharge"], 2, 0, " ", "2026-06-16T00:00:00Z"),
        (["capacitor_bank_discharge"], 2, 0, "local_regression", "PRIVATE_INVALID_DATE"),
        (["capacitor_bank_discharge"], 2, 0, "local_regression", "2026-06-16T00:00:00"),
        (["capacitor_bank_discharge"], 2, 0, "local_regression", "2026-06-16T00:00:00+02:00"),
    ],
)
def test_public_suite_refuses_invalid_settings_before_adapter(
    monkeypatch: pytest.MonkeyPatch, names: list[str], steps: int, warmup: int, label: str, utc: str
) -> None:
    """Invalid selections and settings must refuse before any measurement callback."""

    def unexpected(steps: int, warmup: int) -> object:
        raise AssertionError("an invalid suite must not execute its adapter")

    monkeypatch.setattr(suite, "BENCHMARKS", {"capacitor_bank_discharge": unexpected})
    with pytest.raises(BenchmarkSuiteRefusal):
        suite.run_suite(names=names, steps=steps, warmup=warmup, evidence_class=label, generated_utc=utc)


@pytest.mark.parametrize("value", [None, [], {1: "bad"}, {}, {"languages": {}}, {"languages": {"python": None}}])
def test_public_suite_rejects_malformed_registered_result(monkeypatch: pytest.MonkeyPatch, value: object) -> None:
    """A registered callback cannot publish a malformed consumed record."""
    monkeypatch.setattr(suite, "BENCHMARKS", {"declared-test-record": lambda steps, warmup: value})
    with pytest.raises(BenchmarkSuiteRefusal):
        suite.run_suite(
            names=["declared-test-record"],
            steps=2,
            warmup=0,
            evidence_class="test_declaration",
            generated_utc="2026-06-16T00:00:00Z",
        )


@pytest.mark.parametrize("value", [True, "12", -1.0, float("nan"), float("inf"), 10**1000])
def test_registered_gate_metrics_refuse_invalid_numbers(monkeypatch: pytest.MonkeyPatch, value: object) -> None:
    """Unusable metrics from a registered record must never receive a report digest."""
    declaration = copy.deepcopy(DECLARED)
    languages = declaration["languages"]
    assert isinstance(languages, dict)
    languages["python"]["p50_us"] = value
    monkeypatch.setattr(suite, "BENCHMARKS", {"declared-test-record": lambda steps, warmup: declaration})
    with pytest.raises(BenchmarkSuiteRefusal):
        suite.run_suite(
            names=["declared-test-record"],
            steps=2,
            warmup=0,
            evidence_class="test_declaration",
            generated_utc="2026-06-16T00:00:00Z",
        )


@pytest.mark.parametrize("flag", [True, 1, "present", None])
def test_rust_availability_requires_exact_consistent_boolean(flag: object) -> None:
    """An absent Rust row cannot become present through a truthy declaration."""
    declaration = copy.deepcopy(DECLARED)
    declaration["rust_available"] = flag
    with pytest.raises(BenchmarkSuiteRefusal):
        checked_adapter_result(declaration)


@pytest.mark.parametrize("difference", [True, "0", -1.0, float("nan"), float("inf"), 10**1000, None])
def test_present_rust_requires_usable_parity(difference: object) -> None:
    """Rust metrics require a finite nonnegative declared relative difference."""
    declaration = copy.deepcopy(DECLARED)
    languages = declaration["languages"]
    assert isinstance(languages, dict)
    languages["rust"] = dict(languages["python"])
    declaration["rust_available"] = True
    declaration["cross_language_parity"] = {"max_relative_difference": difference}
    with pytest.raises(BenchmarkSuiteRefusal):
        checked_adapter_result(declaration)


def test_coherent_failing_parity_is_retained_without_new_threshold() -> None:
    """A finite mismatch remains a declaration rather than an invented admission."""
    declaration = copy.deepcopy(DECLARED)
    languages = declaration["languages"]
    assert isinstance(languages, dict)
    languages["rust"] = dict(languages["python"])
    declaration["rust_available"] = True
    declaration["cross_language_parity"] = {"max_relative_difference": 0.5}
    result = checked_adapter_result(declaration)
    assert result["cross_language_parity"] == {"max_relative_difference": 0.5}


@pytest.mark.parametrize(
    "value",
    [True, "12", -1.0, float("nan"), float("inf"), 10**1000],
    ids=["bool", "text", "negative", "nan", "infinite", "overflow"],
)
def test_latency_normalization_refuses_unusable_harness_statistics(value: object) -> None:
    """Consumed statistic declarations cannot become finite-looking gate observations."""
    statistics: dict[str, object] = {"mean_us": value, "median_us": 12.0, "p95_us": 14.0, "p99_us": 20.0}
    with pytest.raises(BenchmarkSuiteRefusal):
        _language_metrics(statistics)


@pytest.mark.parametrize("mean", [10.0, 1e-310], ids=["percentile-order", "reciprocal-overflow"])
def test_normalization_refuses_order_or_derived_overflow(mean: float) -> None:
    """Ordering and reciprocal overflow refuse without changing the zero-mean convention."""
    statistics: dict[str, object] = {
        "mean_us": mean,
        "median_us": 21.0 if mean == 10.0 else 12.0,
        "p95_us": 14.0,
        "p99_us": 20.0,
    }
    with pytest.raises(BenchmarkSuiteRefusal):
        _language_metrics(statistics)


@pytest.mark.parametrize("case", ["blank-language", "missing-metric", "blank-metric", "order", "spurious-parity"])
def test_adapter_records_refuse_inconsistent_consumed_declarations(case: str) -> None:
    """Registered metric maps must keep required fields and coherent parity context."""
    declaration = copy.deepcopy(DECLARED)
    languages = declaration["languages"]
    assert isinstance(languages, dict)
    if case == "blank-language":
        languages[" "] = languages.pop("python")
    elif case == "missing-metric":
        languages["python"].pop("p99_us")
    elif case == "blank-metric":
        languages["python"][" "] = 1.0
    elif case == "order":
        languages["python"]["p50_us"] = 99.0
    else:
        declaration["cross_language_parity"] = {"max_relative_difference": 0.0}
    with pytest.raises(BenchmarkSuiteRefusal):
        checked_adapter_result(declaration)


@pytest.mark.parametrize("text", ["[profile]\nrelease=1\n", "profile=1\n", "[invalid\n"])
def test_actual_unusable_release_manifest_is_unavailable(tmp_path: Path, text: str) -> None:
    """Malformed or non-table workspace metadata cannot fabricate release flags."""
    from tools.benchmark_suite_provenance import _rust_release_profile

    manifest = tmp_path / "Cargo.toml"
    manifest.write_text(text, encoding="utf-8")
    assert _rust_release_profile(manifest) == {}


def test_selected_cpuinfo_reads_a_real_declared_model_file(tmp_path: Path) -> None:
    """Native readers parse the selected local declaration without claiming host identity."""
    from tools.benchmark_suite_provenance import _cpu_model

    source = tmp_path / "cpuinfo"
    source.write_text("processor: 0\nmodel name: declared test CPU\n", encoding="utf-8")
    assert _cpu_model(source) == "declared test CPU"


def test_public_report_converts_a_valid_declared_resource_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Observer protocol conversion is a unit test, not a native Windows RSS measurement."""
    from tools import benchmark_suite_provenance as context

    observer = ModuleType("declared-resource-observer")
    observer.__dict__["RUSAGE_SELF"] = 0
    observer.__dict__["getrusage"] = lambda who: SimpleNamespace(ru_maxrss=4096)
    monkeypatch.setattr(context, "resource_module", observer)
    monkeypatch.setattr(suite, "BENCHMARKS", {"declared-test-record": lambda steps, warmup: DECLARED})
    report = suite.run_suite(
        names=["declared-test-record"],
        steps=2,
        warmup=0,
        evidence_class="test_declaration",
        generated_utc="2026-06-16T00:00:00Z",
    )
    assert report["provenance"]["peak_rss_mb"] == 4.0


@pytest.mark.parametrize("raw", [True, "invalid", -1], ids=["bool", "text", "negative"])
def test_public_report_does_not_fabricate_unusable_resource_observation(
    monkeypatch: pytest.MonkeyPatch, raw: object
) -> None:
    """Malformed observer declarations stay unavailable, separate from real timing proofs."""
    from tools import benchmark_suite_provenance as context

    observer = ModuleType("declared-resource-observer")
    observer.__dict__["RUSAGE_SELF"] = 0
    observer.__dict__["getrusage"] = lambda who: SimpleNamespace(ru_maxrss=raw)
    monkeypatch.setattr(context, "resource_module", observer)
    monkeypatch.setattr(suite, "BENCHMARKS", {"declared-test-record": lambda steps, warmup: DECLARED})
    report = suite.run_suite(
        names=["declared-test-record"],
        steps=2,
        warmup=0,
        evidence_class="test_declaration",
        generated_utc="2026-06-16T00:00:00Z",
    )
    assert report["provenance"]["peak_rss_mb"] is None


def test_missing_resource_import_is_explicitly_unavailable(tmp_path: Path) -> None:
    """Actual import denial exercises optional-resource fallback without a zero observation."""
    script = f"""
import sys, importlib.abc
sys.path.insert(0, {str(ROOT)!r})
sys.modules.pop("resource", None)
class MissingResource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "resource":
            raise ModuleNotFoundError("resource denied by test resolver")
        return None
sys.meta_path.insert(0, MissingResource())
from tools.benchmark_suite_provenance import _peak_rss_mb
assert _peak_rss_mb() is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0 and result.stdout == "" and result.stderr == ""


def test_actual_cli_success_and_atomic_output(tmp_path: Path) -> None:
    """A valid executable suite must publish real available-language measurements."""
    target = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_benchmark_suite.py"),
            "--steps",
            "2",
            "--warmup",
            "0",
            "--json-out",
            str(target),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0 and result.stdout == "" and result.stderr == ""
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["production_claim_allowed"] is False
    assert report["benchmarks"]["capacitor_bank_discharge"]["languages"]["python"]["p50_us"] >= 0


def test_real_script_help_exits_without_measurement(tmp_path: Path) -> None:
    """The executable parser help surface must remain available before measurement."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/run_benchmark_suite.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0 and "--warmup" in result.stdout and result.stderr == ""


@pytest.mark.parametrize("args", [["--benchmarks"], ["--steps", "0"], ["--warmup", "-1"]])
def test_real_cli_invalid_settings_have_no_traceback(tmp_path: Path, args: list[str]) -> None:
    """Executable malformed settings must fail without producing files or native text."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/run_benchmark_suite.py"), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1 and result.stdout == ""
    assert result.stderr.startswith("benchmark suite FAILED: ") and "Traceback" not in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_actual_cli_source_alias_is_refused_before_loading_harness(tmp_path: Path) -> None:
    """A copied runner cannot replace itself even with otherwise valid settings."""
    root = tmp_path / "owned"
    (root / "tools").mkdir(parents=True)
    for relative in (
        "tools/__init__.py",
        "tools/benchmark_suite_metrics.py",
        "tools/benchmark_suite_provenance.py",
        "tools/inventory_file_output.py",
        "validation/__init__.py",
        "validation/report_output_paths.py",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    shutil.copytree(
        ROOT / "src/scpn_control",
        root / "src/scpn_control",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    target = root / "tools/run_benchmark_suite.py"
    original = (ROOT / "tools/run_benchmark_suite.py").read_bytes()
    target.write_bytes(original)
    result = subprocess.run(
        [sys.executable, "-I", str(target), "--steps", "2", "--warmup", "0", "--json-out", str(target)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1 and target.read_bytes() == original and result.stdout == ""
    assert result.stderr == "benchmark suite FAILED: inputs, measurement or output could not be processed\n"


@pytest.mark.parametrize("event", ["open", "os.rename"])
def test_publication_native_fault_preserves_predecessor(tmp_path: Path, event: str) -> None:
    """Actual staging/replacement denial through the runtime audit hook keeps old bytes."""
    target = tmp_path / "report.json"
    target.write_bytes(b"retained predecessor\n")
    script = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from tools import run_benchmark_suite as suite
suite.BENCHMARKS = {{"declared-test-record": lambda steps, warmup: {DECLARED!r}}}
def refuse(event, arguments):
    selected = event == "os.rename" and str(arguments[1]) == {str(target)!r}
    staged = event == "open" and str(arguments[0]).startswith({str(target.parent)!r}) and str(arguments[0]).endswith(".tmp")
    if event == {event!r} and (selected or staged):
        raise PermissionError("PRIVATE_PUBLICATION_CANARY")
sys.addaudithook(refuse)
raise SystemExit(suite.main(["--benchmarks", "declared-test-record", "--steps", "2", "--warmup", "0", "--json-out", {str(target)!r}]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 1 and result.stdout == ""
    assert result.stderr == "benchmark suite FAILED: inputs, measurement or output could not be processed\n"
    assert target.read_bytes() == b"retained predecessor\n" and list(tmp_path.iterdir()) == [target]
