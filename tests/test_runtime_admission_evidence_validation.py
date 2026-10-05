# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime admission evidence validation tests.
"""Inspect real persisted probe reports and malformed copies through public API/CLI.

The historical local report is self-sealed evidence of its recorded host only.
Mutated/resealed copies below exercise declaration contracts and do not become
measured production timing, authenticated provenance or current-host admission.
"""

from __future__ import annotations

import doctest
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

import validation.validate_runtime_admission_evidence as runtime_module
from validation.validate_runtime_admission_evidence import DEFAULT_REPORT, main, validate_runtime_admission_evidence


def _canonical_payload_digest(payload: dict[str, object]) -> str:
    """Reseal copied metadata using the documented canonical encoding, not producer authentication."""
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _copy_report(source: Path, destination: Path) -> Path:
    """Preserve a historical report in test-owned scratch before any mutation."""
    shutil.copyfile(source, destination)
    return destination


def _load_report(path: Path) -> dict[str, object]:
    """Read a real JSON object with typed test declarations and no production loader mock."""
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _write_report(path: Path, payload: dict[str, object], *, refresh_digest: bool = True) -> None:
    """Write copied declaration bytes, optionally keeping a deliberately stale self-digest."""
    if refresh_digest:
        payload["payload_sha256"] = _canonical_payload_digest(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_runtime_admission_evidence_admits_repository_report() -> None:
    """Repository runtime-admission evidence is admitted as local fail-closed regression evidence."""
    result = validate_runtime_admission_evidence()

    assert result.status == "pass"
    assert result.errors == ()
    assert result.report_sha256 is not None
    assert result.payload_sha256 is not None
    assert result.benchmark_evidence_class == "local_regression"
    assert result.production_claim_allowed is False
    assert result.admission_status == "fail"
    assert result.admission_error_count is not None
    assert result.admission_error_count > 0


def test_runtime_admission_evidence_rejects_payload_tampering(tmp_path: Path) -> None:
    """Canonical payload digests prevent silent runtime-admission report mutation."""
    report = _copy_report(DEFAULT_REPORT, tmp_path / "runtime.json")
    payload = _load_report(report)
    payload["last_admission_status"] = "pass"
    _write_report(report, payload, refresh_digest=False)

    result = validate_runtime_admission_evidence(report)

    assert result.status == "fail"
    assert "runtime_admission.payload_sha256 does not match canonical payload" in result.errors


def test_runtime_admission_evidence_rejects_local_production_claim(tmp_path: Path) -> None:
    """Local regression runtime-admission evidence cannot be promoted to a production timing claim."""
    report = _copy_report(DEFAULT_REPORT, tmp_path / "runtime.json")
    payload = _load_report(report)
    payload["production_claim_allowed"] = True
    _write_report(report, payload)

    result = validate_runtime_admission_evidence(report)

    assert result.status == "fail"
    assert "local runtime admission evidence must not allow production benchmark claims" in result.errors


def test_runtime_admission_evidence_rejects_missing_context(tmp_path: Path) -> None:
    """Runtime-admission evidence must preserve CPU affinity, load, and isolation context."""
    report = _copy_report(DEFAULT_REPORT, tmp_path / "runtime.json")
    payload = _load_report(report)
    payload["context"] = {}
    _write_report(report, payload)

    result = validate_runtime_admission_evidence(report)

    assert result.status == "fail"
    assert "runtime_admission.context.cpu_affinity must be a non-empty sequence" in result.errors
    assert "runtime_admission.context must record loadavg_start and loadavg_end" in result.errors


def test_runtime_admission_evidence_rejects_non_monotonic_percentiles(tmp_path: Path) -> None:
    """Latency percentile fields must remain internally consistent."""
    report = _copy_report(DEFAULT_REPORT, tmp_path / "runtime.json")
    payload = _load_report(report)
    stats = payload["stats"]
    assert isinstance(stats, dict)
    stats["p99_us"] = 1.0
    stats["p95_us"] = 2.0
    _write_report(report, payload)

    result = validate_runtime_admission_evidence(report)

    assert result.status == "fail"
    assert "runtime_admission.stats percentiles must be monotonic" in result.errors


@pytest.mark.parametrize(
    "section,key,value,fragment",
    [
        ("", "schema_version", [], "schema_version"),
        ("", "evidence_class", {}, "evidence_class"),
        ("", "evidence_class", "unknown", "evidence_class"),
        ("", "production_claim_allowed", 1, "boolean"),
        ("", "command", None, "command"),
        ("", "command", "another_benchmark.py", "command"),
        ("", "last_admission_status", [], "last_admission_status"),
        ("", "last_admission_status", "unknown", "last_admission_status"),
        ("", "last_admission_errors", None, "must be a list"),
        ("", "last_admission_errors", [{}], "non-empty strings"),
        ("", "last_admission_errors", [""], "non-empty strings"),
        ("", "last_admission_errors", [], "fail-closed errors"),
        ("", "last_admission_warnings", {}, "warnings must be a list"),
        ("", "last_admission_warnings", [True], "warnings must contain"),
        ("", "last_admission_warnings", [""], "warnings must contain"),
        ("", "context", None, "context must be an object"),
        ("context", "cpu_affinity", [], "non-empty sequence"),
        ("context", "cpu_affinity", [None], "integer CPU IDs"),
        ("context", "cpu_affinity", [True], "integer CPU IDs"),
        ("context", "cpu_affinity", [-1], "integer CPU IDs"),
        ("context", "platform", None, "platform must be recorded"),
        ("context", "platform", "", "platform must be recorded"),
        ("context", "python", False, "python must be recorded"),
        ("context", "python", "", "python must be recorded"),
        ("context", "isolation_method", [], "isolation_method must be recorded"),
        ("context", "isolation_method", "", "isolation_method must be recorded"),
        ("context", "loadavg_start", None, "record loadavg"),
        ("context", "loadavg_end", [], "record loadavg"),
        ("context", "loadavg_start", [0.0], "three finite"),
        ("context", "loadavg_end", [0.0, False, 0.0], "three finite"),
        ("context", "loadavg_start", [0.0, -1.0, 0.0], "three finite"),
        ("context", "loadavg_start", ["bad"], "three finite"),
        ("", "stats", None, "stats must be an object"),
        ("stats", "samples", True, "positive integer"),
        ("stats", "samples", 0, "positive integer"),
        ("stats", "samples", 1.0, "positive integer"),
        ("stats", "min_us", -1, "min_us must be finite"),
        ("stats", "median_us", None, "median_us must be finite"),
        ("stats", "mean_us", False, "mean_us must be finite"),
        ("stats", "p95_us", [], "p95_us must be finite"),
        ("stats", "p99_us", "fast", "p99_us must be finite"),
        ("stats", "max_us", -1.0, "max_us must be finite"),
        pytest.param("stats", "mean_us", 10**1000, "mean_us must be finite", id="integer-float-overflow"),
        ("stats", "mean_us", 1e30, "between min_us and max_us"),
        ("stats", "min_us", 1e30, "percentiles must be monotonic"),
        ("stats", "median_us", 1e30, "percentiles must be monotonic"),
        ("stats", "p99_us", 1e30, "percentiles must be monotonic"),
    ],
)
def test_public_report_domain_refusals(tmp_path: Path, section: str, key: str, value: object, fragment: str) -> None:
    """Malformed resealed historical copies fail declaration checks instead of crashing or admitting impossible metadata."""
    payload = _load_report(DEFAULT_REPORT)
    target = cast(dict[str, Any], payload[section]) if section else payload
    target[key] = value
    path = tmp_path / "invalid.json"
    _write_report(path, payload)
    result = validate_runtime_admission_evidence(path)
    assert result.status == "fail" and result.report_sha256 is not None
    assert any(fragment in error for error in result.errors), result.errors
    if key in ["evidence_class", "last_admission_status", "production_claim_allowed"] and not isinstance(
        value, str | bool
    ):
        field = {"evidence_class": "benchmark_evidence_class", "last_admission_status": "admission_status"}.get(
            key, key
        )
        assert result.as_dict()[field] is None


@pytest.mark.parametrize("digest", [None, [], "x", "A" * 64])
def test_malformed_self_digest_refuses(tmp_path: Path, digest: object) -> None:
    """Missing, non-string, wrong-length and uppercase self-digests fail the public spelling check."""
    payload = _load_report(DEFAULT_REPORT)
    payload["payload_sha256"] = digest
    path = tmp_path / "digest.json"
    _write_report(path, payload, refresh_digest=False)
    result = validate_runtime_admission_evidence(path)
    assert result.status == "fail" and any("SHA-256 hex" in error for error in result.errors)
    assert result.payload_sha256 == (digest if isinstance(digest, str) else None)


@pytest.mark.parametrize(
    "production_claim,status,errors,passing",
    [
        (False, "fail", ["not realtime"], False),
        (True, "fail", ["not realtime"], False),
        (True, "pass", ["still failed"], False),
        (True, "pass", [], True),
    ],
)
def test_production_declaration_boundary(
    tmp_path: Path, production_claim: bool, status: str, errors: list[str], passing: bool
) -> None:
    """Resealed production-shaped declarations require claim true, probe pass and empty errors without proving realtime qualification."""
    payload = _load_report(DEFAULT_REPORT)
    payload.update(
        evidence_class="production_benchmark",
        production_claim_allowed=production_claim,
        last_admission_status=status,
        last_admission_errors=errors,
    )
    path = tmp_path / "production-shaped.json"
    _write_report(path, payload)
    result = validate_runtime_admission_evidence(path)
    assert result.status == ("pass" if passing else "fail")
    assert result.production_claim_allowed is production_claim


def test_empty_object_is_not_admitted(tmp_path: Path) -> None:
    """A decoded empty object keeps its byte digest but fails all missing mandatory declarations."""
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")
    result = validate_runtime_admission_evidence(path)
    assert result.status == "fail" and result.report_sha256 == hashlib.sha256(b"{}").hexdigest()
    assert result.payload_sha256 is None and result.samples is None and result.admission_error_count is None
    assert any("schema_version" in error for error in result.errors)


@pytest.mark.parametrize("errors,passing", [(["still failed"], False), ([], True)])
def test_local_pass_requires_empty_probe_errors(tmp_path: Path, errors: list[str], passing: bool) -> None:
    """Local PASS cannot contradict its probe errors; an empty-error local declaration grants no production claim."""
    payload = _load_report(DEFAULT_REPORT)
    payload.update(last_admission_status="pass", last_admission_errors=errors)
    path = tmp_path / "local-pass.json"
    _write_report(path, payload)
    result = validate_runtime_admission_evidence(path)
    assert result.status == ("pass" if passing else "fail")
    assert result.production_claim_allowed is False
    if not passing:
        assert "passed runtime admission evidence must not carry admission errors" in result.errors


@pytest.mark.parametrize(
    "blob,fragment",
    [
        (b"[]", "root must be"),
        (b"{", "Expecting"),
        (b"\xff", "utf-8"),
        (b'{"extra":{"x":1,"x":2}}', "duplicate JSON key"),
        (b'{"extra":NaN}', "nonfinite"),
        (b'{"extra":Infinity}', "nonfinite"),
        (b'{"extra":-Infinity}', "nonfinite"),
        (b'{"extra":[1e999]}', "nonfinite"),
    ],
)
def test_decode_failures_refuse_without_digest(tmp_path: Path, blob: bytes, fragment: str) -> None:
    """Invalid encoding/JSON, duplicate keys and nonfinite tokens fail before any byte/payload declaration admission."""
    path = tmp_path / "raw.json"
    path.write_bytes(blob)
    result = validate_runtime_admission_evidence(path)
    assert result.status == "fail" and result.report_sha256 is None and result.payload_sha256 is None
    assert fragment.lower() in result.errors[0].lower()


@pytest.mark.parametrize("directory", [False, True])
def test_missing_or_directory_read_returns_findings(tmp_path: Path, directory: bool) -> None:
    """Real IO failures produce structured report findings and no digest or invented sample count."""
    path = tmp_path / "report.json"
    if directory:
        path.mkdir()
    result = validate_runtime_admission_evidence(path)
    assert result.status == "fail" and result.report_sha256 is None and result.samples is None and result.errors


def test_exact_digest_and_self_seal_scope(tmp_path: Path) -> None:
    """Whitespace changes byte digest alone; resealed finite extra fields pass integrity without producer authentication."""
    payload = _load_report(DEFAULT_REPORT)
    payload["extra_unvalidated"] = {"tag": "resealed test copy", "finite_value": 0.25}
    path = tmp_path / "resealed.json"
    _write_report(path, payload)
    first = validate_runtime_admission_evidence(path)
    assert first.status == "pass" and first.report_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    second = validate_runtime_admission_evidence(path)
    assert second.status == "pass" and second.payload_sha256 == first.payload_sha256
    assert second.report_sha256 != first.report_sha256
    mapping = first.as_dict()
    cast(list[str], mapping["errors"]).append("caller change")
    assert first.errors == () and first.as_dict()["errors"] == []


@pytest.mark.parametrize("json_out", [False, True])
@pytest.mark.parametrize("passing", [False, True])
def test_native_and_registered_runtime_reader(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], json_out: bool, passing: bool
) -> None:
    """Native CLI and actual root validate command invoke the same reader while unrelated gates are explicitly scoped out."""
    path = _copy_report(DEFAULT_REPORT, tmp_path / "runtime.json")
    if not passing:
        path.write_text("{}", encoding="utf-8")
    native_args = ["--report", str(path)] + (["--json-out"] if json_out else [])
    assert main(native_args) == (0 if passing else 1)
    native = capsys.readouterr().out
    args = [
        "validate",
        "--runtime-admission-report",
        str(path),
        "--no-data-manifests",
        "--no-jax-gk-parity",
        "--no-physics-traceability",
        "--no-multi-shot-campaign-evidence",
        "--no-native-formal-certificate",
    ]
    result = subprocess.run(
        [sys.executable, "-m", "scpn_control.cli", *args, *(["--json-out"] if json_out else [])],
        cwd=DEFAULT_REPORT.parents[2],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == (0 if passing else 1), result.stdout + result.stderr
    if json_out:
        assert json.loads(result.stdout)["runtime_admission"] == json.loads(native)
    else:
        expected = f"Runtime admission evidence: {'pass' if passing else 'fail'}"
        assert expected in native and expected in result.stdout
        if not passing:
            assert (
                "ERROR runtime_admission.schema_version" in native
                and "ERROR runtime_admission: runtime_admission.schema_version" in result.stderr
            )


@pytest.mark.parametrize("passing", [False, True])
def test_standalone_stdlib_script_from_other_directory(tmp_path: Path, passing: bool) -> None:
    """Actual -S execution with empty PYTHONPATH reads caller-relative reports and emits admission JSON without dependencies."""
    path = _copy_report(DEFAULT_REPORT, tmp_path / "runtime.json")
    if not passing:
        path.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-S", str(Path(runtime_module.__file__)), "--report", path.name, "--json-out"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == (0 if passing else 1), result.stderr
    assert not result.stderr and json.loads(result.stdout)["status"] == ("pass" if passing else "fail")


def test_actual_admission_probe_producer_is_readable(tmp_path: Path) -> None:
    """Read a fresh two-sample actual host probe report without turning its local timings into a production claim."""
    root = DEFAULT_REPORT.parents[2]
    path = tmp_path / "fresh-probe.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks/bench_runtime_admission.py"),
            "--iterations",
            "2",
            "--warmup",
            "0",
            "--core-snn",
            "0",
            "--core-z3",
            "1",
            "--core-net",
            "2",
            "--core-hb",
            "3",
            "--json-out",
            str(path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    emitted = json.loads(result.stdout)
    actual = validate_runtime_admission_evidence(path)
    assert actual.status == "pass", actual.errors
    assert actual.samples == 2 and actual.benchmark_evidence_class == "local_regression"
    assert actual.production_claim_allowed is False
    assert actual.admission_status == emitted["last_admission_status"]
    assert actual.admission_error_count == len(emitted["last_admission_errors"])
    assert actual.payload_sha256 == emitted["payload_sha256"]


def test_native_example_and_parser_usage_contract() -> None:
    """Owning invalid-object example exercises the public reader; parser unknown options retain exit2."""
    example = doctest.testmod(runtime_module, raise_on_error=True)
    assert example.failed == 0 and example.attempted == 3
    with pytest.raises(SystemExit) as exc:
        main(["--unknown-option"])
    assert exc.value.code == 2
