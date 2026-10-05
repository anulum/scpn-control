# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Multi-shot campaign evidence validation tests.
"""Exercise actual persisted campaign reports and negative copies via public API/CLI.

The historical Python/PyO3 and Rust reports remain untouched. Resealed copies
test metadata admission only: no invented solver, extension, authenticated
digest chain, physical measurement or controller qualification is supplied.
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
from typing import cast

import pytest

import validation.validate_multi_shot_campaign_evidence as campaign_module
from validation.validate_multi_shot_campaign_evidence import (
    DEFAULT_PYTHON_REPORT,
    DEFAULT_RUST_REPORT,
    main,
    validate_multi_shot_campaign_evidence,
)


def _canonical_payload_digest(payload: dict[str, object]) -> str:
    """Reseal copied declarations using the actual canonical encoding, not authentication."""
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _copy_report(source: Path, destination: Path) -> Path:
    """Keep test mutations within scratch copies of actual historical reports."""
    shutil.copyfile(source, destination)
    return destination


def _load_report(path: Path) -> dict[str, object]:
    """Load a real JSON object with typed test metadata and no production loader mock."""
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _write_report(path: Path, payload: dict[str, object], *, refresh_digest: bool = True) -> None:
    """Write negative copied metadata, optionally preserving a deliberately stale seal."""
    if refresh_digest:
        payload["payload_sha256"] = _canonical_payload_digest(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_multi_shot_campaign_evidence_admits_repository_reports() -> None:
    """Repository Python/PyO3 and Rust reports admit the multi-shot campaign evidence gate."""
    result = validate_multi_shot_campaign_evidence()

    assert result.status == "pass"
    assert result.errors == ()
    assert result.admitted_surfaces == ("python", "pyo3", "rust")
    assert result.pyo3_status == "ok"
    assert result.python_report_sha256 is not None
    assert result.rust_report_sha256 is not None


def test_multi_shot_campaign_evidence_rejects_missing_pyo3_surface(tmp_path: Path) -> None:
    """The release gate fails closed when the Python report lacks PyO3 campaign evidence."""
    python_report = _copy_report(DEFAULT_PYTHON_REPORT, tmp_path / "python.json")
    rust_report = _copy_report(DEFAULT_RUST_REPORT, tmp_path / "rust.json")
    payload = _load_report(python_report)
    payload["pyo3_status"] = "unavailable"
    payload["pyo3_result"] = None
    _write_report(python_report, payload)

    result = validate_multi_shot_campaign_evidence(python_report, rust_report)

    assert result.status == "fail"
    assert "multi_shot_campaign.pyo3_status must be 'ok'" in result.errors
    assert "multi_shot_campaign.pyo3_result must be an object" in result.errors


def test_multi_shot_campaign_evidence_rejects_under_counted_digest_chain(tmp_path: Path) -> None:
    """Each admitted surface must preserve the configured number of pulsed-MPC decision digests."""
    python_report = _copy_report(DEFAULT_PYTHON_REPORT, tmp_path / "python.json")
    rust_report = _copy_report(DEFAULT_RUST_REPORT, tmp_path / "rust.json")

    result = validate_multi_shot_campaign_evidence(python_report, rust_report, minimum_digest_count=3)

    assert result.status == "fail"
    assert any("last_pulsed_mpc_admission_digest_count must be at least 3" in error for error in result.errors)


def test_multi_shot_campaign_evidence_rejects_rust_report_without_context(tmp_path: Path) -> None:
    """Rust campaign evidence must record CPU affinity and load context before release admission."""
    python_report = _copy_report(DEFAULT_PYTHON_REPORT, tmp_path / "python.json")
    rust_report = _copy_report(DEFAULT_RUST_REPORT, tmp_path / "rust.json")
    payload = _load_report(rust_report)
    payload["context"] = {}
    _write_report(rust_report, payload)

    result = validate_multi_shot_campaign_evidence(python_report, rust_report)

    assert result.status == "fail"
    assert "multi_shot_campaign.rust.context.cpu_affinity must be recorded" in result.errors
    assert "multi_shot_campaign.rust.context must record loadavg_start and loadavg_end" in result.errors


def test_multi_shot_campaign_evidence_rejects_python_payload_tampering(tmp_path: Path) -> None:
    """Canonical payload digests prevent silent mutation of persisted benchmark claims."""
    python_report = _copy_report(DEFAULT_PYTHON_REPORT, tmp_path / "python.json")
    rust_report = _copy_report(DEFAULT_RUST_REPORT, tmp_path / "rust.json")
    payload = _load_report(python_report)
    steps = payload["steps"]
    assert isinstance(steps, int)
    payload["steps"] = steps + 1
    _write_report(python_report, payload, refresh_digest=False)

    result = validate_multi_shot_campaign_evidence(python_report, rust_report)

    assert result.status == "fail"
    assert "multi_shot_campaign.python.payload_sha256 does not match canonical payload" in result.errors


def _mutated_pair(tmp_path: Path, surface: str, keys: tuple[str, ...], value: object) -> tuple[Path, Path]:
    """Copy a real report pair and reseal a single declared metadata mutation."""
    python = _copy_report(DEFAULT_PYTHON_REPORT, tmp_path / "python.json")
    rust = _copy_report(DEFAULT_RUST_REPORT, tmp_path / "rust.json")
    target = python if surface == "python" else rust
    payload = _load_report(target)
    parent = payload
    for key in keys[:-1]:
        child = parent[key]
        assert isinstance(child, dict)
        parent = cast(dict[str, object], child)
    parent[keys[-1]] = value
    _write_report(target, payload)
    return python, rust


@pytest.mark.parametrize("surface", ["python", "rust"])
@pytest.mark.parametrize(
    "keys,value,fragment",
    [
        (("schema_version",), None, "schema_version"),
        (("evidence_class",), {}, "evidence_class"),
        (("evidence_class",), [], "evidence_class"),
        (("evidence_class",), "unknown", "evidence_class"),
        (("production_claim_allowed",), "true", "production_claim_allowed"),
        (("production_claim_allowed",), True, "local_regression"),
        (("command",), None, "command"),
        (("command",), "unrelated-command", "command"),
        (("payload_sha256",), {}, "payload_sha256"),
        (("context",), [], "context must be an object"),
        (("context", "cpu_affinity"), [], "cpu_affinity"),
        (("context", "cpu_affinity"), "   ", "cpu_affinity"),
        (("context", "loadavg_start"), None, "loadavg_start"),
        (("context", "loadavg_end"), None, "loadavg_start"),
        (("result",), None, "result must be an object"),
        (("result", "last_passed_count"), 0, "last_passed_count"),
        (("result", "last_passed_count"), True, "last_passed_count"),
        (("result", "last_pulsed_mpc_admission_digest_count"), 0, "digest_count"),
        (("result", "last_pulsed_mpc_admission_digest_count"), 1, "digest_count"),
        (("result", "last_pulsed_mpc_admission_digest_count"), 2.0, "digest_count"),
        (("result", "stats"), [], "stats must be an object"),
        (("result", "stats", "samples"), 0, "stats.samples"),
        (("result", "stats", "samples"), True, "stats.samples"),
    ],
)
def test_public_campaign_report_domain_refusals(
    tmp_path: Path, surface: str, keys: tuple[str, ...], value: object, fragment: str
) -> None:
    """Real copied/resealed reports refuse malformed required declarations without a traceback."""
    python, rust = _mutated_pair(tmp_path, surface, keys, value)
    # Keep an intentionally invalid digest declaration rather than overwriting it with the test seal.
    if keys == ("payload_sha256",):
        target = python if surface == "python" else rust
        payload = _load_report(target)
        payload["payload_sha256"] = value
        _write_report(target, payload, refresh_digest=False)
    result = validate_multi_shot_campaign_evidence(python, rust)
    assert result.status == "fail" and result.admitted_surfaces == ()
    assert any(fragment in error for error in result.errors), result.errors
    assert result.python_report_sha256 is not None and result.rust_report_sha256 is not None


@pytest.mark.parametrize(
    "keys,value,fragment",
    [
        (("pyo3_status",), {}, "pyo3_status"),
        (("pyo3_status",), None, "pyo3_status"),
        (("pyo3_status",), "unavailable", "pyo3_status"),
        (("pyo3_result",), None, "pyo3_result must be an object"),
        (("pyo3_result", "last_passed_count"), -1, "last_passed_count"),
        (("pyo3_result", "last_pulsed_mpc_admission_digest_count"), False, "digest_count"),
        (("pyo3_result", "stats"), None, "stats must be an object"),
        (("pyo3_result", "stats", "samples"), 1.0, "stats.samples"),
    ],
)
def test_public_pyo3_declaration_refusals(tmp_path: Path, keys: tuple[str, ...], value: object, fragment: str) -> None:
    """Metadata PyO3 refusals require no invented or mocked extension."""
    python, rust = _mutated_pair(tmp_path, "python", keys, value)
    result = validate_multi_shot_campaign_evidence(python, rust)
    assert result.status == "fail" and any(fragment in e for e in result.errors)
    if keys == ("pyo3_status",):
        assert result.pyo3_status == (value if isinstance(value, str) else None)


@pytest.mark.parametrize("surface", ["python", "rust"])
@pytest.mark.parametrize("digest", [None, 12, "", "a" * 63, "A" * 64, "g" * 64])
def test_digest_spelling_refusals(tmp_path: Path, surface: str, digest: object) -> None:
    """Digest syntax is checked before canonical comparison without coercion."""
    python, rust = _mutated_pair(tmp_path, surface, ("command",), "bench_multi_shot_campaign")
    target = python if surface == "python" else rust
    payload = _load_report(target)
    payload["payload_sha256"] = digest
    _write_report(target, payload, refresh_digest=False)
    result = validate_multi_shot_campaign_evidence(python, rust)
    assert result.status == "fail" and any("SHA-256" in e for e in result.errors)
    declared = result.python_payload_sha256 if surface == "python" else result.rust_payload_sha256
    assert declared == (digest if isinstance(digest, str) else None)


@pytest.mark.parametrize("surface", ["python", "rust"])
@pytest.mark.parametrize(
    "blob",
    [
        b"{",
        b"\xff",
        b"[]",
        b"null",
        b'{"metadata":{"duplicate":1,"duplicate":2}}',
        b'{"extra":NaN}',
        b'{"extra":Infinity}',
        b'{"extra":-Infinity}',
        b'{"extra":1e999}',
        b"[" * 2048 + b"0" + b"]" * 2048,
    ],
)
def test_actual_decoder_refusals(tmp_path: Path, surface: str, blob: bytes) -> None:
    """Actual byte decoding refuses malformed/non-object/nonfinite JSON at any depth."""
    python = _copy_report(DEFAULT_PYTHON_REPORT, tmp_path / "python.json")
    rust = _copy_report(DEFAULT_RUST_REPORT, tmp_path / "rust.json")
    target = python if surface == "python" else rust
    target.write_bytes(blob)
    result = validate_multi_shot_campaign_evidence(python, rust)
    assert result.status == "fail" and result.admitted_surfaces == ()
    assert any(surface + "_report" in e for e in result.errors)
    assert (result.python_report_sha256 if surface == "python" else result.rust_report_sha256) is None


@pytest.mark.parametrize("surface", ["python", "rust"])
@pytest.mark.parametrize("missing", [False, True])
def test_actual_read_refusals(tmp_path: Path, surface: str, missing: bool) -> None:
    """Existing directories and missing actual files become structured report-read findings."""
    target = tmp_path / "absent" if missing else tmp_path
    python = target if surface == "python" else DEFAULT_PYTHON_REPORT
    rust = target if surface == "rust" else DEFAULT_RUST_REPORT
    result = validate_multi_shot_campaign_evidence(python, rust)
    assert result.status == "fail" and any(surface + "_report" in e for e in result.errors)


@pytest.mark.parametrize("surface", ["python", "rust", "both"])
def test_empty_decoded_reports_cannot_admit_surfaces(tmp_path: Path, surface: str) -> None:
    """Exact decoded empty-object hashes remain visible while all required declarations fail."""
    python = _copy_report(DEFAULT_PYTHON_REPORT, tmp_path / "python.json")
    rust = _copy_report(DEFAULT_RUST_REPORT, tmp_path / "rust.json")
    for name, path in [("python", python), ("rust", rust)]:
        if surface in (name, "both"):
            path.write_bytes(b"{}")
    result = validate_multi_shot_campaign_evidence(python, rust)
    assert result.status == "fail" and result.admitted_surfaces == ()
    for name, path in [("python", python), ("rust", rust)]:
        assert getattr(result, name + "_report_sha256") == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("minimum", [0, -1, True, False, None, 2.0, "2", {}])
def test_invalid_public_minimum_is_structured_failure(minimum: object) -> None:
    """Even an ill-typed runtime API argument fails without comparing/coercing arbitrary values."""
    result = validate_multi_shot_campaign_evidence(minimum_digest_count=cast(int, minimum))
    assert result.status == "fail" and result.minimum_digest_count == 1
    assert "minimum_digest_count must be a positive integer" in result.errors


@pytest.mark.parametrize("surface", ["python", "rust"])
def test_stale_otherwise_well_formed_seals_refused(tmp_path: Path, surface: str) -> None:
    """Actual metadata edits with a preserved valid-looking seal fail canonical comparison."""
    python = _copy_report(DEFAULT_PYTHON_REPORT, tmp_path / "python.json")
    rust = _copy_report(DEFAULT_RUST_REPORT, tmp_path / "rust.json")
    target = python if surface == "python" else rust
    payload = _load_report(target)
    payload["extra"] = {"label": "changed"}
    _write_report(target, payload, refresh_digest=False)
    result = validate_multi_shot_campaign_evidence(python, rust)
    assert result.status == "fail" and any("does not match canonical payload" in e for e in result.errors)


@pytest.mark.parametrize("surface", ["python", "rust"])
@pytest.mark.parametrize("flag", [False, True])
def test_production_flag_remains_declared_even_on_other_failure(tmp_path: Path, surface: str, flag: bool) -> None:
    """Copied production labels show flag preservation, never measured or qualified production evidence."""
    python, rust = _mutated_pair(tmp_path, surface, ("evidence_class",), "production_benchmark")
    target = python if surface == "python" else rust
    payload = _load_report(target)
    payload["production_claim_allowed"] = flag
    payload["schema_version"] = "wrong"
    _write_report(target, payload)
    result = validate_multi_shot_campaign_evidence(python, rust)
    assert result.status == "fail" and result.production_claim_allowed is flag


def test_bounded_declaration_nonchecks_and_exact_report_hashes(tmp_path: Path) -> None:
    """Resealed extras/context contents are unchecked; admitted declarations grant no physical proof."""
    python, rust = _mutated_pair(tmp_path, "python", ("context", "cpu_affinity"), [None])
    payload = _load_report(python)
    context = payload["context"]
    assert isinstance(context, dict)
    context["loadavg_start"] = "not-measured"
    payload["unrelated"] = {"finite": 0.125}
    _write_report(python, payload)
    result = validate_multi_shot_campaign_evidence(python, rust)
    assert result.status == "pass" and result.production_claim_allowed is False
    assert result.python_report_sha256 == hashlib.sha256(python.read_bytes()).hexdigest()
    assert result.rust_report_sha256 == hashlib.sha256(rust.read_bytes()).hexdigest()
    assert result.minimum_digest_count == 2
    mapping = result.as_dict()
    errors = mapping["errors"]
    surfaces = mapping["admitted_surfaces"]
    assert isinstance(errors, list) and isinstance(surfaces, list)
    errors.append("caller mutation")
    surfaces.clear()
    assert result.errors == () and result.admitted_surfaces == ("python", "pyo3", "rust")


def test_native_public_example() -> None:
    """Execute the defining public reader's documented empty-report refusal."""
    attempted = doctest.testmod(campaign_module)
    assert attempted.attempted == 2 and attempted.failed == 0


@pytest.mark.parametrize("json_out", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_actual_standalone_from_other_cwd(tmp_path: Path, json_out: bool, fail: bool) -> None:
    """Real -S standalone entry point admits historical metadata or refuses empty reports without dependencies."""
    script = Path(campaign_module.__file__).resolve()
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"{}")
    args = [sys.executable, "-S", str(script)]
    if fail:
        args += ["--python-report", "empty.json", "--rust-report", "empty.json"]
    if json_out:
        args += ["--json-out"]
    result = subprocess.run(
        args,
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == int(fail) and result.stderr == ""
    if json_out:
        payload = json.loads(result.stdout)
        assert payload["status"] == ("fail" if fail else "pass")
        assert payload["admitted_surfaces"] == ([] if fail else ["python", "pyo3", "rust"])
    else:
        assert "Multi-shot campaign evidence: " + ("fail" if fail else "pass") in result.stdout
        assert ("ERROR " in result.stdout) is fail


def test_public_main_json_and_argument_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Exercise public main explicitly plus real argparse rejection with no mocked dispatcher."""
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"{}")
    assert main(["--python-report", str(empty), "--rust-report", str(empty), "--json-out"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "fail"
    with pytest.raises(SystemExit) as error:
        main(["--unknown"])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("fail", [False, True])
def test_public_main_text_result(tmp_path: Path, capsys: pytest.CaptureFixture[str], fail: bool) -> None:
    """Public text main reaches real historical or empty report paths with ordered findings."""
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"{}")
    args = ["--python-report", str(empty), "--rust-report", str(empty)] if fail else []
    assert main(args) == int(fail)
    output = capsys.readouterr()
    assert output.err == "" and ("ERROR " in output.out) is fail
    assert "Multi-shot campaign evidence: " + ("fail" if fail else "pass") in output.out


@pytest.mark.parametrize("json_out", [False, True])
def test_real_root_validate_refuses_empty_campaign_pair(tmp_path: Path, json_out: bool) -> None:
    """Registered real root CLI refuses empty reports and propagates campaign findings."""
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"{}")
    repo = Path(campaign_module.__file__).resolve().parents[1]
    args = [
        sys.executable,
        "-m",
        "scpn_control.cli",
        "validate",
        "--no-data-manifests",
        "--no-jax-gk-parity",
        "--no-physics-traceability",
        "--no-runtime-admission-evidence",
        "--no-native-formal-certificate",
        "--multi-shot-campaign-python-report",
        str(empty),
        "--multi-shot-campaign-rust-report",
        str(empty),
    ]
    if json_out:
        args += ["--json-out"]
    result = subprocess.run(
        args,
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(repo / "src"), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1
    if json_out:
        payload = json.loads(result.stdout)
        assert payload["status"] == "fail" and payload["multi_shot_campaign"]["admitted_surfaces"] == []
    else:
        assert "Multi-shot campaign evidence: fail" in result.stdout
        assert "ERROR multi_shot_campaign:" in result.stderr


def test_real_python_producer_declares_available_surfaces(tmp_path: Path) -> None:
    """Run the actual tiny producer; missing optional PyO3 stays a refusal, never a fabricated surface."""
    repo = Path(campaign_module.__file__).resolve().parents[1]
    path = tmp_path / "actual-python.json"
    args = [
        sys.executable,
        str(repo / "benchmarks/bench_multi_shot_campaign.py"),
        "--steps",
        "2",
        "--warmup",
        "0",
        "--json-out",
        str(path),
    ]
    process = subprocess.run(
        args,
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(repo / "src"), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    payload = _load_report(path)
    assert payload["production_claim_allowed"] is False
    result_section = payload["result"]
    assert isinstance(result_section, dict)
    assert result_section["last_passed_count"] == 2
    assert result_section["last_pulsed_mpc_admission_digest_count"] == 2
    admission = validate_multi_shot_campaign_evidence(path, DEFAULT_RUST_REPORT)
    assert admission.python_report_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert admission.production_claim_allowed is False
    if payload["pyo3_status"] == "ok":
        assert admission.status == "pass"
    else:
        assert admission.status == "fail" and admission.admitted_surfaces == ()
        assert "multi_shot_campaign.pyo3_status must be 'ok'" in admission.errors
