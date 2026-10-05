# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — actual persisted benchmark metadata CLI contracts.
"""Exercise actual copied persisted reports through the owning validator and stdlib CLI."""

from __future__ import annotations

import doctest
import hashlib
import html
import json
import os
import pydoc
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from validation import validate_benchmark_regression_gates as gates


@pytest.fixture
def actual_bundle(tmp_path: Path) -> Path:
    """Copy actual canonical manifest/report bytes into a caller-owned external artifact root."""
    payload = json.loads(gates.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    for entry in payload["entries"]:
        target = tmp_path / entry["report_uri"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gates.ROOT / entry["report_uri"], target)
    manifest = tmp_path / "gates.json"
    shutil.copy2(gates.DEFAULT_MANIFEST, manifest)
    return manifest


def _payload(path: Path) -> dict[str, Any]:
    """Decode a caller-owned copy for deliberate metadata corruption without touching original evidence."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    """Write only the selected copied JSON carrier; no benchmark measurement is generated."""
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _rebind_report(manifest: Path, uri: str) -> None:
    """Bind all copied entries for one deliberately mutated report to its new raw file bytes."""
    payload = _payload(manifest)
    digest = hashlib.sha256((manifest.parent / uri).read_bytes()).hexdigest()
    for entry in payload["entries"]:
        if entry["report_uri"] == uri:
            entry["report_sha256"] = digest
    _write(manifest, payload)


def _cli(manifest: Path, *, relative: bool = False) -> subprocess.CompletedProcess[str]:
    """Run the actual stdlib-only source CLI from the copied artifact root with no site dependencies."""
    return subprocess.run(
        [sys.executable, "-S", str(Path(gates.__file__)), manifest.name if relative else str(manifest)],
        cwd=manifest.parent,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_actual_relative_cli_admits_copied_historical_metadata(actual_bundle: Path) -> None:
    """A relative external manifest admits exact copied historical metadata without new timing or writes."""
    before = {str(p): p.read_bytes() for p in actual_bundle.parent.rglob("*.json")}
    result = _cli(actual_bundle, relative=True)
    payload = json.loads(result.stdout)
    assert result.returncode == 0 and not result.stderr
    assert payload["status"] == "pass" and len(payload["admitted_gates"]) == 4 and not payload["errors"]
    assert payload["manifest_sha256"] == hashlib.sha256(actual_bundle.read_bytes()).hexdigest()
    assert {str(p): p.read_bytes() for p in actual_bundle.parent.rglob("*.json")} == before


@pytest.mark.parametrize(
    "kind", ["missing", "directory", "utf8", "syntax", "array", "duplicate", "nan", "overflow", "integer-limit"]
)
def test_actual_manifest_refusals_are_structured(actual_bundle: Path, kind: str) -> None:
    """Actual source CLI refuses unreadable or malformed copied manifests with safe JSON and byte custody."""
    selected = actual_bundle
    text = selected.read_text(encoding="utf-8")
    if kind == "missing":
        selected = selected.with_name("missing.json")
    elif kind == "directory":
        selected = selected.parent
    else:
        malformed = {
            "utf8": b"\xff",
            "syntax": text.encode() + b"{",
            "array": ("[" + text + "]").encode(),
            "duplicate": text.replace("{", '{"schema_version": "wrong",', 1).encode(),
            "nan": text.replace("{", '{"ignored": NaN,', 1).encode(),
            "overflow": text.replace("{", '{"ignored": 1e400,', 1).encode(),
            "integer-limit": text.replace("{", '{"ignored": ' + "9" * 5000 + ",", 1).encode(),
        }
        selected.write_bytes(malformed[kind])
    before = selected.read_bytes() if selected.is_file() else None
    result = _cli(selected)
    payload = json.loads(result.stdout)
    assert payload == gates.validate_benchmark_regression_gates(selected).as_dict()
    assert result.returncode == 1 and payload["status"] == "fail" and payload["admitted_gates"] == []
    assert payload["errors"] and "Traceback" not in result.stderr
    assert payload["manifest_sha256"] == (hashlib.sha256(before).hexdigest() if before is not None else "")
    assert selected.read_bytes() == before if before is not None else not selected.is_file()


@pytest.mark.parametrize("uri", ["./report.json", "folder//report.json", "folder/./report.json", "report.json/", "."])
def test_actual_cli_refuses_normalized_uri_components(actual_bundle: Path, uri: str) -> None:
    """Literal dot/empty URI components refuse rather than being normalized into another report path."""
    payload = _payload(actual_bundle)
    payload["entries"][0]["report_uri"] = uri
    _write(actual_bundle, payload)
    result = _cli(actual_bundle)
    assert result.returncode == 1 and "report_uri is not repository-relative" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("kind", ["directory", "missing", "symlink-escape", "symlink-loop"])
def test_actual_report_filesystem_refusals(actual_bundle: Path, kind: str) -> None:
    """Actual copied reports and real symlinks refuse unreadability/containment failures safely."""
    payload = _payload(actual_bundle)
    uri = payload["entries"][0]["report_uri"]
    report = actual_bundle.parent / uri
    if kind in {"directory", "missing"}:
        report.rename(report.with_suffix(".preserved"))
        if kind == "directory":
            report.mkdir()
    else:
        link = actual_bundle.parent / "linked"
        link.symlink_to(gates.ROOT if kind == "symlink-escape" else "linked", target_is_directory=True)
        payload["entries"][0]["report_uri"] = "linked/" + uri
        _write(actual_bundle, payload)
    result = _cli(actual_bundle)
    response = json.loads(result.stdout)
    assert response == gates.validate_benchmark_regression_gates(actual_bundle).as_dict()
    assert result.returncode == 1 and response["status"] == "fail" and response["admitted_gates"] == []
    assert response["errors"] and "Traceback" not in result.stderr


@pytest.mark.parametrize("kind", ["utf8", "duplicate", "nan", "overflow", "null-self-digest"])
def test_actual_report_decode_and_self_digest_refusals(actual_bundle: Path, kind: str) -> None:
    """A matching raw checksum does not admit malformed, ambiguous or null-self-digest report metadata."""
    payload = _payload(actual_bundle)
    uri = payload["entries"][0]["report_uri"]
    report = actual_bundle.parent / uri
    text = report.read_text(encoding="utf-8")
    corrupt = {
        "utf8": b"\xff",
        "duplicate": text.replace("{", '{"claim_status": "wrong",', 1).encode(),
        "nan": text.replace("{", '{"ignored": NaN,', 1).encode(),
        "overflow": text.replace("{", '{"ignored": 1e400,', 1).encode(),
        "null-self-digest": text.replace("{", '{"payload_sha256": null,', 1).encode(),
    }
    report.write_bytes(corrupt[kind])
    _rebind_report(actual_bundle, uri)
    result = _cli(actual_bundle)
    response = json.loads(result.stdout)
    assert response == gates.validate_benchmark_regression_gates(actual_bundle).as_dict()
    assert result.returncode == 1 and response["status"] == "fail" and response["errors"]
    assert "Traceback" not in result.stderr


def test_actual_huge_numeric_observation_refuses_without_overflow_trace(actual_bundle: Path) -> None:
    """A decoded integer beyond float range refuses metadata admission without a Python overflow error."""
    payload = _payload(actual_bundle)
    payload["entries"][0]["observed"] = 10**400
    _write(actual_bundle, payload)
    result = _cli(actual_bundle)
    assert json.loads(result.stdout) == gates.validate_benchmark_regression_gates(actual_bundle).as_dict()
    assert result.returncode == 1 and "observed must be finite" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("field", ["payload_sha256", "report_payload_sha256"])
def test_actual_report_optional_canonical_digest_admission(actual_bundle: Path, field: str) -> None:
    """Declared canonical self-digests can bind unchanged copied metrics without claiming a new benchmark run."""
    payload = _payload(actual_bundle)
    uri = payload["entries"][0]["report_uri"]
    report = actual_bundle.parent / uri
    body = _payload(report)
    declared = body.pop("payload_sha256")
    assert isinstance(declared, str) and "report_payload_sha256" not in body
    # The two supported names bind the same unsigned original payload.
    body[field] = declared.upper()
    _write(report, body)
    _rebind_report(actual_bundle, uri)
    result = _cli(actual_bundle)
    assert result.returncode == 0 and json.loads(result.stdout)["status"] == "pass" and not result.stderr


def test_actual_unknown_kernel_metadata_is_not_rt_or_date_validation(actual_bundle: Path) -> None:
    """The real unknown kernel label qualifies as metadata, and a nonblank date label is not freshness proof."""
    payload = _payload(actual_bundle)
    uri = payload["entries"][0]["report_uri"]
    report = actual_bundle.parent / uri
    body = _payload(report)
    context = body["target_hardware"]
    assert context["rt_kernel"] == "unknown"
    context.pop("platform")
    unsigned = {key: value for key, value in body.items() if key != "payload_sha256"}
    encoded = json.dumps(unsigned, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    body["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    _write(report, body)
    _rebind_report(actual_bundle, uri)
    payload = _payload(actual_bundle)
    payload["generated_utc"] = "nonblank-unparsed-date"
    _write(actual_bundle, payload)
    result = _cli(actual_bundle)
    assert result.returncode == 0 and json.loads(result.stdout)["status"] == "pass"


def test_actual_public_argv_and_native_html_contract(actual_bundle: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The public argv API preserves process args; native examples/pydoc inspect the real canonical corpus."""
    before = sys.argv.copy()
    assert gates.main([str(actual_bundle)]) == 0 and sys.argv == before
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    examples = doctest.testmod(gates, raise_on_error=True)
    assert examples.attempted == 3 and examples.failed == 0
    artifact = actual_bundle.parent / "persisted_benchmark_gate.html"
    artifact.write_text(pydoc.HTMLDoc().document(gates), encoding="utf-8")
    visible = html.unescape(artifact.read_text(encoding="utf-8")).replace("\N{NO-BREAK SPACE}", " ")
    assert "manifest_path" in visible and "BenchmarkRegressionGateResult" in visible
    assert "abs_tol=1e-9" in visible and "no transactional snapshot" in visible
