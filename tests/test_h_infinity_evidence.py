# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — DGKF report domains, source binding and read-only CLI regressions.

"""Exercise reports from the real DGKF factory without replacing any provider."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, cast

import pytest

from validation import h_infinity_evidence as evidence
from validation import validate_h_infinity_control as h

# Windows reports a directory opened as a file as a permission error.
DIRECTORY_AS_FILE_ERROR: type[OSError] = PermissionError if os.name == "nt" else IsADirectoryError


@pytest.fixture(scope="module")
def result() -> h.HInfinityValidationResult:
    """Run the actual controller and fixed 20002-frequency sweep once."""
    return h.validate_h_infinity_control()


@pytest.fixture(scope="module")
def payload(result: h.HInfinityValidationResult) -> dict[str, Any]:
    """Observe real source hashes and HEAD for one sealed passing report."""
    return h.build_evidence(result, generated_at="2026-10-03T10:00:00Z")


def reseal(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply the historical UTF-8 seal independently of the production decoder."""
    payload.pop("payload_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def inspect(payload: Mapping[str, Any], observations: Mapping[str, Any]) -> bool:
    """Call the public pure inspector with separately captured actual observations."""
    return evidence.inspect_evidence_payload(
        payload,
        expected_sources=observations["runtime_source_sha256"],
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("result", "gamma"), float("nan"), "finite"),
        (("result", "gamma"), float("inf"), "finite"),
        (("result", "gamma"), True, "finite number"),
        (("result", "gamma"), "5", "finite number"),
        (("result", "gamma"), 10**400, "finite"),
        (("result", "gamma"), 0, "positive"),
        (("result", "normalization_max_residual"), -1, "non-negative"),
        (("result", "normalization_max_residual"), 1, "reported metrics"),
        (("result", "riccati_x_relative_residual"), 1, "reported metrics"),
        (("result", "riccati_y_relative_residual"), 1, "reported metrics"),
        (("result", "controller_formula_relative_error"), 1, "reported metrics"),
        (("result", "spectral_feasibility_margin"), 0, "reported metrics"),
        (("result", "dominant_closed_loop_real_part"), 0, "reported metrics"),
        (("result", "frequency_sweep_peak_over_gamma"), 1, "peak/gamma"),
        (("result", "frequency_samples"), True, "integer 20002"),
        (("result", "frequency_samples"), 20002.0, "integer 20002"),
        (("result", "frequency_samples"), 2, "integer 20002"),
        (("result", "passed"), 1, "must be a boolean"),
        (("result", "passed"), False, "reported metrics"),
        (("result",), [], "declared fields"),
        (("claim_boundary", "production_admission"), True, "remain False"),
        (("claim_boundary", "production_admission"), 0, "remain False"),
        (("claim_boundary", "public_claim_allowed"), True, "remain False"),
        (("claim_boundary", "scientific_admission"), False, "bounded result"),
        (("claim_boundary", "scientific_admission"), 1, "must be a boolean"),
        (("claim_boundary", "model"), "facility", "declared model"),
        (("claim_boundary", "excluded"), [], "exclusions"),
        (("claim_boundary", "frequency_sweep_classification"), "exact proof", "sweep classification"),
        (("claim_boundary",), {}, "declared fields"),
        (("precision",), "float32", "precision and reference"),
        (("reference", "doi"), "wrong", "precision and reference"),
        (("generated_at",), None, "UTC timestamp"),
        (("generated_at",), "", "UTC timestamp"),
        (("generated_at",), "2026-10-03T10:00:00", "UTC timestamp"),
        (("generated_at",), "2026-10-03T10:00:00+01:00", "UTC timestamp"),
        (("source_commit",), "wrong", "Git object hex digest"),
        (("runtime_source_sha256",), {}, "exact owner set"),
        (("runtime_source_sha256", "validation/h_infinity_evidence.py"), "0" * 64, "digest mismatch"),
        (("runtime_source_sha256", "validation/h_infinity_evidence.py"), None, "hex digests"),
    ],
)
def test_resealed_invalid_report(payload: dict[str, Any], path: tuple[str, ...], value: object, message: str) -> None:
    """A matching seal cannot admit malformed metrics, sources or stronger claims."""
    changed = copy.deepcopy(payload)
    parent: Any = changed
    for name in path[:-1]:
        parent = parent[name]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        inspect(reseal(changed), payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("normalization_max_residual", 2e-12),
        ("riccati_x_relative_residual", 2e-8),
        ("riccati_y_relative_residual", 2e-8),
        ("controller_formula_relative_error", 2e-12),
        ("spectral_feasibility_margin", 0),
        ("dominant_closed_loop_real_part", 0),
    ],
)
def test_coherent_failed_metrics_are_not_admitted(
    result: h.HInfinityValidationResult, field: str, value: float
) -> None:
    """Consistent failing results serialize with local admission False but fail the gate."""
    updates: dict[str, Any] = {field: value, "passed": False}
    failed = replace(result, **updates)
    report = h.build_evidence(failed)
    assert inspect(report, report) is False
    assert report["claim_boundary"]["scientific_admission"] is False
    with pytest.raises(ValueError, match="result is not passing"):
        h.validate_evidence_payload(report)


def test_coherent_failed_sampled_gain(result: h.HInfinityValidationResult) -> None:
    """A sampled gain at gamma is a coherent strict-bound failure."""
    failed = replace(result, frequency_sweep_peak=result.gamma, frequency_sweep_peak_over_gamma=1, passed=False)
    report = h.build_evidence(failed)
    assert inspect(report, report) is False
    with pytest.raises(ValueError, match="result is not passing"):
        h.validate_evidence_payload(report)


def test_header_and_observation_refusals(payload: dict[str, Any]) -> None:
    """Unsupported roots, seals and invalid observation sets receive authored refusals."""
    roots: tuple[object, ...] = ([], {"schema_version": "unknown"})
    for root in roots:
        with pytest.raises(ValueError, match="schema_version"):
            inspect(cast(Mapping[str, Any], root), payload)
    for seal in (None, "short", "X" * 64):
        with pytest.raises(ValueError, match="hex digest"):
            inspect(dict(payload, payload_sha256=seal), payload)
    with pytest.raises(ValueError, match="does not match"):
        inspect(dict(payload, generated_at="changed"), payload)
    with pytest.raises(ValueError, match="declared fields"):
        inspect(reseal(dict(payload, extra=1)), payload)
    with pytest.raises(ValueError, match="JSON-serializable"):
        inspect(dict(payload, extra=object()), payload)
    with pytest.raises(ValueError, match="expected source observations"):
        evidence.inspect_evidence_payload(payload, expected_sources={})
    wrong = dict(payload["runtime_source_sha256"])
    wrong["validation/h_infinity_evidence.py"] = "wrong"
    with pytest.raises(ValueError, match="hex digests"):
        evidence.inspect_evidence_payload(payload, expected_sources=wrong)
    with pytest.raises(ValueError, match="source_commit"):
        evidence.inspect_evidence_payload(
            reseal(dict(payload, source_commit=None)), expected_sources=payload["runtime_source_sha256"]
        )


def test_sha256_git_observation_format(payload: dict[str, Any]) -> None:
    """The pure inspector supports a caller-observed SHA-256 Git object identifier."""
    changed = reseal(dict(payload, source_commit="a" * 64))
    assert evidence.inspect_evidence_payload(changed, expected_sources=payload["runtime_source_sha256"]) is True


def test_builder_checks_timestamp_and_consistency(result: h.HInfinityValidationResult) -> None:
    """Serialization refuses invalid timestamps and inconsistent frozen result flags."""
    for stamp in ("", "not-a-timestamp", "2026-10-03T10:00:00"):
        with pytest.raises(ValueError, match="UTC timestamp"):
            h.build_evidence(result, generated_at=stamp)
    with pytest.raises(ValueError, match="reported metrics"):
        h.build_evidence(replace(result, passed=False))
    report = h.build_evidence(result)
    report["result"]["gamma"] = 99
    assert result.gamma != 99


@pytest.mark.parametrize("contents", [b"[]", b"{", b"\xff", b'{"x":1,"x":2}', b'{"x":{"y":1,"y":2}}'])
def test_read_report_refuses_ambiguous_or_malformed_json(tmp_path: Path, contents: bytes) -> None:
    """Actual malformed files refuse without normalization or replacement."""
    path = tmp_path / "invalid.json"
    path.write_bytes(contents)
    with pytest.raises(ValueError):
        evidence.read_report(path)
    assert path.read_bytes() == contents


def test_writer_checked_roundtrip_and_partial_pair(payload: dict[str, Any], tmp_path: Path) -> None:
    """Checked output creates parents, replaces reports and retains direct partial IO."""
    path = tmp_path / "nested" / "report.json"
    markdown = tmp_path / "nested" / "report.md"
    h.write_reports(payload, path, markdown)
    assert evidence.read_report(path) == payload
    assert "Production admission is\n`false`" in markdown.read_text()
    before = (path.read_bytes(), markdown.read_bytes())
    with pytest.raises(ValueError, match="does not match"):
        h.write_reports(dict(payload, precision="float32"), path, markdown)
    assert before == (path.read_bytes(), markdown.read_bytes())
    partial = tmp_path / "partial.json"
    directory = tmp_path / "markdown-directory"
    directory.mkdir()
    with pytest.raises(DIRECTORY_AS_FILE_ERROR):
        h.write_reports(payload, partial, directory)
    assert evidence.read_report(partial) == payload and directory.is_dir()


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_writer_refuses_output_aliases(payload: dict[str, Any], tmp_path: Path, alias: str) -> None:
    """Identical names and real filesystem aliases cannot destroy the JSON output."""
    path = tmp_path / "report.json"
    path.write_text("original")
    other = tmp_path / "report.md"
    if alias == "same":
        other = path
    elif alias == "symlink":
        other.symlink_to(path)
    else:
        os.link(path, other)
    with pytest.raises(ValueError, match="aliases"):
        h.write_reports(payload, path, other)
    assert path.read_text() == "original" and other.read_text() == "original"


def test_writer_protects_declared_runtime_sources(payload: dict[str, Any], tmp_path: Path) -> None:
    """Neither output may overwrite a source whose hash the report observes."""
    source = h.ROOT / "validation/validate_h_infinity_control.py"
    original = source.read_bytes()
    for json_path, markdown in ((source, tmp_path / "report.md"), (tmp_path / "report.json", source)):
        with pytest.raises(ValueError, match="aliases"):
            h.write_reports(payload, json_path, markdown)
    assert source.read_bytes() == original and not (tmp_path / "report.json").exists()


def test_read_only_cli_checks_without_overwrite(
    payload: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The actual check mode accepts valid bytes and refuses resealed stronger claims."""
    path = tmp_path / "existing.json"
    path.write_text(json.dumps(payload))
    original = path.read_bytes()
    assert h.main(["--check-report", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == payload
    assert path.read_bytes() == original and list(tmp_path.iterdir()) == [path]
    bad = copy.deepcopy(payload)
    bad["claim_boundary"]["production_admission"] = True
    path.write_text(json.dumps(reseal(bad)))
    original = path.read_bytes()
    assert h.main(["--check-report", str(path)]) == 1
    captured = capsys.readouterr()
    assert not captured.out and "remain False" in captured.err
    assert path.read_bytes() == original
    assert h.main(["--check-report", str(tmp_path / "absent.json")]) == 1
    assert "H-infinity validation refused" in capsys.readouterr().err
