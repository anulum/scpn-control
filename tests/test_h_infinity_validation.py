# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — normalized DGKF validation tests.

"""Tests for sealed normalized-DGKF scientific evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from validation.validate_h_infinity_control import (
    H_INFINITY_SCHEMA_VERSION,
    HInfinityValidationResult,
    _canonical_bytes,
    build_evidence,
    main,
    validate_evidence_payload,
    validate_h_infinity_control,
)


@pytest.fixture(scope="module")
def result() -> HInfinityValidationResult:
    """Run the bounded validator once for module-scoped evidence tests."""
    return validate_h_infinity_control()


def test_all_bounded_dgkf_checks_pass(result: HInfinityValidationResult) -> None:
    """Every declared bounded numerical and identity gate passes."""
    assert result.passed is True
    assert result.normalization_max_residual <= 1.0e-12
    assert result.riccati_x_relative_residual <= 1.0e-8
    assert result.riccati_y_relative_residual <= 1.0e-8
    assert result.controller_formula_relative_error <= 1.0e-12
    assert result.spectral_feasibility_margin > 0.0
    assert result.dominant_closed_loop_real_part < 0.0
    assert result.frequency_sweep_peak < result.gamma
    assert result.frequency_samples == 20_002


def test_validation_is_deterministic() -> None:
    """Repeated validation yields bit-identical metric values."""
    first = validate_h_infinity_control()
    second = validate_h_infinity_control()
    assert first == second


def test_evidence_roundtrip_and_claim_boundary(result: HInfinityValidationResult) -> None:
    """A sealed payload preserves the explicit non-production boundary."""
    evidence = build_evidence(result, generated_at="2026-08-30T00:00:00Z")
    assert evidence["schema_version"] == H_INFINITY_SCHEMA_VERSION
    assert validate_evidence_payload(evidence)
    assert evidence["claim_boundary"]["production_admission"] is False
    assert evidence["claim_boundary"]["public_claim_allowed"] is False
    assert "arbitrary sampled-data stability" in evidence["claim_boundary"]["excluded"]


def test_tamper_and_nonpassing_result_are_rejected(result: HInfinityValidationResult) -> None:
    """Schema, payload, and seal mutations fail closed."""
    evidence = build_evidence(result, generated_at="2026-08-30T00:00:00Z")
    evidence["result"]["gamma"] = 0.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)

    evidence = build_evidence(result, generated_at="2026-08-30T00:00:00Z")
    evidence["schema_version"] = "wrong"
    with pytest.raises(ValueError, match="schema_version"):
        validate_evidence_payload(evidence)

    evidence = build_evidence(result, generated_at="2026-08-30T00:00:00Z")
    evidence.pop("payload_sha256")
    with pytest.raises(ValueError, match="payload_sha256 must be"):
        validate_evidence_payload(evidence)


def _reseal(evidence: dict[str, object]) -> None:
    evidence.pop("payload_sha256", None)
    evidence["payload_sha256"] = hashlib.sha256(_canonical_bytes(evidence)).hexdigest()


def test_source_coverage_digest_and_pass_status_fail_closed(result: HInfinityValidationResult) -> None:
    """Owner coverage, source bytes, and aggregate pass state are mandatory."""
    evidence = build_evidence(result, generated_at="2026-08-30T00:00:00Z")
    evidence["runtime_source_sha256"] = {}
    _reseal(evidence)
    with pytest.raises(ValueError, match="exact owner set"):
        validate_evidence_payload(evidence)

    evidence = build_evidence(result, generated_at="2026-08-30T00:00:00Z")
    source_digests = evidence["runtime_source_sha256"]
    assert isinstance(source_digests, dict)
    source_digests["src/scpn_control/control/h_infinity_controller.py"] = "0" * 64
    _reseal(evidence)
    with pytest.raises(ValueError, match="runtime source digest mismatch"):
        validate_evidence_payload(evidence)

    evidence = build_evidence(result, generated_at="2026-08-30T00:00:00Z")
    result_payload = evidence["result"]
    assert isinstance(result_payload, dict)
    result_payload["passed"] = False
    _reseal(evidence)
    with pytest.raises(ValueError, match="result is not passing"):
        validate_evidence_payload(evidence)


def test_cli_writes_sealed_json_and_markdown(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI writes a sealed JSON report and its human-readable twin."""
    json_path = tmp_path / "evidence.json"
    markdown_path = tmp_path / "evidence.md"
    assert main(["--json-out", str(json_path), "--markdown-out", str(markdown_path)]) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert validate_evidence_payload(payload)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Normalized DGKF H-infinity validation\n")
    assert "SPDX-License-Identifier" not in "\n".join(markdown.splitlines()[:8])
    assert H_INFINITY_SCHEMA_VERSION in capsys.readouterr().out


def test_cli_no_write_does_not_create_default_side_effect(capsys: pytest.CaptureFixture[str]) -> None:
    """The diagnostic CLI mode prints evidence without writing defaults."""
    assert main(["--no-write"]) == 0
    assert H_INFINITY_SCHEMA_VERSION in capsys.readouterr().out
