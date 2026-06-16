# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption-sequence validation tests
"""Tests for bounded disruption-sequence phase-order validation."""

from __future__ import annotations

import json

import pytest

from validation.validate_disruption_sequence import (
    DISRUPTION_SEQUENCE_SCHEMA_VERSION,
    DisruptionSequenceValidationResult,
    build_evidence,
    default_config,
    validate_disruption_sequence,
    validate_evidence_payload,
)


@pytest.fixture(scope="module")
def result() -> DisruptionSequenceValidationResult:
    """Run the deterministic sequence validation once for this module."""
    return validate_disruption_sequence()


def test_validation_passes_with_bounded_claim_blocked(result: DisruptionSequenceValidationResult) -> None:
    assert result.passed is True
    assert result.production_claim_allowed is False
    assert result.phase_order_passed is True
    assert result.temperature_passed is True
    assert result.current_trace_passed is True
    assert result.halo_force_passed is True
    assert result.mitigation_branch_passed is True


def test_validation_preserves_exact_sequence_relations(result: DisruptionSequenceValidationResult) -> None:
    assert result.total_duration_rel_error < 1e-12
    assert result.wall_heat_load_rel_error < 1e-12
    assert result.vessel_force_rel_error < 1e-12
    assert result.current_monotonic is True
    assert result.current_initial_rel_error < 1e-12


def test_mitigation_branch_changes_density_dependent_current_quench(
    result: DisruptionSequenceValidationResult,
) -> None:
    assert result.mitigated_cq_duration_ms != pytest.approx(result.unmitigated_cq_duration_ms)
    assert result.mitigated_post_tq_temperature_eV == pytest.approx(result.unmitigated_post_tq_temperature_eV)
    assert result.mitigation_metadata["spi_density_target_20"] > result.config.ne_pre_20


def test_validation_is_deterministic() -> None:
    first = validate_disruption_sequence()
    second = validate_disruption_sequence()
    assert first.total_duration_rel_error == second.total_duration_rel_error
    assert first.unmitigated_cq_duration_ms == second.unmitigated_cq_duration_ms
    assert first.mitigated_cq_duration_ms == second.mitigated_cq_duration_ms


def test_default_config_matches_sequence_domain() -> None:
    config = default_config()
    assert config.a < config.R0
    assert config.dBr_over_B_trigger > 0.0


def test_evidence_roundtrip_is_sealed_and_passing(result: DisruptionSequenceValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-disruption-sequence")
    assert evidence["schema_version"] == DISRUPTION_SEQUENCE_SCHEMA_VERSION
    assert evidence["production_claim_allowed"] is False
    assert evidence["passed"] is True
    assert evidence["phase_order"]["current_trace_monotonic"] is True
    assert validate_evidence_payload(evidence) is True


def test_evidence_tamper_is_rejected(result: DisruptionSequenceValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-disruption-sequence")
    evidence["phase_order"]["current_trace_monotonic"] = False
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_disruption_sequence as mod

    output = tmp_path / "disruption_sequence.json"
    assert mod.main(["--json", "--output-json", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == DISRUPTION_SEQUENCE_SCHEMA_VERSION
    assert payload["production_claim_allowed"] is False
    assert payload["passed"] is True
    assert json.loads(capsys.readouterr().out)["payload_sha256"] == payload["payload_sha256"]


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_disruption_sequence as mod

    assert mod.main([]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "production claim allowed: false" in out
