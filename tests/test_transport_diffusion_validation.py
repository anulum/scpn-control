# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Transport heat-diffusion validation tests
"""Tests for the transport heat-diffusion operator and solver validation."""

from __future__ import annotations

import json

import pytest

from validation.validate_transport_diffusion import (
    TRANSPORT_DIFFUSION_SCHEMA_VERSION,
    ConvergenceRecord,
    TransportDiffusionValidationResult,
    bessel_operator_error,
    build_evidence,
    manufactured_steady_state,
    thomas_parity,
    validate_evidence_payload,
    validate_transport_diffusion,
)

_TEST_RESOLUTIONS = (33, 65)


@pytest.fixture(scope="module")
def result() -> TransportDiffusionValidationResult:
    """Run the transport diffusion validation once for the module."""
    return validate_transport_diffusion(resolutions=_TEST_RESOLUTIONS)


# ── Operator eigenvalue (Bessel) ─────────────────────────────────────


def test_operator_error_is_small_and_second_order() -> None:
    err_coarse = bessel_operator_error(33)
    err_fine = bessel_operator_error(65)
    assert err_fine < 1e-3
    # Halving h should roughly quarter the error (second order).
    assert 3.4 < err_coarse / err_fine < 4.6


def test_operator_validation_passes(result: TransportDiffusionValidationResult) -> None:
    assert result.operator_passed is True
    assert result.operator_order == pytest.approx(2.0, abs=0.15)
    assert result.operator_error_finest < result.operator_error_gate


# ── Manufactured steady state ────────────────────────────────────────


def test_manufactured_steady_state_is_small_and_second_order() -> None:
    nrmse_coarse = manufactured_steady_state(33)
    nrmse_fine = manufactured_steady_state(65)
    assert nrmse_fine < 1e-3
    assert 3.0 < nrmse_coarse / nrmse_fine < 5.0


def test_steady_validation_passes(result: TransportDiffusionValidationResult) -> None:
    assert result.steady_passed is True
    assert result.steady_order == pytest.approx(2.0, abs=0.25)
    assert result.steady_nrmse_finest < result.steady_nrmse_gate


# ── Polyglot Thomas parity ───────────────────────────────────────────


def test_thomas_parity_is_bit_identical_when_rust_present(result: TransportDiffusionValidationResult) -> None:
    if result.thomas_available:
        assert result.thomas_record is not None
        assert result.thomas_record.matches is True
        # The Rust and Python Thomas solvers are the same algorithm; expect exact agreement.
        assert result.thomas_record.max_abs_diff < 1e-12
        assert result.thomas_record.max_abs_diff_vs_dense < 1e-9
    else:
        assert result.thomas_record is None
    assert result.thomas_passed is True


def test_thomas_parity_direct_call() -> None:
    record = thomas_parity(nr=49)
    if record is not None:
        assert record.matches is True
        assert record.max_abs_diff < 1e-12


def test_thomas_parity_none_when_extension_missing(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "scpn_control_rs", None)  # forces ImportError
    assert thomas_parity(nr=33) is None


def test_thomas_parity_none_when_binding_absent(monkeypatch) -> None:
    import sys
    import types

    stub = types.ModuleType("scpn_control_rs")  # present but without py_thomas_solve
    monkeypatch.setitem(sys.modules, "scpn_control_rs", stub)
    assert thomas_parity(nr=33) is None


def test_excluding_rust_yields_no_record() -> None:
    res = validate_transport_diffusion(resolutions=_TEST_RESOLUTIONS, include_rust=False)
    assert res.thomas_available is False
    assert res.thomas_record is None
    assert res.thomas_passed is True
    assert res.passed is True


# ── Aggregate result ─────────────────────────────────────────────────


def test_overall_validation_passes(result: TransportDiffusionValidationResult) -> None:
    assert result.passed is True
    assert len(result.operator_records) == len(_TEST_RESOLUTIONS)
    assert len(result.steady_records) == len(_TEST_RESOLUTIONS)


def test_validation_is_deterministic() -> None:
    a = validate_transport_diffusion(resolutions=_TEST_RESOLUTIONS, include_rust=False)
    b = validate_transport_diffusion(resolutions=_TEST_RESOLUTIONS, include_rust=False)
    assert a.operator_error_finest == b.operator_error_finest
    assert a.steady_nrmse_finest == b.steady_nrmse_finest


# ── Input contract ───────────────────────────────────────────────────


def test_validate_requires_two_distinct_resolutions() -> None:
    with pytest.raises(ValueError, match="at least two distinct resolutions"):
        validate_transport_diffusion(resolutions=(33, 33))


def test_validate_rejects_degenerate_resolution() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        validate_transport_diffusion(resolutions=(3, 33))


def test_bessel_operator_error_rejects_nonpositive_chi() -> None:
    with pytest.raises(ValueError, match="chi_value must be positive"):
        bessel_operator_error(33, chi_value=0.0)


def test_bessel_operator_error_rejects_nonnumeric_chi() -> None:
    with pytest.raises(ValueError, match="chi_value must be a finite number"):
        bessel_operator_error(33, chi_value="warm")  # type: ignore[arg-type]


def test_bessel_operator_error_rejects_nonfinite_chi() -> None:
    with pytest.raises(ValueError, match="chi_value must be finite"):
        bessel_operator_error(33, chi_value=float("inf"))


def test_validate_rejects_noninteger_resolution() -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        validate_transport_diffusion(resolutions=(33.5, 65))  # type: ignore[arg-type]


def test_log_log_slope_requires_two_points() -> None:
    from validation.validate_transport_diffusion import _log_log_slope

    with pytest.raises(ValueError, match="at least two resolutions"):
        _log_log_slope([ConvergenceRecord(resolution=33, mesh_spacing=0.03, error=1e-4)])


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: TransportDiffusionValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == TRANSPORT_DIFFUSION_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True


def test_evidence_tamper_is_rejected(result: TransportDiffusionValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["operator_order"] = 99.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: TransportDiffusionValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: TransportDiffusionValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: TransportDiffusionValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "notdigest"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


def test_evidence_thomas_record_serialised(result: TransportDiffusionValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    if result.thomas_available:
        assert evidence["thomas_record"]["matches"] is True
    else:
        assert evidence["thomas_record"] is None


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_transport_diffusion as mod

    assert mod.main(["--resolutions", "33", "65"]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "operator (Bessel)" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_transport_diffusion as mod

    report = tmp_path / "td.json"
    assert mod.main(["--resolutions", "33", "65", "--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == TRANSPORT_DIFFUSION_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "Transport Heat-Diffusion" in report.with_suffix(".md").read_text()


def test_main_report_without_rust_notes_absence(tmp_path) -> None:
    import validation.validate_transport_diffusion as mod

    report = tmp_path / "td_norust.json"
    assert mod.main(["--resolutions", "33", "65", "--no-rust", "--report", str(report)]) == 0
    assert "extension not present" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_transport_diffusion as mod

    real = mod.validate_transport_diffusion
    monkeypatch.setattr(
        mod,
        "validate_transport_diffusion",
        lambda **_: real(resolutions=_TEST_RESOLUTIONS, operator_error_gate=1e-30, include_rust=False),
    )
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
