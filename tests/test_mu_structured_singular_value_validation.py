# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Structured singular value (mu) validation tests
"""Tests for the closed-form structured-singular-value validation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_control.control.mu_synthesis import compute_mu_upper_bound
from validation.validate_mu_structured_singular_value import (
    MU_STRUCTURED_SINGULAR_VALUE_SCHEMA_VERSION,
    MuValidationResult,
    build_evidence,
    diagonal_error,
    full_block_error,
    rank_one_relative_error,
    sandwich_sample,
    scaling_invariance_relative_error,
    validate_evidence_payload,
    validate_mu,
)

_TEST_SIZES = (2, 3, 4)


@pytest.fixture(scope="module")
def result() -> MuValidationResult:
    """Run the mu validation once for the whole module."""
    return validate_mu(sizes=_TEST_SIZES, samples_per_size=3)


# ── Exact identities, checked directly against numpy ──────────────────


def test_full_block_bound_equals_sigma_max() -> None:
    """For a single full block the bound must equal sigma_max to machine precision."""
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    bound = compute_mu_upper_bound(matrix, [(4, "full")])
    sigma_max = float(np.linalg.svd(matrix, compute_uv=False)[0])
    assert bound == pytest.approx(sigma_max, abs=1e-9)


def test_diagonal_bound_equals_max_abs_entry() -> None:
    """For diagonal M with diagonal Delta the bound must equal max|M_ii|."""
    rng = np.random.default_rng(2)
    diag = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    bound = compute_mu_upper_bound(np.diag(diag), [(1, "complex_scalar")] * 4)
    assert bound == pytest.approx(float(np.max(np.abs(diag))), abs=1e-9)


def test_rank_one_bound_equals_sum_abs_products() -> None:
    """For M = u v^H with diagonal Delta the bound must equal sum_i |u_i v_i|."""
    rng = np.random.default_rng(3)
    u = rng.standard_normal(3) + 1j * rng.standard_normal(3)
    v = rng.standard_normal(3) + 1j * rng.standard_normal(3)
    matrix = np.outer(u, np.conjugate(v))
    bound = compute_mu_upper_bound(matrix, [(1, "complex_scalar")] * 3)
    exact = float(np.sum(np.abs(u) * np.abs(v)))
    assert bound == pytest.approx(exact, rel=1e-3)


def test_spectral_sandwich_holds() -> None:
    """rho(M) <= bound <= sigma_max(M) for a diagonal structure."""
    rng = np.random.default_rng(4)
    sample = sandwich_sample(rng, 5, tolerance=1e-6)
    assert sample.within is True
    assert sample.spectral_radius <= sample.bound * (1.0 + 1e-6)
    assert sample.bound <= sample.sigma_max * (1.0 + 1e-6)


# ── Aggregate validation result ──────────────────────────────────────


def test_validation_passes_all_gated_cases(result: MuValidationResult) -> None:
    assert result.passed is True
    gated = {case.name: case for case in result.cases}
    assert gated["full_block_equals_sigma_max"].max_error < 1e-9
    assert gated["diagonal_equals_max_abs_entry"].max_error < 1e-9
    assert gated["rank_one_equals_sum_abs_products"].max_error < 1e-3
    assert gated["spectral_sandwich_rho_le_mu_le_sigma_max"].passed is True
    assert all(case.passed for case in result.cases)


def test_diagnostics_recorded_but_not_gating(result: MuValidationResult) -> None:
    names = {case.name for case in result.diagnostics}
    assert "d_scaling_invariance" in names
    # The gate excludes diagnostics: pass is decided by the exact cases only.
    assert result.passed == all(case.passed for case in result.cases)


def test_case_sample_counts_match_ensemble(result: MuValidationResult) -> None:
    expected = len(_TEST_SIZES) * 3
    for case in result.cases:
        assert case.sample_count == expected
    for case in result.diagnostics:
        assert case.sample_count == expected


def test_validation_is_deterministic_for_fixed_seed() -> None:
    a = validate_mu(sizes=_TEST_SIZES, samples_per_size=3)
    b = validate_mu(sizes=_TEST_SIZES, samples_per_size=3)
    assert [c.max_error for c in a.cases] == [c.max_error for c in b.cases]
    assert a.sandwich_min_margin == b.sandwich_min_margin


# ── Per-case helper guardrails ───────────────────────────────────────


def test_error_helpers_return_small_values() -> None:
    rng = np.random.default_rng(9)
    assert full_block_error(rng, 3) < 1e-9
    assert diagonal_error(rng, 3) < 1e-9
    assert rank_one_relative_error(rng, 3) < 1e-3
    assert scaling_invariance_relative_error(rng, 3) >= 0.0


# ── Input validation ─────────────────────────────────────────────────


def test_validate_rejects_empty_sizes() -> None:
    with pytest.raises(ValueError, match="at least one matrix size"):
        validate_mu(sizes=())


def test_validate_rejects_size_below_two() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        validate_mu(sizes=(1, 3))


def test_validate_rejects_nonpositive_samples() -> None:
    with pytest.raises(ValueError, match="samples_per_size"):
        validate_mu(sizes=_TEST_SIZES, samples_per_size=0)


def test_validate_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        validate_mu(seed=-1, sizes=_TEST_SIZES)


# ── Evidence seal ────────────────────────────────────────────────────


def test_evidence_roundtrip_is_sealed_and_passing(result: MuValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == MU_STRUCTURED_SINGULAR_VALUE_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True
    assert len(evidence["cases"]) == 4
    assert len(evidence["diagnostics"]) == 1


def test_evidence_tamper_is_rejected(result: MuValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["cases"][0]["max_error"] = 1.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: MuValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")


def test_evidence_rejects_unknown_schema(result: MuValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["schema_version"] = "scpn-control.unknown.v9"
    with pytest.raises(ValueError, match="unsupported"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_non_hex_seal(result: MuValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["payload_sha256"] = "xyz"
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        validate_evidence_payload(evidence)


# ── CLI / report writer ──────────────────────────────────────────────


def test_main_text_output_passes(capsys) -> None:
    import validation.validate_mu_structured_singular_value as mod

    assert mod.main(["--sizes", "2", "3", "--samples-per-size", "2"]) == 0
    out = capsys.readouterr().out
    assert "Status: pass" in out
    assert "(diagnostic)" in out


def test_main_json_output_and_report(capsys, tmp_path) -> None:
    import validation.validate_mu_structured_singular_value as mod

    report = tmp_path / "mu.json"
    assert mod.main(["--sizes", "2", "3", "--samples-per-size", "2", "--json-out", "--report", str(report)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == MU_STRUCTURED_SINGULAR_VALUE_SCHEMA_VERSION
    assert report.exists() and report.with_suffix(".md").exists()
    assert validate_evidence_payload(json.loads(report.read_text())) is True
    assert "not gated" in report.with_suffix(".md").read_text()


def test_main_returns_one_on_failure(monkeypatch, capsys) -> None:
    import validation.validate_mu_structured_singular_value as mod

    real = mod.validate_mu
    # Force the rank-one gate to fail with an impossibly tight tolerance.
    monkeypatch.setattr(mod, "validate_mu", lambda **_: real(sizes=(2, 3), samples_per_size=2, rank_one_tol=1e-18))
    assert mod.main([]) == 1
    assert "Status: fail" in capsys.readouterr().out
