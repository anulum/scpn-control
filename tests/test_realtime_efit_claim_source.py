# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — EFIT source admission regression tests.

"""Exercise the public EFIT-lite claim boundary with untrusted references."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest

from scpn_control.control.realtime_efit import (
    EFITLiteClaimEvidence,
    MagneticDiagnostics,
    ReconstructionResult,
    ShapeParams,
    assert_efit_lite_facility_claim_admissible,
    efit_lite_claim_evidence,
    save_efit_lite_claim_evidence,
)


def _result() -> ReconstructionResult:
    """Return a finite reconstruction with a declared geometric solve."""
    return ReconstructionResult(
        psi=np.ones((3, 3), dtype=float),
        p_prime_coeffs=np.ones(1, dtype=float),
        ff_prime_coeffs=np.ones(1, dtype=float),
        shape=ShapeParams(6.2, 1.0, 1.5, 0.2, 0.2, 3.0, 0.5, 0.8, 1.0e6),
        chi_squared=0.0,
        n_iterations=1,
        wall_time_ms=1.0,
    )


def _diagnostics() -> MagneticDiagnostics:
    """Return a finite diagnostic layout for evidence construction."""
    return MagneticDiagnostics([(6.2, 0.0)], [(6.2, 0.0, "Z")], 6.2)


def test_self_reference_cannot_admit_facility_claim() -> None:
    """A result copied into an alleged external reference has no source lineage."""
    result = _result()
    evidence = efit_lite_claim_evidence(
        result,
        _diagnostics(),
        source="efit_reference",
        source_id="caller_claims_independent_source",
        diagnostic_source="caller_claims_measured_diagnostics",
        reference_psi=result.psi.copy(),
        reference_shape=replace(result.shape),
    )
    assert evidence.psi_relative_error == 0.0
    assert evidence.facility_claim_allowed is False
    assert "unverified" in evidence.claim_status
    with pytest.raises(ValueError, match="not admissible"):
        assert_efit_lite_facility_claim_admissible(evidence)


def test_forged_facility_flag_refused_before_persistence(tmp_path: Path) -> None:
    """A forged dataclass cannot turn local metadata into a facility record."""
    evidence = efit_lite_claim_evidence(
        _result(),
        _diagnostics(),
        source="synthetic_regression_reference",
        source_id="bounded_case",
        diagnostic_source="synthetic",
    )
    forged = replace(evidence, facility_claim_allowed=True, claim_status="admission passed")
    with pytest.raises(ValueError, match="not admissible"):
        assert_efit_lite_facility_claim_admissible(forged)
    destination = tmp_path / "new" / "claim.json"
    with pytest.raises(ValueError, match="facility"):
        save_efit_lite_claim_evidence(forged, destination)
    assert not destination.parent.exists()


def test_nonfinite_claim_payload_refused_before_persistence(tmp_path: Path) -> None:
    """Strict JSON never records an invalid numeric claim as evidence."""
    evidence = efit_lite_claim_evidence(
        _result(),
        _diagnostics(),
        source="synthetic_regression_reference",
        source_id="bounded_case",
        diagnostic_source="synthetic",
    )
    destination = tmp_path / "new" / "claim.json"
    with pytest.raises(ValueError, match="finite"):
        save_efit_lite_claim_evidence(replace(evidence, chi_squared=float("nan")), destination)
    assert not destination.parent.exists()
    save_efit_lite_claim_evidence(evidence, destination)
    assert json.loads(destination.read_text(encoding="utf-8"))["facility_claim_allowed"] is False


def test_overflowed_psi_comparison_is_not_evidence() -> None:
    """Finite arrays whose comparison is unrepresentable must fail closed."""
    result = replace(_result(), psi=np.full((3, 3), 1.0e308))
    with pytest.raises(ValueError, match="psi.*finite"):
        efit_lite_claim_evidence(
            result,
            _diagnostics(),
            source="efit_reference",
            source_id="bounded_case",
            diagnostic_source="synthetic",
            reference_psi=np.full((3, 3), 1.0e-308),
        )


def test_persistence_rejects_wrong_type_and_schema_before_writing(tmp_path: Path) -> None:
    """A persisted claim must use the supported evidence object and schema."""
    evidence = efit_lite_claim_evidence(
        _result(),
        _diagnostics(),
        source="synthetic_regression_reference",
        source_id="bounded_case",
        diagnostic_source="synthetic",
    )
    destination = tmp_path / "new" / "claim.json"
    with pytest.raises(ValueError, match="must be EFITLiteClaimEvidence"):
        save_efit_lite_claim_evidence(cast(EFITLiteClaimEvidence, {"claim": "forged"}), destination)
    with pytest.raises(ValueError, match="schema_version is unsupported"):
        save_efit_lite_claim_evidence(replace(evidence, schema_version=99), destination)
    assert not destination.parent.exists()


@pytest.mark.parametrize(
    ("status", "delta", "message"),
    [
        ("picard_converged", None, "requires final_relative_change"),
        ("picard_limit", float("nan"), "final_relative_change must be finite"),
        ("geometric_solve", 0.1, "cannot report final_relative_change"),
        ("forged_status", None, "iteration_status is unsupported"),
    ],
)
def test_claim_builder_rejects_inconsistent_reconstruction_status(
    status: str, delta: float | None, message: str
) -> None:
    """A caller cannot attach an impossible Picard result to claim evidence."""
    declared_status = cast(Literal["unreported", "geometric_solve", "picard_converged", "picard_limit"], status)
    result = replace(_result(), iteration_status=declared_status, final_relative_change=delta)
    with pytest.raises(ValueError, match=message):
        efit_lite_claim_evidence(
            result,
            _diagnostics(),
            source="synthetic_regression_reference",
            source_id="bounded_case",
            diagnostic_source="synthetic",
        )
