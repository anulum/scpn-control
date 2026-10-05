# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Halo Runaway Physics
"""Disruption mitigation claim evidence and admission."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from scpn_control.control._disruption_ensemble import DisruptionMitigationReport
from scpn_control.core._validators import require_non_negative_float

_DISRUPTION_CLAIM_SCHEMA_VERSION = 1
_BOUNDED_CLAIM_STATUS = "bounded halo/runaway ensemble evidence only; mitigation claims blocked"


@dataclass(frozen=True)
class DisruptionMitigationClaimEvidence:
    """Serialisable admission evidence for halo/runaway mitigation claims."""

    schema_version: int
    model_id: str
    source: str
    source_id: str
    ensemble_runs: int
    ensemble_seed: int
    prevention_rate: float
    mean_halo_peak_ma: float
    p95_halo_peak_ma: float
    mean_re_peak_ma: float
    p95_re_peak_ma: float
    mean_tpf_product: float
    passes_iter_limits: bool
    reference_source: str
    reference_dataset_id: str
    reference_artifact_sha256: str
    reference_case_count: int
    risk_after_abs_error: float | None
    detection_lead_time_abs_error_ms: float | None
    halo_current_relative_error: float | None
    runaway_beam_relative_error: float | None
    tbr_abs_error: float | None
    risk_after_abs_tolerance: float | None
    detection_lead_time_abs_tolerance_ms: float | None
    halo_current_relative_tolerance: float | None
    runaway_beam_relative_tolerance: float | None
    tbr_abs_tolerance: float | None
    mitigation_claim_allowed: bool
    claim_status: str


def _non_empty_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_nonnegative_or_none(name: str, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be finite and non-negative")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_unit_interval(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite in [0, 1]")
    return result


def disruption_mitigation_claim_evidence(
    report: DisruptionMitigationReport,
    *,
    source: str,
    source_id: str,
    ensemble_seed: int,
    model_id: str = "halo_runaway_disruption_mitigation",
    reference_artifact_path: str | Path | None = None,
) -> DisruptionMitigationClaimEvidence:
    """Build fail-closed evidence for halo/runaway mitigation claims."""
    if not isinstance(report, DisruptionMitigationReport):
        raise ValueError("report must be DisruptionMitigationReport")
    if report.ensemble_runs < 1:
        raise ValueError("report.ensemble_runs must be positive")
    prevention_rate = _finite_unit_interval("prevention_rate", report.prevention_rate)
    mean_tpf_product = _finite_nonnegative_or_none("mean_tpf_product", report.mean_tpf_product)
    if mean_tpf_product is None:
        raise ValueError("mean_tpf_product must be present")

    if reference_artifact_path is not None:
        from validation.validate_disruption_reference import validate_disruption_reference

        artifact_path = Path(reference_artifact_path)
        validator_report = validate_disruption_reference(artifact_path, require_reference_artifacts=True)
        if validator_report["status"] != "pass":
            raise ValueError("disruption reference artifact failed strict validation")
        raise ValueError("mitigation claim requires independently verified reference comparison")

    return DisruptionMitigationClaimEvidence(
        schema_version=_DISRUPTION_CLAIM_SCHEMA_VERSION,
        model_id=_non_empty_text("model_id", model_id),
        source=_non_empty_text("source", source),
        source_id=_non_empty_text("source_id", source_id),
        ensemble_runs=int(report.ensemble_runs),
        ensemble_seed=int(ensemble_seed),
        prevention_rate=prevention_rate,
        mean_halo_peak_ma=require_non_negative_float("mean_halo_peak_ma", report.mean_halo_peak_ma),
        p95_halo_peak_ma=require_non_negative_float("p95_halo_peak_ma", report.p95_halo_peak_ma),
        mean_re_peak_ma=require_non_negative_float("mean_re_peak_ma", report.mean_re_peak_ma),
        p95_re_peak_ma=require_non_negative_float("p95_re_peak_ma", report.p95_re_peak_ma),
        mean_tpf_product=mean_tpf_product,
        passes_iter_limits=bool(report.passes_iter_limits),
        reference_source="none",
        reference_dataset_id="",
        reference_artifact_sha256="",
        reference_case_count=0,
        risk_after_abs_error=None,
        detection_lead_time_abs_error_ms=None,
        halo_current_relative_error=None,
        runaway_beam_relative_error=None,
        tbr_abs_error=None,
        risk_after_abs_tolerance=None,
        detection_lead_time_abs_tolerance_ms=None,
        halo_current_relative_tolerance=None,
        runaway_beam_relative_tolerance=None,
        tbr_abs_tolerance=None,
        mitigation_claim_allowed=False,
        claim_status=_BOUNDED_CLAIM_STATUS,
    )


def assert_disruption_mitigation_claim_admissible(
    evidence: DisruptionMitigationClaimEvidence,
) -> DisruptionMitigationClaimEvidence:
    """Return evidence only when strict matched-reference admission passed."""
    if not isinstance(evidence, DisruptionMitigationClaimEvidence):
        raise ValueError("evidence must be DisruptionMitigationClaimEvidence")
    if evidence.schema_version != _DISRUPTION_CLAIM_SCHEMA_VERSION:
        raise ValueError("disruption mitigation claim evidence schema_version is unsupported")
    if evidence.mitigation_claim_allowed:
        raise ValueError("mitigation claim requires independently verified reference comparison")
    raise ValueError("disruption mitigation claim is blocked without matched reference evidence")


def save_disruption_mitigation_claim_evidence(
    evidence: DisruptionMitigationClaimEvidence,
    path: str | Path,
) -> None:
    """Persist disruption mitigation claim evidence as deterministic JSON."""
    if not isinstance(evidence, DisruptionMitigationClaimEvidence):
        raise ValueError("evidence must be DisruptionMitigationClaimEvidence")
    if evidence.mitigation_claim_allowed:
        raise ValueError("mitigation claim requires independently verified reference comparison")
    if evidence.claim_status != _BOUNDED_CLAIM_STATUS:
        raise ValueError("bounded disruption claim status is inconsistent")
    reference_fields = (
        evidence.risk_after_abs_error,
        evidence.detection_lead_time_abs_error_ms,
        evidence.halo_current_relative_error,
        evidence.runaway_beam_relative_error,
        evidence.tbr_abs_error,
        evidence.risk_after_abs_tolerance,
        evidence.detection_lead_time_abs_tolerance_ms,
        evidence.halo_current_relative_tolerance,
        evidence.runaway_beam_relative_tolerance,
        evidence.tbr_abs_tolerance,
    )
    if (
        evidence.reference_source != "none"
        or evidence.reference_dataset_id
        or evidence.reference_artifact_sha256
        or evidence.reference_case_count != 0
        or any(value is not None for value in reference_fields)
    ):
        raise ValueError("bounded disruption evidence cannot carry unverified reference comparisons")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")
