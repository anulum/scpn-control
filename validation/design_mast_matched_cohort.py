#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fail-closed MAST matched-cohort design gate
"""Design a statistically explicit MAST cohort without admitting proxy labels.

L2F-23a consumes the self-digested L2F-22 proxy-sensitivity report and a
self-digested design specification. It computes binomial-precision and AUC
CI/power planning minima, but it never selects shots. The report remains blocked
until authoritative labels, diagnostic availability, and distribution-informed
matching calipers exist.

The AUC calculation is an explicitly labelled large-sample planning
approximation. It must be replaced or independently reviewed before a final
analysis protocol is frozen. No raw traces or shot artefacts are opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import cast

from validation.audit_mast_label_proxy_sensitivity import AUDIT_SCHEMA
from validation.mast_source_object_manifest import canonical_json_sha256

COHORT_DESIGN_SCHEMA = "scpn-control.mast-matched-cohort-design.v1.0.0"
COHORT_DESIGN_REPORT_SCHEMA = "scpn-control.mast-matched-cohort-design-report.v1.0.0"

INDEPENDENT_LABEL_AUTHORITIES = frozenset({"facility", "defuse", "independent_manual_review"})
EXCLUDED_OUTCOMES = ("aborted_no_plasma", "ambiguous")
CANONICAL_CHANNELS = (
    "BT_T",
    "Ip_MA",
    "beta_N",
    "dBdt_gauss_per_s",
    "locked_mode_amp",
    "n1_amp",
    "n2_amp",
    "ne_1e19",
    "q95",
    "time_s",
    "vertical_position_m",
)
REQUIRED_EXACT_MATCH_FIELDS = frozenset({"campaign", "configuration", "diagnostic_availability", "programme_class"})
_CLAIM_FIELDS = frozenset(
    {
        "scientific_validation",
        "independent_label_validation",
        "cohort_admission",
        "training_admission",
        "facility_prediction",
        "control_admission",
    }
)
_SHA256_LENGTH = 64
_MAX_GROUP_SIZE = 1_000_000


class CohortDesignError(ValueError):
    """Raised when a design input is ambiguous or breaches an evidence gate."""


@dataclass(frozen=True)
class CohortDesignSpec:
    """Validated planning assumptions for a future matched cohort."""

    confidence_level: float
    target_power: float
    expected_sensitivity: float
    expected_specificity: float
    max_sensitivity_ci_half_width: float
    max_specificity_ci_half_width: float
    target_auc: float
    null_auc: float
    max_auc_ci_half_width: float
    control_to_case_ratio: int
    admissible_label_authorities: tuple[str, ...]
    required_channels: tuple[str, ...]
    exact_match_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject weak labels, under-specified matching, and invalid targets."""
        _unit_interval(self.confidence_level, "confidence_level", lower_open=True)
        if self.confidence_level <= 0.5:
            raise CohortDesignError("confidence_level must exceed 0.5")
        _unit_interval(self.target_power, "target_power", lower_open=True)
        if self.target_power <= 0.5:
            raise CohortDesignError("target_power must exceed 0.5")
        _unit_interval(self.expected_sensitivity, "expected_sensitivity", lower_open=True, upper_open=True)
        _unit_interval(self.expected_specificity, "expected_specificity", lower_open=True, upper_open=True)
        _half_width(self.max_sensitivity_ci_half_width, "max_sensitivity_ci_half_width")
        _half_width(self.max_specificity_ci_half_width, "max_specificity_ci_half_width")
        _unit_interval(self.null_auc, "null_auc", lower_open=True, upper_open=True)
        _require_no_discrimination_null_auc(self.null_auc)
        _unit_interval(self.target_auc, "target_auc", lower_open=True, upper_open=True)
        if self.target_auc <= self.null_auc:
            raise CohortDesignError("target_auc must exceed null_auc")
        _half_width(self.max_auc_ci_half_width, "max_auc_ci_half_width")
        if isinstance(self.control_to_case_ratio, bool) or self.control_to_case_ratio < 1:
            raise CohortDesignError("control_to_case_ratio must be a positive integer")
        _sorted_unique(self.admissible_label_authorities, "admissible_label_authorities")
        if not self.admissible_label_authorities:
            raise CohortDesignError("admissible_label_authorities must not be empty")
        weak = set(self.admissible_label_authorities) - INDEPENDENT_LABEL_AUTHORITIES
        if weak:
            raise CohortDesignError(f"non-independent label authorities are forbidden: {sorted(weak)}")
        _sorted_unique(self.required_channels, "required_channels")
        if self.required_channels != tuple(sorted(CANONICAL_CHANNELS)):
            raise CohortDesignError("required_channels must equal the canonical 11-channel contract")
        _sorted_unique(self.exact_match_fields, "exact_match_fields")
        missing_match = REQUIRED_EXACT_MATCH_FIELDS - set(self.exact_match_fields)
        if missing_match:
            raise CohortDesignError(f"exact_match_fields lacks required fields: {sorted(missing_match)}")


def _unit_interval(
    value: float,
    field: str,
    *,
    lower_open: bool = False,
    upper_open: bool = False,
) -> None:
    if not math.isfinite(value):
        raise CohortDesignError(f"{field} must be finite")
    lower_ok = value > 0.0 if lower_open else value >= 0.0
    upper_ok = value < 1.0 if upper_open else value <= 1.0
    if not lower_ok or not upper_ok:
        bounds = "(0, 1)" if lower_open and upper_open else "(0, 1]" if lower_open else "[0, 1]"
        raise CohortDesignError(f"{field} must be within {bounds}")


def _half_width(value: float, field: str) -> None:
    if not math.isfinite(value) or not 0.0 < value < 0.5:
        raise CohortDesignError(f"{field} must be finite and within (0, 0.5)")


def _require_no_discrimination_null_auc(null_auc: float) -> None:
    if null_auc != 0.5:
        raise CohortDesignError("null_auc must equal 0.5 for the no-discrimination power model")


def _sorted_unique(values: tuple[str, ...], field: str) -> None:
    if any(not value.strip() for value in values) or tuple(sorted(set(values))) != values:
        raise CohortDesignError(f"{field} must be non-empty strings, unique, and sorted")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise CohortDesignError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _load_json_with_sha(path: Path) -> tuple[Mapping[str, object], str]:
    """Parse and hash one immutable byte snapshot of a JSON evidence file."""
    try:
        raw = path.read_bytes()
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, CohortDesignError) as exc:
        raise CohortDesignError(f"cannot read verified JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CohortDesignError(f"JSON root must be an object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> Mapping[str, object]:
    payload, _ = _load_json_with_sha(path)
    return payload


def _required_string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise CohortDesignError(f"{field} must be a non-empty string")
    return value


def _required_float(payload: Mapping[str, object], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CohortDesignError(f"{field} must be numeric")
    return float(value)


def _required_int(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CohortDesignError(f"{field} must be an integer")
    return value


def _string_tuple(payload: Mapping[str, object], field: str) -> tuple[str, ...]:
    value = payload.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise CohortDesignError(f"{field} must be an array of strings")
    return tuple(cast(list[str], value))


def _validate_sha256(value: str, *, field: str) -> str:
    if len(value) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in value):
        raise CohortDesignError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _verify_self_digest(payload: Mapping[str, object]) -> str:
    digest = _validate_sha256(_required_string(payload, "payload_sha256"), field="payload_sha256")
    digest_input = dict(payload)
    digest_input["payload_sha256"] = None
    if canonical_json_sha256(digest_input) != digest:
        raise CohortDesignError("payload_sha256 does not match the JSON payload")
    return digest


def _load_design_spec(path: Path) -> tuple[CohortDesignSpec, str, str]:
    payload, file_digest = _load_json_with_sha(path)
    expected = {
        "schema_version",
        "machine",
        "design_status",
        "confidence_level",
        "target_power",
        "expected_sensitivity",
        "expected_specificity",
        "max_sensitivity_ci_half_width",
        "max_specificity_ci_half_width",
        "target_auc",
        "null_auc",
        "max_auc_ci_half_width",
        "control_to_case_ratio",
        "admissible_label_authorities",
        "excluded_outcomes",
        "required_channels",
        "exact_match_fields",
        "caliper_policy_status",
        "payload_sha256",
    }
    if set(payload) != expected:
        raise CohortDesignError("cohort-design fields do not match the schema")
    if _required_string(payload, "schema_version") != COHORT_DESIGN_SCHEMA:
        raise CohortDesignError("unsupported cohort-design schema")
    if _required_string(payload, "machine") != "MAST":
        raise CohortDesignError("cohort-design machine must be MAST")
    if _required_string(payload, "design_status") != "draft_owner_ratification_required":
        raise CohortDesignError("cohort-design status must remain draft_owner_ratification_required")
    if _string_tuple(payload, "excluded_outcomes") != EXCLUDED_OUTCOMES:
        raise CohortDesignError("excluded_outcomes must be aborted_no_plasma and ambiguous")
    if _required_string(payload, "caliper_policy_status") != "required_not_supplied":
        raise CohortDesignError("matching calipers must remain required_not_supplied in L2F-23a")
    digest = _verify_self_digest(payload)
    spec = CohortDesignSpec(
        confidence_level=_required_float(payload, "confidence_level"),
        target_power=_required_float(payload, "target_power"),
        expected_sensitivity=_required_float(payload, "expected_sensitivity"),
        expected_specificity=_required_float(payload, "expected_specificity"),
        max_sensitivity_ci_half_width=_required_float(payload, "max_sensitivity_ci_half_width"),
        max_specificity_ci_half_width=_required_float(payload, "max_specificity_ci_half_width"),
        target_auc=_required_float(payload, "target_auc"),
        null_auc=_required_float(payload, "null_auc"),
        max_auc_ci_half_width=_required_float(payload, "max_auc_ci_half_width"),
        control_to_case_ratio=_required_int(payload, "control_to_case_ratio"),
        admissible_label_authorities=_string_tuple(payload, "admissible_label_authorities"),
        required_channels=_string_tuple(payload, "required_channels"),
        exact_match_fields=_string_tuple(payload, "exact_match_fields"),
    )
    return spec, digest, file_digest


def wilson_half_width(*, expected_rate: float, sample_size: int, confidence_level: float) -> float:
    """Return the Wilson-score interval half-width at an expected rate."""
    if sample_size < 1:
        raise CohortDesignError("sample_size must be positive")
    _unit_interval(expected_rate, "expected_rate", lower_open=True, upper_open=True)
    _unit_interval(confidence_level, "confidence_level", lower_open=True)
    if confidence_level <= 0.5:
        raise CohortDesignError("confidence_level must exceed 0.5")
    z = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    z2 = z * z
    denominator = 1.0 + z2 / sample_size
    variance = expected_rate * (1.0 - expected_rate) / sample_size + z2 / (4.0 * sample_size**2)
    return z * math.sqrt(variance) / denominator


def minimum_binomial_sample_size(
    *,
    expected_rate: float,
    confidence_level: float,
    max_half_width: float,
    max_n: int = _MAX_GROUP_SIZE,
) -> int:
    """Find the minimum count whose Wilson half-width meets the target."""
    _half_width(max_half_width, "max_half_width")
    if max_n < 1:
        raise CohortDesignError("max_n must be positive")

    def meets(n: int) -> bool:
        return (
            wilson_half_width(
                expected_rate=expected_rate,
                sample_size=n,
                confidence_level=confidence_level,
            )
            <= max_half_width
        )

    high = 1
    while high < max_n and not meets(high):
        high = min(high * 2, max_n)
    if not meets(high):
        raise CohortDesignError("binomial precision target exceeds max_n")
    low = 1
    while low < high:
        middle = (low + high) // 2
        if meets(middle):
            high = middle
        else:
            low = middle + 1
    return low


def _auc_standard_error(*, auc: float, positive_count: int, negative_count: int) -> float:
    if positive_count < 2 or negative_count < 2:
        raise CohortDesignError("AUC planning requires at least two samples per class")
    _unit_interval(auc, "auc", lower_open=True, upper_open=True)
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    variance = (
        auc * (1.0 - auc) + (positive_count - 1) * (q1 - auc * auc) + (negative_count - 1) * (q2 - auc * auc)
    ) / (positive_count * negative_count)
    # Valid AUCs and class counts make this algebraically positive; retain the
    # guard against future formula changes or exotic numeric implementations.
    if variance <= 0.0 or not math.isfinite(variance):  # pragma: no cover
        raise CohortDesignError("AUC planning variance must be finite and positive")
    return math.sqrt(variance)


def _auc_design_metrics(
    *,
    target_auc: float,
    null_auc: float,
    positive_count: int,
    negative_count: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    """Return alternative SE, two-sided CI half-width, and one-sided power."""
    _require_no_discrimination_null_auc(null_auc)
    alternative_se = _auc_standard_error(
        auc=target_auc,
        positive_count=positive_count,
        negative_count=negative_count,
    )
    confidence_z = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    ci_half_width = confidence_z * alternative_se
    null_se = math.sqrt((positive_count + negative_count + 1.0) / (12.0 * positive_count * negative_count))
    one_sided_z = NormalDist().inv_cdf(confidence_level)
    rejection_threshold = null_auc + one_sided_z * null_se
    power = NormalDist().cdf((target_auc - rejection_threshold) / alternative_se)
    return alternative_se, ci_half_width, power


def minimum_auc_group_sizes(
    *,
    target_auc: float,
    null_auc: float,
    confidence_level: float,
    target_power: float,
    max_ci_half_width: float,
    control_to_case_ratio: int,
    max_positive_count: int = _MAX_GROUP_SIZE,
) -> tuple[int, int, float, float, float]:
    """Find minimum class counts meeting the declared AUC planning targets."""
    _unit_interval(target_auc, "target_auc", lower_open=True, upper_open=True)
    _unit_interval(null_auc, "null_auc", lower_open=True, upper_open=True)
    _require_no_discrimination_null_auc(null_auc)
    if target_auc <= null_auc:
        raise CohortDesignError("target_auc must exceed null_auc")
    _unit_interval(confidence_level, "confidence_level", lower_open=True)
    _unit_interval(target_power, "target_power", lower_open=True)
    _half_width(max_ci_half_width, "max_ci_half_width")
    if isinstance(control_to_case_ratio, bool) or control_to_case_ratio < 1:
        raise CohortDesignError("control_to_case_ratio must be a positive integer")
    if max_positive_count < 2:
        raise CohortDesignError("max_positive_count must be at least two")

    def metrics(positive_count: int) -> tuple[int, float, float, float]:
        negative_count = positive_count * control_to_case_ratio
        se, half_width, power = _auc_design_metrics(
            target_auc=target_auc,
            null_auc=null_auc,
            positive_count=positive_count,
            negative_count=negative_count,
            confidence_level=confidence_level,
        )
        return negative_count, se, half_width, power

    def meets(positive_count: int) -> bool:
        _, _, half_width, power = metrics(positive_count)
        return half_width <= max_ci_half_width and power >= target_power

    high = 2
    while high < max_positive_count and not meets(high):
        high = min(high * 2, max_positive_count)
    if not meets(high):
        raise CohortDesignError("AUC planning target exceeds max_positive_count")
    low = 2
    while low < high:
        middle = (low + high) // 2
        if meets(middle):
            high = middle
        else:
            low = middle + 1
    negative_count, se, half_width, power = metrics(low)
    return low, negative_count, se, half_width, power


def _proxy_audit(path: Path) -> tuple[Mapping[str, object], str, str]:
    payload, file_digest = _load_json_with_sha(path)
    if _required_string(payload, "schema_version") != AUDIT_SCHEMA:
        raise CohortDesignError("unsupported proxy-sensitivity report schema")
    if payload.get("status") != "blocked":
        raise CohortDesignError("proxy-sensitivity report must remain status:blocked")
    digest = _verify_self_digest(payload)
    for field in (
        "source_dataset_report_file_sha256",
        "source_dataset_report_payload_sha256",
        "source_label_algorithm_sha256",
    ):
        _validate_sha256(_required_string(payload, field), field=field)
    verified = _required_int(payload, "verified_shot_artifact_count")
    stable = _required_int(payload, "stable_shot_count")
    unstable = _required_int(payload, "unstable_shot_count")
    if verified <= 0 or stable < 0 or unstable < 0 or stable + unstable != verified:
        raise CohortDesignError("proxy-sensitivity shot counts are inconsistent")
    unstable_ids = payload.get("unstable_shot_ids")
    if not isinstance(unstable_ids, list) or len(unstable_ids) != unstable:
        raise CohortDesignError("unstable_shot_ids does not match unstable_shot_count")
    claims = payload.get("claim_boundary")
    if (
        not isinstance(claims, Mapping)
        or set(claims) != _CLAIM_FIELDS
        or not all(value is False for value in claims.values())
    ):
        raise CohortDesignError("proxy-sensitivity claim boundary must keep every admission claim false")
    comparison = payload.get("programme_comparison")
    if not isinstance(comparison, Mapping) or comparison.get("independent_validation_claim") is not False:
        raise CohortDesignError("programme comparison must not assert independent validation")
    return payload, digest, file_digest


def design_matched_cohort(
    *,
    design_spec_path: Path,
    proxy_sensitivity_path: Path,
    generated_at: str,
) -> dict[str, object]:
    """Build a raw-data-free, fail-closed statistical cohort-design report."""
    if not generated_at:
        raise CohortDesignError("generated_at must be non-empty")
    spec, design_digest, design_file_digest = _load_design_spec(design_spec_path)
    proxy, proxy_digest, proxy_file_digest = _proxy_audit(proxy_sensitivity_path)

    sensitivity_n = minimum_binomial_sample_size(
        expected_rate=spec.expected_sensitivity,
        confidence_level=spec.confidence_level,
        max_half_width=spec.max_sensitivity_ci_half_width,
    )
    specificity_n = minimum_binomial_sample_size(
        expected_rate=spec.expected_specificity,
        confidence_level=spec.confidence_level,
        max_half_width=spec.max_specificity_ci_half_width,
    )
    auc_positive_n, auc_negative_n, auc_se, auc_half_width, auc_power = minimum_auc_group_sizes(
        target_auc=spec.target_auc,
        null_auc=spec.null_auc,
        confidence_level=spec.confidence_level,
        target_power=spec.target_power,
        max_ci_half_width=spec.max_auc_ci_half_width,
        control_to_case_ratio=spec.control_to_case_ratio,
    )
    required_positive = max(sensitivity_n, auc_positive_n)
    required_negative = max(specificity_n, auc_negative_n)
    required_total = required_positive + required_negative
    verified = _required_int(proxy, "verified_shot_artifact_count")
    unstable = _required_int(proxy, "unstable_shot_count")
    blockers = [
        "design_parameters_require_owner_and_statistical_ratification",
        "authoritative_label_manifest_not_supplied",
        "diagnostic_availability_manifest_not_supplied",
        "distribution_informed_matching_calipers_not_supplied",
        "proxy_labels_are_not_cohort_admissible",
    ]
    if unstable:
        blockers.append("proxy_outcomes_are_parameter_sensitive")
    if verified < required_total:
        blockers.append("verified_proxy_population_is_smaller_than_design_minimum")

    report: dict[str, object] = {
        "schema_version": COHORT_DESIGN_REPORT_SCHEMA,
        "status": "blocked",
        "blockers": blockers,
        "design_spec": {
            "schema_version": COHORT_DESIGN_SCHEMA,
            "file_sha256": design_file_digest,
            "payload_sha256": design_digest,
            "status": "draft_owner_ratification_required",
        },
        "source_proxy_audit": {
            "schema_version": AUDIT_SCHEMA,
            "file_sha256": proxy_file_digest,
            "payload_sha256": proxy_digest,
            "dataset_id": _required_string(proxy, "dataset_id"),
            "verified_shot_count": verified,
            "stable_shot_count": _required_int(proxy, "stable_shot_count"),
            "unstable_shot_count": unstable,
            "capacity_only": True,
            "label_authority_admission": False,
        },
        "statistical_design": {
            "confidence_level": spec.confidence_level,
            "target_power": spec.target_power,
            "binomial_precision": {
                "method": "Wilson score interval at predeclared expected rate",
                "sensitivity": {
                    "expected_rate": spec.expected_sensitivity,
                    "max_ci_half_width": spec.max_sensitivity_ci_half_width,
                    "minimum_positive_count": sensitivity_n,
                },
                "specificity": {
                    "expected_rate": spec.expected_specificity,
                    "max_ci_half_width": spec.max_specificity_ci_half_width,
                    "minimum_negative_count": specificity_n,
                },
            },
            "auc_planning": {
                "method": "Hanley-McNeil large-sample SE plus normal-approximation one-sided power",
                "approximation_only": True,
                "target_auc": spec.target_auc,
                "null_auc": spec.null_auc,
                "max_ci_half_width": spec.max_auc_ci_half_width,
                "control_to_case_ratio": spec.control_to_case_ratio,
                "minimum_positive_count": auc_positive_n,
                "minimum_negative_count": auc_negative_n,
                "standard_error_at_minimum": auc_se,
                "ci_half_width_at_minimum": auc_half_width,
                "power_at_minimum": auc_power,
            },
            "required_positive_count": required_positive,
            "required_negative_count": required_negative,
            "required_total_count": required_total,
        },
        "matching_contract": {
            "admissible_label_authorities": list(spec.admissible_label_authorities),
            "forbidden_label_authorities": ["ip_proxy", "programme_metadata"],
            "excluded_outcomes": list(EXCLUDED_OUTCOMES),
            "required_channels": list(spec.required_channels),
            "exact_match_fields": list(spec.exact_match_fields),
            "caliper_policy_status": "required_not_supplied",
            "matching_with_replacement": False,
            "post_outcome_matching_variables_permitted": False,
        },
        "population_comparison": {
            "verified_proxy_shot_count": verified,
            "required_total_count": required_total,
            "nominal_count_shortfall": max(required_total - verified, 0),
            "interpretation": "capacity comparison only; proxy outcomes do not populate either cohort class",
        },
        "admitted_shot_ids": [],
        "split_status": "not_started_l2f24",
        "claim_boundary": {
            "scientific_validation": False,
            "independent_label_validation": False,
            "cohort_admission": False,
            "training_admission": False,
            "facility_prediction": False,
            "control_admission": False,
        },
        "generated_at": generated_at,
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-spec", type=Path, required=True)
    parser.add_argument("--proxy-sensitivity", type=Path, required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run L2F-23a and write the deterministic design report."""
    args = _parse_args(argv)
    report = design_matched_cohort(
        design_spec_path=args.design_spec,
        proxy_sensitivity_path=args.proxy_sensitivity,
        generated_at=args.generated_at,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    design = cast(Mapping[str, object], report["statistical_design"])
    print(
        "matched-cohort design blocked: "
        f"minimum={design['required_total_count']} "
        f"verified_proxy={cast(Mapping[str, object], report['source_proxy_audit'])['verified_shot_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
