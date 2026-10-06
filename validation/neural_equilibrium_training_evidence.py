# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium trainer
"""Validate launch/result declarations and bind result templates to a launch digest.

Validators return the supplied mapping on success. Null-field SHA-256 binds
finite JSON consistency, not authenticity. Supported result section schemas,
typed policy/diagnostic declarations and exact launch/dataset bindings must
remain intact. Templates never establish measured latency, cost or admission.

Non-object inputs refuse through the same public validators used by writers:

>>> validate_training_report([])
Traceback (most recent call last):
...
ValueError: training report must be an object and require_executed a boolean
>>> validate_result_templates([])
Traceback (most recent call last):
...
ValueError: result templates must be an object
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from validation.neural_equilibrium_training_inputs import (
    EXECUTION_HOST_POLICY,
    REQUIRED_ADMISSION_CERTIFICATE_FIELDS,
    REQUIRED_GPU_COST_FIELDS,
    REQUIRED_HOLDOUT_METRICS,
    REQUIRED_LATENCY_FIELDS,
    RESULT_TEMPLATES_SCHEMA,
    SPLITS,
    STORAGE_OUTPUT_ROOTS,
    TARGET_KEYS,
    TRAINING_SCHEMA,
    _path_is_relative_to,
    _sha256_file,
)
from validation.report_output_paths import refuse_link_loop


def _sha256_json(payload: dict[str, Any]) -> str:
    """Hash finite sorted compact ASCII JSON; unsupported/nonfinite/deep values refuse."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    """Recognise only lowercase64hex strings, without type or whitespace coercion."""
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _payload_digest(payload: dict[str, Any]) -> str:
    """Recompute the launch/template self-digest with its hash field bound to literal null."""
    return _sha256_json({**payload, "payload_sha256": None})


def _require(condition: bool, field: str, message: str, errors: list[dict[str, str]]) -> None:
    """Append an authored field diagnostic when a declaration condition is false."""
    if not condition:
        errors.append({"field": field, "error": message})


def _require_string_members(
    values: Any,
    expected: tuple[str, ...],
    field: str,
    errors: list[dict[str, str]],
) -> None:
    """Require distinct trimmed string members and every expected contract entry, retaining unknown extras."""
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        errors.append({"field": field, "error": "must be a list of strings"})
        return
    if len(set(values)) != len(values) or any(not item or item != item.strip() for item in values):
        errors.append({"field": field, "error": "must contain distinct nonempty trimmed strings"})
    missing = [item for item in expected if item not in values]
    if missing:
        errors.append({"field": field, "error": f"missing required entries: {', '.join(missing)}"})


def build_result_templates(report: dict[str, Any]) -> dict[str, Any]:
    """Validate a launch and describe future result schemas bound to its digest/dataset.

    The returned fresh mapping is a declaration template, not measured holdout,
    latency, cost or predictive/facility admission evidence. Launch self-digests
    must match; a template cannot repair or promote an invalid launch.
    """
    validate_training_report(report)
    templates: dict[str, Any] = {
        "schema_version": RESULT_TEMPLATES_SCHEMA,
        "claim_boundary": (
            "These are result schemas for a later admitted compute run. They are not executed training evidence."
        ),
        "training_report_payload_sha256": report["payload_sha256"],
        "expected_dataset_sha256": report["dataset_sha256"],
        "expected_weight_path_policy": "weights are written to workstation or external cloud compute storage, not storage-host dataset storage",
        "holdout_metrics": {
            "schema_version": "scpn-control.mast-efm-neural-equilibrium-holdout-metrics.v1",
            "required_splits": list(SPLITS),
            "required_metrics": list(REQUIRED_HOLDOUT_METRICS),
            "acceptance_policy": (
                "compact train, validation, and test metrics must be emitted before predictive admission is requested"
            ),
        },
        "latency_metrics": {
            "schema_version": "scpn-control.mast-efm-neural-equilibrium-latency-metrics.v1",
            "required_fields": list(REQUIRED_LATENCY_FIELDS),
            "acceptance_policy": "latency is evidence only after hardware, precision, batch size, and sample count are recorded",
        },
        "gpu_cost": {
            "schema_version": "scpn-control.mast-efm-neural-equilibrium-gpu-cost.v1",
            "required_fields": list(REQUIRED_GPU_COST_FIELDS),
            "acceptance_policy": "cost reports must distinguish planning estimates from measured billing evidence",
        },
        "admission_certificate": {
            "schema_version": "scpn-control.mast-efm-neural-equilibrium-admission-certificate.v1",
            "required_fields": list(REQUIRED_ADMISSION_CERTIFICATE_FIELDS),
            "admission_status_enum": ["blocked", "pass", "fail"],
            "acceptance_policy": (
                "certificate stays blocked until the strict neural-equilibrium reference gate admits the exact weights"
            ),
        },
    }
    templates["payload_sha256"] = _sha256_json({**templates, "payload_sha256": None})
    return templates


def validate_training_report(report: dict[str, Any], *, require_executed: bool = False) -> dict[str, Any]:
    """Validate finite launch declarations, policy and source/compute consistency.

    Returns the supplied mapping unchanged on success; malformed/tampered fields
    raise ValueError. Execute-mode metrics must be present and null or finite
    nonnegative values; null means unobserved, not a passing tolerance. Dry-run
    reports cannot declare weights/holdout results. A self-digest is consistency,
    not authenticity. require_executed additionally requires actual selected local
    weight bytes matching SHA-256; other validation is declaration-only. Scientific
    admission_ready and strict_artefact_emitted must remain literally False.
    """
    if not isinstance(report, dict) or not isinstance(require_executed, bool):
        raise ValueError("training report must be an object and require_executed a boolean")
    try:
        _sha256_json(report)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"training report must be finite JSON: {exc}") from exc
    errors: list[dict[str, str]] = []
    _require(report.get("schema_version") == TRAINING_SCHEMA, "schema_version", "unsupported schema_version", errors)
    _require(report.get("status") in ("prepared", "executed"), "status", "must be prepared or executed", errors)
    _require(
        report.get("execution_mode") in ("dry_run", "execute"), "execution_mode", "must be dry_run or execute", errors
    )
    _require(_is_sha256(report.get("payload_sha256")), "payload_sha256", "must be a SHA-256 hex digest", errors)
    if _is_sha256(report.get("payload_sha256")):
        _require(
            report["payload_sha256"] == _payload_digest(report),
            "payload_sha256",
            "does not match canonical report payload",
            errors,
        )
    _require(_is_sha256(report.get("dataset_sha256")), "dataset_sha256", "must be a SHA-256 hex digest", errors)
    _require(
        isinstance(report.get("claim_boundary"), str)
        and "not predictive EFIT/P-EFIT admission evidence" in report["claim_boundary"],
        "claim_boundary",
        "must preserve predictive admission block",
        errors,
    )
    _require(
        report.get("execution_host_policy") == EXECUTION_HOST_POLICY,
        "execution_host_policy",
        "must preserve storage-host storage-only policy",
        errors,
    )
    _require(report.get("admission_ready") is False, "admission_ready", "launch report cannot self-admit", errors)
    _require(
        report.get("strict_artefact_emitted") is False,
        "strict_artefact_emitted",
        "strict reference artefact must remain false in launch report",
        errors,
    )
    _require_string_members(report.get("required_targets"), TARGET_KEYS, "required_targets", errors)

    path_text = report.get("weights_path")
    if not isinstance(path_text, str) or not path_text or path_text != path_text.strip() or "\x00" in path_text:
        errors.append({"field": "weights_path", "error": "must declare a nonempty trimmed weights path without NUL"})
    else:
        weights_path = Path(path_text)
        try:
            refuse_link_loop(weights_path)
            weights_path.resolve()
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append({"field": "weights_path", "error": f"cannot resolve weights path: {exc}"})
        _require(weights_path.suffix == ".npz", "weights_path", "must declare a .npz output", errors)
        _require(
            not any(_path_is_relative_to(weights_path, root) for root in STORAGE_OUTPUT_ROOTS),
            "weights_path",
            "must not write weights under storage-host dataset storage",
            errors,
        )
    pre_run = report.get("pre_run_admission")
    if not isinstance(pre_run, dict):
        errors.append({"field": "pre_run_admission", "error": "must be an object"})
        pre_run = {}
    else:
        _require(
            pre_run.get("required_for_execute") is True,
            "pre_run_admission.required_for_execute",
            "must be true",
            errors,
        )
        _require(
            isinstance(pre_run.get("errors"), list),
            "pre_run_admission.errors",
            "must list pre-run admission errors",
            errors,
        )

    _require(
        isinstance(report.get("dataset_exists_on_this_host"), bool),
        "dataset_exists_on_this_host",
        "must be a boolean",
        errors,
    )
    observed = pre_run.get("dataset_sha256_verified")
    _require(isinstance(observed, bool), "pre_run_admission.dataset_sha256_verified", "must be a boolean", errors)
    if observed is True:
        _require(
            report.get("dataset_exists_on_this_host") is True,
            "dataset_exists_on_this_host",
            "verified local digest requires local dataset",
            errors,
        )
    aggregate: list[str] = []
    for section in ("source_provenance", "compute_execution"):
        child = pre_run.get(section)
        if (
            not isinstance(child, dict)
            or child.get("status") not in ("pass", "fail")
            or not isinstance(child.get("errors"), list)
            or any(not isinstance(v, str) for v in child["errors"])
        ):
            errors.append(
                {"field": f"pre_run_admission.{section}", "error": "must declare pass/fail and string errors"}
            )
            continue
        _require(
            (child["status"] == "pass") == (child["errors"] == []),
            f"pre_run_admission.{section}",
            "status must match errors",
            errors,
        )
        aggregate.extend(child["errors"])
    _require(
        pre_run.get("errors") == aggregate,
        "pre_run_admission.errors",
        "must retain source and compute errors",
        errors,
    )
    _require(
        pre_run.get("status") == ("pass" if not aggregate else "fail"),
        "pre_run_admission.status",
        "must match source and compute errors",
        errors,
    )
    if report.get("execution_mode") == "execute":
        _require(report.get("status") == "executed", "status", "execute mode must have executed status", errors)
        _require(pre_run.get("status") == "pass", "pre_run_admission.status", "execute mode requires pass", errors)
        _require(
            pre_run.get("dataset_sha256_verified") is True,
            "pre_run_admission.dataset_sha256_verified",
            "execute requires verified local SHA-256",
            errors,
        )
        _require(
            report.get("dataset_exists_on_this_host") is True,
            "dataset_exists_on_this_host",
            "execute requires dataset",
            errors,
        )
        _require(_is_sha256(report.get("weights_sha256")), "weights_sha256", "execute requires weights digest", errors)
        holdout = report.get("holdout_metrics")
        if not isinstance(holdout, dict):
            errors.append({"field": "holdout_metrics", "error": "execute requires holdout metrics"})
        else:
            for split in SPLITS:
                split_metrics = holdout.get(split)
                if not isinstance(split_metrics, dict):
                    errors.append({"field": f"holdout_metrics.{split}", "error": "missing split metrics"})
                    continue
                for metric in REQUIRED_HOLDOUT_METRICS:
                    value = split_metrics.get(metric)
                    _require(
                        metric in split_metrics and (value is None or _finite_nonnegative_number(value)),
                        f"holdout_metrics.{split}.{metric}",
                        "must be present and null or finite nonnegative numeric",
                        errors,
                    )
    else:
        _require(report.get("status") == "prepared", "status", "dry_run mode must have prepared status", errors)
        _require(report.get("weights_sha256") is None, "weights_sha256", "dry_run must not declare weights", errors)
        _require(
            report.get("holdout_metrics") is None, "holdout_metrics", "dry_run must not declare holdout metrics", errors
        )

    if require_executed:
        _require(report.get("execution_mode") == "execute", "execution_mode", "executed report required", errors)

    if errors:
        raise ValueError("; ".join(f"{error['field']}: {error['error']}" for error in errors))
    if require_executed:
        weights = Path(report["weights_path"])
        if not weights.is_file() or _sha256_file(weights) != report["weights_sha256"]:
            raise ValueError("executed weights_path must exist and match weights_sha256 on this host")
    return report


def validate_result_templates(
    templates: dict[str, Any],
    *,
    training_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate finite future-result schema declarations and optional launch binding.

    Required distinct string members, exact splits/status enumeration, self-digest
    and claim/output-policy boundaries are enforced. The supplied mapping is
    returned unchanged; unsupported/invalid JSON or bindings raise ValueError.
    This does not generate or validate actual latency/billing/admission results.
    """
    if not isinstance(templates, dict):
        raise ValueError("result templates must be an object")
    try:
        _sha256_json(templates)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"result templates must be finite JSON: {exc}") from exc
    errors: list[dict[str, str]] = []
    _require(
        templates.get("schema_version") == RESULT_TEMPLATES_SCHEMA,
        "schema_version",
        "unsupported schema_version",
        errors,
    )
    _require(_is_sha256(templates.get("payload_sha256")), "payload_sha256", "must be a SHA-256 hex digest", errors)
    if _is_sha256(templates.get("payload_sha256")):
        _require(
            templates["payload_sha256"] == _payload_digest(templates),
            "payload_sha256",
            "does not match canonical template payload",
            errors,
        )
    _require(
        _is_sha256(templates.get("training_report_payload_sha256")),
        "training_report_payload_sha256",
        "must be a SHA-256 hex digest",
        errors,
    )
    _require(
        _is_sha256(templates.get("expected_dataset_sha256")),
        "expected_dataset_sha256",
        "must be a SHA-256 hex digest",
        errors,
    )
    _require(
        isinstance(templates.get("expected_weight_path_policy"), str)
        and "not storage-host dataset storage" in templates["expected_weight_path_policy"],
        "expected_weight_path_policy",
        "must forbid storage-host dataset storage weight output",
        errors,
    )
    _require(
        isinstance(templates.get("claim_boundary"), str)
        and "not executed training evidence" in templates["claim_boundary"],
        "claim_boundary",
        "must preserve non-executed boundary",
        errors,
    )

    template_specs = {
        "holdout_metrics": ("required_metrics", REQUIRED_HOLDOUT_METRICS),
        "latency_metrics": ("required_fields", REQUIRED_LATENCY_FIELDS),
        "gpu_cost": ("required_fields", REQUIRED_GPU_COST_FIELDS),
        "admission_certificate": ("required_fields", REQUIRED_ADMISSION_CERTIFICATE_FIELDS),
    }
    for section, (key, expected) in template_specs.items():
        payload = templates.get(section)
        if not isinstance(payload, dict):
            errors.append({"field": section, "error": "must be an object"})
            continue
        _require(
            payload.get("schema_version") == f"scpn-control.mast-efm-neural-equilibrium-{section.replace('_', '-')}.v1",
            f"{section}.schema_version",
            "unsupported result schema",
            errors,
        )
        _require(
            isinstance(payload.get("acceptance_policy"), str) and bool(payload["acceptance_policy"]),
            f"{section}.acceptance_policy",
            "must be a non-empty string",
            errors,
        )
        _require_string_members(payload.get(key), expected, f"{section}.{key}", errors)

    holdout = templates.get("holdout_metrics")
    if isinstance(holdout, dict):
        _require(
            holdout.get("required_splits") == list(SPLITS),
            "holdout_metrics.required_splits",
            "must preserve train/validation/test",
            errors,
        )
    certificate = templates.get("admission_certificate", {})
    if isinstance(certificate, dict):
        _require(
            certificate.get("admission_status_enum") == ["blocked", "pass", "fail"],
            "admission_certificate.admission_status_enum",
            "must be ['blocked', 'pass', 'fail']",
            errors,
        )

    if training_report is not None:
        validate_training_report(training_report)
        _require(
            templates.get("training_report_payload_sha256") == training_report.get("payload_sha256"),
            "training_report_payload_sha256",
            "does not match launch report",
            errors,
        )
        _require(
            templates.get("expected_dataset_sha256") == training_report.get("dataset_sha256"),
            "expected_dataset_sha256",
            "does not match launch report",
            errors,
        )

    if errors:
        raise ValueError("; ".join(f"{error['field']}: {error['error']}" for error in errors))
    return templates


def _finite_nonnegative_number(value: Any) -> bool:
    """Admit a genuine finite nonnegative metric, excluding booleans and overflowing integers."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value)) and value >= 0
    except OverflowError:
        return False
