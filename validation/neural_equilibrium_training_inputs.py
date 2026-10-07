# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium trainer
"""Validate training controls, plan/dataset metadata and declared execution provenance.

Campaign bindings compare consumed metadata fields without numeric coercion.
Source digests are consistency checks, not authenticated measurements; blocked,
missing or malformed source declarations retain FAIL admission diagnostics.
Explicit controls are validated without reading selected paths:

>>> inputs = TrainingInputs(Path("dataset.json"), Path("plan.json"), Path("dataset.npz"), Path("weights.npz"))
>>> inputs.execute, inputs.compute_host_kind
(False, 'unspecified')
>>> TrainingInputs(Path("dataset.json"), Path("plan.json"), Path("dataset.npz"), Path("weights.bin"))
Traceback (most recent call last):
...
ValueError: weights_out must have an explicit .npz suffix
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from validation.mast_efm_feature_audit_reporting import validate_audit_dataset_bindings
from validation.mast_efm_original_source_policy import AUDIT_SCHEMA as ORIGINAL_AUDIT_SCHEMA
from validation.mast_efm_original_source_reporting import validate_original_audit_bindings
from validation.neural_equilibrium_campaign_inputs import canonical_campaign_digest
from validation.neural_equilibrium_dataset_contracts import DATASET_SCHEMA as DATASET_SCHEMA
from validation.neural_equilibrium_dataset_contracts import FEATURE_NAMES as FEATURE_NAMES
from validation.neural_equilibrium_dataset_contracts import TARGET_KEYS as TARGET_KEYS
from validation.plan_neural_equilibrium_training_campaign import REPORT_SCHEMA as _PLAN_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PLAN_SCHEMA: str = _PLAN_SCHEMA
TRAINING_SCHEMA = "scpn-control.mast-efm-neural-equilibrium-training.v1"
RESULT_TEMPLATES_SCHEMA = "scpn-control.mast-efm-neural-equilibrium-result-templates.v1"
EXECUTION_HOST_POLICY = (
    "The storage host is storage-only; execute training only on this workstation or external cloud compute with the storage-host dataset "
    "mounted read-only or copied to admitted compute storage."
)
DEFAULT_DATASET_REPORT = ROOT / "validation" / "reports" / "mast_efm_neural_equilibrium_dataset.json"
DEFAULT_CAMPAIGN_PLAN = ROOT / "validation" / "reports" / "neural_equilibrium_training_campaign_plan.json"
DEFAULT_FEATURE_PROVENANCE_REPORT = ROOT / "validation" / "reports" / "mast_efm_feature_provenance_audit.json"
DEFAULT_ORIGINAL_SOURCE_REPORT = ROOT / "validation" / "reports" / "mast_efm_original_feature_source_audit.json"
DEFAULT_DATASET_PATH = Path("/data/SCPN-CONTROL/processed/neural_equilibrium/mast_efm_supervised_dataset.npz")
DEFAULT_WEIGHTS_OUT = Path("artifacts/neural_equilibrium/mast_efm_full_output_baseline_weights.npz")
DEFAULT_JSON_OUT = ROOT / "validation" / "reports" / "mast_efm_neural_equilibrium_training_launch.json"
DEFAULT_MD_OUT = ROOT / "validation" / "reports" / "mast_efm_neural_equilibrium_training_launch.md"
DEFAULT_TEMPLATES_JSON_OUT = ROOT / "validation" / "reports" / "mast_efm_neural_equilibrium_result_templates.json"
DEFAULT_TEMPLATES_MD_OUT = ROOT / "validation" / "reports" / "mast_efm_neural_equilibrium_result_templates.md"
SPLITS = ("train", "validation", "test")
ADMITTED_COMPUTE_HOST_KINDS = ("workstation", "external_cloud")
STORAGE_ONLY_HOST_MARKERS = ("storage_host",)
STORAGE_OUTPUT_ROOTS = (
    Path("/data"),
    Path("/data/SCPN-CONTROL"),
)
REQUIRED_HOLDOUT_METRICS = (
    "psi_rmse_Wb_per_rad",
    "pprime_rmse_Pa_per_Wb_rad",
    "q_profile_rmse",
    "lcfs_r_rmse_m",
    "lcfs_z_rmse_m",
    "magnetic_axis_rmse_m",
)
REQUIRED_LATENCY_FIELDS = (
    "hardware_label",
    "accelerator_kind",
    "precision",
    "batch_size",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "sample_count",
)
REQUIRED_GPU_COST_FIELDS = (
    "compute_provider",
    "gpu_model",
    "gpu_count",
    "wall_time_hours",
    "gpu_hours",
    "storage_gb",
    "currency",
    "estimated_cost",
)
REQUIRED_ADMISSION_CERTIFICATE_FIELDS = (
    "dataset_sha256",
    "weights_sha256",
    "training_report_sha256",
    "holdout_metrics_sha256",
    "latency_metrics_sha256",
    "source_provenance_payload_sha256",
    "strict_reference_report_sha256",
    "admission_status",
)


@dataclass(frozen=True)
class TrainingInputs:
    """Select real report/tensor/weight paths and literal execution controls.

    Dataset metadata must remain predictive-blocked and its campaign/source
    bindings are validated before fitting. Copied compute tensor locations may
    differ from planned storage; selected bytes must match dataset SHA. Execute
    is explicit and never implied by host kind. Ridge alpha is finite/positive,
    component count a positive genuine integer and output has an explicit .npz
    suffix. Missing physical source tensors cannot establish scientific admission.
    """

    dataset_report: Path
    campaign_plan: Path
    dataset_path: Path
    weights_out: Path
    feature_provenance_report: Path = DEFAULT_FEATURE_PROVENANCE_REPORT
    original_source_report: Path = DEFAULT_ORIGINAL_SOURCE_REPORT
    compute_host_kind: str = "unspecified"
    compute_host_label: str = ""
    execute: bool = False
    ridge_alpha: float = 1.0e-6
    max_flux_components: int = 32

    def __post_init__(self) -> None:
        """Refuse coerced controls, invalid conditioning and non-NPZ weight outputs before IO."""
        for field in (
            "dataset_report",
            "campaign_plan",
            "dataset_path",
            "weights_out",
            "feature_provenance_report",
            "original_source_report",
        ):
            value = getattr(self, field)
            if not isinstance(value, Path) or "\x00" in str(value):
                raise ValueError(f"{field} must be a Path without NUL")
        if self.weights_out.suffix != ".npz":
            raise ValueError("weights_out must have an explicit .npz suffix")
        if not isinstance(self.execute, bool):
            raise ValueError("execute must be a boolean")
        if isinstance(self.ridge_alpha, bool) or not isinstance(self.ridge_alpha, int | float):
            raise ValueError("ridge_alpha must be finite and strictly positive")
        try:
            alpha = float(self.ridge_alpha)
        except OverflowError as exc:
            raise ValueError("ridge_alpha must be finite and strictly positive") from exc
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("ridge_alpha must be finite and strictly positive")
        if (
            isinstance(self.max_flux_components, bool)
            or not isinstance(self.max_flux_components, int)
            or self.max_flux_components <= 0
        ):
            raise ValueError("max_flux_components must be a positive integer")
        if not isinstance(self.compute_host_kind, str) or self.compute_host_kind not in {
            "unspecified",
            *ADMITTED_COMPUTE_HOST_KINDS,
        }:
            raise ValueError("compute_host_kind must be unspecified, workstation or external_cloud")
        if not isinstance(self.compute_host_label, str) or self.compute_host_label != self.compute_host_label.strip():
            raise ValueError("compute_host_label must be a trimmed string")


def _sha256_file(path: Path) -> str:
    """Stream one selected local byte file into SHA-256; IO failures propagate to the public caller."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    """Read finite duplicate-free UTF-8 metadata and translate supported read/depth failures."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Refuse duplicate keys at every JSON object depth."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
        if not isinstance(payload, dict):
            raise ValueError("metadata root must be an object")
        json.dumps(payload, allow_nan=False)
    except (OSError, ValueError, RecursionError, RuntimeError) as exc:
        raise ValueError(f"cannot read training metadata {path}: {exc}") from exc
    return payload


def _validate_reports(dataset_report: dict[str, Any], campaign_plan: dict[str, Any]) -> None:
    """Require a self-consistent prepared plan whose MAST and acquisition bindings match the dataset.

    Dataset metadata has passed read_campaign_dataset_report. Copied/mounted
    compute dataset paths may differ from the planned storage locator; its
    selected bytes still must match dataset SHA. This is declaration admission,
    not authentication of measurement provenance or physical source authority.
    """
    if campaign_plan.get("schema_version") != CAMPAIGN_PLAN_SCHEMA:
        raise ValueError("campaign plan has unsupported schema_version")
    if campaign_plan.get("status") != "prepared":
        raise ValueError("campaign plan must be prepared before training")
    if campaign_plan.get("payload_sha256") != canonical_campaign_digest(campaign_plan):
        raise ValueError("campaign plan payload_sha256 does not match its contents")
    mast = campaign_plan.get("mast_efm_dataset")
    package = campaign_plan.get("compute_execution_package")
    if not isinstance(mast, dict) or not isinstance(package, dict):
        raise ValueError("campaign plan must declare MAST dataset and compute package")
    for field in (
        "reference_dataset_id",
        "equilibria_count",
        "grid_shape",
        "split_counts",
        "fallback_features",
        "ragged_target_policy",
    ):
        if json.dumps(mast.get(field), sort_keys=True) != json.dumps(dataset_report[field], sort_keys=True):
            raise ValueError(f"campaign plan {field} does not match the dataset report")
    payload = mast.get("payload")
    if not isinstance(payload, dict) or payload.get("sha256") != dataset_report["dataset_sha256"]:
        raise ValueError("campaign plan dataset payload SHA-256 does not match the dataset report")
    if (
        package.get("dataset_sha256") != dataset_report["dataset_sha256"]
        or package.get("status") != "prepared_not_executed"
    ):
        raise ValueError("campaign compute package must preserve prepared state and dataset SHA-256")
    if package.get("admitted_compute_host_kinds") != list(ADMITTED_COMPUTE_HOST_KINDS) or package.get(
        "forbidden_training_hosts"
    ) != ["storage host"]:
        raise ValueError("campaign compute package must preserve admitted host and storage-only policy")
    lanes = campaign_plan.get("prepared_dataset_lanes")
    if not isinstance(lanes, list) or any(not isinstance(lane, dict) for lane in lanes):
        raise ValueError("campaign dataset lanes must be objects")
    public_lanes = [lane for lane in lanes if lane.get("id") == "qlknn_qualikiz_neural_transport"]
    if len(public_lanes) != 1:
        raise ValueError("campaign must declare exactly one public acquisition lane")
    summary = public_lanes[0].get("public_data_summary")
    if not isinstance(summary, dict) or summary.get("status") != "pass":
        raise ValueError("campaign public-data acquisition must pass before training preparation")

    for field in ("records", "files", "local_files", "deferred_files", "deferred_bytes"):
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"campaign public acquisition {field} must be a nonnegative integer")
    if (
        summary["records"] < 1
        or summary["files"] < 1
        or summary["files"] != summary["local_files"] + summary["deferred_files"]
    ):
        raise ValueError("campaign public acquisition counters must declare nonempty consistent coverage")
    manifests = summary.get("manifests")
    if (
        not isinstance(manifests, list)
        or len(manifests) != summary["records"]
        or any(not isinstance(m, dict) for m in manifests)
    ):
        raise ValueError("campaign public acquisition manifest count must match records")


def _path_is_relative_to(path: Path, root: Path) -> bool:
    """Test canonical containment for storage output policy; resolution failure is not proof of membership."""
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, ValueError, RuntimeError):
        return False
    return True


def _display_path(path: Path) -> str:
    """Render contained repository paths relatively and preserve external path spellings."""
    if not path.is_absolute():
        return str(path)
    try:
        return path.resolve(strict=False).relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _validate_feature_provenance(
    report: dict[str, Any], dataset_report: dict[str, Any], *, dataset_report_sha256: str
) -> list[str]:
    """Retain source diagnostics and require full audit bindings to the exact captured dataset declaration."""
    errors: list[str] = []
    if report.get("payload_sha256") != canonical_campaign_digest(report):
        errors.append("provenance payload_sha256 does not match its contents")
    if report.get("schema_version") != "scpn-control.mast-efm-feature-provenance-audit.v1":
        errors.append("feature provenance report has unsupported schema_version")
    if report.get("reference_dataset_id") != dataset_report.get("reference_dataset_id"):
        errors.append("feature provenance report does not match the dataset reference_dataset_id")
    if report.get("blocked_features") != []:
        errors.append("feature provenance report still has blocked features")
    feature_status = report.get("feature_status")
    if not isinstance(feature_status, dict) or not feature_status:
        errors.append("feature provenance report has no feature_status entries")
    else:
        if not {"Ip_MA", "Bt_T", "ffprime_scale"}.issubset(feature_status):
            errors.append("feature provenance must declare all three sourced features")
        unresolved = [
            str(name)
            for name, status in feature_status.items()
            if not isinstance(status, dict) or status.get("status") != "resolved"
        ]
        if unresolved:
            errors.append(f"feature provenance report has unresolved features: {', '.join(sorted(unresolved))}")
    try:
        validate_audit_dataset_bindings(report, dataset_report, dataset_report_sha256=dataset_report_sha256)
    except ValueError as exc:
        errors.append(f"feature provenance bindings invalid: {exc}")
    return errors


def _validate_original_source_provenance(
    report: dict[str, Any], dataset_report: dict[str, Any], *, dataset_report_sha256: str
) -> list[str]:
    """Retain readiness diagnostics and consume complete v2 original/converted dataset bindings."""
    errors: list[str] = []
    if report.get("payload_sha256") != canonical_campaign_digest(report):
        errors.append("provenance payload_sha256 does not match its contents")
    if report.get("schema_version") != ORIGINAL_AUDIT_SCHEMA:
        errors.append("original source report has unsupported schema_version")
    if report.get("reference_dataset_id") != dataset_report.get("reference_dataset_id"):
        errors.append("original source report does not match the dataset reference_dataset_id")
    if report.get("status") != "source_ready":
        errors.append("original source report is not source_ready")
    if report.get("can_rebuild_dataset_now") is not True:
        errors.append("original source report does not admit rebuild readiness")
    if report.get("blocked_features") != []:
        errors.append("original source report still has blocked features")
    try:
        validate_original_audit_bindings(report, dataset_report, dataset_report_sha256=dataset_report_sha256)
    except ValueError as exc:
        errors.append(f"original source bindings invalid: {exc}")
    return errors


def _source_provenance_admission(
    inputs: TrainingInputs, dataset_report: dict[str, Any], *, dataset_report_sha256: str
) -> dict[str, Any]:
    """Inspect actual audit files and exact selected dataset bindings, retaining invalid source diagnostics as FAIL."""
    errors: list[str] = []
    reports: list[dict[str, Any] | None] = []
    if dataset_report["fallback_features"]:
        errors.append("dataset report still declares fallback features")
    validators: list[tuple[Path, Callable[[dict[str, Any], dict[str, Any]], list[str]], str]] = [
        (
            inputs.feature_provenance_report,
            partial(_validate_feature_provenance, dataset_report_sha256=dataset_report_sha256),
            "feature provenance",
        ),
        (
            inputs.original_source_report,
            partial(_validate_original_source_provenance, dataset_report_sha256=dataset_report_sha256),
            "original source",
        ),
    ]
    for path, validate, label in validators:
        report = None
        if path.is_file():
            try:
                report = _load_json_object(path)
                errors.extend(validate(report, dataset_report))
            except ValueError as exc:
                errors.append(f"{label} report invalid: {exc}")
        else:
            errors.append(f"{label} report is missing: {path}")
        reports.append(report)
    feature_report, original_report = reports
    return {
        "status": "pass" if not errors else "fail",
        "feature_provenance_report": _display_path(inputs.feature_provenance_report),
        "feature_provenance_payload_sha256": None if feature_report is None else feature_report.get("payload_sha256"),
        "original_source_report": _display_path(inputs.original_source_report),
        "original_source_payload_sha256": None if original_report is None else original_report.get("payload_sha256"),
        "errors": errors,
    }


def _compute_execution_admission(inputs: TrainingInputs, dataset_sha256: str | None) -> dict[str, Any]:
    """Report explicit host, verified local digest, output-root and dataset-overwrite refusals before fitting."""
    errors: list[str] = []
    host_label = inputs.compute_host_label or socket.gethostname()
    host_label_lower = host_label.lower()
    if inputs.compute_host_kind not in ADMITTED_COMPUTE_HOST_KINDS:
        errors.append("compute host kind must be explicitly declared as workstation or external_cloud before --execute")
    if any(marker in host_label_lower for marker in STORAGE_ONLY_HOST_MARKERS):
        errors.append("The storage host is storage-only and is not an admitted training host")
    if dataset_sha256 is None:
        errors.append("dataset payload SHA-256 must be verified before --execute")
    if any(_path_is_relative_to(inputs.weights_out, root) for root in STORAGE_OUTPUT_ROOTS):
        errors.append(
            "weights_out must not be under storage-host dataset storage; use workstation or cloud compute storage"
        )
    try:
        for selected in (
            inputs.dataset_path,
            inputs.dataset_report,
            inputs.campaign_plan,
            inputs.feature_provenance_report,
            inputs.original_source_report,
        ):
            if inputs.weights_out.resolve() == selected.resolve() or (
                inputs.weights_out.exists() and selected.exists() and inputs.weights_out.samefile(selected)
            ):
                errors.append("weights_out must not overwrite the selected dataset or training metadata")
    except (OSError, ValueError, RuntimeError) as exc:
        errors.append(f"cannot resolve training output custody: {exc}")
    return {
        "status": "pass" if not errors else "fail",
        "compute_host_kind": inputs.compute_host_kind,
        "compute_host_label": host_label,
        "admitted_compute_host_kinds": list(ADMITTED_COMPUTE_HOST_KINDS),
        "storage_only_host_markers": list(STORAGE_ONLY_HOST_MARKERS),
        "forbidden_output_roots": [str(root) for root in STORAGE_OUTPUT_ROOTS],
        "errors": errors,
    }


def _pre_run_admission(
    inputs: TrainingInputs,
    dataset_report: dict[str, Any],
    dataset_sha256: str | None,
    *,
    dataset_report_sha256: str,
) -> dict[str, Any]:
    """Aggregate source and compute observations; actual fitting requires their complete PASS."""
    source = _source_provenance_admission(inputs, dataset_report, dataset_report_sha256=dataset_report_sha256)
    compute = _compute_execution_admission(inputs, dataset_sha256)
    errors = [*source["errors"], *compute["errors"]]
    return {
        "status": "pass" if not errors else "fail",
        "required_for_execute": True,
        "dataset_sha256_verified": dataset_sha256 == dataset_report.get("dataset_sha256"),
        "source_provenance": source,
        "compute_execution": compute,
        "errors": errors,
    }
