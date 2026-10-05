#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium trainer
"""Prepare or execute deterministic full-output MAST EFM baseline training.

The facade retains public trainer/report/template APIs while controls and
metadata, tensor/numerical work, evidence declarations and persistence have
separate owners. Both dry-run and execution inspect present selected bytes;
missing tensors allow a planning report only. Scientific admission stays blocked.

Generate a real plan from canonical repository metadata and inspect a dry-run
without writing weights or updating checked-in reports:

>>> import json
>>> from tempfile import TemporaryDirectory
>>> from validation.plan_neural_equilibrium_training_campaign import CampaignInputs, build_plan
>>> with TemporaryDirectory() as directory:
...     folder = Path(directory)
...     dataset_report = DEFAULT_DATASET_REPORT
...     plan = build_plan(CampaignInputs(dataset_report, folder / "missing-storage", ROOT / "validation/reference_data/qlknn"))
...     plan_path = folder / "plan.json"
...     _ = plan_path.write_text(json.dumps(plan))
...     launch = build_training_report(TrainingInputs(dataset_report, plan_path, folder / "missing.npz", folder / "weights.npz"))
...     template = build_result_templates(launch)
...     verified = validate_result_templates(template, training_report=launch)
...     print(launch["status"], launch["execution_mode"], launch["admission_ready"])
...     print(verified is template, (folder / "weights.npz").exists())
prepared dry_run False
True False
"""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.mast_efm_feature_audit_inputs import read_dataset_declaration
from validation.neural_equilibrium_training_arrays import _dataset_metadata, _execute_training, _load_dataset
from validation.neural_equilibrium_training_evidence import _sha256_json
from validation.neural_equilibrium_training_evidence import build_result_templates as build_result_templates
from validation.neural_equilibrium_training_evidence import validate_result_templates as validate_result_templates
from validation.neural_equilibrium_training_evidence import validate_training_report as validate_training_report
from validation.neural_equilibrium_training_inputs import (
    ADMITTED_COMPUTE_HOST_KINDS as ADMITTED_COMPUTE_HOST_KINDS,
)
from validation.neural_equilibrium_training_inputs import (
    CAMPAIGN_PLAN_SCHEMA as CAMPAIGN_PLAN_SCHEMA,
)
from validation.neural_equilibrium_training_inputs import (
    DATASET_SCHEMA as DATASET_SCHEMA,
)
from validation.neural_equilibrium_training_inputs import (
    DEFAULT_CAMPAIGN_PLAN as DEFAULT_CAMPAIGN_PLAN,
)
from validation.neural_equilibrium_training_inputs import (
    DEFAULT_DATASET_PATH as DEFAULT_DATASET_PATH,
)
from validation.neural_equilibrium_training_inputs import (
    DEFAULT_DATASET_REPORT as DEFAULT_DATASET_REPORT,
)
from validation.neural_equilibrium_training_inputs import (
    DEFAULT_FEATURE_PROVENANCE_REPORT as DEFAULT_FEATURE_PROVENANCE_REPORT,
)
from validation.neural_equilibrium_training_inputs import (
    DEFAULT_JSON_OUT as DEFAULT_JSON_OUT,
)
from validation.neural_equilibrium_training_inputs import (
    DEFAULT_MD_OUT as DEFAULT_MD_OUT,
)
from validation.neural_equilibrium_training_inputs import (
    DEFAULT_ORIGINAL_SOURCE_REPORT as DEFAULT_ORIGINAL_SOURCE_REPORT,
)
from validation.neural_equilibrium_training_inputs import (
    DEFAULT_TEMPLATES_JSON_OUT as DEFAULT_TEMPLATES_JSON_OUT,
)
from validation.neural_equilibrium_training_inputs import (
    DEFAULT_TEMPLATES_MD_OUT as DEFAULT_TEMPLATES_MD_OUT,
)
from validation.neural_equilibrium_training_inputs import (
    DEFAULT_WEIGHTS_OUT as DEFAULT_WEIGHTS_OUT,
)
from validation.neural_equilibrium_training_inputs import (
    EXECUTION_HOST_POLICY as EXECUTION_HOST_POLICY,
)
from validation.neural_equilibrium_training_inputs import (
    FEATURE_NAMES as FEATURE_NAMES,
)
from validation.neural_equilibrium_training_inputs import (
    RESULT_TEMPLATES_SCHEMA as RESULT_TEMPLATES_SCHEMA,
)
from validation.neural_equilibrium_training_inputs import (
    TARGET_KEYS as TARGET_KEYS,
)
from validation.neural_equilibrium_training_inputs import (
    TRAINING_SCHEMA as TRAINING_SCHEMA,
)
from validation.neural_equilibrium_training_inputs import (
    TrainingInputs as TrainingInputs,
)
from validation.neural_equilibrium_training_inputs import (
    _display_path,
    _load_json_object,
    _pre_run_admission,
    _validate_reports,
)
from validation.neural_equilibrium_training_inputs import (
    _sha256_file as _sha256_file,
)
from validation.neural_equilibrium_training_rendering import ensure_distinct_outputs
from validation.neural_equilibrium_training_rendering import write_report as write_report
from validation.neural_equilibrium_training_rendering import write_result_templates as write_result_templates


def build_training_report(inputs: TrainingInputs) -> dict[str, Any]:
    """Prepare a launch from real metadata or execute the existing validated baseline.

    Campaign digest/public acquisition and every MAST binding must match. Selected
    local bytes and decoded tensor layout/splits/masks are checked before fitting.
    Missing tensors permit dry-run metadata with FAIL admission; --execute requires
    actual local bytes and complete source/compute PASS. Source audits are checked
    as declarations/digests, not authenticated remote physics. Generated launches
    are validated before return; executed weights are locally SHA-bound. Paths and
    reads are sequential observations, not an atomic filesystem snapshot.
    Selected tensor decoding uses the same captured NPZ bytes whose SHA was
    verified; the reported SHA is assigned only after successful verification.
    """
    dataset_report, dataset_report_sha256 = read_dataset_declaration(inputs.dataset_report)
    campaign_plan = _load_json_object(inputs.campaign_plan)
    _validate_reports(dataset_report, campaign_plan)
    try:
        inputs.dataset_path.resolve()
        dataset_exists = inputs.dataset_path.is_file()
        if not dataset_exists and (inputs.dataset_path.exists() or inputs.dataset_path.is_symlink()):
            raise ValueError("selected dataset_path must be a regular file when present")
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"cannot inspect selected training dataset: {exc}") from exc
    if inputs.execute and not dataset_exists:
        raise FileNotFoundError(f"dataset payload is required for --execute: {inputs.dataset_path}")

    dataset_sha256: str | None = None
    dataset_metadata: dict[str, Any] | None = None
    execution_payload: dict[str, Any]
    if dataset_exists:
        data = _load_dataset(inputs.dataset_path, dataset_report["dataset_sha256"])
        dataset_sha256 = dataset_report["dataset_sha256"]
        dataset_metadata = _dataset_metadata(data, dataset_report)
        if inputs.execute:
            pre_run_admission = _pre_run_admission(
                inputs, dataset_report, dataset_sha256, dataset_report_sha256=dataset_report_sha256
            )
            if pre_run_admission["status"] != "pass":
                raise ValueError("pre-run admission failed before --execute: " + "; ".join(pre_run_admission["errors"]))
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    execution_payload, _ = _execute_training(data, inputs)
            except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                raise ValueError(f"baseline numerical fit refused: {exc}") from exc
        else:
            execution_payload = {
                "execution_mode": "dry_run",
                "weights_path": str(inputs.weights_out),
                "weights_sha256": None,
                "holdout_metrics": None,
            }
    else:
        execution_payload = {
            "execution_mode": "dry_run",
            "weights_path": str(inputs.weights_out),
            "weights_sha256": None,
            "holdout_metrics": None,
        }
    pre_run_admission = _pre_run_admission(
        inputs, dataset_report, dataset_sha256, dataset_report_sha256=dataset_report_sha256
    )

    fallback_features = list(dataset_report["fallback_features"])
    blocked_before_admission = [
        "run --execute on workstation or external cloud compute and publish holdout metrics for train, validation, and test splits",
        "validate the exact trained weight checksum through the strict neural-equilibrium reference gate",
    ]
    if fallback_features:
        blocked_before_admission.insert(
            0,
            "replace fallback Ip_MA, Bt_T, and ffprime_scale with acquired or documented public inputs",
        )
    report: dict[str, Any] = {
        "schema_version": TRAINING_SCHEMA,
        "status": "executed" if inputs.execute else "prepared",
        "admission_ready": False,
        "strict_artefact_emitted": False,
        "claim_boundary": (
            "This report prepares or executes a deterministic repository baseline. "
            "It is not predictive EFIT/P-EFIT admission evidence."
        ),
        "dataset_report": _display_path(inputs.dataset_report),
        "campaign_plan": _display_path(inputs.campaign_plan),
        "dataset_path": str(inputs.dataset_path),
        "dataset_exists_on_this_host": dataset_exists,
        "dataset_sha256": dataset_sha256 or dataset_report["dataset_sha256"],
        "dataset_metadata": dataset_metadata,
        "execution_host_policy": EXECUTION_HOST_POLICY,
        "feature_provenance_report": _display_path(inputs.feature_provenance_report),
        "original_source_report": _display_path(inputs.original_source_report),
        "pre_run_admission": pre_run_admission,
        "required_targets": list(TARGET_KEYS),
        "fallback_features": fallback_features,
        "blocked_before_admission": blocked_before_admission,
        "run_command": (
            "python validation/train_mast_efm_neural_equilibrium.py --execute "
            f"--compute-host-kind {inputs.compute_host_kind if inputs.compute_host_kind != 'unspecified' else 'workstation'} "
            f"--compute-host-label {shlex.quote(inputs.compute_host_label)} "
            f"--dataset-report {shlex.quote(str(inputs.dataset_report))} "
            f"--campaign-plan {shlex.quote(str(inputs.campaign_plan))} "
            f"--dataset-path {shlex.quote(str(inputs.dataset_path))} --weights-out {shlex.quote(str(inputs.weights_out))} "
            f"--feature-provenance-report {shlex.quote(str(inputs.feature_provenance_report))} "
            f"--original-source-report {shlex.quote(str(inputs.original_source_report))} "
            f"--ridge-alpha {float(inputs.ridge_alpha)!r} --max-flux-components {inputs.max_flux_components}"
        ),
        **execution_payload,
    }
    report["payload_sha256"] = _sha256_json({**report, "payload_sha256": None})
    validate_training_report(report, require_executed=inputs.execute)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit argv or process controls; help/usage retain argparse0/2.

    --execute is required for fitting. Default outputs are historical canonical
    report locations; isolated callers must supply all four report output paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-report", default=DEFAULT_DATASET_REPORT, type=Path)
    parser.add_argument("--campaign-plan", default=DEFAULT_CAMPAIGN_PLAN, type=Path)
    parser.add_argument("--feature-provenance-report", default=DEFAULT_FEATURE_PROVENANCE_REPORT, type=Path)
    parser.add_argument("--original-source-report", default=DEFAULT_ORIGINAL_SOURCE_REPORT, type=Path)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, type=Path)
    parser.add_argument("--weights-out", default=DEFAULT_WEIGHTS_OUT, type=Path)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT, type=Path)
    parser.add_argument("--report-out", default=DEFAULT_MD_OUT, type=Path)
    parser.add_argument("--templates-json-out", default=DEFAULT_TEMPLATES_JSON_OUT, type=Path)
    parser.add_argument("--templates-report-out", default=DEFAULT_TEMPLATES_MD_OUT, type=Path)
    parser.add_argument(
        "--compute-host-kind", default="unspecified", choices=("unspecified", *ADMITTED_COMPUTE_HOST_KINDS)
    )
    parser.add_argument("--compute-host-label", default="")
    parser.add_argument("--ridge-alpha", default=1.0e-6, type=float)
    parser.add_argument("--max-flux-components", default=32, type=int)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare/write or explicitly execute; return0 on success and1 on supported refusal.

    All report outputs are checked against each other and protected inputs/weights
    before execution. Failures print authored FAIL stderr, with no implied rollback:
    sequential writes may leave earlier outputs. No GPU/storage reservation or
    remote fetch is performed. Argparse retains help0/usage2.
    """
    args = parse_args(argv)
    try:
        ensure_distinct_outputs(
            [args.json_out, args.report_out, args.templates_json_out, args.templates_report_out],
            protected=[
                args.dataset_report,
                args.campaign_plan,
                args.dataset_path,
                args.weights_out,
                args.feature_provenance_report,
                args.original_source_report,
            ],
        )
        report = build_training_report(
            TrainingInputs(
                dataset_report=args.dataset_report,
                campaign_plan=args.campaign_plan,
                dataset_path=args.dataset_path,
                weights_out=args.weights_out,
                feature_provenance_report=args.feature_provenance_report,
                original_source_report=args.original_source_report,
                compute_host_kind=args.compute_host_kind,
                compute_host_label=args.compute_host_label,
                execute=args.execute,
                ridge_alpha=args.ridge_alpha,
                max_flux_components=args.max_flux_components,
            )
        )
        write_report(report, args.json_out, args.report_out)
        write_result_templates(build_result_templates(report), args.templates_json_out, args.templates_report_out)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
