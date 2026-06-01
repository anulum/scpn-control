#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural-equilibrium training campaign planner
"""Publish run-ready neural-equilibrium dataset and GPU-budget campaign plans."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.build_mast_efm_neural_equilibrium_dataset import DATASET_SCHEMA as MAST_DATASET_SCHEMA
from validation.validate_public_data_acquisition import validate_public_data_acquisition_directory

REPORT_SCHEMA = "scpn-control.neural-equilibrium-training-campaign-plan.v1"
DEFAULT_SAS_ROOT = Path("/mnt/data_sas/DATASETS/SCPN-CONTROL")
DEFAULT_MAST_REPORT = ROOT / "validation" / "reports" / "mast_efm_neural_equilibrium_dataset.json"
DEFAULT_PUBLIC_DATA_ROOT = ROOT / "validation" / "reference_data" / "qlknn"
DEFAULT_JSON_OUT = ROOT / "validation" / "reports" / "neural_equilibrium_training_campaign_plan.json"
DEFAULT_MD_OUT = ROOT / "validation" / "reports" / "neural_equilibrium_training_campaign_plan.md"


@dataclass(frozen=True)
class CampaignInputs:
    """Inputs for generating the campaign plan."""

    mast_dataset_report: Path
    sas_root: Path
    public_data_root: Path
    require_sas_payload: bool = False
    verified_sas_payload: bool = False


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _validate_mast_report(path: Path) -> dict[str, Any]:
    report = _load_json_object(path)
    if report.get("schema_version") != MAST_DATASET_SCHEMA:
        raise ValueError("MAST EFM dataset report has unsupported schema_version")
    if report.get("status") != "blocked":
        raise ValueError("MAST EFM dataset report must preserve blocked predictive-admission state")
    if int(report.get("equilibria_count", 0)) <= 0:
        raise ValueError("MAST EFM dataset report must contain equilibria")
    dataset_sha = report.get("dataset_sha256")
    if not isinstance(dataset_sha, str) or len(dataset_sha) != 64:
        raise ValueError("MAST EFM dataset report must bind a dataset_sha256")
    return report


def _sas_payload_status(
    report: dict[str, Any],
    sas_root: Path,
    require_payload: bool,
    verified_payload: bool,
) -> dict[str, Any]:
    relative = Path(str(report["dataset_path"]))
    absolute = sas_root / relative
    exists = absolute.is_file()
    available = bool(exists or verified_payload)
    if require_payload and not available:
        raise FileNotFoundError(f"SAS dataset payload is missing: {absolute}")
    return {
        "relative_path": str(relative),
        "absolute_path": str(absolute),
        "exists_on_this_host": exists,
        "verified_available": available,
        "sha256": report["dataset_sha256"],
    }


def _public_data_summary(root: Path) -> dict[str, Any]:
    report = validate_public_data_acquisition_directory(root)
    return {
        "status": report["status"],
        "records": int(report.get("records", 0)),
        "files": int(report.get("files", 0)),
        "local_files": int(report.get("local_files", 0)),
        "deferred_files": int(report.get("deferred_files", 0)),
        "deferred_bytes": int(report.get("deferred_bytes", 0)),
        "manifests": report.get("manifests", []),
    }


def _gpu_budget_table(equilibria_count: int, deferred_bytes: int) -> list[dict[str, Any]]:
    """Return planning GPU budgets with explicit assumptions.

    Estimates are deliberately planning ranges. They are not benchmark claims.
    They assume JAX/PyTorch x64-capable kernels, checkpointed training, compact
    model sweeps, and repository evidence generation rather than large
    foundation-model-scale runs.
    """

    mast_scale = max(float(equilibria_count) / 527.0, 1.0)
    qlknn_scale = max(float(deferred_bytes) / 309_688_648_974.0, 1.0)
    return [
        {
            "scenario": "mast_efm_readiness_smoke",
            "target": "load dataset, verify splits, run one short fit/evaluation dry campaign",
            "gpu_class": "single 16-24 GB CUDA GPU or CPU fallback",
            "minimum_gpu_hours": round(0.0 * mast_scale, 2),
            "nominal_gpu_hours": round(1.0 * mast_scale, 2),
            "upper_gpu_hours": round(3.0 * mast_scale, 2),
            "storage_tb": 0.05,
            "blocking_condition": "full-output trainer still required before predictive admission",
        },
        {
            "scenario": "mast_efm_single_seed_full_output",
            "target": "one full-output neural-equilibrium training run with flux, pressure, q-profile, LCFS, and axis heads",
            "gpu_class": "single 24-48 GB CUDA GPU",
            "minimum_gpu_hours": round(2.0 * mast_scale, 2),
            "nominal_gpu_hours": round(6.0 * mast_scale, 2),
            "upper_gpu_hours": round(12.0 * mast_scale, 2),
            "storage_tb": 0.1,
            "blocking_condition": "requires implementation of full-output trainer and admitted input-feature provenance",
        },
        {
            "scenario": "mast_efm_multiseed_ablation",
            "target": "five seeds, architecture sweep, uncertainty calibration, and holdout reports",
            "gpu_class": "one to four 24-80 GB CUDA GPUs",
            "minimum_gpu_hours": round(30.0 * mast_scale, 2),
            "nominal_gpu_hours": round(80.0 * mast_scale, 2),
            "upper_gpu_hours": round(180.0 * mast_scale, 2),
            "storage_tb": 0.5,
            "blocking_condition": "requires single-seed trainer and stable holdout metric schema",
        },
        {
            "scenario": "qlknn_qualikiz_payload_processing",
            "target": "download, checksum, preprocess, split, and train neural-transport baselines",
            "gpu_class": "single A10/A100-class GPU for first pass; A100/H100 for sweeps",
            "minimum_gpu_hours": round(100.0 * qlknn_scale, 2),
            "nominal_gpu_hours": round(350.0 * qlknn_scale, 2),
            "upper_gpu_hours": round(900.0 * qlknn_scale, 2),
            "storage_tb": 2.0,
            "blocking_condition": "large numeric payloads must be pulled to SAS and checksum-verified first",
        },
        {
            "scenario": "external_efit_pefit_or_diiid_equilibrium_set",
            "target": "matched EFIT/P-EFIT or documented public equilibrium artefacts converted into strict reference reports",
            "gpu_class": "single 24-80 GB CUDA GPU after CPU-side conversion",
            "minimum_gpu_hours": 20.0,
            "nominal_gpu_hours": 120.0,
            "upper_gpu_hours": 400.0,
            "storage_tb": 1.0,
            "blocking_condition": "requires acquired public or collaborator-provided matched reconstruction artefacts",
        },
        {
            "scenario": "publication_grade_equilibrium_campaign",
            "target": "multi-dataset training, seed repeats, uncertainty, latency, and strict admission evidence",
            "gpu_class": "multi-GPU A100/H100-class allocation",
            "minimum_gpu_hours": 500.0,
            "nominal_gpu_hours": 1500.0,
            "upper_gpu_hours": 4000.0,
            "storage_tb": 4.0,
            "blocking_condition": "requires at least one admitted external equilibrium reference set beyond MAST EFM candidate data",
        },
    ]


def build_plan(inputs: CampaignInputs) -> dict[str, Any]:
    """Build the deterministic campaign plan."""

    mast_report = _validate_mast_report(inputs.mast_dataset_report)
    sas_payload = _sas_payload_status(
        mast_report,
        inputs.sas_root,
        inputs.require_sas_payload,
        inputs.verified_sas_payload,
    )
    public_data = _public_data_summary(inputs.public_data_root)
    deferred_bytes = int(public_data["deferred_bytes"])
    equilibria_count = int(mast_report["equilibria_count"])
    gpu_budgets = _gpu_budget_table(equilibria_count, deferred_bytes)
    plan: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "prepared",
        "claim_boundary": (
            "This report prepares data-processing and training campaigns. It is not predictive EFIT/P-EFIT "
            "admission evidence and does not launch GPU training."
        ),
        "sas_root": str(inputs.sas_root),
        "mast_efm_dataset": {
            "status": "prepared",
            "reference_dataset_id": mast_report["reference_dataset_id"],
            "equilibria_count": equilibria_count,
            "grid_shape": mast_report["grid_shape"],
            "split_counts": mast_report["split_counts"],
            "fallback_features": mast_report["fallback_features"],
            "ragged_target_policy": mast_report["ragged_target_policy"],
            "payload": sas_payload,
            "ready_to_run_checks": [
                "python validation/plan_neural_equilibrium_training_campaign.py --require-sas-payload",
                "python validation/train_mast_efm_neural_equilibrium.py",
                "python validation/build_mast_efm_neural_equilibrium_dataset.py --candidate-report "
                f"{inputs.sas_root / mast_report['candidate_report']} --sas-root {inputs.sas_root} --output-npz "
                f"{inputs.sas_root / mast_report['dataset_path']} --json-out "
                "validation/reports/mast_efm_neural_equilibrium_dataset.json --report-out "
                "validation/reports/mast_efm_neural_equilibrium_dataset.md",
            ],
            "blocked_before_admission": [
                "full-output trainer must be executed on admitted storage and publish holdout metrics",
                "fallback Ip_MA, Bt_T, and ffprime_scale inputs must be replaced by acquired or documented public inputs",
                "strict neural-equilibrium reference admission must pass on the exact trained weight checksum",
            ],
        },
        "prepared_dataset_lanes": [
            {
                "id": "mast_efm_neural_equilibrium",
                "status": "prepared_on_sas",
                "next_action": "run the dry-run trainer locally, then execute explicitly on admitted storage when compute is reserved",
            },
            {
                "id": "qlknn_qualikiz_neural_transport",
                "status": "manifested_large_payloads_deferred",
                "next_action": "download deferred payloads to SAS, verify checksums, then build processed transport tensors",
                "public_data_summary": public_data,
            },
            {
                "id": "external_efit_pefit_or_diiid_equilibrium",
                "status": "external_material_required",
                "next_action": "acquire matched public EFIT/P-EFIT, GEQDSK, or MDSplus-derived reconstruction artefacts",
            },
            {
                "id": "sparc_or_public_geqdsk_equilibrium",
                "status": "external_material_required",
                "next_action": "seal redistributable GEQDSK/EQDSK artefacts with source policy and SHA-256 manifests",
            },
        ],
        "gpu_budget_estimates": gpu_budgets,
        "run_order": [
            "Re-run the MAST EFM dataset readiness check before any campaign.",
            "Run the MAST EFM trainer in dry-run mode and inspect the launch report.",
            "Use explicit --execute only on admitted storage and reserved compute.",
            "Run a smoke campaign and publish compact metrics before spending multi-seed GPU budget.",
            "Pull QLKNN/QuaLiKiz large payloads to SAS only when storage and GPU allocation are reserved.",
            "Keep all predictive and facility claims blocked until strict admission reports pass.",
        ],
    }
    plan["payload_sha256"] = _payload_sha256({**plan, "payload_sha256": None})
    return plan


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def write_report(plan: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    """Write JSON and Markdown campaign reports."""

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Neural-Equilibrium Training Campaign Plan",
        "",
        f"Schema: `{plan['schema_version']}`",
        f"Status: `{plan['status']}`",
        f"SAS root: `{plan['sas_root']}`",
        "",
        "## Claim boundary",
        "",
        plan["claim_boundary"],
        "",
        "## MAST EFM dataset",
        "",
        f"- Dataset status: `{plan['mast_efm_dataset']['status']}`",
        f"- Equilibria: {plan['mast_efm_dataset']['equilibria_count']}",
        f"- Grid shape: {plan['mast_efm_dataset']['grid_shape'][0]} x {plan['mast_efm_dataset']['grid_shape'][1]}",
        f"- Split counts: `{json.dumps(plan['mast_efm_dataset']['split_counts'], sort_keys=True)}`",
        f"- Dataset SHA-256: `{plan['mast_efm_dataset']['payload']['sha256']}`",
        f"- SAS payload: `{plan['mast_efm_dataset']['payload']['absolute_path']}`",
        f"- Exists on this host: `{plan['mast_efm_dataset']['payload']['exists_on_this_host']}`",
        f"- Verified available: `{plan['mast_efm_dataset']['payload']['verified_available']}`",
        "",
        "## Prepared dataset lanes",
        "",
        "| Lane | Status | Next action |",
        "|---|---|---|",
    ]
    for lane in plan["prepared_dataset_lanes"]:
        lines.append(f"| `{lane['id']}` | `{lane['status']}` | {lane['next_action']} |")
    lines.extend(
        [
            "",
            "## GPU budget estimates",
            "",
            "| Scenario | GPU class | Minimum GPU-h | Nominal GPU-h | Upper GPU-h | Storage TB | Blocking condition |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for budget in plan["gpu_budget_estimates"]:
        lines.append(
            "| "
            f"`{budget['scenario']}` | {budget['gpu_class']} | {budget['minimum_gpu_hours']} | "
            f"{budget['nominal_gpu_hours']} | {budget['upper_gpu_hours']} | {budget['storage_tb']} | "
            f"{budget['blocking_condition']} |"
        )
    lines.extend(["", "## Run order", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(plan["run_order"], start=1))
    lines.append("")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mast-dataset-report", default=DEFAULT_MAST_REPORT, type=Path)
    parser.add_argument("--sas-root", default=DEFAULT_SAS_ROOT, type=Path)
    parser.add_argument("--public-data-root", default=DEFAULT_PUBLIC_DATA_ROOT, type=Path)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT, type=Path)
    parser.add_argument("--report-out", default=DEFAULT_MD_OUT, type=Path)
    parser.add_argument("--require-sas-payload", action="store_true")
    parser.add_argument("--verified-sas-payload", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Write the campaign plan."""

    args = parse_args()
    plan = build_plan(
        CampaignInputs(
            mast_dataset_report=args.mast_dataset_report,
            sas_root=args.sas_root,
            public_data_root=args.public_data_root,
            require_sas_payload=args.require_sas_payload,
            verified_sas_payload=args.verified_sas_payload,
        )
    )
    write_report(plan, args.json_out, args.report_out)


if __name__ == "__main__":
    main()
