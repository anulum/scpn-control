#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural-equilibrium training campaign planner
"""Prepare deterministic metadata plans and finite GPU budget ranges without training.

The public acquisition reader must PASS. Dataset counts, grids, split totals and
optional producer digests are checked; selected local dataset bytes must match
SHA-256. Remote operator attestation is recorded separately from local hashing.
Preparation does not admit predictive EFIT/P-EFIT or facility use. Budgets are
planning assumptions, not measured performance. Reports bind finite canonical
JSON with the payload_sha256 field set to null and refuse stale digests.

>>> from tempfile import TemporaryDirectory
>>> with TemporaryDirectory() as tmp:
...     inputs = CampaignInputs(DEFAULT_MAST_REPORT, Path(tmp), DEFAULT_PUBLIC_DATA_ROOT)
...     plan = build_plan(inputs)
...     (plan['status'], plan['mast_efm_dataset']['payload']['sha256_verified_on_this_host'])
('prepared', False)
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.neural_equilibrium_campaign_inputs import (
    CampaignPlanError as CampaignPlanError,
)
from validation.neural_equilibrium_campaign_inputs import (
    canonical_campaign_digest,
    inspect_storage_payload,
    read_campaign_dataset_report,
    summarise_campaign_public_data,
)

REPORT_SCHEMA = "scpn-control.neural-equilibrium-training-campaign-plan.v1"
EXECUTION_HOST_POLICY = (
    "The storage host is storage-only; execute training only on this workstation or external cloud compute with storage-mounted "
    "or copied data."
)
DEFAULT_STORAGE_ROOT = Path("/data/SCPN-CONTROL")
DEFAULT_COMPUTE_WEIGHTS_OUT = Path("artifacts/neural_equilibrium/mast_efm_full_output_baseline_weights.npz")
DEFAULT_MAST_REPORT = ROOT / "validation" / "reports" / "mast_efm_neural_equilibrium_dataset.json"
DEFAULT_PUBLIC_DATA_ROOT = ROOT / "validation" / "reference_data" / "qlknn"
DEFAULT_JSON_OUT = ROOT / "validation" / "reports" / "neural_equilibrium_training_campaign_plan.json"
DEFAULT_MD_OUT = ROOT / "validation" / "reports" / "neural_equilibrium_training_campaign_plan.md"


@dataclass(frozen=True)
class CampaignInputs:
    """Immutable path choices and literal boolean storage controls.

    Parameters
    ----------
    mast_dataset_report : Path
        UTF-8 blocked supervised-dataset metadata, not tensors.
    storage_root : Path
        Local lookup root; relative dataset paths cannot escape it.
    public_data_root : Path
        Offline acquisition manifest tree; aggregate FAIL refuses preparation.
    require_storage_payload : bool
        Require local SHA-bound bytes or explicit remote operator attestation.
    verified_storage_payload : bool
        Acknowledge operator verification elsewhere; never hash remote storage.

    Raises
    ------
    CampaignPlanError
        A path is not a Path or a control is not a literal boolean.
    """

    mast_dataset_report: Path
    storage_root: Path
    public_data_root: Path
    require_storage_payload: bool = False
    verified_storage_payload: bool = False

    def __post_init__(self) -> None:
        """Reject runtime type coercion before paths or acknowledgement flags are used."""
        for field in ("mast_dataset_report", "storage_root", "public_data_root"):
            if not isinstance(getattr(self, field), Path):
                raise CampaignPlanError(f"{field} must be a Path")
        for field in ("require_storage_payload", "verified_storage_payload"):
            if not isinstance(getattr(self, field), bool):
                raise CampaignPlanError(f"{field} must be a boolean")


def _gpu_budget_table(equilibria_count: int, deferred_bytes: int) -> list[dict[str, Any]]:
    """Return planning GPU budgets with explicit assumptions.

    Estimates are deliberately planning ranges. They are not benchmark claims.
    They assume JAX/PyTorch x64-capable kernels, checkpointed training, compact
    model sweeps, and repository evidence generation rather than large
    foundation-model-scale runs.
    """
    try:
        mast_scale = max(float(equilibria_count) / 527.0, 1.0)
        qlknn_scale = max(float(deferred_bytes) / 309_688_648_974.0, 1.0)
    except OverflowError as exc:
        raise CampaignPlanError("declared counts exceed finite planning budgets") from exc
    return [
        {
            "scenario": "mast_efm_readiness_smoke",
            "target": "load dataset, verify splits, run one short fit/evaluation dry campaign",
            "gpu_class": "single 16-24 GB ROCm-capable GPU or CPU fallback",
            "minimum_gpu_hours": round(0.0 * mast_scale, 2),
            "nominal_gpu_hours": round(1.0 * mast_scale, 2),
            "upper_gpu_hours": round(3.0 * mast_scale, 2),
            "storage_tb": 0.05,
            "blocking_condition": "full-output trainer still required before predictive admission",
        },
        {
            "scenario": "mast_efm_single_seed_full_output",
            "target": "one full-output neural-equilibrium training run with flux, pressure, q-profile, LCFS, and axis heads",
            "gpu_class": "single 24-48 GB ROCm-capable GPU",
            "minimum_gpu_hours": round(2.0 * mast_scale, 2),
            "nominal_gpu_hours": round(6.0 * mast_scale, 2),
            "upper_gpu_hours": round(12.0 * mast_scale, 2),
            "storage_tb": 0.1,
            "blocking_condition": "requires implementation of full-output trainer and admitted input-feature provenance",
        },
        {
            "scenario": "mast_efm_multiseed_ablation",
            "target": "five seeds, architecture sweep, uncertainty calibration, and holdout reports",
            "gpu_class": "one to four 24-80 GB ROCm-capable GPUs",
            "minimum_gpu_hours": round(30.0 * mast_scale, 2),
            "nominal_gpu_hours": round(80.0 * mast_scale, 2),
            "upper_gpu_hours": round(180.0 * mast_scale, 2),
            "storage_tb": 0.5,
            "blocking_condition": "requires single-seed trainer and stable holdout metric schema",
        },
        {
            "scenario": "qlknn_qualikiz_payload_processing",
            "target": "download, checksum, preprocess, split, and train neural-transport baselines",
            "gpu_class": "single 24-80 GB accelerator for first pass; multiple accelerators for sweeps",
            "minimum_gpu_hours": round(100.0 * qlknn_scale, 2),
            "nominal_gpu_hours": round(350.0 * qlknn_scale, 2),
            "upper_gpu_hours": round(900.0 * qlknn_scale, 2),
            "storage_tb": 2.0,
            "blocking_condition": "large numeric payloads must be pulled to storage-host storage and checksum-verified first",
        },
        {
            "scenario": "external_efit_pefit_or_diiid_equilibrium_set",
            "target": "matched EFIT/P-EFIT or documented public equilibrium artefacts converted into strict reference reports",
            "gpu_class": "single 24-80 GB accelerator after CPU-side conversion",
            "minimum_gpu_hours": 20.0,
            "nominal_gpu_hours": 120.0,
            "upper_gpu_hours": 400.0,
            "storage_tb": 1.0,
            "blocking_condition": "requires acquired public or collaborator-provided matched reconstruction artefacts",
        },
        {
            "scenario": "publication_grade_equilibrium_campaign",
            "target": "multi-dataset training, seed repeats, uncertainty, latency, and strict admission evidence",
            "gpu_class": "multiple 24-80 GB accelerators",
            "minimum_gpu_hours": 500.0,
            "nominal_gpu_hours": 1500.0,
            "upper_gpu_hours": 4000.0,
            "storage_tb": 4.0,
            "blocking_condition": "requires at least one admitted external equilibrium reference set beyond MAST EFM candidate data",
        },
    ]


def build_plan(inputs: CampaignInputs) -> dict[str, Any]:
    """Build a finite declaration plan after public acquisition and local custody checks.

    Returns
    -------
    dict[str, Any]
        Fresh JSON-compatible metadata with null-field canonical digest. The
        historical prepared_on_storage lane label describes declared storage;
        payload availability_basis records what was actually observed.

    Raises
    ------
    CampaignPlanError
        Dataset metadata, selected bytes, public acquisition or budgets refuse.
    FileNotFoundError
        Required storage is absent and no remote attestation is supplied.
    """
    mast_report = read_campaign_dataset_report(inputs.mast_dataset_report)
    storage_payload = inspect_storage_payload(
        mast_report,
        inputs.storage_root,
        inputs.require_storage_payload,
        inputs.verified_storage_payload,
    )
    public_data = summarise_campaign_public_data(inputs.public_data_root, ROOT)
    deferred_bytes = public_data["deferred_bytes"]
    equilibria_count = mast_report["equilibria_count"]
    candidate_path = shlex.quote(str(inputs.storage_root / mast_report["candidate_report"]))
    dataset_path = shlex.quote(str(inputs.storage_root / mast_report["dataset_path"]))
    storage_root = shlex.quote(str(inputs.storage_root))
    gpu_budgets = _gpu_budget_table(equilibria_count, deferred_bytes)
    admission_blockers = [
        "full-output trainer must be executed on workstation or external cloud compute and publish holdout metrics",
        "strict neural-equilibrium reference admission must pass on the exact trained weight checksum",
    ]
    if mast_report["fallback_features"]:
        admission_blockers.insert(
            1,
            "fallback Ip_MA, Bt_T, and ffprime_scale inputs must be replaced by acquired or documented public inputs",
        )
    plan: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "prepared",
        "claim_boundary": (
            "This report prepares data-processing and training campaigns. It is not predictive EFIT/P-EFIT "
            "admission evidence and does not launch GPU training."
        ),
        "execution_host_policy": EXECUTION_HOST_POLICY,
        "storage_root": str(inputs.storage_root),
        "mast_efm_dataset": {
            "status": "prepared",
            "reference_dataset_id": mast_report["reference_dataset_id"],
            "equilibria_count": equilibria_count,
            "grid_shape": mast_report["grid_shape"],
            "split_counts": mast_report["split_counts"],
            "fallback_features": mast_report["fallback_features"],
            "ragged_target_policy": mast_report["ragged_target_policy"],
            "payload": storage_payload,
            "ready_to_run_checks": [
                "python validation/plan_neural_equilibrium_training_campaign.py --require-storage-payload",
                "python validation/train_mast_efm_neural_equilibrium.py",
                "python validation/build_mast_efm_neural_equilibrium_dataset.py --candidate-report "
                f"{candidate_path} --storage-root {storage_root} --output-npz "
                f"{dataset_path} --json-out "
                "validation/reports/mast_efm_neural_equilibrium_dataset.json --report-out "
                "validation/reports/mast_efm_neural_equilibrium_dataset.md",
            ],
            "blocked_before_admission": admission_blockers,
        },
        "compute_execution_package": {
            "status": "prepared_not_executed",
            "dataset_sha256": mast_report["dataset_sha256"],
            "dataset_path": str(inputs.storage_root / mast_report["dataset_path"]),
            "weights_out": DEFAULT_COMPUTE_WEIGHTS_OUT.as_posix(),
            "admitted_compute_host_kinds": ["workstation", "external_cloud"],
            "forbidden_training_hosts": ["storage host"],
            "source_provenance_reports": [
                "validation/reports/mast_efm_feature_provenance_audit.json",
                "validation/reports/mast_efm_original_feature_source_audit.json",
            ],
            "result_template_reports": [
                "validation/reports/mast_efm_neural_equilibrium_result_templates.json",
                "validation/reports/mast_efm_neural_equilibrium_result_templates.md",
            ],
            "exact_command": (
                "python validation/train_mast_efm_neural_equilibrium.py --execute "
                "--compute-host-kind workstation "
                f"--dataset-path {dataset_path} "
                f"--weights-out {DEFAULT_COMPUTE_WEIGHTS_OUT.as_posix()}"
            ),
            "pre_run_admission_gates": [
                "dataset SHA-256 must match the supervised dataset report",
                "converted feature-provenance audit must have no blocked features",
                "original public-source audit must be source_ready",
                "compute host must be explicitly declared as workstation or external_cloud",
                "weights_out must not be under storage-host dataset storage",
            ],
        },
        "prepared_dataset_lanes": [
            {
                "id": "mast_efm_neural_equilibrium",
                "status": "prepared_on_storage",
                "next_action": (
                    "run the dry-run trainer locally, then execute explicitly on this workstation or external cloud "
                    "compute when compute is reserved"
                ),
            },
            {
                "id": "qlknn_qualikiz_neural_transport",
                "status": "manifested_large_payloads_deferred",
                "next_action": "download deferred payloads to storage-host storage, verify checksums, then build processed transport tensors",
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
            "Inspect the compute execution package and result templates before reserving GPU time.",
            "Use explicit --execute only on workstation or external cloud compute with reserved GPU capacity.",
            "Run a smoke campaign and publish compact metrics before spending multi-seed GPU budget.",
            "Pull QLKNN/QuaLiKiz large payloads to storage-host storage only when storage and GPU allocation are reserved.",
            "Keep all predictive and facility claims blocked until strict admission reports pass.",
        ],
    }
    plan["payload_sha256"] = canonical_campaign_digest(plan)
    return plan


def _render_markdown(plan: dict[str, Any]) -> str:
    """Render declared plan sections; write_report wraps malformed runtime shapes."""
    lines = [
        "# Neural-Equilibrium Training Campaign Plan",
        "",
        f"Schema: `{plan['schema_version']}`",
        f"Status: `{plan['status']}`",
        f"Storage root: `{plan['storage_root']}`",
        "",
        "## Claim boundary",
        "",
        plan["claim_boundary"],
        "",
        "## Execution host policy",
        "",
        plan["execution_host_policy"],
        "",
        "## MAST EFM dataset",
        "",
        f"- Dataset status: `{plan['mast_efm_dataset']['status']}`",
        f"- Equilibria: {plan['mast_efm_dataset']['equilibria_count']}",
        f"- Grid shape: {plan['mast_efm_dataset']['grid_shape'][0]} x {plan['mast_efm_dataset']['grid_shape'][1]}",
        f"- Split counts: `{json.dumps(plan['mast_efm_dataset']['split_counts'], sort_keys=True)}`",
        f"- Dataset SHA-256: `{plan['mast_efm_dataset']['payload']['sha256']}`",
        f"- storage-host payload: `{plan['mast_efm_dataset']['payload']['absolute_path']}`",
        f"- Exists on this host: `{plan['mast_efm_dataset']['payload']['exists_on_this_host']}`",
        f"- Verified available: `{plan['mast_efm_dataset']['payload']['verified_available']}`",
        f"- Availability basis: `{plan['mast_efm_dataset']['payload']['availability_basis']}`",
        f"- SHA-256 verified on this host: `{plan['mast_efm_dataset']['payload']['sha256_verified_on_this_host']}`",
        "",
        "## Compute execution package",
        "",
        f"- Status: `{plan['compute_execution_package']['status']}`",
        f"- Weights output: `{plan['compute_execution_package']['weights_out']}`",
        f"- Dataset SHA-256: `{plan['compute_execution_package']['dataset_sha256']}`",
        f"- Admitted compute hosts: `{json.dumps(plan['compute_execution_package']['admitted_compute_host_kinds'])}`",
        f"- Forbidden training hosts: `{json.dumps(plan['compute_execution_package']['forbidden_training_hosts'])}`",
        "",
        "```bash",
        plan["compute_execution_package"]["exact_command"],
        "```",
        "",
        "### Pre-run admission gates",
        "",
    ]
    lines.extend(f"- {item}" for item in plan["compute_execution_package"]["pre_run_admission_gates"])
    lines.extend(
        [
            "",
            "## Prepared dataset lanes",
            "",
            "| Lane | Status | Next action |",
            "|---|---|---|",
        ]
    )
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
    return "\n".join(lines)


def write_report(plan: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    """Persist digest-bound JSON and rendered Markdown to distinct explicit paths.

    Invalid schema/status, stale/non-finite payloads and unrenderable shapes are
    refused before either output is written. Supported IO/path errors become
    CampaignPlanError. Writes are sequential, not a two-file transaction; a
    later IO failure may leave the JSON file and must not count as a complete
    report. A matching self-digest proves consistency, not scientific admission.
    """
    try:
        if (
            not isinstance(plan, dict)
            or plan.get("schema_version") != REPORT_SCHEMA
            or plan.get("status") != "prepared"
        ):
            raise CampaignPlanError("campaign report must have the supported schema and prepared status")
        if plan.get("payload_sha256") != canonical_campaign_digest(plan):
            raise CampaignPlanError("campaign report payload_sha256 does not match its contents")
        encoded = json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
        markdown = _render_markdown(plan)
        if json_out.resolve() == markdown_out.resolve() or (
            json_out.exists() and markdown_out.exists() and json_out.samefile(markdown_out)
        ):
            raise CampaignPlanError("JSON and Markdown outputs must be distinct paths")
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(encoded, encoding="utf-8")
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(markdown, encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError, RuntimeError) as exc:
        raise CampaignPlanError(f"cannot write campaign report: {exc}") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse path/output controls from argv or the process arguments.

    The verified-storage flag is an explicit remote operator attestation.
    Default outputs remain the historical canonical report paths; callers
    needing an isolated inspection must supply both output paths. Argparse
    usage errors retain SystemExit(2); --help retains SystemExit(0).

    >>> parse_args(['--verified-storage-payload']).verified_storage_payload
    True
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mast-dataset-report", default=DEFAULT_MAST_REPORT, type=Path)
    parser.add_argument("--storage-root", default=DEFAULT_STORAGE_ROOT, type=Path)
    parser.add_argument("--public-data-root", default=DEFAULT_PUBLIC_DATA_ROOT, type=Path)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT, type=Path)
    parser.add_argument("--report-out", default=DEFAULT_MD_OUT, type=Path)
    parser.add_argument("--require-storage-payload", action="store_true")
    parser.add_argument("--verified-storage-payload", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare/write metadata only; return zero on success or one on domain/IO refusal.

    Failures print authored FAIL text to stderr. Training, fetching, provenance
    admission and weight creation are never performed. Argparse exits preserve
    its documented usage/help codes.
    """
    args = parse_args(argv)
    try:
        plan = build_plan(
            CampaignInputs(
                mast_dataset_report=args.mast_dataset_report,
                storage_root=args.storage_root,
                public_data_root=args.public_data_root,
                require_storage_payload=args.require_storage_payload,
                verified_storage_payload=args.verified_storage_payload,
            )
        )
        write_report(plan, args.json_out, args.report_out)
    except (CampaignPlanError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
