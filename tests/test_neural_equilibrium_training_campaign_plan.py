# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural-equilibrium campaign-plan tests
"""Exercise the public neural-equilibrium campaign planner and its reports."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from validation.build_mast_efm_neural_equilibrium_dataset import DATASET_SCHEMA
from validation.plan_neural_equilibrium_training_campaign import (
    REPORT_SCHEMA,
    CampaignInputs,
    CampaignPlanError,
    build_plan,
    main,
    parse_args,
    write_report,
)
from validation.validate_public_data_acquisition import SCHEMA_VERSION as PUBLIC_DATA_SCHEMA


def _write_mast_report(path: Path) -> None:
    """Write declaration-only fixture metadata without creating numerical tensors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": DATASET_SCHEMA,
                "status": "blocked",
                "reference_dataset_id": "mast-efm-test",
                "equilibria_count": 12,
                "grid_shape": [65, 129],
                "split_counts": {"train": 8, "validation": 2, "test": 2},
                "fallback_features": [],
                "ragged_target_policy": {
                    "keys": ["lcfs_r_m", "lcfs_z_m", "lcfs_valid_mask"],
                    "padding": "NaN for coordinates and False for validity mask",
                    "point_count_key": "lcfs_point_count",
                    "max_lcfs_points": 157,
                },
                "candidate_report": "converted/neural_equilibrium_reference/candidate.json",
                "dataset_path": "processed/neural_equilibrium/mast_efm_supervised_dataset.npz",
                "dataset_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )


def _write_public_data_manifest(root: Path) -> None:
    """Write deferred advertisement metadata; no remote bytes are retrieved."""
    manifest_dir = root / "zenodo_1"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "files_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": PUBLIC_DATA_SCHEMA,
                "source": "zenodo",
                "doi": "10.5281/zenodo.1",
                "title": "fixture",
                "license": "cc-by-4.0",
                "record_sha256": "b" * 64,
                "large_numeric_files_downloaded": False,
                "large_numeric_files_policy": "deferred: pull multi-GB arrays on the storage target",
                "files": [
                    {
                        "key": "large.nc",
                        "size_bytes": 1024,
                        "checksum": "md5:" + "c" * 32,
                        "download_url": "https://zenodo.org/api/records/1/files/large.nc/content",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_build_plan_prepares_mast_and_deferred_public_data_lanes(tmp_path: Path) -> None:
    """Keep the dataset, execution-host, and deferred-payload contracts intact."""
    mast_report = tmp_path / "mast.json"
    public_root = tmp_path / "public"
    _write_mast_report(mast_report)
    _write_public_data_manifest(public_root)

    plan = build_plan(
        CampaignInputs(mast_dataset_report=mast_report, storage_root=tmp_path, public_data_root=public_root)
    )

    assert plan["schema_version"] == REPORT_SCHEMA
    assert plan["status"] == "prepared"
    assert plan["mast_efm_dataset"]["status"] == "prepared"
    assert plan["mast_efm_dataset"]["payload"]["exists_on_this_host"] is False
    assert plan["mast_efm_dataset"]["payload"]["verified_available"] is False
    assert "The storage host is storage-only" in plan["execution_host_policy"]
    package = plan["compute_execution_package"]
    assert package["status"] == "prepared_not_executed"
    assert package["weights_out"] == "artifacts/neural_equilibrium/mast_efm_full_output_baseline_weights.npz"
    assert package["admitted_compute_host_kinds"] == ["workstation", "external_cloud"]
    assert package["forbidden_training_hosts"] == ["storage host"]
    assert any(
        "weights_out must not be under storage-host dataset storage" in item
        for item in package["pre_run_admission_gates"]
    )
    assert plan["prepared_dataset_lanes"][0]["status"] == "prepared_on_storage"
    assert "dry-run trainer" in plan["prepared_dataset_lanes"][0]["next_action"]
    assert "workstation or external cloud" in plan["prepared_dataset_lanes"][0]["next_action"]
    assert all("fallback" not in item for item in plan["mast_efm_dataset"]["blocked_before_admission"])
    assert plan["prepared_dataset_lanes"][1]["status"] == "manifested_large_payloads_deferred"
    assert plan["prepared_dataset_lanes"][1]["public_data_summary"]["deferred_bytes"] == 1024
    assert {budget["scenario"] for budget in plan["gpu_budget_estimates"]} >= {
        "mast_efm_single_seed_full_output",
        "qlknn_qualikiz_payload_processing",
        "publication_grade_equilibrium_campaign",
    }
    assert len(plan["payload_sha256"]) == 64


def test_build_plan_can_require_storage_payload(tmp_path: Path) -> None:
    """Require an explicit verified-storage acknowledgement when requested."""
    mast_report = tmp_path / "mast.json"
    public_root = tmp_path / "public"
    _write_mast_report(mast_report)
    _write_public_data_manifest(public_root)

    with pytest.raises(FileNotFoundError, match="storage-host dataset payload is missing"):
        build_plan(
            CampaignInputs(
                mast_dataset_report=mast_report,
                storage_root=tmp_path,
                public_data_root=public_root,
                require_storage_payload=True,
            )
        )

    plan = build_plan(
        CampaignInputs(
            mast_dataset_report=mast_report,
            storage_root=tmp_path,
            public_data_root=public_root,
            require_storage_payload=True,
            verified_storage_payload=True,
        )
    )
    assert plan["mast_efm_dataset"]["payload"]["verified_available"] is True


def test_write_report_records_gpu_budget_table(tmp_path: Path) -> None:
    """Write vendor-neutral budgets to both public report formats."""
    mast_report = tmp_path / "mast.json"
    public_root = tmp_path / "public"
    _write_mast_report(mast_report)
    _write_public_data_manifest(public_root)
    plan = build_plan(
        CampaignInputs(mast_dataset_report=mast_report, storage_root=tmp_path, public_data_root=public_root)
    )

    write_report(plan, tmp_path / "plan.json", tmp_path / "plan.md")

    markdown = (tmp_path / "plan.md").read_text(encoding="utf-8")
    assert "Neural-Equilibrium Training Campaign Plan" in markdown
    assert "mast_efm_single_seed_full_output" in markdown
    assert "qlknn_qualikiz_payload_processing" in markdown
    assert "Compute execution package" in markdown
    assert "--compute-host-kind workstation" in markdown
    assert "predictive EFIT/P-EFIT" in markdown
    assert "The storage host is storage-only" in markdown
    report = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    bound_payload = {**report, "payload_sha256": None}
    encoded_payload = json.dumps(bound_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    assert report["payload_sha256"] == hashlib.sha256(encoded_payload).hexdigest()
    assert len(report["gpu_budget_estimates"]) == 6
    assert all("ROCm-capable" in budget["gpu_class"] for budget in report["gpu_budget_estimates"][:3])
    for output in (markdown, json.dumps(report)):
        assert not any(vendor_class in output for vendor_class in ("CUDA", "A10", "A100", "H100"))


@pytest.fixture
def campaign_inputs(tmp_path: Path) -> CampaignInputs:
    """Select isolated real metadata and storage locations for public entrypoint tests."""
    mast = tmp_path / "mast.json"
    public = tmp_path / "public"
    _write_mast_report(mast)
    _write_public_data_manifest(public)
    return CampaignInputs(mast, tmp_path / "storage", public)


def _change_report(inputs: CampaignInputs, change: Callable[[dict[str, Any]], object]) -> None:
    """Mutate declared input JSON, never a production helper or numerical kernel."""
    report = json.loads(inputs.mast_dataset_report.read_text())
    change(report)
    inputs.mast_dataset_report.write_text(json.dumps(report))


def _reseal(plan: dict[str, Any]) -> None:
    """Independently recompute the documented null-field digest for structural refusal probes."""
    encoded = json.dumps({**plan, "payload_sha256": None}, sort_keys=True, separators=(",", ":")).encode()
    plan["payload_sha256"] = hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "old", "schema_version"),
        ("status", "pass", "blocked predictive"),
        ("payload_sha256", "0" * 64, "payload_sha256"),
        ("equilibria_count", True, "equilibria_count"),
        ("equilibria_count", 12.0, "equilibria_count"),
        ("equilibria_count", "12", "equilibria_count"),
        ("equilibria_count", 0, "equilibria_count"),
        ("dataset_sha256", "z" * 64, "dataset_sha256"),
        ("dataset_sha256", None, "dataset_sha256"),
        ("dataset_sha256", "A" * 64, "dataset_sha256"),
        ("reference_dataset_id", "", "reference_dataset_id"),
        ("reference_dataset_id", " spaced ", "reference_dataset_id"),
        ("reference_dataset_id", "newline\nvalue", "reference_dataset_id"),
        ("dataset_path", "../escape.npz", "storage-relative"),
        ("dataset_path", "/absolute.npz", "storage-relative"),
        ("dataset_path", "C:escape.npz", "storage-relative"),
        ("dataset_path", "dir\\payload.npz", "storage-relative"),
        ("dataset_path", "dir//payload.npz", "storage-relative"),
        ("dataset_path", "./payload.npz", "storage-relative"),
        ("candidate_report", None, "candidate_report"),
        ("candidate_report", "dir\x00/payload.json", "candidate_report"),
        ("grid_shape", [1], "grid_shape"),
        ("grid_shape", {}, "grid_shape"),
        ("grid_shape", [True, 129], "grid_shape"),
        ("split_counts", [], "split_counts"),
        ("split_counts", {"train": 12}, "split_counts"),
        ("split_counts", {"train": 8, "validation": 2, "test": True}, "split_counts"),
        ("split_counts", {"train": 9, "validation": 2, "test": 2}, "sum"),
        ("fallback_features", None, "fallback_features"),
        ("fallback_features", ["Ip_MA", "Ip_MA"], "distinct"),
        ("fallback_features", [1], "trimmed string"),
        ("ragged_target_policy", [], "ragged_target_policy"),
    ],
)
def test_public_plan_refuses_invalid_consumed_metadata(
    campaign_inputs: CampaignInputs, field: str, value: Any, message: str
) -> None:
    """Reject schema, identity, path, count and nested declaration drift without coercion."""
    _change_report(campaign_inputs, lambda r: r.update({field: value}))
    with pytest.raises(CampaignPlanError, match=message):
        build_plan(campaign_inputs)


@pytest.mark.parametrize(
    ("field", "value"), [("keys", []), ("padding", None), ("point_count_key", ""), ("max_lcfs_points", -1)]
)
def test_public_plan_refuses_incomplete_ragged_policy(campaign_inputs: CampaignInputs, field: str, value: Any) -> None:
    """Validate the ragged declaration consumed by downstream reports."""
    _change_report(campaign_inputs, lambda r: r["ragged_target_policy"].update({field: value}))
    with pytest.raises(CampaignPlanError, match="ragged_target_policy"):
        build_plan(campaign_inputs)


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b"{",
        b"\xff",
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1e9999}',
        b'{"x":{"a":1,"a":2}}',
        b"[" * 10000 + b"0" + b"]" * 10000,
    ],
)
def test_public_plan_reports_real_json_decode_refusals(campaign_inputs: CampaignInputs, raw: bytes) -> None:
    """Exercise UTF-8, root, duplicate, nonfinite and actual decoder depth failures."""
    campaign_inputs.mast_dataset_report.write_bytes(raw)
    with pytest.raises(CampaignPlanError, match="dataset report|duplicate JSON"):
        build_plan(campaign_inputs)


@pytest.mark.parametrize("kind", ["missing", "directory", "null", "loop"])
def test_public_plan_reports_real_input_path_refusals(campaign_inputs: CampaignInputs, kind: str) -> None:
    """Exercise real OS/path failures without replacing production readers."""
    path = campaign_inputs.mast_dataset_report
    path.unlink()
    if kind == "directory":
        path.mkdir()
    elif kind == "null":
        path = Path("invalid\x00.json")
    elif kind == "loop":
        path.symlink_to(path.name)
    with pytest.raises(CampaignPlanError, match="cannot read dataset report"):
        build_plan(CampaignInputs(path, campaign_inputs.storage_root, campaign_inputs.public_data_root))


@pytest.mark.parametrize(
    "field",
    ["mast_dataset_report", "storage_root", "public_data_root", "require_storage_payload", "verified_storage_payload"],
)
def test_public_inputs_refuse_truthiness_and_path_coercion(campaign_inputs: CampaignInputs, field: str) -> None:
    """Literal path/control contracts apply even to callers bypassing static typing."""
    values: dict[str, Any] = dict(vars(campaign_inputs))
    values[field] = "false"
    with pytest.raises(CampaignPlanError, match=field):
        CampaignInputs(**values)


@pytest.mark.parametrize("kind", ["invalid_url", "empty", "missing", "partial"])
def test_public_plan_refuses_failed_acquisition(campaign_inputs: CampaignInputs, kind: str) -> None:
    """A real FAIL/partial acquisition report cannot prepare the public lane or outer plan."""
    manifest = campaign_inputs.public_data_root / "zenodo_1/files_manifest.json"
    if kind in {"invalid_url", "partial"}:
        payload = json.loads(manifest.read_text())
        payload["files"][0]["download_url"] = "https://zenodo.org/api/records/2/files/large.nc/content"
        if kind == "partial":
            bad = campaign_inputs.public_data_root / "bad/files_manifest.json"
            bad.parent.mkdir()
            bad.write_text(json.dumps(payload))
        else:
            manifest.write_text(json.dumps(payload))
    else:
        manifest.unlink()
        if kind == "missing":
            campaign_inputs = CampaignInputs(
                campaign_inputs.mast_dataset_report,
                campaign_inputs.storage_root,
                campaign_inputs.public_data_root / "absent",
            )
    with pytest.raises(CampaignPlanError, match="public-data acquisition failed"):
        build_plan(campaign_inputs)


def test_public_plan_binds_actual_selected_storage_bytes(campaign_inputs: CampaignInputs) -> None:
    """Hash a multi-chunk local byte fixture, without presenting it as physical training data."""
    payload = campaign_inputs.storage_root / "processed/neural_equilibrium/mast_efm_supervised_dataset.npz"
    payload.parent.mkdir(parents=True)
    content = b"local custody fixture; no physical data\n" * 35000
    payload.write_bytes(content)
    _change_report(campaign_inputs, lambda r: r.update(dataset_sha256=hashlib.sha256(content).hexdigest()))
    plan = build_plan(campaign_inputs)
    storage = plan["mast_efm_dataset"]["payload"]
    assert storage["availability_basis"] == "local_sha256"
    assert storage["sha256_verified_on_this_host"] is True
    assert storage["exists_on_this_host"] is True
    assert storage["remote_operator_attestation"] is False
    assert storage["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize(
    "kind", ["wrong_sha", "directory", "escape", "candidate_escape", "dangling", "loop", "null_root"]
)
def test_public_plan_storage_attestation_cannot_hide_local_refusals(campaign_inputs: CampaignInputs, kind: str) -> None:
    """Even explicit remote acknowledgement cannot bypass an invalid selected local binding."""
    payload = campaign_inputs.storage_root / "processed/neural_equilibrium/mast_efm_supervised_dataset.npz"
    payload.parent.mkdir(parents=True)
    if kind == "wrong_sha":
        payload.write_bytes(b"wrong local custody bytes")
    elif kind == "directory":
        payload.mkdir()
    elif kind == "escape":
        outside = campaign_inputs.storage_root.parent / "outside"
        outside.write_bytes(b"outside")
        payload.symlink_to(outside)
    elif kind == "candidate_escape":
        outside = campaign_inputs.storage_root.parent / "outside.json"
        outside.write_text("{}")
        candidate = campaign_inputs.storage_root / "converted/neural_equilibrium_reference/candidate.json"
        candidate.parent.mkdir(parents=True)
        candidate.symlink_to(outside)
    elif kind == "dangling":
        payload.symlink_to("absent.npz")
    elif kind == "loop":
        payload.symlink_to(payload.name)
    storage_root = Path("invalid\x00") if kind == "null_root" else campaign_inputs.storage_root
    with pytest.raises(CampaignPlanError, match="storage payload"):
        build_plan(
            CampaignInputs(
                campaign_inputs.mast_dataset_report,
                storage_root,
                campaign_inputs.public_data_root,
                verified_storage_payload=True,
            )
        )


def test_public_plan_attestation_is_explicit_and_shell_arguments_are_quoted(campaign_inputs: CampaignInputs) -> None:
    """Preserve remote acknowledgement without claiming local hashing, and keep spaces/literals in paths."""
    storage = campaign_inputs.storage_root / "literal $(never-execute) space"
    plan = build_plan(
        CampaignInputs(campaign_inputs.mast_dataset_report, storage, campaign_inputs.public_data_root, True, True)
    )
    payload = plan["mast_efm_dataset"]["payload"]
    assert payload["availability_basis"] == "remote_operator_attestation"
    assert payload["verified_available"] is True
    assert payload["exists_on_this_host"] is False
    assert payload["sha256_verified_on_this_host"] is False
    command = shlex.split(plan["compute_execution_package"]["exact_command"])
    assert command[command.index("--dataset-path") + 1] == str(storage / payload["relative_path"])
    assert not storage.exists()


@pytest.mark.parametrize("kind", ["count_overflow", "large_finite", "public_bytes_overflow", "finite_scaled"])
def test_public_plan_budget_ranges_remain_finite(campaign_inputs: CampaignInputs, kind: str) -> None:
    """Reject conversion/product overflow and exercise declared scaled planning ranges without benchmarking."""
    count = 10**400 if kind == "count_overflow" else 10**308 if kind == "large_finite" else 1054
    _change_report(
        campaign_inputs,
        lambda r: r.update(
            equilibria_count=count,
            split_counts={"train": count, "validation": 0, "test": 0},
            fallback_features=["Ip_MA"],
            extra={"finite": 1.25},
        ),
    )
    if kind == "public_bytes_overflow":
        manifest = campaign_inputs.public_data_root / "zenodo_1/files_manifest.json"
        data = json.loads(manifest.read_text())
        data["files"][0]["size_bytes"] = 10**400
        manifest.write_text(json.dumps(data))
    if kind in {"finite_scaled", "large_finite"}:
        plan = build_plan(campaign_inputs)
        if kind == "finite_scaled":
            assert plan["gpu_budget_estimates"][0]["nominal_gpu_hours"] == 2.0
        assert json.dumps(plan, allow_nan=False)
        assert any("fallback" in b for b in plan["mast_efm_dataset"]["blocked_before_admission"])
    else:
        with pytest.raises(CampaignPlanError, match="finite planning budgets"):
            build_plan(campaign_inputs)


def test_public_plan_uses_canonical_metadata_without_modifying_reports(tmp_path: Path) -> None:
    """Exercise actual three acquisition declarations and the published blocked dataset, without downloads."""
    root = Path(__file__).resolve().parents[1]
    dataset = root / "validation/reports/mast_efm_neural_equilibrium_dataset.json"
    before = dataset.read_bytes()
    plan = build_plan(CampaignInputs(dataset, tmp_path, root / "validation/reference_data/qlknn"))
    public = plan["prepared_dataset_lanes"][1]["public_data_summary"]
    assert public["records"] == 3 and public["deferred_files"] == 52
    assert public["deferred_bytes"] == 309_688_648_974
    assert all(str(m["path"]).startswith("validation/reference_data/") for m in public["manifests"])
    assert plan["mast_efm_dataset"]["equilibria_count"] == 527
    assert dataset.read_bytes() == before


@pytest.mark.parametrize(
    "kind", ["digest", "schema", "status", "missing_section", "grid", "nonfinite", "nonjson", "depth"]
)
def test_public_writer_refuses_invalid_payload_before_outputs(
    campaign_inputs: CampaignInputs, tmp_path: Path, kind: str
) -> None:
    """A self-digest is checked before persistence; malformed render shapes return authored errors."""
    plan = build_plan(campaign_inputs)
    if kind == "digest":
        plan["storage_root"] = "changed"
    elif kind == "schema":
        plan["schema_version"] = "old"
    elif kind == "status":
        plan["status"] = "admitted"
    elif kind == "missing_section":
        del plan["compute_execution_package"]
        _reseal(plan)
    elif kind == "grid":
        plan["mast_efm_dataset"]["grid_shape"] = []
        _reseal(plan)
    elif kind == "nonfinite":
        plan["extra"] = float("nan")
    elif kind == "nonjson":
        plan["extra"] = {1, 2}
    else:
        deep: dict[str, Any] = {}
        plan["extra"] = deep
        for _ in range(10000):
            child: dict[str, Any] = {}
            deep["child"] = child
            deep = child
    json_out, md_out = tmp_path / "refused.json", tmp_path / "refused.md"
    with pytest.raises(CampaignPlanError, match="cannot write campaign report"):
        write_report(plan, json_out, md_out)
    assert not json_out.exists() and not md_out.exists()


@pytest.mark.parametrize(
    "kind", ["same", "symlink_alias", "hardlink_alias", "directory", "blocked_parent", "null", "second_write"]
)
def test_public_writer_reports_real_output_custody_failures(
    campaign_inputs: CampaignInputs, tmp_path: Path, kind: str
) -> None:
    """Exercise actual OS/collision refusals and document sequential second-output failure."""
    plan = build_plan(campaign_inputs)
    json_out, md_out = tmp_path / "output.json", tmp_path / "output.md"
    if kind == "same":
        md_out = json_out
    elif kind == "symlink_alias":
        md_out.symlink_to(json_out)
    elif kind == "hardlink_alias":
        json_out.write_text("original custody")
        md_out.hardlink_to(json_out)
    elif kind == "directory":
        json_out.mkdir()
    elif kind == "blocked_parent":
        json_out.write_text("parent")
        json_out = json_out / "child.json"
    elif kind == "null":
        json_out = Path("invalid\x00.json")
    else:
        md_out.mkdir()
    with pytest.raises(CampaignPlanError, match="cannot write campaign report"):
        write_report(plan, json_out, md_out)
    if kind == "second_write":
        assert json.loads(json_out.read_text())["payload_sha256"] == plan["payload_sha256"]
    elif kind == "hardlink_alias":
        assert json_out.read_text() == "original custody"


def _cli_args(inputs: CampaignInputs, json_out: Path, md_out: Path) -> list[str]:
    """Pass all paths explicitly so CLI probes never overwrite canonical report pins."""
    return [
        "--mast-dataset-report",
        str(inputs.mast_dataset_report),
        "--storage-root",
        str(inputs.storage_root),
        "--public-data-root",
        str(inputs.public_data_root),
        "--json-out",
        str(json_out),
        "--report-out",
        str(md_out),
    ]


def test_actual_cli_and_inprocess_entrypoint_write_bound_reports(
    campaign_inputs: CampaignInputs, tmp_path: Path
) -> None:
    """Exercise installed Python script from a foreign cwd and explicit main arguments."""
    json_out, md_out = tmp_path / "cli.json", tmp_path / "cli.md"
    args = _cli_args(campaign_inputs, json_out, md_out)
    assert main(args) == 0
    assert "SHA-256 verified on this host: `False`" in md_out.read_text()
    script = Path(__file__).resolve().parents[1] / "validation/plan_neural_equilibrium_training_campaign.py"
    proc = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(script.parents[1] / "src")),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0 and "Traceback" not in proc.stderr
    plan = json.loads(json_out.read_text())
    assert plan["status"] == "prepared"
    assert plan["compute_execution_package"]["status"] == "prepared_not_executed"
    assert not campaign_inputs.storage_root.exists()


@pytest.mark.parametrize("kind", ["bad_public", "missing_required", "output", "bad_dataset"])
def test_actual_cli_refusals_are_authored_and_nonzero(
    campaign_inputs: CampaignInputs, tmp_path: Path, kind: str
) -> None:
    """Refused metadata, absent required storage and OS output failures cannot report CLI success."""
    json_out, md_out = tmp_path / "cli.json", tmp_path / "cli.md"
    if kind == "bad_public":
        (campaign_inputs.public_data_root / "zenodo_1/files_manifest.json").write_text("[]")
    elif kind == "output":
        md_out.mkdir()
    elif kind == "bad_dataset":
        campaign_inputs.mast_dataset_report.write_text("[]")
    args = _cli_args(campaign_inputs, json_out, md_out)
    if kind == "missing_required":
        args.append("--require-storage-payload")
    assert main(args) == 1
    script = Path(__file__).resolve().parents[1] / "validation/plan_neural_equilibrium_training_campaign.py"
    proc = subprocess.run(
        [sys.executable, str(script), *args], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 1 and "FAIL:" in proc.stderr and "Traceback" not in proc.stderr
    if kind != "output":
        assert not json_out.exists() and not md_out.exists()


def test_actual_cli_usage_and_defaults_preserve_argument_contract() -> None:
    """Exercise normal parser defaults and actual help/usage exits without creating reports."""
    args = parse_args([])
    assert args.require_storage_payload is False and args.verified_storage_payload is False
    with pytest.raises(SystemExit) as exc:
        main(["--unknown-campaign-option"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_actual_trainer_consumes_new_plan_in_dry_run_only(tmp_path: Path) -> None:
    """Exercise the real downstream reader with canonical metadata and absent tensors; create no weights."""
    from validation.train_mast_efm_neural_equilibrium import TrainingInputs, build_training_report

    root = Path(__file__).resolve().parents[1]
    dataset = root / "validation/reports/mast_efm_neural_equilibrium_dataset.json"
    storage = tmp_path / "absent-storage"
    plan = build_plan(CampaignInputs(dataset, storage, root / "validation/reference_data/qlknn"))
    json_out, md_out = tmp_path / "plan.json", tmp_path / "plan.md"
    write_report(plan, json_out, md_out)
    weights = tmp_path / "never-written-weights.npz"
    launch = build_training_report(
        TrainingInputs(
            dataset_report=dataset,
            campaign_plan=json_out,
            dataset_path=storage / plan["mast_efm_dataset"]["payload"]["relative_path"],
            weights_out=weights,
        )
    )
    assert launch["execution_mode"] == "dry_run"
    assert launch["dataset_exists_on_this_host"] is False
    assert launch["pre_run_admission"]["status"] == "fail"
    assert launch["holdout_metrics"] is None and launch["admission_ready"] is False
    assert not weights.exists() and not storage.exists()
