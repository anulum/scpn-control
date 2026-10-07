# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium training tests

"""Exercise real public training/report/CLI paths; legacy NPZ fixtures are engineering regressions, not physical evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from zipfile import ZipFile

import numpy as np
import pytest
from mast_efm_zarr_fixtures import dataset_from_reference, write_zarr

from validation.audit_mast_efm_feature_provenance import build_audit
from validation.audit_mast_efm_feature_provenance import write_report as write_feature_audit
from validation.audit_mast_efm_original_feature_sources import build_original_feature_source_audit
from validation.audit_mast_efm_original_feature_sources import write_report as write_original_audit
from validation.build_mast_efm_neural_equilibrium_dataset import (
    FEATURE_NAMES,
    DatasetInput,
    build_dataset,
)
from validation.build_mast_efm_neural_equilibrium_dataset import write_report as write_dataset_report
from validation.convert_mast_efm_neural_equilibrium_reference import convert_shot_zarr
from validation.neural_equilibrium_campaign_inputs import canonical_campaign_digest
from validation.neural_equilibrium_dataset_contracts import CANDIDATE_SCHEMA, TARGET_KEYS
from validation.plan_neural_equilibrium_training_campaign import CampaignInputs, build_plan
from validation.train_mast_efm_neural_equilibrium import (
    RESULT_TEMPLATES_SCHEMA,
    TRAINING_SCHEMA,
    TrainingInputs,
    build_result_templates,
    build_training_report,
    main,
    parse_args,
    validate_result_templates,
    validate_training_report,
    write_report,
    write_result_templates,
)


def _complete_source_declarations(dataset: Path, report_path: Path, plan_path: Path, audit_path: Path) -> None:
    """Generate full producer/auditor declarations around the retained numerical regression tensors.

    Converted source files are engineering fixtures copied from those tensors,
    with explicit current/field channels and positive RMS. The public producer
    emits a separate shadow dataset, then its declaration is bound to the legacy
    regression NPZ whose original feature/target values remain unchanged. This
    exercises format and source-declaration custody, not authentic acquisition
    or proof that legacy synthetic features equal reference-derived features.
    """
    storage = dataset.parent
    with np.load(dataset, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    declarations: list[dict[str, Any]] = []
    for shot_id in (1, 2, 3):
        selected = data["shot_id"] == shot_id
        reference: dict[str, Any] = {key: data[key][selected] for key in (*TARGET_KEYS, "shot_id", "time_s")}
        reference.update(
            r_grid_m=data["r_grid_m"],
            z_grid_m=data["z_grid_m"],
            Ip_MA=data["features"][selected, 0],
            Bt_T=data["features"][selected, 1],
            ffprime_rms_T_rad=np.ones(np.count_nonzero(selected)),
        )
        path = storage / "converted" / f"reference_{shot_id}.npz"
        store = write_zarr(storage / f"mast/level1/shot_{shot_id}/efm.zarr", dataset_from_reference(reference))
        convert_shot_zarr(shot_id=shot_id, zarr_path=store, output_path=path)
        declarations.append(
            {
                "shot_id": shot_id,
                "output_path": path.as_posix(),
                "source_path": "retained_engineering_numerical_fixture",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "selected_time_count": int(np.count_nonzero(selected)),
                "grid_shape": list(data["psirz_Wb_per_rad"].shape[1:]),
                "lcfs_points": data["lcfs_r_m"].shape[1],
                "status": "reference_candidate",
            }
        )
    candidate: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA,
        "status": "pass",
        "source": "documented_public_reference",
        "fixture_kind": "engineering_schema_contract",
        "reference_dataset_id": "mast-efm-test",
        "reference_equilibria_count": len(data["shot_id"]),
        "target_schema_status": "reference_only_no_prediction_metrics",
        "admission_ready": False,
        "errors": [],
        "shots": declarations,
    }
    candidate["payload_sha256"] = canonical_campaign_digest(candidate)
    cp = storage / "converted/candidate.json"
    cp.write_text(json.dumps(candidate))
    report = build_dataset(DatasetInput(cp, storage, storage / "producer-shadow.npz", (1,), (2,), (3,)))
    report.update(
        dataset_path=dataset.name,
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        fixture_kind="retained_engineering_training_tensor_contract",
    )
    report["payload_sha256"] = canonical_campaign_digest(report)
    write_dataset_report(report, report_path, report_path.with_suffix(".md"))
    write_feature_audit(build_audit(report_path, storage), audit_path, audit_path.with_suffix(".md"))
    original = storage / "original_source.json"
    write_original_audit(
        build_original_feature_source_audit(report_path, storage), original, original.with_suffix(".md")
    )
    root = Path(__file__).resolve().parents[1]
    plan_path.write_text(
        json.dumps(build_plan(CampaignInputs(report_path, storage, root / "validation/reference_data/qlknn")))
    )


def _write_payloads(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Retain original local NPZ engineering fixture values and bind actual format/plan/source metadata.

    This is a declared schema-contract fixture, not authenticated public MAST
    data or predictive evidence. The same public trainer/readers execute without
    substituted kernels. No canonical reports or physical corpus are changed.
    """
    dataset = tmp_path / "mast_efm_supervised_dataset.npz"
    n = 6
    features = np.column_stack([np.linspace(1.0 + col, 2.0 + col, n) for col in range(len(FEATURE_NAMES))])
    split = np.array(["train", "train", "train", "validation", "test", "test"])
    z, r = 3, 4
    base = np.arange(n * z * r, dtype=np.float64).reshape(n, z, r) / 10.0
    psirz_mask = np.ones_like(base, dtype=bool)
    base[0, 0, 0] = np.nan
    psirz_mask[0, 0, 0] = False
    lcfs_r = np.tile(np.linspace(0.4, 0.8, 5), (n, 1))
    lcfs_z = np.tile(np.linspace(-0.2, 0.2, 5), (n, 1))
    lcfs_mask = np.ones((n, 5), dtype=bool)
    lcfs_mask[0, 4] = False
    lcfs_r[0, 4] = np.nan
    lcfs_z[0, 4] = np.nan
    np.savez_compressed(
        dataset,
        features=features,
        feature_names=np.asarray(FEATURE_NAMES),
        split=split,
        shot_id=np.array([1, 1, 1, 2, 3, 3]),
        time_s=np.linspace(0.0, 0.5, n),
        r_grid_m=np.linspace(0.0, 1.0, r),
        z_grid_m=np.linspace(-1.0, 1.0, z),
        psirz_Wb_per_rad=base,
        psirz_valid_mask=psirz_mask,
        psi_axis_Wb_per_rad=np.linspace(0.0, 0.1, n),
        psi_boundary_Wb_per_rad=np.linspace(1.0, 1.1, n),
        pprime_Pa_per_Wb_rad=np.column_stack([np.linspace(1.0, 2.0, n), np.linspace(2.0, 3.0, n)]),
        pprime_valid_mask=np.ones((n, 2), dtype=bool),
        q_profile=np.column_stack([np.linspace(2.0, 3.0, n), np.linspace(3.0, 4.0, n)]),
        q_profile_valid_mask=np.ones((n, 2), dtype=bool),
        lcfs_r_m=lcfs_r,
        lcfs_z_m=lcfs_z,
        lcfs_valid_mask=lcfs_mask,
        lcfs_point_count=np.array([4, 5, 5, 5, 5, 5]),
        magnetic_axis_r_m=np.linspace(0.6, 0.7, n),
        magnetic_axis_z_m=np.linspace(-0.01, 0.01, n),
    )

    dataset_report = tmp_path / "dataset.json"
    campaign_plan = tmp_path / "plan.json"
    feature_provenance = tmp_path / "feature_provenance.json"
    _complete_source_declarations(dataset, dataset_report, campaign_plan, feature_provenance)
    original_source = tmp_path / "original_source.json"
    write_original_audit(
        build_original_feature_source_audit(dataset_report, tmp_path),
        original_source,
        original_source.with_suffix(".md"),
    )
    for source_path in (feature_provenance, original_source):
        payload = json.loads(source_path.read_text())
        payload["payload_sha256"] = hashlib.sha256(
            json.dumps({**payload, "payload_sha256": None}, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        source_path.write_text(json.dumps(payload))
    weights = tmp_path / "weights.npz"
    return dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source


def test_training_report_default_is_dry_run_and_does_not_write_weights(tmp_path: Path) -> None:
    """Exercise actual NPZ decoding in dry-run; no output weights or physical admission is produced."""
    dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source = _write_payloads(tmp_path)

    report = build_training_report(
        TrainingInputs(
            dataset_report=dataset_report,
            campaign_plan=campaign_plan,
            dataset_path=dataset,
            weights_out=weights,
            feature_provenance_report=feature_provenance,
            original_source_report=original_source,
        )
    )

    assert report["schema_version"] == TRAINING_SCHEMA
    assert report["status"] == "prepared"
    assert report["execution_mode"] == "dry_run"
    assert report["dataset_exists_on_this_host"] is True
    assert report["dataset_metadata"]["split_counts"] == {"train": 3, "validation": 1, "test": 2}
    assert report["fallback_features"] == []
    assert "The storage host is storage-only" in report["execution_host_policy"]
    assert report["pre_run_admission"]["source_provenance"]["status"] == "pass"
    assert report["pre_run_admission"]["compute_execution"]["status"] == "fail"
    assert any("compute host kind" in item for item in report["pre_run_admission"]["errors"])
    assert all("fallback" not in item for item in report["blocked_before_admission"])
    assert any("workstation or external cloud" in item for item in report["blocked_before_admission"])
    assert report["holdout_metrics"] is None
    assert validate_training_report(report) is report
    assert not weights.exists()


def test_training_report_execute_writes_weights_and_holdout_metrics(tmp_path: Path) -> None:
    """Run the existing baseline on its retained engineering fixture and verify actual coefficient serialization."""
    dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source = _write_payloads(tmp_path)

    report = build_training_report(
        TrainingInputs(
            dataset_report=dataset_report,
            campaign_plan=campaign_plan,
            dataset_path=dataset,
            weights_out=weights,
            feature_provenance_report=feature_provenance,
            original_source_report=original_source,
            compute_host_kind="workstation",
            compute_host_label="workstation-fixture",
            execute=True,
            max_flux_components=2,
        )
    )

    assert report["status"] == "executed"
    assert report["execution_mode"] == "execute"
    assert report["pre_run_admission"]["status"] == "pass"
    assert report["weights_sha256"]
    assert report["holdout_metrics"]["validation"]["psi_rmse_Wb_per_rad"] is not None
    assert report["holdout_metrics"]["test"]["magnetic_axis_rmse_m"] is not None
    assert validate_training_report(report, require_executed=True) is report
    with np.load(weights, allow_pickle=False) as payload:
        assert payload["flux_components"].shape == (2, 12)
        assert payload["axis_regression"].shape[1] == 2


def test_training_execute_refuses_storage_output_and_unadmitted_host(tmp_path: Path) -> None:
    """The actual pre-run pipeline refuses storage output or undeclared compute before fitting."""
    dataset, dataset_report, campaign_plan, _, feature_provenance, original_source = _write_payloads(tmp_path)

    with pytest.raises(ValueError, match="compute host kind"):
        build_training_report(
            TrainingInputs(
                dataset_report=dataset_report,
                campaign_plan=campaign_plan,
                dataset_path=dataset,
                weights_out=Path("/data/SCPN-CONTROL/models/weights.npz"),
                feature_provenance_report=feature_provenance,
                original_source_report=original_source,
                execute=True,
            )
        )

    with pytest.raises(ValueError, match="weights_out must not be under storage-host dataset storage"):
        build_training_report(
            TrainingInputs(
                dataset_report=dataset_report,
                campaign_plan=campaign_plan,
                dataset_path=dataset,
                weights_out=Path("/data/SCPN-CONTROL/models/weights.npz"),
                feature_provenance_report=feature_provenance,
                original_source_report=original_source,
                compute_host_kind="external_cloud",
                compute_host_label="cloud-fixture",
                execute=True,
            )
        )


def test_training_execute_refuses_failed_source_provenance(tmp_path: Path) -> None:
    """The actual pipeline refuses a changed feature-source report before creating weights."""
    dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source = _write_payloads(tmp_path)
    feature_payload = json.loads(feature_provenance.read_text(encoding="utf-8"))
    feature_payload["blocked_features"] = ["Ip_MA"]
    feature_provenance.write_text(json.dumps(feature_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="feature provenance report still has blocked features"):
        build_training_report(
            TrainingInputs(
                dataset_report=dataset_report,
                campaign_plan=campaign_plan,
                dataset_path=dataset,
                weights_out=weights,
                feature_provenance_report=feature_provenance,
                original_source_report=original_source,
                compute_host_kind="workstation",
                compute_host_label="workstation-fixture",
                execute=True,
            )
        )


def test_result_templates_bind_training_report_and_required_outputs(tmp_path: Path) -> None:
    """Bind future-result templates to a validated actual launch and exercise both output formats."""
    dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source = _write_payloads(tmp_path)
    report = build_training_report(
        TrainingInputs(
            dataset_report=dataset_report,
            campaign_plan=campaign_plan,
            dataset_path=dataset,
            weights_out=weights,
            feature_provenance_report=feature_provenance,
            original_source_report=original_source,
        )
    )

    templates = build_result_templates(report)
    write_result_templates(templates, tmp_path / "templates.json", tmp_path / "templates.md")

    assert templates["schema_version"] == RESULT_TEMPLATES_SCHEMA
    assert templates["expected_dataset_sha256"] == report["dataset_sha256"]
    assert "psi_rmse_Wb_per_rad" in templates["holdout_metrics"]["required_metrics"]
    assert "p99_ms" in templates["latency_metrics"]["required_fields"]
    assert "strict_reference_report_sha256" in templates["admission_certificate"]["required_fields"]
    assert validate_result_templates(templates, training_report=report) is templates
    markdown = (tmp_path / "templates.md").read_text(encoding="utf-8")
    assert "MAST EFM Neural-Equilibrium Result Templates" in markdown


def test_training_report_validation_rejects_digest_and_policy_tampering(tmp_path: Path) -> None:
    """Reject changed launch content carrying the old self-digest through the public validator."""
    dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source = _write_payloads(tmp_path)
    report = build_training_report(
        TrainingInputs(
            dataset_report=dataset_report,
            campaign_plan=campaign_plan,
            dataset_path=dataset,
            weights_out=weights,
            feature_provenance_report=feature_provenance,
            original_source_report=original_source,
        )
    )

    tampered = dict(report)
    tampered["blocked_before_admission"] = []
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_training_report(tampered)

    tampered = dict(report)
    tampered["execution_host_policy"] = "training may run anywhere"
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_training_report(tampered)


def test_result_templates_validation_rejects_report_binding_drift(tmp_path: Path) -> None:
    """Reject changed result bindings/fields through the actual public template validator."""
    dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source = _write_payloads(tmp_path)
    report = build_training_report(
        TrainingInputs(
            dataset_report=dataset_report,
            campaign_plan=campaign_plan,
            dataset_path=dataset,
            weights_out=weights,
            feature_provenance_report=feature_provenance,
            original_source_report=original_source,
        )
    )
    templates = build_result_templates(report)

    drifted = dict(templates)
    drifted["training_report_payload_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_result_templates(drifted, training_report=report)

    drifted = json.loads(json.dumps(templates))
    drifted["holdout_metrics"]["required_metrics"].remove("q_profile_rmse")
    with pytest.raises(ValueError, match="payload_sha256"):
        validate_result_templates(drifted)


def test_write_report_records_execute_command_and_admission_boundary(tmp_path: Path) -> None:
    """Persist a real dry-run launch and inspect explicit commands and scientific claim boundaries."""
    dataset, dataset_report, campaign_plan, weights, feature_provenance, original_source = _write_payloads(tmp_path)
    report = build_training_report(
        TrainingInputs(
            dataset_report=dataset_report,
            campaign_plan=campaign_plan,
            dataset_path=dataset,
            weights_out=weights,
            feature_provenance_report=feature_provenance,
            original_source_report=original_source,
        )
    )

    write_report(report, tmp_path / "report.json", tmp_path / "report.md")

    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "MAST EFM Neural-Equilibrium Training Launch" in markdown
    assert "--execute" in markdown
    assert "not predictive EFIT/P-EFIT admission evidence" in markdown
    assert "The storage host is storage-only" in markdown


@pytest.fixture
def canonical_inputs(tmp_path: Path) -> TrainingInputs:
    """Select actual published metadata, a freshly generated real plan and absent local tensor storage."""
    root = Path(__file__).resolve().parents[1]
    dataset = root / "validation/reports/mast_efm_neural_equilibrium_dataset.json"
    plan = build_plan(CampaignInputs(dataset, tmp_path / "missing-storage", root / "validation/reference_data/qlknn"))
    plan_path = tmp_path / "fresh-plan.json"
    plan_path.write_text(json.dumps(plan))
    return TrainingInputs(dataset, plan_path, tmp_path / "missing-dataset.npz", tmp_path / "uncreated-weights.npz")


def _reseal(payload: dict[str, Any]) -> None:
    """Independently bind the documented canonical null-field digest for isolated declaration refusals."""
    encoded = json.dumps({**payload, "payload_sha256": None}, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()


def test_actual_canonical_dry_run_and_templates(canonical_inputs: TrainingInputs, tmp_path: Path) -> None:
    """Run canonical metadata through the real launch/template/writer chain without physical data or weights."""
    report = build_training_report(canonical_inputs)
    assert report["execution_mode"] == "dry_run" and report["admission_ready"] is False
    assert report["pre_run_admission"]["status"] == "fail"
    # The actual preserved source files carry stale self-digests; declarations cannot bypass checking.
    assert report["pre_run_admission"]["source_provenance"]["status"] == "fail"
    assert any("payload_sha256" in error for error in report["pre_run_admission"]["errors"])
    assert validate_training_report(report) is report
    templates = build_result_templates(report)
    assert validate_result_templates(templates, training_report=report) is templates
    write_report(report, tmp_path / "launch.json", tmp_path / "launch.md")
    write_result_templates(templates, tmp_path / "templates.json", tmp_path / "templates.md")
    assert json.loads((tmp_path / "launch.json").read_text())["payload_sha256"] == report["payload_sha256"]
    assert not canonical_inputs.weights_out.exists() and not canonical_inputs.dataset_path.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("execute", "false", "boolean"),
        ("ridge_alpha", True, "ridge_alpha"),
        ("ridge_alpha", "1e-6", "ridge_alpha"),
        ("ridge_alpha", 0, "ridge_alpha"),
        ("ridge_alpha", float("nan"), "ridge_alpha"),
        ("ridge_alpha", 10**400, "ridge_alpha"),
        ("max_flux_components", True, "positive integer"),
        ("max_flux_components", 0, "positive integer"),
        ("compute_host_kind", [], "compute_host_kind"),
        ("compute_host_kind", "storage_host", "compute_host_kind"),
        ("compute_host_label", 1, "trimmed string"),
        ("compute_host_label", " spaced ", "trimmed string"),
        ("dataset_path", "path", "must be a Path"),
        ("dataset_path", Path("bad\x00.npz"), "NUL"),
        ("weights_out", Path("weights.txt"), ".npz"),
    ],
)
def test_public_training_controls_refuse_coercion(
    canonical_inputs: TrainingInputs, field: str, value: Any, message: str
) -> None:
    """Reject runtime controls before metadata reads or numerical execution."""
    with pytest.raises(ValueError, match=message):
        replace(canonical_inputs, **{field: value})


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b"{",
        b"\xff",
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1e9999}',
        b'{"a":{"x":1,"x":2}}',
        b"[" * 10000 + b"0" + b"]" * 10000,
    ],
)
def test_public_training_plan_reader_refuses_actual_decoder_errors(
    canonical_inputs: TrainingInputs, raw: bytes
) -> None:
    """Real selected plan bytes must be finite unique UTF-8 JSON; no patched decoder or private helper is used."""
    canonical_inputs.campaign_plan.write_bytes(raw)
    with pytest.raises(ValueError, match="cannot read training metadata"):
        build_training_report(canonical_inputs)


@pytest.mark.parametrize(
    "kind",
    [
        "missing_plan",
        "plan_directory",
        "plan_loop",
        "dataset_directory",
        "dataset_dangling",
        "dataset_loop",
        "execute_missing",
    ],
)
def test_public_training_reports_actual_path_and_absence_refusals(canonical_inputs: TrainingInputs, kind: str) -> None:
    """Selected path errors and missing execute tensors refuse before any weights write."""
    inputs = canonical_inputs
    if kind.startswith("plan_") or kind == "missing_plan":
        inputs.campaign_plan.unlink()
        if kind == "plan_directory":
            inputs.campaign_plan.mkdir()
        elif kind == "plan_loop":
            inputs.campaign_plan.symlink_to(inputs.campaign_plan.name)
    elif kind == "dataset_directory":
        inputs.dataset_path.mkdir()
    elif kind == "dataset_dangling":
        inputs.dataset_path.symlink_to("absent-target.npz")
    elif kind == "dataset_loop":
        inputs.dataset_path.symlink_to(inputs.dataset_path.name)
    else:
        inputs = replace(inputs, execute=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        build_training_report(inputs)
    assert not inputs.weights_out.exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p.update(schema_version="old"), "schema_version"),
        (lambda p: p.update(status="failed"), "prepared"),
        (lambda p: p.update(storage_root="changed"), "payload_sha256"),
        (lambda p: p.update(mast_efm_dataset=[]), "MAST dataset"),
        (lambda p: p["mast_efm_dataset"].update(reference_dataset_id="wrong"), "reference_dataset_id"),
        (lambda p: p["mast_efm_dataset"].update(payload={}), "payload SHA"),
        (lambda p: p["compute_execution_package"].update(dataset_sha256="0" * 64), "compute package"),
        (
            lambda p: p["compute_execution_package"].update(admitted_compute_host_kinds=["storage_host"]),
            "host and storage",
        ),
        (lambda p: p.update(prepared_dataset_lanes={}), "dataset lanes"),
        (lambda p: p.update(prepared_dataset_lanes=[1]), "dataset lanes"),
        (lambda p: p.update(prepared_dataset_lanes=[]), "exactly one"),
        (lambda p: p["prepared_dataset_lanes"][1].update(public_data_summary=[]), "public-data acquisition"),
        (
            lambda p: p["prepared_dataset_lanes"][1]["public_data_summary"].update(status="fail"),
            "public-data acquisition",
        ),
        (lambda p: p["prepared_dataset_lanes"][1]["public_data_summary"].update(records=True), "nonnegative integer"),
        (lambda p: p["prepared_dataset_lanes"][1]["public_data_summary"].update(files=0), "consistent coverage"),
        (lambda p: p["prepared_dataset_lanes"][1]["public_data_summary"].update(manifests=[]), "manifest count"),
    ],
)
def test_public_training_refuses_actual_plan_binding_drift(
    canonical_inputs: TrainingInputs, change: Callable[[dict[str, Any]], object], message: str
) -> None:
    """Mutate actual canonical-plan declarations; even resealing cannot bypass semantic source/counter checks."""
    plan = json.loads(canonical_inputs.campaign_plan.read_text())
    change(plan)
    if message != "payload_sha256":
        _reseal(plan)
    canonical_inputs.campaign_plan.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=message):
        build_training_report(canonical_inputs)


@pytest.mark.parametrize(
    ("kind", "change", "message"),
    [
        ("feature", lambda p: p.update(schema_version="old"), "unsupported"),
        ("feature", lambda p: p.update(reference_dataset_id="wrong"), "reference_dataset_id"),
        ("feature", lambda p: p.update(blocked_features=["Ip_MA"]), "blocked features"),
        ("feature", lambda p: p.update(feature_status={}), "no feature_status"),
        ("feature", lambda p: p.update(feature_status={"Ip_MA": {"status": "resolved"}}), "all three"),
        ("feature", lambda p: p["feature_status"]["Ip_MA"].update(status="unresolved"), "unresolved features"),
        ("original", lambda p: p.update(schema_version="old"), "unsupported"),
        ("original", lambda p: p.update(reference_dataset_id="wrong"), "reference_dataset_id"),
        ("original", lambda p: p.update(status="blocked"), "not source_ready"),
        ("original", lambda p: p.update(can_rebuild_dataset_now="true"), "rebuild readiness"),
        ("original", lambda p: p.update(blocked_features=["Ip_MA"]), "blocked features"),
    ],
)
def test_public_training_preserves_real_provenance_refusals(
    canonical_inputs: TrainingInputs,
    tmp_path: Path,
    kind: str,
    change: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    """Isolate malformed declarations copied from actual source audits; no passing physical provenance is invented."""
    source = (
        canonical_inputs.feature_provenance_report if kind == "feature" else canonical_inputs.original_source_report
    )
    payload = json.loads(source.read_text())
    change(payload)
    _reseal(payload)
    selected = tmp_path / "declared-source.json"
    selected.write_text(json.dumps(payload))
    inputs = (
        replace(canonical_inputs, feature_provenance_report=selected)
        if kind == "feature"
        else replace(canonical_inputs, original_source_report=selected)
    )
    report = build_training_report(inputs)
    assert report["pre_run_admission"]["source_provenance"]["status"] == "fail"
    assert any(message in e for e in report["pre_run_admission"]["errors"])
    assert not inputs.weights_out.exists()


@pytest.mark.parametrize(
    "kind",
    ["missing_sources", "invalid_source", "storage_label", "dataset_alias", "hardlink_dataset_alias", "weights_loop"],
)
def test_actual_source_and_compute_custody_refusals(
    canonical_inputs: TrainingInputs, tmp_path: Path, kind: str
) -> None:
    """Exercise real source IO and compute/output policy paths without substituting host or storage helpers."""
    inputs = canonical_inputs
    if kind == "missing_sources":
        inputs = replace(
            inputs,
            feature_provenance_report=tmp_path / "absent-feature.json",
            original_source_report=tmp_path / "absent-original.json",
        )
    elif kind == "invalid_source":
        source = tmp_path / "invalid-source.json"
        source.write_text("[]")
        inputs = replace(inputs, feature_provenance_report=source)
    elif kind == "storage_label":
        inputs = replace(inputs, compute_host_kind="workstation", compute_host_label="storage_host-local")
    elif kind == "dataset_alias":
        inputs = replace(inputs, weights_out=inputs.dataset_path)
    elif kind == "weights_loop":
        inputs.weights_out.symlink_to(inputs.weights_out.name)
        with pytest.raises(ValueError, match="cannot resolve weights path"):
            build_training_report(inputs)
        return
    else:
        # Byte custody is observed before tensor loading; an actual invalid selected archive refuses.
        inputs.dataset_path.write_bytes(b"invalid actual archive bytes")
        inputs.weights_out.hardlink_to(inputs.dataset_path)
        with pytest.raises(ValueError, match="SHA-256"):
            build_training_report(inputs)
        return
    report = build_training_report(inputs)
    assert report["pre_run_admission"]["status"] == "fail"
    if kind in {"dataset_alias", "weights_loop"}:
        assert any("custody" in e or "overwrite" in e for e in report["pre_run_admission"]["errors"])


def _cli_args(inputs: TrainingInputs, output: Path) -> list[str]:
    """Specify every actual CLI input/output, preserving canonical report pins."""
    return [
        "--dataset-report",
        str(inputs.dataset_report),
        "--campaign-plan",
        str(inputs.campaign_plan),
        "--dataset-path",
        str(inputs.dataset_path),
        "--weights-out",
        str(inputs.weights_out),
        "--json-out",
        str(output / "launch.json"),
        "--report-out",
        str(output / "launch.md"),
        "--templates-json-out",
        str(output / "templates.json"),
        "--templates-report-out",
        str(output / "templates.md"),
    ]


def test_actual_standalone_cli_writes_all_four_reports(canonical_inputs: TrainingInputs, tmp_path: Path) -> None:
    """Exercise real main and a cold foreign-cwd process, without executing training or updating canonical reports."""
    args = _cli_args(canonical_inputs, tmp_path)
    assert main(args) == 0
    script = Path(__file__).resolve().parents[1] / "validation/train_mast_efm_neural_equilibrium.py"
    proc = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(script.parents[1] / "src")),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0 and "Traceback" not in proc.stderr
    launch = json.loads((tmp_path / "launch.json").read_text())
    assert launch["execution_mode"] == "dry_run" and launch["pre_run_admission"]["status"] == "fail"
    assert not canonical_inputs.weights_out.exists()


@pytest.mark.parametrize("kind", ["execute_absent", "same_outputs", "protected_input", "second_write", "bad_plan"])
def test_actual_cli_refuses_before_false_success(canonical_inputs: TrainingInputs, tmp_path: Path, kind: str) -> None:
    """Actual CLI failure codes cover absent execution input, output custody, IO and invalid metadata."""
    args = _cli_args(canonical_inputs, tmp_path)
    if kind == "execute_absent":
        args.append("--execute")
    elif kind == "same_outputs":
        args[args.index("--report-out") + 1] = args[args.index("--json-out") + 1]
    elif kind == "protected_input":
        args[args.index("--json-out") + 1] = str(canonical_inputs.campaign_plan)
    elif kind == "second_write":
        (tmp_path / "launch.md").mkdir()
    else:
        canonical_inputs.campaign_plan.write_text("[]")
    assert main(args) == 1 and not canonical_inputs.weights_out.exists()


def test_actual_cli_argument_defaults_help_and_usage() -> None:
    """Exercise real parsing and help/usage exits without writing historical default reports."""
    assert parse_args([]).execute is False
    with pytest.raises(SystemExit) as exc:
        main(["--unknown-training-option"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def _format_inputs(tmp_path: Path, *, execute: bool = False) -> TrainingInputs:
    """Route the retained NPZ engineering fixture through the actual public trainer and provenance paths."""
    dataset, dataset_report, plan, weights, feature, original = _write_payloads(tmp_path)
    return TrainingInputs(
        dataset_report,
        plan,
        dataset,
        weights,
        feature,
        original,
        compute_host_kind="workstation",
        compute_host_label="schema-contract-fixture",
        execute=execute,
        max_flux_components=2,
    )


def _bind_selected_bytes(inputs: TrainingInputs) -> None:
    """Rebind modified selected test bytes and regenerate their actual public plan, without changing canonical evidence."""
    metadata = json.loads(inputs.dataset_report.read_text())
    metadata["dataset_sha256"] = hashlib.sha256(inputs.dataset_path.read_bytes()).hexdigest()
    _reseal(metadata)
    inputs.dataset_report.write_text(json.dumps(metadata))
    audit = json.loads(inputs.feature_provenance_report.read_text())
    audit.update(
        dataset_sha256=metadata["dataset_sha256"],
        dataset_payload_sha256=metadata["payload_sha256"],
        dataset_report_sha256=hashlib.sha256(inputs.dataset_report.read_bytes()).hexdigest(),
    )
    _reseal(audit)
    write_feature_audit(audit, inputs.feature_provenance_report, inputs.feature_provenance_report.with_suffix(".md"))
    write_original_audit(
        build_original_feature_source_audit(inputs.dataset_report, inputs.dataset_path.parent),
        inputs.original_source_report,
        inputs.original_source_report.with_suffix(".md"),
    )
    root = Path(__file__).resolve().parents[1]
    plan = build_plan(
        CampaignInputs(inputs.dataset_report, inputs.dataset_path.parent, root / "validation/reference_data/qlknn")
    )
    inputs.campaign_plan.write_text(json.dumps(plan))


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing", "missing required keys"),
        ("features", "features"),
        ("names", "feature_names"),
        ("split_rank", "split labels"),
        ("split_unknown", "split labels"),
        ("split_counts", "split_counts"),
        ("shots", "shot_id"),
        ("shot_leak", "shot-held-out"),
        ("time", "time_s"),
        ("grid", "strictly increasing"),
        ("flux_shape", "target/boolean mask"),
        ("flux_mask", "boolean mask"),
        ("flux_nonfinite", "finite on valid"),
        ("profile_rank", "profile columns"),
        ("axis", "magnetic_axis"),
        ("lcfs_count", "positive integer"),
        ("lcfs_padding", "contiguous valid"),
    ],
)
def test_actual_public_npz_contract_refusals(tmp_path: Path, kind: str, message: str) -> None:
    """Mutate the retained actual archive format and exercise whole decoding/admission through public training."""
    inputs = _format_inputs(tmp_path)
    with np.load(inputs.dataset_path, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    if kind == "missing":
        del data["q_profile"]
    elif kind == "features":
        data["features"] = data["features"].astype(bool)
    elif kind == "names":
        data["feature_names"] = data["feature_names"][::-1]
    elif kind == "split_rank":
        data["split"] = data["split"][:, None]
    elif kind == "split_unknown":
        data["split"][0] = "other"
    elif kind == "split_counts":
        data["split"][0] = "test"
    elif kind == "shots":
        data["shot_id"][0] = 0
    elif kind == "shot_leak":
        data["shot_id"][:] = 1
    elif kind == "time":
        data["time_s"][0] = -1
    elif kind == "grid":
        data["r_grid_m"][1] = data["r_grid_m"][0]
    elif kind == "flux_shape":
        data["psirz_Wb_per_rad"] = data["psirz_Wb_per_rad"][:, :1, :]
    elif kind == "flux_mask":
        data["psirz_valid_mask"] = data["psirz_valid_mask"].astype(str)
    elif kind == "flux_nonfinite":
        data["psirz_Wb_per_rad"][1, 0, 0] = np.inf
    elif kind == "profile_rank":
        data["q_profile"] = data["q_profile"][:, 0]
    elif kind == "axis":
        data["magnetic_axis_z_m"][0] = np.nan
    elif kind == "lcfs_count":
        data["lcfs_point_count"] = data["lcfs_point_count"].astype(np.float64)
    else:
        data["lcfs_valid_mask"][0] = np.asarray([True, False, True, True, False])
        data["lcfs_point_count"][0] = 3
    np.savez_compressed(inputs.dataset_path, **data)
    _bind_selected_bytes(inputs)
    with pytest.raises(ValueError, match=message):
        build_training_report(inputs)
    assert not inputs.weights_out.exists()


@pytest.mark.parametrize(
    "kind",
    [
        "invalid_archive",
        "object_archive",
        "npy_instead_of_npz",
        "insufficient_train",
        "numerical_overflow",
        "dataset_output_alias",
    ],
)
def test_actual_format_and_numeric_execution_refusals(tmp_path: Path, kind: str) -> None:
    """Actual malformed formats, held-out counts, overflow and custody fail without substituted kernels."""
    inputs = _format_inputs(tmp_path, execute=True)
    with np.load(inputs.dataset_path, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    if kind == "invalid_archive":
        inputs.dataset_path.write_bytes(b"invalid selected NPZ archive")
    elif kind == "object_archive":
        data["features"] = data["features"].astype(object)
        np.savez_compressed(inputs.dataset_path, **data)
    elif kind == "npy_instead_of_npz":
        with inputs.dataset_path.open("wb") as handle:
            np.save(handle, data["features"], allow_pickle=False)
    elif kind == "insufficient_train":
        data["split"] = np.asarray(["train", "validation", "test", "test", "test", "test"])
        data["shot_id"] = np.asarray([1, 2, 3, 3, 3, 3])
        metadata = json.loads(inputs.dataset_report.read_text())
        metadata["split_counts"] = {"train": 1, "validation": 1, "test": 4}
        inputs.dataset_report.write_text(json.dumps(metadata))
        np.savez_compressed(inputs.dataset_path, **data)
        _complete_source_declarations(
            inputs.dataset_path, inputs.dataset_report, inputs.campaign_plan, inputs.feature_provenance_report
        )
    elif kind == "numerical_overflow":
        data["features"][:, 0] = np.finfo(np.float64).max
        np.savez_compressed(inputs.dataset_path, **data)
    else:
        inputs = replace(inputs, weights_out=inputs.dataset_path)
    _bind_selected_bytes(inputs)
    with pytest.raises(ValueError):
        build_training_report(inputs)


def test_actual_baseline_missing_observations_stay_unobserved(tmp_path: Path) -> None:
    """Original numerical pipeline fills from training observations and keeps entirely masked metrics null."""
    inputs = _format_inputs(tmp_path, execute=True)
    with np.load(inputs.dataset_path, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    data["features"][:, 0] = 1.0
    for key, mask_key in [("pprime_Pa_per_Wb_rad", "pprime_valid_mask"), ("q_profile", "q_profile_valid_mask")]:
        data[mask_key][:] = False
        data[key][:] = np.nan
    np.savez_compressed(inputs.dataset_path, **data)
    _bind_selected_bytes(inputs)
    report = build_training_report(inputs)
    assert report["holdout_metrics"]["test"]["q_profile_rmse"] is None
    assert report["holdout_metrics"]["train"]["pprime_rmse_Pa_per_Wb_rad"] is None
    assert report["admission_ready"] is False and report["strict_artefact_emitted"] is False
    assert validate_training_report(report, require_executed=True) is report


@pytest.fixture
def executed_format_report(tmp_path: Path) -> dict[str, Any]:
    """Produce an actual bound engineering launch; this is not canonical physical training evidence."""
    return build_training_report(_format_inputs(tmp_path, execute=True))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "old"),
        ("status", []),
        ("execution_mode", []),
        ("payload_sha256", None),
        ("claim_boundary", []),
        ("execution_host_policy", "The storage host is storage-only, but allowed"),
        ("admission_ready", True),
        ("strict_artefact_emitted", True),
        ("required_targets", []),
        ("required_targets", [1]),
        ("weights_path", None),
        ("weights_path", ""),
        ("weights_path", "bad\x00.npz"),
        ("weights_path", "/data/SCPN-CONTROL/models/weights.npz"),
        ("pre_run_admission", []),
        ("dataset_exists_on_this_host", "true"),
        ("holdout_metrics", []),
        ("holdout_metrics", {"train": {}}),
    ],
)
def test_public_launch_validator_refuses_invalid_declarations(
    executed_format_report: dict[str, Any], field: str, value: Any
) -> None:
    """Resealing real producer output cannot bypass typed policy, target or execution declarations."""
    report = json.loads(json.dumps(executed_format_report))
    report[field] = value
    if field != "payload_sha256":
        _reseal(report)
    with pytest.raises(ValueError):
        validate_training_report(report)


@pytest.mark.parametrize("value", [True, "0.1", -1.0, 10**400])
def test_public_launch_validator_refuses_invalid_metrics(executed_format_report: dict[str, Any], value: Any) -> None:
    """Metrics cannot become booleans, text, negative values or overflowing integers after resealing."""
    report = json.loads(json.dumps(executed_format_report))
    report["holdout_metrics"]["test"]["q_profile_rmse"] = value
    _reseal(report)
    with pytest.raises(ValueError, match="finite nonnegative"):
        validate_training_report(report)


@pytest.mark.parametrize(
    "kind",
    [
        "weight_tamper",
        "missing_weights",
        "child_type",
        "child_status",
        "errors_type",
        "verified_type",
        "missing_required",
        "finite",
        "nonjson",
        "root",
    ],
)
def test_public_launch_custody_and_json_refusals(executed_format_report: dict[str, Any], kind: str) -> None:
    """Actual produced launches retain weight custody, child diagnostic consistency and finite JSON contracts."""
    report = json.loads(json.dumps(executed_format_report))
    if kind == "weight_tamper":
        Path(report["weights_path"]).write_bytes(b"tampered actual weight bytes")
    elif kind == "missing_weights":
        Path(report["weights_path"]).unlink()
    elif kind == "child_type":
        report["pre_run_admission"]["source_provenance"] = []
    elif kind == "child_status":
        report["pre_run_admission"]["source_provenance"]["status"] = "fail"
    elif kind == "errors_type":
        report["pre_run_admission"]["errors"] = None
    elif kind == "verified_type":
        report["pre_run_admission"]["dataset_sha256_verified"] = "true"
    elif kind == "missing_required":
        del report["holdout_metrics"]["test"]["q_profile_rmse"]
    elif kind == "finite":
        report["extra"] = float("nan")
    elif kind == "nonjson":
        report["extra"] = {1, 2}
    else:
        with pytest.raises(ValueError, match="must be an object"):
            validate_training_report(cast(Any, []))
        return
    if kind not in {"finite", "nonjson"}:
        _reseal(report)
    with pytest.raises(ValueError):
        validate_training_report(report, require_executed=True)


@pytest.mark.parametrize(
    "kind",
    [
        "root",
        "nonfinite",
        "nonjson",
        "missing_section",
        "schema",
        "policy_type",
        "boundary_type",
        "members",
        "split",
        "enum",
        "digest",
        "binding",
        "dataset_binding",
    ],
)
def test_public_template_contract_refusals(canonical_inputs: TrainingInputs, kind: str) -> None:
    """Actual producer templates refuse malformed schema/policy/binding declarations through their public validator."""
    launch = build_training_report(canonical_inputs)
    templates = build_result_templates(launch)
    if kind == "root":
        with pytest.raises(ValueError, match="must be an object"):
            validate_result_templates(cast(Any, []))
        return
    if kind == "nonfinite":
        templates["extra"] = float("inf")
    elif kind == "nonjson":
        templates["extra"] = {1, 2}
    elif kind == "missing_section":
        templates["holdout_metrics"] = []
        templates["admission_certificate"] = []
    elif kind == "schema":
        templates["latency_metrics"]["schema_version"] = "wrong-schema"
    elif kind == "policy_type":
        templates["expected_weight_path_policy"] = [templates["expected_weight_path_policy"]]
    elif kind == "boundary_type":
        templates["claim_boundary"] = [templates["claim_boundary"]]
    elif kind == "members":
        templates["holdout_metrics"]["required_metrics"].append(templates["holdout_metrics"]["required_metrics"][0])
    elif kind == "split":
        templates["holdout_metrics"]["required_splits"] = ["train", "test"]
    elif kind == "enum":
        templates["admission_certificate"]["admission_status_enum"] = ["pass"]
    elif kind == "digest":
        templates["payload_sha256"] = None
    elif kind == "binding":
        templates["training_report_payload_sha256"] = "a" * 64
    else:
        templates["expected_dataset_sha256"] = "a" * 64
    if kind not in {"root", "nonfinite", "nonjson", "digest"}:
        _reseal(templates)
    with pytest.raises(ValueError):
        validate_result_templates(templates, training_report=launch)


def test_actual_executed_report_persistence(executed_format_report: dict[str, Any], tmp_path: Path) -> None:
    """Persist actual executed format-regression weights/holdout declarations, then refuse altered weight bytes."""
    output = tmp_path / "executed-report.json"
    markdown = tmp_path / "executed-report.md"
    write_report(executed_format_report, output, markdown)
    assert json.loads(output.read_text()) == executed_format_report
    assert "## Holdout metrics" in markdown.read_text()
    Path(executed_format_report["weights_path"]).write_bytes(b"changed persisted weights")
    with pytest.raises(ValueError, match="cannot write training launch report"):
        write_report(executed_format_report, tmp_path / "refused.json", tmp_path / "refused.md")
    assert not (tmp_path / "refused.json").exists()


@pytest.mark.parametrize("kind", ["same_path", "symlink", "hardlink", "second_io", "invalid_template"])
def test_actual_template_writer_custody_and_io(canonical_inputs: TrainingInputs, tmp_path: Path, kind: str) -> None:
    """Actual public template writes refuse aliases and invalid declarations, and expose sequential IO partial output."""
    templates = build_result_templates(build_training_report(canonical_inputs))
    json_out = tmp_path / "future-results.json"
    markdown_out = tmp_path / "future-results.md"
    if kind == "same_path":
        markdown_out = json_out
    elif kind == "symlink":
        markdown_out.symlink_to(json_out)
    elif kind == "hardlink":
        json_out.write_text("original custody bytes")
        os.link(json_out, markdown_out)
    elif kind == "second_io":
        markdown_out.mkdir()
    else:
        templates["holdout_metrics"] = None
        _reseal(templates)
    before = json_out.read_bytes() if json_out.exists() else None
    with pytest.raises(ValueError, match="cannot write training result templates"):
        write_result_templates(templates, json_out, markdown_out)
    if kind == "second_io":
        assert json.loads(json_out.read_text()) == templates
    elif before is None:
        assert not json_out.exists()
    else:
        assert json_out.read_bytes() == before


def test_actual_relative_source_paths_and_fallback_refusal(tmp_path: Path) -> None:
    """Public training renders genuine relative paths and retains fallback declarations as source-admission blockers."""
    inputs = _format_inputs(tmp_path)
    metadata = json.loads(inputs.dataset_report.read_text())
    metadata["fallback_features"] = ["Ip_MA"]
    del metadata["feature_source_policy"]["Ip_MA"]
    _reseal(metadata)
    inputs.dataset_report.write_text(json.dumps(metadata))
    root = Path(__file__).resolve().parents[1]
    inputs.campaign_plan.write_text(
        json.dumps(
            build_plan(CampaignInputs(inputs.dataset_report, tmp_path, root / "validation/reference_data/qlknn"))
        )
    )
    relative = replace(
        inputs, feature_provenance_report=Path("validation/reports/mast_efm_feature_provenance_audit.json")
    )
    report = build_training_report(relative)
    assert report["pre_run_admission"]["source_provenance"]["feature_provenance_report"] == str(
        relative.feature_provenance_report
    )
    assert "dataset report still declares fallback features" in report["pre_run_admission"]["errors"]
    assert "replace fallback Ip_MA" in report["blocked_before_admission"][0]
    assert not inputs.weights_out.exists()


def test_actual_large_ridge_keeps_unpenalised_intercept(tmp_path: Path) -> None:
    """Actual ridge execution retains each training mean with large finite regularisation instead of cancelling intercept count."""
    inputs = replace(_format_inputs(tmp_path, execute=True), ridge_alpha=1.0e100)
    report = build_training_report(inputs)
    with (
        np.load(inputs.dataset_path, allow_pickle=False) as data,
        np.load(inputs.weights_out, allow_pickle=False) as weights,
    ):
        expected_axis = np.column_stack([data["magnetic_axis_r_m"][:3], data["magnetic_axis_z_m"][:3]]).mean(axis=0)
        np.testing.assert_allclose(weights["axis_regression"][-1], expected_axis, rtol=1.0e-12)
        np.testing.assert_allclose(
            weights["q_profile_regression"][-1], data["q_profile"][:3].mean(axis=0), rtol=1.0e-12
        )
        assert np.all(np.isfinite(weights["flux_regression"]))
    assert report["holdout_metrics"]["test"]["magnetic_axis_rmse_m"] < 1.0
    assert report["admission_ready"] is False


def test_actual_nonfinite_fit_refuses_before_weights(tmp_path: Path) -> None:
    """The actual NumPy fit refuses nonfinite coefficients from finite ill-conditioned declarations without mocking kernels."""
    inputs = replace(_format_inputs(tmp_path, execute=True), ridge_alpha=1.0e-12)
    with np.load(inputs.dataset_path, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    data["features"][1, 0] += 1.0e-6
    data["q_profile"][:3] = np.asarray([[1.0e306, 1.0e306], [-1.0e306, -1.0e306], [1.0e306, 1.0e306]])
    np.savez_compressed(inputs.dataset_path, **data)
    _bind_selected_bytes(inputs)
    with pytest.raises(ValueError, match="finite coefficients and predictions before weights persistence"):
        build_training_report(inputs)
    assert not inputs.weights_out.exists()


def test_actual_launch_command_preserves_selected_inputs_and_controls(tmp_path: Path) -> None:
    """Round-trip the public launch command through the real parser, including quoted paths and compute controls."""
    selected = tmp_path / "operator's compute folder"
    selected.mkdir()
    inputs = replace(
        _format_inputs(selected),
        compute_host_kind="external_cloud",
        compute_host_label="cloud label $(literal)",
        ridge_alpha=0.125,
        max_flux_components=1,
    )
    report = build_training_report(inputs)
    args = parse_args(shlex.split(report["run_command"])[2:])
    for field in (
        "dataset_report",
        "campaign_plan",
        "dataset_path",
        "weights_out",
        "feature_provenance_report",
        "original_source_report",
        "compute_host_kind",
        "compute_host_label",
        "ridge_alpha",
        "max_flux_components",
    ):
        assert getattr(args, field) == getattr(inputs, field)
    assert args.execute is True and report["execution_mode"] == "dry_run"
    assert not inputs.weights_out.exists()


@pytest.mark.parametrize("key", ["features", "magnetic_axis_z_m", "q_profile", "psirz_Wb_per_rad"])
def test_public_trainer_requires_finite_computation_dtype(tmp_path: Path, key: str) -> None:
    """Real NPZ observations finite in a wider storage dtype must remain finite in the trainer's float64 representation."""
    inputs = _format_inputs(tmp_path)
    with np.load(inputs.dataset_path, allow_pickle=False) as payload:
        data = {name: payload[name] for name in payload.files}
    data[key] = data[key].astype(np.longdouble)
    data[key][1] = np.finfo(np.longdouble).max
    assert np.all(np.isfinite(data[key][1]))
    np.savez_compressed(inputs.dataset_path, **data)
    _bind_selected_bytes(inputs)
    if np.finfo(np.longdouble).max > np.finfo(np.float64).max:
        with (
            pytest.raises(ValueError, match="finite"),
            pytest.warns(RuntimeWarning, match="overflow encountered in cast"),
        ):
            build_training_report(inputs)
    else:
        # Some native platforms use float64 for longdouble. Its largest value
        # remains representable, so dry-run shape admission remains supported.
        report = build_training_report(inputs)
        assert report["dataset_metadata"]["equilibria_count"] == 6
        assert report["admission_ready"] is False
    assert not inputs.weights_out.exists()


@pytest.mark.parametrize("kind", ["shape", "nonfinite"])
def test_public_trainer_exercises_shared_producer_feature_contract(tmp_path: Path, kind: str) -> None:
    """The real public NPZ consumer enforces the producer's exact12-column finite feature matrix contract before weights."""
    inputs = _format_inputs(tmp_path)
    with np.load(inputs.dataset_path, allow_pickle=False) as payload:
        data = {name: payload[name] for name in payload.files}
    if kind == "shape":
        data["features"] = data["features"][:, :-1]
    else:
        data["features"][1, 0] = np.nan
    np.savez_compressed(inputs.dataset_path, **data)
    _bind_selected_bytes(inputs)
    with pytest.raises(ValueError, match="features|feature matrix"):
        build_training_report(inputs)
    assert not inputs.weights_out.exists()


@pytest.mark.parametrize("kind", ["shape", "dtype"])
def test_public_trainer_requires_real_aligned_time_values(tmp_path: Path, kind: str) -> None:
    """The actual NPZ consumer rejects short or boolean time vectors before numerical fitting."""
    inputs = _format_inputs(tmp_path)
    with np.load(inputs.dataset_path, allow_pickle=False) as payload:
        data = {name: payload[name] for name in payload.files}
    if kind == "shape":
        data["time_s"] = data["time_s"][:-1]
    else:
        data["time_s"] = data["time_s"].astype(bool)
    np.savez_compressed(inputs.dataset_path, **data)
    _bind_selected_bytes(inputs)
    with pytest.raises(ValueError, match="time_s must be finite real values with shape"):
        build_training_report(inputs)
    assert not inputs.weights_out.exists()


@pytest.mark.parametrize("rebind", [False, True])
def test_public_trainer_binds_exact_archive_container_bytes(tmp_path: Path, rebind: bool) -> None:
    """Actual public supervised loading binds the archive digest even when a container-only change preserves all arrays."""
    inputs = _format_inputs(tmp_path)
    with np.load(inputs.dataset_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    prior = hashlib.sha256(inputs.dataset_path.read_bytes()).hexdigest()
    with ZipFile(inputs.dataset_path, "a") as archive:
        archive.comment = b"engineering supervised container custody regression"
    current = hashlib.sha256(inputs.dataset_path.read_bytes()).hexdigest()
    assert current != prior
    with np.load(inputs.dataset_path, allow_pickle=False) as payload:
        assert set(payload.files) == set(arrays)
        for name in payload.files:
            if arrays[name].dtype.kind in "fiub":
                assert np.array_equal(payload[name], arrays[name], equal_nan=True)
            else:
                assert np.array_equal(payload[name], arrays[name])
    if rebind:
        _bind_selected_bytes(inputs)
        report = build_training_report(inputs)
        assert report["dataset_sha256"] == current and report["status"] == "prepared"
    else:
        with pytest.raises(ValueError, match="dataset payload SHA-256 does not match the dataset report"):
            build_training_report(inputs)
    assert not inputs.weights_out.exists()


@pytest.mark.parametrize(
    "field",
    [
        "dataset_report_sha256",
        "dataset_payload_sha256",
        "dataset_sha256",
        "dataset_path",
        "candidate_report",
        "reference_dataset_id",
        "reference_count",
        "shot_id",
        "reference_path",
        "reference_sha256",
        "equilibria_count",
    ],
)
def test_full_source_audit_must_bind_the_exact_selected_dataset_before_fitting(tmp_path: Path, field: str) -> None:
    """A self-consistent real producer audit for different bytes or shot bindings cannot pass dry-run or execute admission."""
    inputs = _format_inputs(tmp_path)
    audit = json.loads(inputs.feature_provenance_report.read_text())
    if field in {"dataset_report_sha256", "dataset_payload_sha256", "dataset_sha256"}:
        audit[field] = "0" * 64
    elif field == "dataset_path":
        audit[field] = "different/dataset.npz"
    elif field == "candidate_report":
        audit[field] = "different/candidate.json"
    elif field == "reference_dataset_id":
        audit[field] = "different-engineering-source"
    elif field == "reference_count":
        audit["shots"].pop()
        audit["reference_count"] -= 1
        for entry in audit["feature_status"].values():
            entry["reference_count"] -= 1
            entry["complete_reference_count"] -= 1
    elif field == "shot_id":
        audit["shots"][0][field] = 42
    elif field == "reference_path":
        audit["shots"][0][field] = "different/reference.npz"
    elif field == "reference_sha256":
        audit["shots"][0][field] = "0" * 64
    else:
        audit["shots"][0][field] += 1
        for key in ("Ip_MA", "Bt_T", "ffprime_rms_T_rad"):
            audit["shots"][0]["shapes"][key] = [audit["shots"][0][field]]
    _reseal(audit)
    write_feature_audit(audit, inputs.feature_provenance_report, inputs.feature_provenance_report.with_suffix(".md"))
    dry_run = build_training_report(inputs)
    source = dry_run["pre_run_admission"]["source_provenance"]
    assert source["status"] == "fail" and any(field in error for error in source["errors"])
    with pytest.raises(ValueError, match="pre-run admission"):
        build_training_report(replace(inputs, execute=True))
    assert not inputs.weights_out.exists()


def test_feature_audit_binds_json_bytes_when_only_whitespace_changes(tmp_path: Path) -> None:
    """Equal decoded metadata with different real JSON bytes requires a new audit capture before fitting."""
    inputs = _format_inputs(tmp_path)
    before = json.loads(inputs.dataset_report.read_text())
    inputs.dataset_report.write_text(json.dumps(before, indent=4) + "\n")
    assert json.loads(inputs.dataset_report.read_text()) == before
    report = build_training_report(inputs)
    assert any("dataset_report_sha256" in error for error in report["pre_run_admission"]["errors"])
    with pytest.raises(ValueError, match="pre-run admission"):
        build_training_report(replace(inputs, execute=True))
    assert not inputs.weights_out.exists()
    audit = build_audit(inputs.dataset_report, inputs.dataset_path.parent)
    write_feature_audit(audit, inputs.feature_provenance_report, inputs.feature_provenance_report.with_suffix(".md"))
    source = build_training_report(inputs)["pre_run_admission"]["source_provenance"]
    assert source["status"] == "fail" and any("dataset_report_sha256" in error for error in source["errors"])
    write_original_audit(
        build_original_feature_source_audit(inputs.dataset_report, inputs.dataset_path.parent),
        inputs.original_source_report,
        inputs.original_source_report.with_suffix(".md"),
    )
    assert build_training_report(inputs)["pre_run_admission"]["source_provenance"]["status"] == "pass"
